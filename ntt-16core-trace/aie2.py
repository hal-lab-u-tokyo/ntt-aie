# vector_scalar_mul/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import math
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import memref, arith

import aie.utils.trace as trace_utils


def ntt(trace_size):
    logN = 9
    N = 1 << logN
    N_in_bytes = N * 4
    p = 3329
    barrett_w = math.ceil(math.log2(p))
    barrett_u = math.floor(pow(2, 2 * barrett_w) / p)

    n_column = 4
    n_row = 4
    n_core = n_column * n_row
    data_percolumn = N // n_column
    data_percore = N // n_core
    data_percore_log2 = int(math.log2(data_percore))
    
    buffer_depth = 2

    @device(AIEDevice.npu1_4col)
    def device_body():
        memRef_ty_vec = T.memref(N, T.i32())
        memRef_ty_column = T.memref(data_percolumn, T.i32())
        memRef_ty_core = T.memref(data_percore, T.i32())
        memRef_ty_core_half = T.memref(data_percore // 2, T.i32())
        memRef_ty_scalar = T.memref(1, T.i32())

        # AIE Core Function declarations
        # void ntt_stage0_to_Nminus5(int32_t *a_in, int32_t *root_in, int32_t *c_out0, int32_t *c_out1, int32_t N, int32_t logN, int32_t N_all, int32_t core_idx, int32_t p, int32_t w, int32_t u) {
        ntt_stage0_to_Nminus5 = external_func(
            "ntt_stage0_to_Nminus5",
            inputs=[memRef_ty_core, memRef_ty_vec, memRef_ty_core_half, memRef_ty_core_half, T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32(), T.i32()],
        )
        # void ntt_1stage(int32_t idx_stage, int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
        ntt_1stage = external_func(
            "ntt_1stage",
            inputs=[T.i32(), T.i32(), T.i32(), T.i32(), memRef_ty_core_half, memRef_ty_core_half, memRef_ty_core_half, memRef_ty_core_half, memRef_ty_vec, T.i32(), T.i32(), T.i32()],
        )

        # void swap(int32_t *a, int32_t *b, int32_t N) {
        swap_buff = external_func(
            "swap_buff",
            inputs=[memRef_ty_core_half, memRef_ty_core_half, T.i32()],
        )
        
        # void write_back(int32_t *to, int32_t *a, int32_t *b, int32_t N_ab) {
        write_back = external_func(
            "write_back",
            inputs=[memRef_ty_core, memRef_ty_core_half, memRef_ty_core_half, T.i32()],
        )

        trace_event0 = external_func(
            "trace_event0",
            inputs=[],
        )

        trace_event1 = external_func(
            "trace_event1",
            inputs=[],
        )

        # Tile declarations
        ShimTiles = []
        MemTiles = []
        ComputeTiles = []
        for c in range(n_column):
            ShimTiles.append(tile(c, 0))
            MemTiles.append(tile(c, 1))
            ComputeTiles.append([])
            for r in range(n_row):
                ComputeTiles[c].append(tile(c, r+2))

        # Input Array
        of_ins = []
        of_ins_core = [[] for c in range(n_column)]
        of_ins_names = [f"in{c}" for c in range(n_column)]
        of_ins_core_names = [[f"in{c}_{r}" for r in range(n_row)] for c in range(n_column)]
        for c in range(n_column):
            # Create link ShimTile -> MemTile
            of_ins.append(object_fifo(of_ins_names[c], ShimTiles[c], MemTiles[c], buffer_depth, memRef_ty_column))
            for r in range(n_row):
                # Create link MemTile -> ComputeTile
                of_ins_core[c].append(object_fifo(of_ins_core_names[c][r], MemTiles[c], ComputeTiles[c][r], buffer_depth, memRef_ty_core))
            object_fifo_link(of_ins[c], of_ins_core[c])

        # Input Root
        of_inroots = []
        of_inroots_core = []
        of_inroots_name = [f"inroots{c}" for c in range(n_column)]
        of_inroots_core_names = [f"inroots_core{c}" for c in range(n_column)]
        for c in range(n_column):
            of_inroots.append(object_fifo(of_inroots_name[c], ShimTiles[c], MemTiles[c], buffer_depth, memRef_ty_vec))
            of_inroots_core.append(object_fifo(of_inroots_core_names[c], MemTiles[c], ComputeTiles[c][0:n_row], buffer_depth, memRef_ty_vec))
            object_fifo_link(of_inroots[c], of_inroots_core[c])

        # Output Array
        of_outs = []
        of_outs_names = [f"out{c}" for c in range(n_column)]
        for c in range(n_column):
            of_outs.append(object_fifo(of_outs_names[c], ComputeTiles[c][0], ShimTiles[c], buffer_depth, memRef_ty_column))


        # Buffer
        buffs_a0 = [[] for c in range(n_column)]
        buffs_a1 = [[] for c in range(n_column)]
        buffs_a0_names = [[f"buffa0_{c}_{r}" for r in range(n_row)] for c in range(n_column)]
        buffs_a1_names = [[f"buffa1_{c}_{r}" for r in range(n_row)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row):
                buffs_a0[c].append(Buffer(ComputeTiles[c][r], [data_percore // 2], T.i32(), buffs_a0_names[c][r]))
                buffs_a1[c].append(Buffer(ComputeTiles[c][r], [data_percore // 2], T.i32(), buffs_a1_names[c][r]))
        
        # Lock
        of_lock_up = [[] for c in range(n_column)]
        of_lock_up_names = [[f"lock_up{c}{r}_{c}{r+1}" for r in range(0, n_row - 1)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row - 1):
                of_lock_up[c].append(object_fifo(of_lock_up_names[c][r], ComputeTiles[c][r], ComputeTiles[c][r + 1], 1, memRef_ty_scalar))

        of_lock_down = [[] for c in range(n_column)]
        of_lock_down_names = [[f"lock_down{c}{r+1}_{c}{r}" for r in range(0, n_row - 1)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row - 1):
                of_lock_down[c].append(object_fifo(of_lock_down_names[c][r], ComputeTiles[c][r + 1], ComputeTiles[c][r], 1, memRef_ty_scalar))

        of_lock_left = [[] for r in range(n_row)]
        of_lock_left_names = [[f"lock_left{c+1}{r}_{c}{r}" for c in range(0, n_column - 1)] for r in range(n_row)]
        for r in range(n_row):
            for c in range(n_column - 1):
                of_lock_left[r].append(object_fifo(of_lock_left_names[r][c], ComputeTiles[c+1][r], ComputeTiles[c][r], 1, memRef_ty_scalar))

        of_lock_left_additional = []
        of_lock_left_additional_names = [f"lock_left_addirional{3}{r}_{2}{r}" for r in range(n_row)]
        for r in range(n_row):
            of_lock_left_additional.append(object_fifo(of_lock_left_additional_names[r], ComputeTiles[2][r], ComputeTiles[3][r], 1, memRef_ty_scalar))

        of_lock_left_additional2 = []
        of_lock_left_additional2_names = [f"lock_left_addirional2_{3}{r}_{2}{r}" for r in range(n_row)]
        for r in range(n_row):
            of_lock_left_additional2.append(object_fifo(of_lock_left_additional2_names[r], ComputeTiles[3][r], ComputeTiles[2][r], 1, memRef_ty_scalar))

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            packetflow(0, ComputeTiles[0][0], WireBundle.Trace, 0, ShimTiles[0], WireBundle.DMA, 1, keep_pkt_header=True) # core trace

        # Set up compute tiles
        for c in range(n_column):
            for r in range(n_row):
                @core(ComputeTiles[c][r], "ntt_core.o")
                def core_body():
                    # Effective while(1)
                    core_idx = n_row * c + r
                    for _ in for_(sys.maxsize):
                        call(trace_event0, [])

                        # Number of sub-vector "tile" iterations
                        elem_in = of_ins_core[c][r].acquire(ObjectFifoPort.Consume, 1)
                        elem_root = of_inroots_core[c].acquire(ObjectFifoPort.Consume, 1)
                        call(trace_event1, [])
                        call(trace_event0, [])

                        # ============================
                        #    NTT Stage 0 to n-5
                        # ============================
                        
                        call(ntt_stage0_to_Nminus5, [elem_in, elem_root, buffs_a0[c][r], buffs_a1[c][r], data_percore, data_percore_log2, N, core_idx, p, barrett_w, barrett_u])

                        # ============================
                        #    NTT Stage n-4
                        # ============================
                        # void ntt_1stage(int32_t idx_stage, int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
                        if r % 2 == 0:                        
                            call(ntt_1stage, [3, data_percore, core_idx, n_core, buffs_a0[c][r], buffs_a0[c][r+1], buffs_a0[c][r], buffs_a0[c][r+1], elem_root, p, barrett_w, barrett_u])
                        else:
                            call(ntt_1stage, [3, data_percore, core_idx, n_core, buffs_a1[c][r-1], buffs_a1[c][r], buffs_a1[c][r-1], buffs_a1[c][r], elem_root, p, barrett_w, barrett_u])

                        # ============================
                        #    Swap
                        # ============================
                        if r == 1:                 
                            of_lock_down[c][0].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(data_percore // 2):
                                v0 = memref.load(buffs_a0[c][1], [i])
                                v1 = memref.load(buffs_a0[c][2], [i])
                                memref.store(v1, buffs_a0[c][1], [i])
                                memref.store(v0, buffs_a0[c][2], [i])
                                yield_([]) 
                            of_lock_down[c][0].release(ObjectFifoPort.Produce, 1)
                        elif r == 2:
                            of_lock_up[c][2].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(data_percore // 2):
                                v0 = memref.load(buffs_a1[c][1], [i])
                                v1 = memref.load(buffs_a1[c][2], [i])
                                memref.store(v1, buffs_a1[c][1], [i])
                                memref.store(v0, buffs_a1[c][2], [i])
                                yield_([]) 
                            of_lock_up[c][2].release(ObjectFifoPort.Produce, 1)
                        else:
                            # Dummy
                            for i in for_(16):
                                v0 = memref.load(buffs_a0[c][r], [i])
                                memref.store(v0, buffs_a0[c][r], [i])
                                yield_([]) 
                        
                        # ============================
                        #    NTT Stage n-3
                        # ============================
                        # void ntt_1stage(int32_t idx_stage, int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
                        if r == 0:
                            of_lock_down[c][0].acquire(ObjectFifoPort.Consume, 1) 
                        if r == 3:
                            of_lock_up[c][2].acquire(ObjectFifoPort.Consume, 1) 
                        
                        if r % 2 == 0:                        
                            call(ntt_1stage, [2, data_percore, core_idx, n_core, buffs_a0[c][r], buffs_a0[c][r+1], buffs_a0[c][r], buffs_a0[c][r+1], elem_root, p, barrett_w, barrett_u])
                        else:
                            call(ntt_1stage, [2, data_percore, core_idx, n_core, buffs_a1[c][r-1], buffs_a1[c][r], buffs_a1[c][r-1], buffs_a1[c][r], elem_root, p, barrett_w, barrett_u])

                        if r == 0:
                            of_lock_down[c][0].release(ObjectFifoPort.Consume, 1) 
                        if r == 3:
                            of_lock_up[c][2].release(ObjectFifoPort.Consume, 1) 

                        # ============================
                        #    NTT Stage n-2
                        # ============================
                        if c == 1:      
                            call(ntt_1stage, [1, data_percore, core_idx - n_row, n_core // 2, buffs_a0[c-1][r], buffs_a0[c][r], buffs_a0[c-1][r], buffs_a0[c][r], elem_root, p, barrett_w, barrett_u])
                            call(ntt_1stage, [1, data_percore, core_idx - n_row, n_core // 2, buffs_a1[c-1][r], buffs_a1[c][r], buffs_a1[c-1][r], buffs_a1[c][r], elem_root, p, barrett_w, barrett_u])
                        if c == 3:      
                            of_lock_left[r][2].acquire(ObjectFifoPort.Produce, 1)                  
                            call(ntt_1stage, [1, data_percore, core_idx - n_row * 2, n_core // 2, buffs_a0[c-1][r], buffs_a0[c][r], buffs_a0[c-1][r], buffs_a0[c][r], elem_root, p, barrett_w, barrett_u])
                            call(ntt_1stage, [1, data_percore, core_idx - n_row * 2, n_core // 2, buffs_a1[c-1][r], buffs_a1[c][r], buffs_a1[c-1][r], buffs_a1[c][r], elem_root, p, barrett_w, barrett_u])
                            of_lock_left[r][2].release(ObjectFifoPort.Produce, 1)                  
                        else:
                            # dummy
                            for i in for_(16):
                                v0 = memref.load(buffs_a0[c][r], [i])
                                memref.store(v0, buffs_a0[c][r], [i])
                                yield_([])   

                        # ============================
                        #    Swap
                        # ============================
                        if c == 2:                 
                            of_lock_left[r][1].acquire(ObjectFifoPort.Produce, 1)
                            of_lock_left[r][2].acquire(ObjectFifoPort.Consume, 1)                  
                            of_lock_left_additional[r].acquire(ObjectFifoPort.Produce, 1)                  
                            call(swap_buff, [buffs_a0[1][r], buffs_a0[2][r], data_percore // 2])
                            call(swap_buff, [buffs_a1[1][r], buffs_a1[2][r], data_percore // 2])
                            of_lock_left[r][2].release(ObjectFifoPort.Consume, 1)                  
                            of_lock_left[r][1].release(ObjectFifoPort.Produce, 1)
                            of_lock_left_additional[r].release(ObjectFifoPort.Produce, 1)                  
                        else:
                            # Dummy
                            for i in for_(16):
                                v0 = memref.load(buffs_a0[c][r], [i])
                                memref.store(v0, buffs_a0[c][r], [i])
                                yield_([])

                        # ============================
                        #    NTT Stage n-1
                        # ============================
                        if c == 1:                
                            of_lock_left[r][1].acquire(ObjectFifoPort.Consume, 1)
                            of_lock_left[r][0].acquire(ObjectFifoPort.Produce, 1)
                            call(ntt_1stage, [0, data_percore, core_idx - n_row, n_core // 2, buffs_a0[c-1][r], buffs_a0[c][r], buffs_a0[c-1][r], buffs_a0[c][r], elem_root, p, barrett_w, barrett_u])
                            call(ntt_1stage, [0, data_percore, core_idx - n_row, n_core // 2, buffs_a1[c-1][r], buffs_a1[c][r], buffs_a1[c-1][r], buffs_a1[c][r], elem_root, p, barrett_w, barrett_u])
                            of_lock_left[r][1].release(ObjectFifoPort.Consume, 1)
                            of_lock_left[r][0].release(ObjectFifoPort.Produce, 1)
                        elif c == 3:
                            of_lock_left_additional[r].acquire(ObjectFifoPort.Consume, 1) 
                            of_lock_left_additional2[r].acquire(ObjectFifoPort.Produce, 1) 
                            call(ntt_1stage, [0, data_percore, core_idx - n_row * 2, n_core // 2, buffs_a0[c-1][r], buffs_a0[c][r], buffs_a0[c-1][r], buffs_a0[c][r], elem_root, p, barrett_w, barrett_u])
                            call(ntt_1stage, [0, data_percore, core_idx - n_row * 2, n_core // 2, buffs_a1[c-1][r], buffs_a1[c][r], buffs_a1[c-1][r], buffs_a1[c][r], elem_root, p, barrett_w, barrett_u])
                            of_lock_left_additional[r].release(ObjectFifoPort.Consume, 1) 
                            of_lock_left_additional2[r].release(ObjectFifoPort.Produce, 1)                  
                        else:
                            # dummy
                            for i in for_(16):
                                v0 = memref.load(buffs_a0[c][r], [i])
                                memref.store(v0, buffs_a0[c][r], [i])
                                yield_([])   

                        # ============================
                        #    Write Back
                        # ============================
                        
                        if c == 0:
                            of_lock_left[r][0].acquire(ObjectFifoPort.Consume, 1)  
                        elif c == 2:
                            of_lock_left_additional2[r].acquire(ObjectFifoPort.Consume, 1) 
                        
                        if r == 0:
                            elem_out = of_outs[c].acquire(ObjectFifoPort.Produce, 1)
                            v0 = arith.constant(c, T.i32())
                            memref.store(v0, elem_out, [0])
                            of_outs[c].release(ObjectFifoPort.Produce, 1)
                        
                        if c  == 0:
                            of_lock_left[r][0].release(ObjectFifoPort.Consume, 1)                  
                        elif c == 2:
                            of_lock_left_additional2[r].release(ObjectFifoPort.Consume, 1) 

                        of_ins_core[c][r].release(ObjectFifoPort.Consume, 1)
                        of_inroots_core[c].release(ObjectFifoPort.Consume, 1)
                        call(trace_event1, [])
                        yield_([])

        # To/from AIE-array data movement
        @FuncOp.from_py_func(memRef_ty_vec, memRef_ty_vec, memRef_ty_vec)
        def sequence(input, root, output):
            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTiles[0][0],
                    ShimTiles[0],
                    ddr_id=2,
                    size=trace_size,
                    offset=N_in_bytes,
                )
            
            for c in range(n_column):
                size = N // n_column
                offset = c * size
                npu_dma_memcpy_nd(metadata=of_outs_names[c], bd_id=c, mem=output, sizes=[1, 1, 1, size], offsets=[0, 0, 0, offset])
                npu_dma_memcpy_nd(metadata=of_ins_names[c], bd_id=n_column+c, mem=input, sizes=[1, 1, 1, size], offsets=[0, 0, 0, offset])
                npu_dma_memcpy_nd(metadata=of_inroots_name[c], bd_id=2*n_column+c, mem=root, sizes=[1, 1, 1, N])
            npu_sync(column=0, row=0, direction=0, channel=0)

trace_size = 0 if (len(sys.argv) < 2) else int(sys.argv[1])
with mlir_mod_ctx() as ctx:
    ntt(trace_size)
    print(ctx.module)
