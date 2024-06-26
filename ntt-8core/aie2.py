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


def ntt():
    # AIETile parameters
    n_column = 2
    n_row = 4
    n_core = n_column * n_row
    buffer_depth = 2

    # NTT parameters
    logN = 11
    N = 1 << logN
    N_in_bytes = N * 4
    p = 3329
    barrett_w = math.ceil(math.log2(p))
    barrett_u = math.floor(pow(2, 2 * barrett_w) / p)    
    
    N_percore = N // n_core
    log2_N_percore = int(math.log2(N_percore))

    @device(AIEDevice.npu1_2col)
    def device_body():
        memRef_ty_vec = T.memref(N, T.i32())
        memRef_ty_column = T.memref(N // n_column, T.i32())
        memRef_ty_core = T.memref(N // n_core, T.i32())
        memRef_ty_core_half = T.memref(N // (n_core * 2), T.i32())
        memRef_ty_scalar = T.memref(1, T.i32())
        
        # AIE Core Function declarations
        # void ntt_stage_0_to_logN(int32_t N_all, int32_t N, int32_t logN, int32_t core_idx, int32_t *in_a, int32_t *root_in, int32_t *out0, int32_t *out1, int32_t p, int32_t w, int32_t u) {
        ntt_stage_0_to_logN = external_func(
            "ntt_stage_0_to_logN",
            inputs=[T.i32(), T.i32(), T.i32(), T.i32(), memRef_ty_core, memRef_ty_vec, memRef_ty_core_half, memRef_ty_core_half, T.i32(), T.i32(), T.i32()],
        )
        
        # void ntt_1stage(int32_t idx_stage, int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
        ntt_1stage = external_func(
            "ntt_1stage",
            inputs=[T.i32(), T.i32(), T.i32(), T.i32(), memRef_ty_core_half, memRef_ty_core_half, memRef_ty_core_half, memRef_ty_core_half, memRef_ty_vec, T.i32(), T.i32(), T.i32()],
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
        of_outs_core = [[] for c in range(n_column)]
        of_outs_names = [f"out{c}" for c in range(n_column)]
        of_outs_core_names = [[f"out{c}_{r}" for r in range(n_row)] for c in range(n_column)]
        for c in range(n_column):
            of_outs.append(object_fifo(of_outs_names[c], MemTiles[c], ShimTiles[c], buffer_depth, memRef_ty_column))
            for r in range(n_row):
                of_outs_core[c].append(object_fifo(of_outs_core_names[c][r], ComputeTiles[c][r], MemTiles[c], buffer_depth, memRef_ty_core))
            object_fifo_link(of_outs_core[c], of_outs[c])
        
        # Local Buff
        of_buffs = [[] for c in range(n_column)]
        of_buffs_names = [[f"buff{c}_{r}" for r in range(n_row)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row):
                of_buffs[c].append(object_fifo(of_buffs_names[c][r], ComputeTiles[c][r], ComputeTiles[c][r], 2, memRef_ty_core_half))
        

        # Link between ComputeTiles
        of_up = [[] for c in range(n_column)]
        of_down = [[] for c in range(n_column)]
        of_up_names = [[f"up{c}_{r}{r+1}" for r in range(0, n_row - 1)] for c in range(n_column)]
        of_down_names = [[f"down{c}_{r+1}{r}" for r in range(0, n_row - 1)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row - 1):
                of_up[c].append(object_fifo(of_up_names[c][r], ComputeTiles[c][r], ComputeTiles[c][r + 1], buffer_depth, memRef_ty_core_half))
                of_down[c].append(object_fifo(of_down_names[c][r], ComputeTiles[c][r + 1], ComputeTiles[c][r], buffer_depth, memRef_ty_core_half))
        of_up2 = [[] for c in range(n_column)]
        of_down2 = [[] for c in range(n_column)]
        of_up2_names = [[f"up2{c}_{r}{r+1}" for r in range(n_row - 1)] for c in range(n_column)]
        of_down2_names = [[f"down2{c}_{r+1}{r}" for r in range(n_row - 1)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row - 1):
                of_up2[c].append(object_fifo(of_up2_names[c][r], ComputeTiles[c][r], ComputeTiles[c][r + 1], buffer_depth, memRef_ty_core_half))
                of_down2[c].append(object_fifo(of_down2_names[c][r], ComputeTiles[c][r + 1], ComputeTiles[c][r], buffer_depth, memRef_ty_core_half))
        
        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTiles[0][0], WireBundle.Trace, 0, ShimTiles[0], WireBundle.DMA, 1)

        # Compute tile 
        for c in range(n_column):
            for r in range(n_row):
                @core(ComputeTiles[c][r], "ntt_core.o")
                def core_body():
                    # Effective while(1)
                    core_idx = n_column * c + r
                    for _ in for_(2):
                        # ============================
                        #    NTT Stage 0 to n-3
                        # ============================
                        # Acquire
                        elem_buff_local = of_buffs[c][r].acquire(ObjectFifoPort.Produce, 1) 
                        elem_out_next = of_up[c][r].acquire(ObjectFifoPort.Produce, 1) if r % 2 == 0 else of_down[c][r-1].acquire(ObjectFifoPort.Produce, 1) 
                        elem_in = of_ins_core[c][r].acquire(ObjectFifoPort.Consume, 1)
                        elem_root = of_inroots_core[c].acquire(ObjectFifoPort.Consume, 1)
                        
                        # Call NTT kernel
                        if r % 2 == 0:
                            call(ntt_stage_0_to_logN, [N, N_percore, log2_N_percore, core_idx, elem_in, elem_root, elem_buff_local, elem_out_next, p, barrett_w, barrett_u])
                        else:
                            call(ntt_stage_0_to_logN, [N, N_percore, log2_N_percore, core_idx, elem_in, elem_root, elem_out_next, elem_buff_local, p, barrett_w, barrett_u])
                        
                        # Release
                        of_ins_core[c][r].release(ObjectFifoPort.Consume, 1)
                        if r % 2 == 0:
                            of_up[c][r].release(ObjectFifoPort.Produce, 1)
                        else:
                            of_down[c][r-1].release(ObjectFifoPort.Produce, 1)

                        # ============================
                        #    NTT Stage n-2
                        # ============================
                        # Acquire
                        # Out
                        # r == 0: *local, of_up2[c][0]
                        # r == 1: *of_down2[c][0], of_up2[c][1], 
                        # r == 2: of_down2[c][1], *of_up2[c][2]
                        # r == 3: of_down2[c][2], *local
                        sw_elem_out0 = {
                            0: of_buffs[c][0],
                            1: of_down2[c][0], # *
                            2: of_down2[c][1],
                            3: of_down2[c][2]
                        }
                        sw_elem_out1 = {
                            0: of_up2[c][0],
                            1: of_up2[c][1],
                            2: of_up2[c][2], # *
                            3: of_buffs[c][3]
                        }
                        elem_out0 = sw_elem_out0.get(r).acquire(ObjectFifoPort.Produce, 1)
                        elem_out1 = sw_elem_out1.get(r).acquire(ObjectFifoPort.Produce, 1)
                        elem_in_next = of_down[c][r].acquire(ObjectFifoPort.Consume, 1) if r % 2 == 0 else of_up[c][r-1].acquire(ObjectFifoPort.Consume, 1) 
                        
                        # Call NTT kernel
                        # void ntt_stage_N_2(int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
                        if r % 2 == 0:
                            call(ntt_1stage, [1, N_percore, core_idx, n_core, elem_out0, elem_out1, elem_buff_local, elem_in_next, elem_root, p, barrett_w, barrett_u])
                        else:
                            call(ntt_1stage, [1, N_percore, core_idx, n_core, elem_out0, elem_out1, elem_in_next, elem_buff_local, elem_root, p, barrett_w, barrett_u])
                        
                        # Release
                        of_inroots_core[c].release(ObjectFifoPort.Consume, 1)
                        if r == 0:
                            sw_elem_out0.get(r).release(ObjectFifoPort.Produce, 1) 
                            sw_elem_out1.get(r).release(ObjectFifoPort.Produce, 1)
                            of_down[c][r].release(ObjectFifoPort.Consume, 1)
                        elif r == 1:
                            sw_elem_out1.get(r).release(ObjectFifoPort.Produce, 1)
                            of_up[c][r-1].release(ObjectFifoPort.Consume, 1)
                        elif r == 2:
                            sw_elem_out0.get(r).release(ObjectFifoPort.Produce, 1) 
                            of_down[c][r].release(ObjectFifoPort.Consume, 1)
                        elif r == 3:
                            sw_elem_out0.get(r).release(ObjectFifoPort.Produce, 1) 
                            sw_elem_out1.get(r).release(ObjectFifoPort.Produce, 1)
                            of_up[c][r-1].release(ObjectFifoPort.Consume, 1)

                        # Transfer
                        # r == 1 of_up2[c][0]->of_up2[c][1]
                        # r == 1 of_down2[c][1]->of_down[c][0]
                        # r == 2 of_up2[c][1]->of_up[c][2]
                        # r == 2 of_down2[c][2]->of_down2[c][1] 
                        if r == 1:
                            elem_in = of_down2[c][1].acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_down[c][0].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(N_percore//2):
                                v0 = memref.load(elem_in, [i])
                                memref.store(v0, elem_out, [i]) 
                                yield_([])
                            of_down2[c][1].release(ObjectFifoPort.Consume, 1)
                            of_down[c][0].release(ObjectFifoPort.Produce, 1)
                            
                            elem_in = of_up2[c][0].acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_up2[c][1].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(N_percore//2):
                                v0 = memref.load(elem_in, [i])
                                memref.store(v0, elem_out, [i]) 
                                yield_([])
                            of_up2[c][0].release(ObjectFifoPort.Consume, 1)
                            of_up2[c][1].release(ObjectFifoPort.Produce, 1)
                        elif r == 2:
                            elem_in = of_up2[c][1].acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_up[c][2].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(N_percore//2):
                                v0 = memref.load(elem_in, [i])
                                memref.store(v0, elem_out, [i]) 
                                yield_([])
                            of_up2[c][1].release(ObjectFifoPort.Consume, 1)
                            of_up[c][2].release(ObjectFifoPort.Produce, 1)

                            elem_in = of_down2[c][2].acquire(ObjectFifoPort.Consume, 1)
                            elem_out = of_down2[c][1].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(N_percore//2):
                                v0 = memref.load(elem_in, [i])
                                memref.store(v0, elem_out, [i]) 
                                yield_([])
                            of_down2[c][2].release(ObjectFifoPort.Consume, 1)
                            of_down2[c][1].release(ObjectFifoPort.Produce, 1)

                        # ============================
                        #    NTT Stage n-1
                        # ============================
                        # Acquire

                        # Call NTT kernel
                        # In
                        # r == 0: *local, of_down[c][0]
                        # r == 1: *of_down2[c][0], of_down2[c][1]
                        # r == 2: of_up2[c][1], *of_up2[c][2]
                        # r == 3: of_up[c][2], *local
                        # Out
                        # r == 0: local, of_up[c][0]
                        # r == 1: of_down[c][0], local
                        # r == 2: local, of_up[c][2]
                        # r == 3: of_down[2], local
                        sw_elem_in0 = {
                            0: of_buffs[c][0],
                            1: of_down2[c][0], #*
                            2: of_up2[c][1],
                            3: of_up[c][2]
                        }
                        sw_elem_in1 = {
                            0: of_down[c][0],
                            1: of_down2[c][1],
                            2: of_up2[c][2], #*
                            3: of_buffs[c][r]
                        }
                        sw_elem_out0 = {
                            0: of_buffs[c][0],
                            1: of_down[c][0],
                            2: of_buffs[c][2],
                            3: of_down[c][2]
                        }
                        sw_elem_out1 = {
                            0: of_up[c][0],
                            1: of_buffs[c][1],
                            2: of_up[c][2],
                            3: of_buffs[c][3]
                        }
                        elem_in0 = sw_elem_in0.get(r).acquire(ObjectFifoPort.Consume, 1) if r != 1 else sw_elem_in0.get(r).acquire(ObjectFifoPort.Produce, 1) 
                        elem_in1 = sw_elem_in1.get(r).acquire(ObjectFifoPort.Consume, 1) if r != 2 else sw_elem_in1.get(r).acquire(ObjectFifoPort.Produce, 1)     
                        elem_out0 = sw_elem_out0.get(r).acquire(ObjectFifoPort.Produce, 1)
                        elem_out1 = sw_elem_out1.get(r).acquire(ObjectFifoPort.Produce, 1)
                        
                        # Call NTT kernel
                        # void ntt_stage_N_1(int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
                        call(ntt_1stage, [0, N_percore, core_idx, n_core, elem_out0, elem_out1, elem_in0, elem_in1, elem_root, p, barrett_w, barrett_u])
                        
                        # Release
                        if r == 0:
                            sw_elem_in1.get(r).release(ObjectFifoPort.Consume, 1) 
                            sw_elem_out1.get(r).release(ObjectFifoPort.Produce, 1)
                        elif r == 1:
                            sw_elem_in0.get(r).release(ObjectFifoPort.Produce, 1)
                            sw_elem_in1.get(r).release(ObjectFifoPort.Consume, 1) 
                            sw_elem_out0.get(r).release(ObjectFifoPort.Produce, 1)
                        elif r == 2:
                            sw_elem_in0.get(r).release(ObjectFifoPort.Consume, 1) 
                            sw_elem_in1.get(r).release(ObjectFifoPort.Produce, 1) 
                            sw_elem_out1.get(r).release(ObjectFifoPort.Produce, 1)
                        elif r == 3:
                            sw_elem_in0.get(r).release(ObjectFifoPort.Consume, 1) 
                            sw_elem_out0.get(r).release(ObjectFifoPort.Produce, 1)

                        # Write Back
                        sw_result0 = {
                            0: of_buffs[c][0],
                            1: of_up[c][0],
                            2: of_buffs[c][2],
                            3: of_up[c][2]
                        }
                        sw_result1 = {
                            0: of_down[c][0],
                            1: of_buffs[c][1],
                            2: of_down[c][2],
                            3: of_buffs[c][3]
                        }
                        if r % 2 == 0:
                            elem_out1 = sw_result1.get(r).acquire(ObjectFifoPort.Consume, 1) 
                        else:
                            elem_out0 = sw_result0.get(r).acquire(ObjectFifoPort.Consume, 1) 
                        elem_out_local = of_outs_core[c][r].acquire(ObjectFifoPort.Produce, 1)
                        for i in for_(N_percore//2):
                            v0 = memref.load(elem_out0, [i])
                            v1 = memref.load(elem_out1, [i])
                            memref.store(v0, elem_out_local, [i])
                            memref.store(v1, elem_out_local, [i + N_percore // 2])
                            yield_([])
                        
                        of_outs_core[c][r].release(ObjectFifoPort.Produce, 1)
                        sw_result0.get(r).release(ObjectFifoPort.Consume, 1) 
                        sw_result1.get(r).release(ObjectFifoPort.Consume, 1) 

                        """

                        # Write Back
                        elem_out_local = of_outs_core[c][r].acquire(ObjectFifoPort.Produce, 1)
                        for i in for_(N_percore//2):
                            v0 = memref.load(elem_in0, [i])
                            v1 = memref.load(elem_in1, [i])
                            memref.store(v0, elem_out_local, [i])
                            memref.store(v1, elem_out_local, [i + N_percore//2])
                            yield_([])
                        
                        # Release
                        of_buffs[c][r].release(ObjectFifoPort.Produce, 1) 
                        of_outs_core[c][r].release(ObjectFifoPort.Produce, 1)
                        sw_elem_in0.get(r).release(ObjectFifoPort.Consume, 1) if r != 1 else sw_elem_in0.get(r).release(ObjectFifoPort.Produce, 1) 
                        sw_elem_in1.get(r).release(ObjectFifoPort.Consume, 1) if r != 2 else sw_elem_in1.get(r).release(ObjectFifoPort.Produce, 1) 
                        """
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

trace_size = 0
with mlir_mod_ctx() as ctx:
    ntt()
    print(ctx.module)
