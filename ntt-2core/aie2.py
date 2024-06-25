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
    n_column = 1
    n_row = 2
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

    @device(AIEDevice.npu1_1col)
    def device_body():
        memRef_ty_vec = T.memref(N, T.i32())
        memRef_ty_column = T.memref(N // n_column, T.i32())
        memRef_ty_core = T.memref(N // n_core, T.i32())
        memRef_ty_core_half = T.memref(N // (n_core * 2), T.i32())
        memRef_ty_scalar = T.memref(1, T.i32())
        
        # AIE Core Function declarations
        # void ntt_stage_0_to_N_2(int32_t N_all, int32_t N, int32_t logN, int32_t core_idx, int32_t *a_in, int32_t *root_in, int32_t *c_out0, int32_t *c_out1, int32_t p, int32_t w, int32_t u) {
        ntt_stage_0_to_N_2 = external_func(
            "ntt_stage_0_to_N_2",
            inputs=[T.i32(), T.i32(), T.i32(), T.i32(), memRef_ty_core, memRef_ty_vec, memRef_ty_core_half, memRef_ty_core_half, T.i32(), T.i32(), T.i32()],
        )
        # void ntt_stage_N_1(int32_t N, int32_t *in_a0, int32_t *in_a1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
        ntt_stage_N_1 = external_func(
            "ntt_stage_N_1",
            inputs=[T.i32(), memRef_ty_core_half, memRef_ty_core_half, memRef_ty_vec, T.i32(), T.i32(), T.i32()],
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
        of_outs_core = [[[] for r in range(n_row)] for c in range(n_column)]
        of_outs_names = [f"out{c}" for c in range(n_column)]
        of_outs_core_names = [[[f"out{c}_{r}first", f"out{c}_{r}second"] for r in range(n_row)] for c in range(n_column)]
        for c in range(n_column):
            of_outs.append(object_fifo(of_outs_names[c], MemTiles[c], ShimTiles[c], buffer_depth, memRef_ty_column))
            for r in range(n_row):
                for i in range(2):
                    of_outs_core[c][r].append(object_fifo(of_outs_core_names[c][r][i], ComputeTiles[c][r], MemTiles[c], buffer_depth, memRef_ty_core_half))
            object_fifo_link([item for sublist in of_outs_core[c] for item in sublist], of_outs[c])

        # Link between ComputeTiles
        of_up = [[] for c in range(n_column)]
        of_down = [[] for c in range(n_column)]
        of_up_names = [[f"up{c}_{2*r}{2*r+1}" for r in range(n_row//2)] for c in range(n_column)]
        of_down_names = [[f"down{c}_{2*r+1}{2*r}" for r in range(n_row//2)] for c in range(n_column)]
        for c in range(n_column):
            for r in range(n_row // 2):
                of_up[c].append(object_fifo(of_up_names[c][r], ComputeTiles[c][2 * r], ComputeTiles[c][2 * r + 1], buffer_depth, memRef_ty_core_half))
                of_down[c].append(object_fifo(of_down_names[c][r], ComputeTiles[c][2 * r + 1], ComputeTiles[c][2 * r], buffer_depth, memRef_ty_core_half))
        
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
                        #    NTT Stage 0 to n-2
                        # ============================
                        # Acquire
                        elem_out_local = of_outs_core[c][r][0].acquire(ObjectFifoPort.Produce, 1) if r % 2 == 0 else of_outs_core[c][r][1].acquire(ObjectFifoPort.Produce, 1) 
                        elem_out_next = of_up[c][r//2].acquire(ObjectFifoPort.Produce, 1) if r % 2 == 0 else of_down[c][r//2].acquire(ObjectFifoPort.Produce, 1) 
                        elem_in = of_ins_core[c][r].acquire(ObjectFifoPort.Consume, 1)
                        elem_root = of_inroots_core[c].acquire(ObjectFifoPort.Consume, 1)
                        
                        # Call NTT kernel
                        if r % 2 == 0:
                            call(ntt_stage_0_to_N_2, [N, N_percore, log2_N_percore, core_idx, elem_in, elem_root, elem_out_local, elem_out_next, p, barrett_w, barrett_u])
                        else:
                            call(ntt_stage_0_to_N_2, [N, N_percore, log2_N_percore, core_idx, elem_in, elem_root, elem_out_next, elem_out_local, p, barrett_w, barrett_u])
                        
                        # Release
                        of_ins_core[c][r].release(ObjectFifoPort.Consume, 1)
                        if r % 2 == 0:
                            of_up[c][r//2].release(ObjectFifoPort.Produce, 1)
                        else:
                            of_down[c][r//2].release(ObjectFifoPort.Produce, 1)

                        # ============================
                        #    NTT Stage n-1
                        # ============================
                        # Acquire
                        elem_in_next = of_down[c][r//2].acquire(ObjectFifoPort.Consume, 1) if r % 2 == 0 else of_up[c][r//2].acquire(ObjectFifoPort.Consume, 1) 

                        # Call NTT kernel
                        # void ntt_stage_N_1(int32_t N, int32_t *in_a0, int32_t *in_a1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
                        if r % 2 == 0:
                            call(ntt_stage_N_1, [N_percore, elem_out_local, elem_in_next, elem_root,  p, barrett_w, barrett_u])
                        else:
                            call(ntt_stage_N_1, [N_percore, elem_in_next, elem_out_local, elem_root,  p, barrett_w, barrett_u])
                        
                        # Release
                        of_inroots_core[c].release(ObjectFifoPort.Consume, 1)
                        if r % 2 == 0:
                            of_down[c][r//2].release(ObjectFifoPort.Consume, 1)
                        else:
                            of_up[c][r//2].release(ObjectFifoPort.Consume, 1)

                        # Write Back
                        elem_result_from = of_up[c][r//2].acquire(ObjectFifoPort.Produce, 1) if r % 2 == 0 else of_down[c][r//2].acquire(ObjectFifoPort.Produce, 1) 
                        idx_result_export = 1 if r % 2 == 0 else 0
                        elem_result_write = of_outs_core[c][r][idx_result_export].acquire(ObjectFifoPort.Produce, 1)
                        for i in for_(N//2):
                            v0 = memref.load(elem_result_from, [i])
                            memref.store(v0, elem_result_write, [i])
                            yield_([])
                        
                        # Release
                        of_outs_core[c][r][idx_result_export].release(ObjectFifoPort.Produce, 1)
                        if r % 2 == 0:
                            of_up[c][r//2].release(ObjectFifoPort.Produce, 1)
                        else:
                            of_down[c][r//2].release(ObjectFifoPort.Produce, 1)
                        
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
