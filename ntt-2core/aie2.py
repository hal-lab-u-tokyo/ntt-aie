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
    logN = 8
    N = 1 << logN
    N_in_bytes = N * 4
    p = 3329
    barrett_w = math.ceil(math.log2(p))
    barrett_u = math.floor(pow(2, 2 * barrett_w) / p)
    n_column = 1
    n_row = 2
    n_core = n_column * n_row
    
    buffer_depth = 2

    @device(AIEDevice.npu1_1col)
    def device_body():
        memRef_ty_vec = T.memref(N, T.i32())
        memRef_ty_column = T.memref(N // n_column, T.i32())
        memRef_ty_core = T.memref(N // n_core, T.i32())
        memRef_ty_scalar = T.memref(1, T.i32())

        
        # AIE Core Function declarations
        ntt_stage0_to_Nminus5 = external_func(
            "ntt_stage0_to_Nminus5",
            inputs=[T.i32(), memRef_ty_vec, memRef_ty_vec, memRef_ty_vec, T.i32(), T.i32(), T.i32(), T.i32(), T.i32()],
        )

        # Tile declarations
        ShimTiles = []
        MemTiles = []
        ComputeTiles = []
        for c in range(n_column):
            ShimTiles.append(tile(c, 0))
            MemTiles.append(tile(c, 1))
            ComputeTiles.append([])
            for r in range(0, n_row):
                ComputeTiles[c].append(tile(c, r+2))
        
        # AIE-array data movement with object fifos
        of_ins_host = []
        of_outs_host = []
        of_ins_core = []
        of_outs_core = []
        of_ins_host_names = [f"in{i}" for i in range(n_column)]
        of_outs_host_names = [f"out{i}" for i in range(n_column)]
        of_ins_core_names = [[f"in{i}_{j}" for j in range(n_row)] for i in range(n_column)]
        of_outs_core_names = [[f"out{i}_{j}" for j in range(n_row)] for i in range(n_column)]
        for i in range(n_column):
            of_ins_core.append([])
            of_outs_core.append([])
            of_in_i_host = object_fifo(of_ins_host_names[i], ShimTiles[i], MemTiles[i], buffer_depth, memRef_ty_column)
            of_out_i_host = object_fifo(of_outs_host_names[i], MemTiles[i], ShimTiles[i], buffer_depth, memRef_ty_column)
            of_ins_host.append(of_in_i_host)
            of_outs_host.append(of_out_i_host)
            for j in range(n_row):
                of_in_ij = object_fifo(of_ins_core_names[i][j], MemTiles[i], ComputeTiles[i][j], buffer_depth, memRef_ty_core)
                of_out_ij = object_fifo(of_outs_core_names[i][j], ComputeTiles[i][j], MemTiles[i], buffer_depth, memRef_ty_core)
                of_ins_core[i].append(of_in_ij)
                of_outs_core[i].append(of_out_ij)
            object_fifo_link(of_ins_host[i], of_ins_core[i])
            object_fifo_link(of_outs_core[i], of_outs_host[i])
        
        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            flow(ComputeTile[0][0], WireBundle.Trace, 0, ShimTile[0], WireBundle.DMA, 1)

        # Compute tile 
        for column in range(n_column):
            for row in range(n_row):
                @core(ComputeTiles[column][row])
                def core_body():
                    # Effective while(1)
                    for _ in for_(2):
                        elem_out = of_outs_core[column][row].acquire(ObjectFifoPort.Produce, 1)
                        elem_in = of_ins_core[column][row].acquire(ObjectFifoPort.Consume, 1)
                        for i in for_(N // n_core):
                            v0 = memref.load(elem_in, [i])
                            v1 = arith.addi(v0, v0)
                            memref.store(v1, elem_out, [i])
                            yield_([])
                        of_ins_core[column][row].release(ObjectFifoPort.Consume, 1)
                        of_outs_core[column][row].release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    
        # To/from AIE-array data movement
        @FuncOp.from_py_func(memRef_ty_vec, memRef_ty_vec, memRef_ty_vec)
        def sequence(A, root, C):
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
                npu_dma_memcpy_nd(metadata=of_outs_host_names[c], bd_id=c, mem=C, sizes=[1, 1, 1, size], offsets=[0, 0, 0, offset])
                npu_dma_memcpy_nd(metadata=of_ins_host_names[c], bd_id=n_column+c, mem=A, sizes=[1, 1, 1, size], offsets=[0, 0, 0, offset])
            npu_sync(column=0, row=0, direction=0, channel=0)

trace_size = 0
with mlir_mod_ctx() as ctx:
    ntt()
    print(ctx.module)
