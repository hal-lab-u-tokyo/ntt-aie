# vector_vector_add/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import memref, arith

import sys


def my_vector_add():
    N = 256
    n_column = 4
    n_row = 4
    n_array = n_column * n_row
    ndata_array = N // n_array
    ndata_array_half = ndata_array // 2

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_4col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(AIEDevice.npu1_4col)
    def device_body():
        memRef_ty_column = T.memref(N//n_column, T.i32())
        memRef_ty_core = T.memref(ndata_array, T.i32())
        memRef_ty_core_half = T.memref(ndata_array_half, T.i32())
        
        # AIE Core Function declarations

        # Tile declarations
        ShimTiles = []
        MemTiles = []
        ComputeTiles = []
        for c in range(0, n_column):
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
        for i in range(0, n_column):
            of_ins_core.append([])
            of_outs_core.append([])
            of_in_i_host = object_fifo(of_ins_host_names[i], ShimTiles[i], MemTiles[i], buffer_depth, memRef_ty_column)
            of_out_i_host = object_fifo(of_outs_host_names[i], MemTiles[i], ShimTiles[i], buffer_depth, memRef_ty_column)
            of_ins_host.append(of_in_i_host)
            of_outs_host.append(of_out_i_host)
            for j in range(0, n_row):
                of_in_ij = object_fifo(of_ins_core_names[i][j], MemTiles[i], ComputeTiles[i][j], buffer_depth, memRef_ty_core)
                of_out_ij = object_fifo(of_outs_core_names[i][j], ComputeTiles[i][j], MemTiles[i], buffer_depth, memRef_ty_core)
                of_ins_core[i].append(of_in_ij)
                of_outs_core[i].append(of_out_ij)
            object_fifo_link(of_ins_host[i], of_ins_core[i])
            object_fifo_link(of_outs_core[i], of_outs_host[i])
        
        # lef-right (row)
        ntt_left_right = []
        ntt_right_left = []
        for i in range(0, n_column):
            ntt_left_right.append([])
            ntt_right_left.append([])
            for j in range(0, n_row - 1):
                ntt_left_right[i].append(object_fifo(f"ct_{i}{j + 2}_{i}{j + 3}", ComputeTiles[i][j], ComputeTiles[i][j + 1], buffer_depth, memRef_ty_core_half))
                ntt_right_left[i].append(object_fifo(f"ct_{i}{j + 3}_{i}{j + 2}", ComputeTiles[i][j + 1], ComputeTiles[i][j], buffer_depth, memRef_ty_core_half))
                
        # top-bottom
        ntt_top_bottom = []
        ntt_bottom_top = []
        for i in range(0, n_column - 1):
            ntt_top_bottom.append([])
            ntt_bottom_top.append([])
            for j in range(0, n_row):
                ntt_top_bottom[i].append(object_fifo(f"ct_{i}{j + 2}_{i + 1}{j + 2}", ComputeTiles[i][j], ComputeTiles[i + 1][j], buffer_depth, memRef_ty_core_half))
                ntt_bottom_top[i].append(object_fifo(f"ct_{i + 1}{j + 2}_{i}{j + 2}", ComputeTiles[i + 1][j], ComputeTiles[i][j], buffer_depth, memRef_ty_core_half))

        # Buffer
        buffs = []
        for i in range(0, n_column):
            buffs.append([])
            for j in range(0, n_row):
                buffs[i].append(Buffer(ComputeTiles[i][j], [N], T.i32(), f"buffComputeTile{i}{j+2}"))
        
        # Lock
        lock_ct0 = lock(ComputeTiles[0][0], lock_id = 0, init = 1)
        lock_ct0_cons = lock(ComputeTiles[0][0], lock_id = 1, init = 0)

        # Compute tile 
        for column in range(0, n_column):
            for row in range(0, n_row):
                @core(ComputeTiles[column][row])
                def core_body():
                    # Effective while(1)
                    for _ in for_(2):
                        core_idx = column * n_row + row
                        if row % 2 == 0:
                            # Stage 0 - (n-5) (inside core)
                            in_from_mem = of_ins_core[column][row].acquire(ObjectFifoPort.Consume, 1)
                            for i in for_(ndata_array):
                                # NTT within a core
                                v0 = memref.load(in_from_mem, [i])
                                memref.store(v0, buffs[column][row], [i])
                                yield_([])
                            of_ins_core[column][row].release(ObjectFifoPort.Consume, 1)
                            
                            # Stage n-4
                            for i in for_(ndata_array_half):
                                v0 = memref.load(buffs[column][row], [i])
                                v1 = memref.load(buffs[column][row + 1], [i])
                                # NTT with right
                                memref.store(v0, buffs[column][row], [i])
                                memref.store(v1, buffs[column][row + 1], [i])
                                yield_([])
                            
                            # 1st Swap
                            if row == 2:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column][row - 1], [i])
                                    memref.store(v0, buffs[column][row], [i])
                                    yield_([])

                            # Stage n-3
                            for i in for_(ndata_array_half):
                                v0 = memref.load(buffs[column][row], [i])
                                v1 = memref.load(buffs[column][row + 1], [i])
                                # NTT with right
                                memref.store(v0, buffs[column][row], [i])
                                memref.store(v1, buffs[column][row + 1], [i])
                                yield_([])
                            
                            # Stage n-2
                            if column % 2 == 1:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column][row], [i])
                                    v1 = memref.load(buffs[column - 1][row], [i])
                                    # NTT with right
                                    memref.store(v0, buffs[column][row], [i])
                                    memref.store(v1, buffs[column - 1][row], [i])
                                    yield_([])
                            
                            # 2nd Swap
                            if column == 2:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column - 1][row], [i])
                                    v1 = memref.load(buffs[column][row], [i])
                                    memref.store(v1, buffs[column - 1][row], [i])
                                    memref.store(v0, buffs[column][row], [i])
                                    yield_([])

                            # Stage n-1
                            if column % 2 == 1:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column][row], [i])
                                    v1 = memref.load(buffs[column - 1][row], [i])
                                    # NTT with right
                                    memref.store(v0, buffs[column][row], [i])
                                    memref.store(v1, buffs[column - 1][row], [i])
                                    yield_([])


                        else:
                            # Stage 0 - (n-5) (inside core)
                            in_from_mem = of_ins_core[column][row].acquire(ObjectFifoPort.Consume, 1)
                            for i in for_(ndata_array):
                                # NTT within a core
	                            v0 = memref.load(in_from_mem, [i])
                                # Write to left
	                            memref.store(v0, buffs[column][row], [i])
	                            yield_([])
                            of_ins_core[column][row].release(ObjectFifoPort.Consume, 1)
                            
                            # Stage n-4
                            for i in for_(ndata_array_half):
                                v0 = memref.load(buffs[column][row - 1], [i + ndata_array_half])
                                v1 = memref.load(buffs[column][row], [i + ndata_array_half])
	                            # NTT with left
                                memref.store(v0, buffs[column][row - 1], [i + ndata_array_half])
                                memref.store(v1, buffs[column][row], [i + ndata_array_half])
                                yield_([])

                            # 1st Swap
                            if row == 1:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column][row + 1], [i])
                                    memref.store(v0, buffs[column][row], [i])
                                    yield_([])

                            # Stage n-3
                            for i in for_(ndata_array_half):
                                v0 = memref.load(buffs[column][row - 1], [i + ndata_array_half])
                                v1 = memref.load(buffs[column][row], [i + ndata_array_half])
	                            # NTT with left
                                memref.store(v0, buffs[column][row - 1], [i + ndata_array_half])
                                memref.store(v1, buffs[column][row], [i + ndata_array_half])
                                yield_([])

                            # Stage n-2
                            if column % 2 == 1:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column - 1][row], [i])
                                    v1 = memref.load(buffs[column][row], [i])
                                    # NTT with right
                                    memref.store(v0, buffs[column - 1][row], [i])
                                    memref.store(v1, buffs[column][row], [i])
                                    yield_([])

                            # 2nd Swap
                            if column == 2:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column - 1][row], [i])
                                    v1 = memref.load(buffs[column][row], [i])
                                    memref.store(v1, buffs[column - 1][row], [i])
                                    memref.store(v0, buffs[column][row], [i])
                                    yield_([])

                            # Stage n-1
                            if column % 2 == 1:
                                for i in for_(ndata_array):
                                    v0 = memref.load(buffs[column - 1][row], [i])
                                    v1 = memref.load(buffs[column][row], [i])
                                    # NTT with right
                                    memref.store(v0, buffs[column - 1][row], [i])
                                    memref.store(v1, buffs[column][row], [i])
                                    yield_([])

                        # Write back
                        out_to_mem = of_outs_core[column][row].acquire(ObjectFifoPort.Produce, 1)
                        for i in for_(ndata_array):
                            v0 = memref.load(buffs[column][row], [i])
                            memref.store(v0, out_to_mem, [i])
                            yield_([])
                        of_outs_core[column][row].release(ObjectFifoPort.Produce, 1)

                        yield_([])
                    
        # To/from AIE-array data movement
        tensor_ty_N = T.memref(N, T.i32())

        @FuncOp.from_py_func(tensor_ty_N, tensor_ty_N)
        def sequence(A, C):
            for c in range(0, n_column):
                size = N // n_column
                offset = c * size
                npu_dma_memcpy_nd(metadata=of_outs_host_names[c], bd_id=c, mem=C, sizes=[1, 1, 1, size], offsets=[0, 0, 0, offset])
                npu_dma_memcpy_nd(metadata=of_ins_host_names[c], bd_id=n_column+c, mem=A, sizes=[1, 1, 1, size], offsets=[0, 0, 0, offset])
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
