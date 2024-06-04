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

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_4col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        memRef_ty_column = T.memref(N//n_column, T.i32())
        memRef_ty_core = T.memref(N // n_array, T.i32())

        # AIE Core Function declarations

        # Tile declarations
        _0_ShimTile = tile(0, 0)
        _1_ShimTile = tile(1, 0)
        _2_ShimTile = tile(2, 0)
        _3_ShimTile = tile(3, 0)
        _0_MemTile = tile(0, 1)
        _1_MemTile = tile(1, 1)
        _2_MemTile = tile(2, 1)
        _3_MemTile = tile(3, 1)
        ComputeTile0 = tile(0, 2)
        ComputeTile1 = tile(0, 3)
        ComputeTile2 = tile(0, 4)
        ComputeTile3 = tile(0, 5)
        ComputeTile4 = tile(1, 2)
        ComputeTile5 = tile(1, 3)
        ComputeTile6 = tile(1, 4)
        ComputeTile7 = tile(1, 5)
        ComputeTile8 = tile(2, 2)
        ComputeTile9 = tile(2, 3)
        ComputeTile10 = tile(2, 4)
        ComputeTile11 = tile(2, 5)
        ComputeTile12 = tile(3, 2)
        ComputeTile13 = tile(3, 3)
        ComputeTile14 = tile(3, 4)
        ComputeTile15 = tile(3, 5)        
        ShimTiles = [_0_ShimTile, _1_ShimTile, _2_ShimTile, _3_ShimTile]
        MemTiles = [_0_MemTile, _1_MemTile, _2_MemTile, _3_MemTile]
        ComputeTiles = [[ComputeTile0, ComputeTile1, ComputeTile2, ComputeTile3], [ComputeTile4, ComputeTile5, ComputeTile6, ComputeTile7], [ComputeTile8, ComputeTile9, ComputeTile10, ComputeTile11], [ComputeTile12, ComputeTile13, ComputeTile14, ComputeTile15]]
        
        # AIE-array data movement with object fifos
        of_ins_host = []
        of_outs_host = []
        of_ins_core = []
        of_outs_core = []
        of_ins_host_names = ["in0", "in1", "in2", "in3"]
        of_outs_host_names = ["out0", "out1", "out2", "out3"]
        of_ins_core_names = [["in0_0", "in0_1", "in0_2", "in0_3"], ["in1_0", "in1_1", "in1_2", "in1_3"], ["in2_0", "in2_1", "in2_2", "in2_3"], ["in3_0", "in3_1", "in3_2", "in3_3"]]
        of_outs_core_names = [["out0_0", "out0_1", "out0_2", "out0_3"], ["out1_0", "out1_1", "out1_2", "out1_3"], ["out2_0", "out2_1", "out2_2", "out2_3"], ["out3_0", "out3_1", "out3_2", "out3_3"]]
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


        # Compute tile 
        for column in range(0, n_column):
            for row in range(0, n_row):
                @core(ComputeTiles[column][row])
                def core_body():
                    # Effective while(1)
                    for _ in for_(2):
                        elem_in0 = of_ins_core[column][row].acquire(ObjectFifoPort.Consume, 1)
                        elem_out = of_outs_core[column][row].acquire(ObjectFifoPort.Produce, 1)
                        idx = column * n_row + row
                        for i in for_(N//n_array):
                            v0 = memref.load(elem_in0, [i])
                            v1 = arith.addi(v0, arith.constant(idx, T.i32()))
                            memref.store(v1, elem_out, [i])
                            yield_([])
                        of_ins_core[column][row].release(ObjectFifoPort.Consume, 1)
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
