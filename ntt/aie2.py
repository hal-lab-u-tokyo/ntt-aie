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
        memRef_ty = T.memref(N//2, T.i32())
        memRef_ty_div4 = T.memref(N // 8, T.i32())

        # AIE Core Function declarations

        # Tile declarations
        _0_ShimTile = tile(0, 0)
        _1_ShimTile = tile(1, 0)
        _0_MemTile = tile(0, 1)
        _1_MemTile = tile(1, 1)
        ComputeTile0 = tile(0, 2)
        ComputeTile1 = tile(0, 3)
        ComputeTile2 = tile(0, 4)
        ComputeTile3 = tile(0, 5)
        ComputeTile4 = tile(1, 2)
        ComputeTile5 = tile(1, 3)
        ComputeTile6 = tile(1, 4)
        ComputeTile7 = tile(1, 5)        
        
        MemTiles = [_0_MemTile, _1_MemTile]
        ComputeTiles = [[ComputeTile0, ComputeTile1, ComputeTile2, ComputeTile3], [ComputeTile4, ComputeTile5, ComputeTile6, ComputeTile7]]
        
        # AIE-array data movement with object fifos
        of_in0 = object_fifo("in0", _0_ShimTile, _0_MemTile, buffer_depth, memRef_ty)
        of_in1 = object_fifo("in1", _1_ShimTile, _1_MemTile, buffer_depth, memRef_ty)
        of_out0 = object_fifo("out0", _0_MemTile, _0_ShimTile, buffer_depth, memRef_ty)
        of_out1 = object_fifo("out1", _1_MemTile, _1_ShimTile, buffer_depth, memRef_ty)
        of_ins_host = [of_in0, of_in1]
        of_outs_host = [of_out0, of_out1]

        of_ins = []
        of_outs = []
        of_ins_core_names = [["in0_0", "in0_1", "in0_2", "in0_3"], ["in1_0", "in1_1", "in1_2", "in1_3"]]
        of_outs_core_names = [["out0_0", "out0_1", "out0_2", "out0_3"], ["out1_0", "out1_1", "out1_2", "out1_3"]]
        for i in range(0, 2):
            of_ins.append([])
            of_outs.append([])
            for j in range(0, 4):
                of_in_ij = object_fifo(of_ins_core_names[i][j], MemTiles[i], ComputeTiles[i][j], buffer_depth, memRef_ty_div4)
                of_out_ij = object_fifo(of_outs_core_names[i][j], ComputeTiles[i][j], MemTiles[i], buffer_depth, memRef_ty_div4)
                of_ins[i].append(of_in_ij)
                of_outs[i].append(of_out_ij)
            object_fifo_link(of_ins_host[i], of_ins[i])
            object_fifo_link(of_outs[i], of_outs_host[i])


        # Compute tile 2
        for column in range(0, 2):
            for row in range(0, 4):
                @core(ComputeTiles[column][row])
                def core_body():
                    # Effective while(1)
                    for _ in for_(2):
                        elem_in0 = of_ins[column][row].acquire(ObjectFifoPort.Consume, 1)
                        elem_out = of_outs[column][row].acquire(ObjectFifoPort.Produce, 1)
                        idx = column * 4 + row
                        for i in for_(N//8):
                            v0 = memref.load(elem_in0, [i])
                            v1 = arith.addi(v0, arith.constant(idx, T.i32()))
                            memref.store(v1, elem_out, [i])
                            yield_([])
                        of_ins[column][row].release(ObjectFifoPort.Consume, 1)
                        of_outs[column][row].release(ObjectFifoPort.Produce, 1)
                        yield_([])
                    
        # To/from AIE-array data movement
        tensor_ty = T.memref(N//2, T.i32())

        @FuncOp.from_py_func(tensor_ty, tensor_ty, tensor_ty, tensor_ty)
        def sequence(A0, A1, C0, C1):
            npu_dma_memcpy_nd(metadata="out0", bd_id=0, mem=C0, sizes=[1, 1, 1, N//2])
            npu_dma_memcpy_nd(metadata="out1", bd_id=1, mem=C1, sizes=[1, 1, 1, N//2])
            npu_dma_memcpy_nd(metadata="in0", bd_id=2, mem=A0, sizes=[1, 1, 1, N//2])
            npu_dma_memcpy_nd(metadata="in1", bd_id=3, mem=A1, sizes=[1, 1, 1, N//2])
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
