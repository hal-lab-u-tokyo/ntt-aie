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
        memRef_ty = T.memref(N, T.i32())
        memRef_ty_div4 = T.memref(N // 4, T.i32())

        # AIE Core Function declarations

        # Tile declarations
        _0_ShimTile = tile(0, 0)
        _0_MemTile = tile(0, 1)
        ComputeTile0 = tile(0, 2)
        ComputeTile1 = tile(0, 3)
        ComputeTile2 = tile(0, 4)
        ComputeTile3 = tile(0, 5)        
        
        ComputeTiles = [ComputeTile0, ComputeTile1, ComputeTile2, ComputeTile3]
        
        # AIE-array data movement with object fifos
        of_in0 = object_fifo("in0", _0_ShimTile, _0_MemTile, buffer_depth, memRef_ty)
        of_in0_0 = object_fifo("in0_0", _0_MemTile, ComputeTile0, buffer_depth, memRef_ty_div4)
        of_in0_1 = object_fifo("in0_1", _0_MemTile, ComputeTile1, buffer_depth, memRef_ty_div4)
        of_in0_2 = object_fifo("in0_2", _0_MemTile, ComputeTile2, buffer_depth, memRef_ty_div4)
        of_in0_3 = object_fifo("in0_3", _0_MemTile, ComputeTile3, buffer_depth, memRef_ty_div4)
        object_fifo_link(of_in0, [of_in0_0, of_in0_1, of_in0_2, of_in0_3])

        of_out0 = object_fifo("out0", _0_MemTile, _0_ShimTile, buffer_depth, memRef_ty)
        of_out0_0 = object_fifo("out0_0", ComputeTile0, _0_MemTile, buffer_depth, memRef_ty_div4)
        of_out0_1 = object_fifo("out0_1", ComputeTile1, _0_MemTile, buffer_depth, memRef_ty_div4)
        of_out0_2 = object_fifo("out0_2", ComputeTile2, _0_MemTile, buffer_depth, memRef_ty_div4)
        of_out0_3 = object_fifo("out0_3", ComputeTile3, _0_MemTile, buffer_depth, memRef_ty_div4)
        object_fifo_link([of_out0_0, of_out0_1, of_out0_2, of_out0_3], of_out0)

        of_ins = [of_in0_0, of_in0_1, of_in0_2, of_in0_3]
        of_outs = [of_out0_0, of_out0_1, of_out0_2, of_out0_3]

        # Set up compute tiles

        # Compute tile 2
        for ct in range(0, 4):
            @core(ComputeTiles[ct])
            def core_body():
                # Effective while(1)
                for _ in for_(2):
                    elem_in0 = of_ins[ct].acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_outs[ct].acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(N//4):
                        v0 = memref.load(elem_in0, [i])
                        v1 = arith.addi(v0, arith.constant(ct, T.i32()))
                        memref.store(v1, elem_out, [i])
                        yield_([])
                    of_ins[ct].release(ObjectFifoPort.Consume, 1)
                    of_outs[ct].release(ObjectFifoPort.Produce, 1)
                    yield_([])
                    
        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.i32())

        @FuncOp.from_py_func(tensor_ty, tensor_ty)
        def sequence(A, C):
            npu_dma_memcpy_nd(metadata="out0", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="in0", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
