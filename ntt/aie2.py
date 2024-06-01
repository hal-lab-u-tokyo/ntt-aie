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
    n = 16
    N_div_n = N // n

    buffer_depth = 2

    if len(sys.argv) != 3:
        raise ValueError("[ERROR] Need 2 command line arguments (Device name, Col)")

    if sys.argv[1] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[1] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))

    @device(dev)
    def device_body():
        memRef_ty = T.memref(n, T.i32())

        # AIE Core Function declarations

        # Tile declarations
        _0_ShimTile = tile(0, 0)
        _0_ComputeTile2 = tile(0, 2)
        
        #_1_ShimTile = tile(1, 0)
        #_1_ComputeTile2 = tile(1, 2)
        
        
        # AIE-array data movement with object fifos
        of_in0 = object_fifo("in0", _0_ShimTile, _0_ComputeTile2, buffer_depth, memRef_ty)
        of_out = object_fifo("out", _0_ComputeTile2, _0_ShimTile, buffer_depth, memRef_ty)

        # Set up compute tiles

        # Compute tile 2
        @core(_0_ComputeTile2)
        def core_body():
            # Effective while(1)
            for _ in for_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                for _ in for_(N_div_n):
                    elem_in0 = of_in0.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    for i in for_(n):
                        v0 = memref.load(elem_in0, [i])
                        v2 = arith.addi(v0, v0)
                        memref.store(v2, elem_out, [i])
                        yield_([])
                    of_in0.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                yield_([])

        # To/from AIE-array data movement
        tensor_ty = T.memref(N, T.i32())

        @FuncOp.from_py_func(tensor_ty, tensor_ty)
        def sequence(A, C):
            npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="in0", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_sync(column=0, row=0, direction=0, channel=0)


with mlir_mod_ctx() as ctx:
    my_vector_add()
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
