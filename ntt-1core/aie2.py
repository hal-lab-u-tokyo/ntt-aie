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
    logN = 7
    N = 1 << logN
    N_in_bytes = N * 4
    p = 3329
    barrett_w = math.ceil(math.log2(p))
    barrett_u = math.floor(pow(2, 2 * barrett_w) / p)
    
    buffer_depth = 2

    @device(AIEDevice.npu1_1col)
    def device_body():
        memRef_vec = T.memref(N, T.i32())
        memRef_scalar = T.memref(1, T.i32())

        # AIE Core Function declarations
        # void ntt_stage0_to_Nminus5(int32_t *a_in, int32_t *root_in, int32_t *c_out0, int32_t *c_out1, int32_t N, int32_t logN, int32_t p, int32_t w, int32_t u) {
        ntt_stage0_to_Nminus5 = external_func(
            "ntt_stage0_to_Nminus5_1core",
            inputs=[memRef_vec, memRef_vec, memRef_vec, T.i32(), T.i32(), T.i32(), T.i32(), T.i32()],
        )

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)

        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, buffer_depth, memRef_vec)
        of_root = object_fifo("inroot", ShimTile, ComputeTile2, buffer_depth, memRef_vec)
        #of_prime = object_fifo("inprime", ShimTile, ComputeTile2, buffer_depth, memRef_scalar)
        of_out = object_fifo("out", ComputeTile2, ShimTile, buffer_depth, memRef_vec)

        # Set up a circuit-switched flow from core to shim for tracing information
        if trace_size > 0:
            #flow(ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1)
            packetflow(0, ComputeTile2, WireBundle.Trace, 0, ShimTile, WireBundle.DMA, 1, keep_pkt_header=True) # core trace
 
        # Buffer
        buff2 = Buffer(ComputeTile2, [N], T.i32(), "buff2")

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "ntt_core.o")
        def core_body():
            # Effective while(1)
            for _ in for_(sys.maxsize):
                # Number of sub-vector "tile" iterations
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                elem_root = of_root.acquire(ObjectFifoPort.Consume, 1)
                call(ntt_stage0_to_Nminus5, [elem_in, elem_root, elem_out, N, logN, p, barrett_w, barrett_u])
                of_in.release(ObjectFifoPort.Consume, 1)
                of_root.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)
                yield_([])

        # To/from AIE-array data movement
        @FuncOp.from_py_func(memRef_vec, memRef_vec, memRef_vec)
        def sequence(A, root, C):
            if trace_size > 0:
                trace_utils.configure_simple_tracing_aie2(
                    ComputeTile2,
                    ShimTile,
                    ddr_id=2,
                    size=trace_size,
                    offset=N_in_bytes,
                )
            npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=C, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=A, sizes=[1, 1, 1, N])
            npu_dma_memcpy_nd(metadata="inroot", bd_id=2, mem=root, sizes=[1, 1, 1, N])
            npu_sync(column=0, row=0, direction=0, channel=0)

trace_size = 0 if (len(sys.argv) < 2) else int(sys.argv[1])
with mlir_mod_ctx() as ctx:
    ntt(trace_size)
    print(ctx.module)
