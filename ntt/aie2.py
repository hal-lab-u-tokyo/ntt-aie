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
        
        # lef-right
        of_ct_02_03 = object_fifo("ct_02_03", ComputeTiles[0][0], ComputeTiles[0][1], buffer_depth, memRef_ty_core)
        of_ct_03_04 = object_fifo("ct_03_04", ComputeTiles[0][1], ComputeTiles[0][2], buffer_depth, memRef_ty_core)
        of_ct_04_05 = object_fifo("ct_04_05", ComputeTiles[0][2], ComputeTiles[0][3], buffer_depth, memRef_ty_core)
        of_ct_12_13 = object_fifo("ct_12_13", ComputeTiles[1][0], ComputeTiles[1][1], buffer_depth, memRef_ty_core)
        of_ct_13_14 = object_fifo("ct_13_14", ComputeTiles[1][1], ComputeTiles[1][2], buffer_depth, memRef_ty_core)
        of_ct_14_15 = object_fifo("ct_14_15", ComputeTiles[1][2], ComputeTiles[1][3], buffer_depth, memRef_ty_core)
        of_ct_22_23 = object_fifo("ct_22_23", ComputeTiles[2][0], ComputeTiles[2][1], buffer_depth, memRef_ty_core)
        of_ct_23_24 = object_fifo("ct_23_24", ComputeTiles[2][1], ComputeTiles[2][2], buffer_depth, memRef_ty_core)
        of_ct_24_25 = object_fifo("ct_24_25", ComputeTiles[2][2], ComputeTiles[2][3], buffer_depth, memRef_ty_core)
        of_ct_32_33 = object_fifo("ct_32_33", ComputeTiles[3][0], ComputeTiles[3][1], buffer_depth, memRef_ty_core)
        of_ct_33_34 = object_fifo("ct_33_34", ComputeTiles[3][1], ComputeTiles[3][2], buffer_depth, memRef_ty_core)
        of_ct_34_35 = object_fifo("ct_34_35", ComputeTiles[3][2], ComputeTiles[3][3], buffer_depth, memRef_ty_core)
        # top-bottom
        of_ct_02_12 = object_fifo("ct_02_12", ComputeTiles[0][0], ComputeTiles[1][0], buffer_depth, memRef_ty_core)
        of_ct_03_13 = object_fifo("ct_03_13", ComputeTiles[0][1], ComputeTiles[1][1], buffer_depth, memRef_ty_core)
        of_ct_04_14 = object_fifo("ct_04_14", ComputeTiles[0][2], ComputeTiles[1][2], buffer_depth, memRef_ty_core)
        of_ct_05_15 = object_fifo("ct_05_15", ComputeTiles[0][3], ComputeTiles[1][3], buffer_depth, memRef_ty_core)
        of_ct_12_22 = object_fifo("ct_12_22", ComputeTiles[1][0], ComputeTiles[2][0], buffer_depth, memRef_ty_core)
        of_ct_13_23 = object_fifo("ct_13_23", ComputeTiles[1][1], ComputeTiles[2][1], buffer_depth, memRef_ty_core)
        of_ct_14_24 = object_fifo("ct_14_24", ComputeTiles[1][2], ComputeTiles[2][2], buffer_depth, memRef_ty_core)
        of_ct_15_25 = object_fifo("ct_15_25", ComputeTiles[1][3], ComputeTiles[2][3], buffer_depth, memRef_ty_core)
        of_ct_22_32 = object_fifo("ct_22_32", ComputeTiles[2][0], ComputeTiles[3][0], buffer_depth, memRef_ty_core)
        of_ct_23_33 = object_fifo("ct_23_33", ComputeTiles[2][1], ComputeTiles[3][1], buffer_depth, memRef_ty_core)
        of_ct_24_34 = object_fifo("ct_24_34", ComputeTiles[2][2], ComputeTiles[3][2], buffer_depth, memRef_ty_core)
        of_ct_25_35 = object_fifo("ct_25_35", ComputeTiles[2][3], ComputeTiles[3][3], buffer_depth, memRef_ty_core)
        ntt_left_right = [of_ct_02_03, of_ct_04_05, of_ct_12_13, of_ct_14_15, of_ct_22_23, of_ct_24_25, of_ct_32_33, of_ct_34_35]
        ntt_top_bottom = [of_ct_02_12, of_ct_03_13, of_ct_04_14, of_ct_05_15, of_ct_22_32, of_ct_23_33, of_ct_24_34, of_ct_25_35]
        swap_left_right = [of_ct_03_04, of_ct_13_14, of_ct_23_24, of_ct_33_34]
        swap_top_bottom = [of_ct_12_22, of_ct_13_23, of_ct_14_24, of_ct_15_25]
            
        # Buffer 
        aComputeTile15 = Buffer(ComputeTiles[3][3], [16], T.i32(), "aComputeTile15") 

        # Compute tile 
        for column in range(0, n_column):
            for row in range(0, n_row):
                @core(ComputeTiles[column][row])
                def core_body():
                    # Effective while(1)
                    for _ in for_(2):
                        # Init value
                        core_idx = column * n_row + row
                        if row % 2 == 0:
                            in_from_mem = of_ins_core[column][row].acquire(ObjectFifoPort.Consume, 1)
                            out_to_right = ntt_left_right[column * 2 + row // 2].acquire(ObjectFifoPort.Produce, 1)
                            out_to_mem = of_outs_core[column][row].acquire(ObjectFifoPort.Produce, 1)
                            for i in for_(N//n_array):
	                              v0 = memref.load(in_from_mem, [i])
	                              # NTT within a core
	                              memref.store(v0, out_to_right, [i])
	                              memref.store(v0, out_to_mem, [i])
	                              yield_([])
                            ntt_left_right[column * 2 + row // 2].release(ObjectFifoPort.Produce, 1)
                            of_ins_core[column][row].release(ObjectFifoPort.Consume, 1)
                            of_outs_core[column][row].release(ObjectFifoPort.Produce, 1)
                        else:
                            in_from_mem = of_ins_core[column][row].acquire(ObjectFifoPort.Consume, 1)
                            out_to_mem = of_outs_core[column][row].acquire(ObjectFifoPort.Produce, 1)
                            in_from_left = ntt_left_right[column * 2 + row // 2].acquire(ObjectFifoPort.Consume, 1)
                            for i in for_(N//n_array):
	                              v0 = memref.load(in_from_mem, [i])
	                              # NTT within a core
	                              v1 = memref.load(in_from_left, [i])
	                              v2 = arith.addi(v0, v1)
	                              memref.store(v2, out_to_mem, [i])
	                              yield_([])
                            ntt_left_right[column * 2 + row // 2].release(ObjectFifoPort.Consume, 1)
                            of_ins_core[column][row].release(ObjectFifoPort.Consume, 1)
                            of_outs_core[column][row].release(ObjectFifoPort.Produce, 1)

                        # NTT Left-right
                        # Swap left-right
                        # NTT left-right
                        # NTT top-bottom
                        # Swap top-bottom
                        # NTT top-bottom
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
