# This file includes modifications to original code from the mlir-aie project:
# https://github.com/Xilinx/mlir-aie
# Licensed under the Apache License, Version 2.0.

# ===========================
# VITIS related variables
# ===========================
VITIS_ROOT ?= $(shell realpath $(dir $(shell which vitis))/../)
VITIS_AIETOOLS_DIR ?= ${VITIS_ROOT}/aietools
VITIS_AIE_INCLUDE_DIR ?= ${VITIS_ROOT}/aietools/data/versal_prod/lib
VITIS_AIE2_INCLUDE_DIR ?= ${VITIS_ROOT}/aietools/data/aie_ml/lib

CHESSCC1_FLAGS = -f -p me -P ${VITIS_AIE_INCLUDE_DIR} -I ${VITIS_AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${VITIS_AIE2_INCLUDE_DIR} -I ${VITIS_AIETOOLS_DIR}/include -D__AIENGINE__=2 -D__AIEARCH__=20
CHESS_FLAGS = -P ${VITIS_AIE_INCLUDE_DIR}

CHESSCCWRAP1_FLAGS = aie -I ${VITIS_AIETOOLS_DIR}/include 
CHESSCCWRAP2_FLAGS = aie2 -I ${VITIS_AIETOOLS_DIR}/include 


SRC_DIR := $(shell dirname $(realpath Makefile))/src
BUILD_DIR := build
targetname := ntt

all: ${BUILD_DIR}/final.xclbin ${BUILD_DIR}/insts.txt

${BUILD_DIR}/aie.mlir: ${SRC_DIR}/aie2.py
	mkdir -p ${BUILD_DIR}
	python3 $< > $@
	
${BUILD_DIR}/ntt_core.o: ${SRC_DIR}/aie_core.cc
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

${BUILD_DIR}/final.xclbin: ${BUILD_DIR}/aie.mlir ${BUILD_DIR}/ntt_core.o
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

clean: 
	rm -rf build _build ${targetname}.exe

