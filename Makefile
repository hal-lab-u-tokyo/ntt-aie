##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/src
LOGN = 8

# VITIS related variables
VITIS_ROOT ?= $(shell realpath $(dir $(shell which vitis))/../)
VITIS_AIETOOLS_DIR ?= ${VITIS_ROOT}/aietools
VITIS_AIE_INCLUDE_DIR ?= ${VITIS_ROOT}/aietools/data/versal_prod/lib
VITIS_AIE2_INCLUDE_DIR ?= ${VITIS_ROOT}/aietools/data/aie_ml/lib

CHESSCC1_FLAGS = -f -p me -P ${VITIS_AIE_INCLUDE_DIR} -I ${VITIS_AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${VITIS_AIE2_INCLUDE_DIR} -I ${VITIS_AIETOOLS_DIR}/include -D__AIENGINE__=2 -D__AIEARCH__=20
CHESS_FLAGS = -P ${VITIS_AIE_INCLUDE_DIR}

CHESSCCWRAP1_FLAGS = aie -I ${VITIS_AIETOOLS_DIR}/include 
CHESSCCWRAP2_FLAGS = aie2 -I ${VITIS_AIETOOLS_DIR}/include 

TEST_POWERSHELL := $(shell command -v powershell.exe >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(TEST_POWERSHELL),yes)
	powershell = powershell.exe
else
	powershell = 
endif

all: build/final.xclbin build/insts.txt

targetname = vectorScalar
trace_size = 32768

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< > $@

build/aie_trace.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${trace_size} > $@
	
build/ntt_core.o: ${srcdir}/aie_core.cc
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -c $< -o ${@F}

build/final.xclbin: build/aie.mlir build/ntt_core.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

build/final_trace.xclbin: build/aie_trace.mlir build/ntt_core.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-cdo --no-compile-host --xclbin-name=${@F} \
				--aie-generate-npu --npu-insts-name=insts.txt $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif 

run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

trace-wsl: build/final_trace.xclbin build/insts.txt 

trace-win: 
	.\test.exe -x ..\build\final_trace.xclbin -i ..\build\insts.txt -k MLIR_AIE

trace-export:
	mkdir -p trace
	${srcdir}/../utils/parse_trace.py --filename buildMSVS/trace.txt --mlir build/aie_trace.mlir --colshift 1 > ../profile/trace/trace_1core_n${LOGN}.json

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean: 
	rm -rf build _build ${targetname}.exe

