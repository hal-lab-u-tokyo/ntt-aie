##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../makefile-common

all: build/final.xclbin build/insts.txt

targetname = vectorScalar
trace_size = 8192

build/aie.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< > $@

build/aie_trace.mlir: ${srcdir}/aie2.py
	mkdir -p ${@D}
	python3 $< ${trace_size} > $@
	
build/ntt_core.o: ${srcdir}/ntt_core.cc
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

host: cpu_ntt.cc
	g++ -o host $<

modmul: cpu_modmul.cc
	g++ -o modmul $<

run: ${targetname}.exe build/final.xclbin build/insts.txt 
	${powershell} ./$< -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

trace-wsl: build/final_trace.xclbin build/insts.txt 

trace-win: 
	.\test.exe -x ..\build\final_trace.xclbin -i ..\build\insts.txt -k MLIR_AIE

trace-export:
	mkdir -p trace
	../utils/parse_trace.py --filename buildMSVS/trace.txt --mlir build/aie_trace.mlir --colshift 1 > trace/trace_vs.json

run_py: build/final.xclbin build/insts.txt
	${powershell} python3 ${srcdir}/test.py -x build/final.xclbin -i build/insts.txt -k MLIR_AIE

clean: 
	rm -rf build _build ${targetname}.exe

