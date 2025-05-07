# NTT on AMD AI Engine

This repository contains the implementation of NTT (Number Theoretic Transform) algorithm on AMD AI Engine. 

## Prerequisites
Setup for the AMD AI Engine development environment is required.
Please refer to [Getting Started for AMD Ryzen™ AI on Linux](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux).

## Build and Run
### Windows + WSL2
In WSL2,
```
# move to the directory mounted to C: drive
cd /mnt/c/
git clone git@github.com:hal-lab-u-tokyo/ntt-aie.git
cd ntt-aie
make clean
# Build device code
make
```

In Windows,
```
cd c:\ntt-aie
mkdir buildMSVS
# Build host code
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
# Run
.\test.exe -x ..\build\final.xclbin -k MLIR_AIE -i ..\build\insts.txt -v 1
```

### Ubuntu
TODO

## License
This project is licensed under the Apache License 2.0.
It includes modified source code from the [mlir-aie project](https://github.com/Xilinx/mlir-aie) by Xilinx.  
Please refer to the [LICENSE](./LICENSE) for details.

## Reference
```
@INPROCEEDINGS{nozaki-mcsoc24,
  author={Nozaki, Ai and Kojima, Takuya and Nakamura, Hiroshi and Takase, Hideki},
  title={A Study on Number Theoretic Transform Acceleration on AMD AI Engine}, 
  booktitle={2024 IEEE 17th International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC)}, 
  year={2024},
  pages={325-331},
  doi={10.1109/MCSoC64144.2024.00060}}
```