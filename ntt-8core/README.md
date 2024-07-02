# How to build and run
## Build on WSL
```
make
```

## Build and Run on Powershell
```
mkdir buildMSVS
cd buildMSVS
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

Run!
```
.\test.exe -x ..\build\final.xclbin -i ..\build\insts.txt -k MLIR_AIE
```

# How to trace
## WSL
```
make trace-wsl
```

## Powershell
```
mkdir buildMSVS
cd buildMSVS
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

Run!
```
.\test.exe -x ..\build\final_trace.xclbin -i ..\build\insts.txt -k MLIR_AIE
```
## Export Result on WSL
```
make trace-export
```
