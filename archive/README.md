# Build
## WSL
- setup environments
```
cd /mnt/c/Technical/ntt-aie
source setup_env.sh
sudo setup_net.sh
```
- build
```
cd vector_vector_add
make
```

## Powershell
- ensure 
```
PS C:\Technical> ls


    Directory: C:\Technical


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         5/22/2024  10:46 AM                mlir-aie
d-----         5/26/2024  11:05 PM                ntt-aie
d-----        12/29/2023   3:11 AM                thirdParty
d-----        12/29/2023   3:21 AM                XRT
d-----        12/29/2023   3:29 AM                xrtIPUfromDLL

```
- build
```
cd C:/Technical/ntt-aie
mkdir buildMSVS
cd buildMSVS
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

# Run
```
.\test.exe -x ..\build\final.xclbin -k MLIR_AIE -i ..\build\insts.txt -v 1
```