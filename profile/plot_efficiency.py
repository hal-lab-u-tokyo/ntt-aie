import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import math


x_gpu = []
x_aie = []
y_gpu = []
y_aie = []

fname = "/mnt/c/Technical/ntt-aie/profile/kerneltime/gpu.csv"
df = pd.read_csv(fname, header=None)
if df.empty:
    print(f"{fname} is empty, skipped")
csv_n = df.iloc[:, 0]
csv_exectime = df.iloc[:, 1]
assert len(csv_n) == len(csv_exectime)
for i in range(len(csv_n)):
    n = csv_n[i]
    us = csv_exectime[i]

    # Cal Efficiency
    flops_ntt = 5.5 * math.log2(n) * n
    flops_real = flops_ntt / (1000 * us) ## GFLOPS
    efficiency = flops_real / 4280

    x_gpu.append(n)
    y_gpu.append(efficiency)

fname = "/mnt/c/Technical/ntt-aie/profile/kerneltime/aie.csv"
df = pd.read_csv(fname, header=None)
if df.empty:
    print(f"{fname} is empty, skipped")
csv_n = df.iloc[:, 0]
csv_exectime = df.iloc[:, 1]
assert len(csv_n) == len(csv_exectime)
for i in range(len(csv_n)):
    n = csv_n[i]
    us = csv_exectime[i]

    # Cal Efficiency
    flops_ntt = 5.5 * math.log2(n) * n
    flops_real = flops_ntt / (1000 * us) ## GFLOPS
    efficiency = flops_real / 88

    x_aie.append(n)
    y_aie.append(efficiency)

print(x_gpu)
print(y_gpu)
print(x_aie)
print(y_aie)

def format_func(value, tick_number):
    return f'$2^{{{int(math.log2(value))}}}$'

fig, ax = plt.subplots(figsize=(10, 6))

plt.plot(x_gpu, y_gpu, marker='o', color='tab:blue', label='NVIDIA A100')
plt.plot(x_aie, y_aie, marker='o', color='tab:green', label='Ryzen AI Engine')
plt.xlabel('Data size', fontsize=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xticks([2 ** i for i in [8, 15, 16, 17]], fontsize=14)
#plt.xscale('log')
plt.yticks(fontsize=18)
plt.ylabel('Efficiency', fontsize=20)

# Right y-axis
#pltr= plt.twinx()
#pltr.plot(x, y_performance, marker='o', color='g')
#pltr.set_ylabel('Performance (GOPS)', fontsize=20, color='g')


plt.tick_params(labelsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("/mnt/c/Technical/ntt-aie/profile/efficiency.png", dpi=500, bbox_inches='tight')
