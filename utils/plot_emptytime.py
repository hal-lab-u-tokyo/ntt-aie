import os
import pandas as pd
import matplotlib.pyplot as plt
import math

directory_path = "/mnt/c/Technical/ntt-aie/profile/empty"

data = {}

for i in range(4, 5):
    for j in range(8, 13):
        n = 2 ** j
        core = 2 ** i
        fname = f"empty_{2 ** i}core_n{j}.csv"
        filepath = os.path.join(directory_path, fname)

        # Read CSV
        print("reading...", filepath)
        if not os.path.exists(filepath):
            print("skip")
            continue
        df = pd.read_csv(filepath, header=None)
        if df.empty:
            print(f"{filename} is empty, skipped")
            continue
        
        column_data = df.iloc[:, 0]
        filtered_data = column_data[(column_data != column_data.max()) & (column_data != column_data.min())]
        mean_execution_time = filtered_data.mean()
        
        if core not in data:
            data[core] = {}
        data[core][n] = mean_execution_time

# Empty Kernel
filename_empty = "/mnt/c/Technical/ntt-aie/profile/dummy.csv"
df = pd.read_csv(filename_empty, header=None)
if df.empty:
    print(f"{filename_empty} is empty, skipped")
time_empty_avg = df.mean().values[0]
print(time_empty_avg)

print(data)


fig, ax = plt.subplots(figsize=(10, 6))

for core, n_data in data.items():
    n_values = sorted(n_data.keys())
    execution_times = [n_data[logn] for logn in n_values]
    plt.plot(n_values, execution_times, marker='o', label=f"{core} Tiles")

#plt.axhline(y=time_empty_avg, color='darkred', linestyle='--')

def format_func(value, tick_number):
    return f'$2^{{{int(math.log2(value))}}}$'

plt.xticks([2 ** i for i in range(8, 14)], fontsize=16)
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel('Data size', fontsize=20)
plt.ylabel('Execution Time (us)', fontsize=20)
plt.tick_params(labelsize=16)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("/mnt/c/Technical/ntt-aie/profile/exectime.png", dpi=500)