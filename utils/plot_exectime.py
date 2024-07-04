import os
import pandas as pd
import matplotlib.pyplot as plt

directory_path = "/mnt/c/Technical/ntt-aie/profile/exectime"

data = {}

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        parts = filename.split('_')
        core = parts[1]  # e.g., 'core1'
        logn = int(parts[2].replace('logn', '').replace('.csv', ''))  # e.g., '10' (as integer)
        n = 2 ** logn
        
        file_path = os.path.join(directory_path, filename)
        
        print("reading...", file_path)
        df = pd.read_csv(file_path, header=None)
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

plt.figure(figsize=(10, 6))

print(data)
for core, n_data in data.items():
    n_values = sorted(n_data.keys())
    execution_times = [n_data[logn] for logn in n_values]
    plt.plot(n_values, execution_times, marker='o', label=core)

#plt.axhline(y=time_empty_avg, color='darkred', linestyle='--')

plt.xticks(n_values)

plt.xlabel('Data size', fontsize=20)
plt.ylabel('Execution Time (us)', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("/mnt/c/Technical/ntt-aie/profile/exectime.png", dpi=500)