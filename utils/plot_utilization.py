import os
import pandas as pd
import matplotlib.pyplot as plt

directory_path = "/mnt/c/Technical/ntt-aie/profile/exectime"

data = {}

for i in range(3):
    for j in range(8, 12):
        n = 2 ** j
        fname = f"ntt_{2 ** i}core_logn{j}.csv"
        filepath = os.path.join(directory_path, fname)

        # Read CSV
        print("reading...", filepath)
        df = pd.read_csv(filepath, header=None)
        if df.empty:
            print(f"{fname} is empty, skip")
            continue
        
        # Average of values
        avg_value = df.mean().values[0]

        # Cal Utilization
        flops_actually = 8 * n * j / (avg_value * 0.0001)
        flops_peak = 400000000
        utilization = flops_actually / flops_peak
        print(f"avg:{avg_value}, actualy:{flops_actually}, peak:{flops_peak}, utilization:{utilization}")

        # Save data
        if i not in data:
            data[i] = {}
        data[i][n] = utilization

print(data)

plt.figure(figsize=(10, 6))

print(data)
for core, n_data in data.items():
    label_name = f"{(core)}Core"
    n_values = sorted(n_data.keys())
    execution_times = [n_data[logn] for logn in n_values]
    plt.plot(n_values, execution_times, marker='o', label=label_name)

plt.xticks(n_values)

plt.xlabel('Data size', fontsize=20)
plt.ylabel('Efficiency', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("efficiency.png", dpi=500)
