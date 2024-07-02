import json
import os
import pandas as pd
import matplotlib.pyplot as plt

directory_path = "/mnt/c/Technical/ntt-aie/profile/trace"

data = {}

for i in range(3):
    for j in range(7, 12):
        n = 2 ** j
        fname = f"trace_{2 ** i}core_n{j}.json"
        filepath = os.path.join(directory_path, fname)

        # Read CSV
        print("reading...", filepath)
        if not os.path.exists(filepath):
            print("skip")
            continue
        with open(filepath) as f:
            js = json.load(f)
            
            event0 = []
            event1 = []
            for entry in js:
                name = entry['name']
                phase = entry['ph']
                if name == 'Event0' and phase == 'E':
                    event0.append(entry['ts'])
                if name == 'Event1' and phase == 'E':
                    event1.append(entry['ts'])
            event0 = event0[1:]

            # Raw execution time
            clock = []
            datalen = min(len(event0), len(event1))
            for k in range(datalen):
                clock.append(event0[k] - event1[k])
            clock_average = sum(clock) / len(clock)
            exectime = clock_average / (1.25 * 1000)

            # Cal Utilization
            flops_actually = 8 * n * j / (exectime * 0.0001)
            flops_peak = (2 ** i) * 20000000 # N * 20GHz
            utilization = flops_actually / flops_peak
            print(f"exectime:{exectime}")
            print(f"actualy:{flops_actually}")
            print(f"peak:{flops_peak}")
            print(f"utilization:{utilization}")

        
            if i not in data:
                data[i] = {}
            data[i][n] = utilization

        """
        # Average of values filtering max and min
        column_data = df.iloc[:, 0]
        filtered_data = column_data[(column_data != column_data.max()) & (column_data != column_data.min())]
        avg_value = filtered_data.mean()

        # Cal Utilization
        

        # Save data
        if i not in data:
            data[i] = {}
        data[i][n] = utilization
        """

print(data)
"""
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
plt.savefig("/mnt/c/Technical/ntt-aie/profile/efficiency.png", dpi=500)
"""