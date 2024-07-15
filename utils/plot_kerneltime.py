import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

def plot_ncore_exectime():
    directory_path = "/mnt/c/Technical/ntt-aie/profile/exectime"
    data = {}
    for i in range(5):
        for j in range(9, 14):
            n = 2 ** j
            core = 2 ** i
            fname = f"ntt_{2 ** i}core_logn{j}.csv"
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
    
    for core, n_data in data.items():
        n_values = sorted(n_data.keys())
        execution_times = [n_data[logn] for logn in n_values]
        plt.plot(n_values, execution_times, marker='o', label=f"{core} Tiles")

def plot_16core_kerneltime():
    kerneltime = {}
    directory_path = "/mnt/c/Technical/ntt-aie/profile/trace"
    for j in range(7, 13):
        n = 2 ** j
        fname = f"trace_16core_n{j}.json"
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

            # Raw execution time
            clock = []
            for k in range(len(event1) - 1):
                clock.append(event1[k+1] - event1[k])
            clock_average = sum(clock) / len(clock)
            exectime = clock_average / (1.25 * 1000) # us 
            #print(f"\texectime    :{exectime} us")

            kerneltime[n] = exectime
    print(kerneltime)
    plt.plot(kerneltime.keys(), kerneltime.values(), marker='o', label="16core Kernel Only", color='purple')


fig, ax = plt.subplots(figsize=(10, 6))
#plot_ncore_exectime()
plot_16core_kerneltime()

def format_func(value, tick_number):
    return f'$2^{{{int(math.log2(value))}}}$'

plt.xticks([2 ** i for i in range(7, 13)])
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel('Data size', fontsize=20)
plt.ylabel('Execution Time (us)', fontsize=20)
plt.tick_params(labelsize=16)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("/mnt/c/Technical/ntt-aie/profile/exectime_with_kerneltime.png", dpi=500)