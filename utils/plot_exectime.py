import os
import pandas as pd
import matplotlib.pyplot as plt

# ディレクトリのパスを指定
directory_path = "/mnt/c/Technical/ntt-aie/profile"

# コアごとにデータを保存するための辞書を作成
data = {}

# ディレクトリ内のファイルを読み込む
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        # ファイル名からコア番号とlogn値を抽出
        parts = filename.split('_')
        core = parts[1]  # e.g., 'core1'
        logn = int(parts[2].replace('logn', '').replace('.csv', ''))  # e.g., '10' (as integer)
        n = 2 ** logn
        
        # ファイルのフルパスを作成
        file_path = os.path.join(directory_path, filename)
        
        # CSVファイルを読み込む
        print("reading...", file_path)
        df = pd.read_csv(file_path, header=None)
        if df.empty:
            print(f"{filename} is empty, skipped")
            continue
        
        # 実行時間の平均を計算
        mean_execution_time = df.mean().values[0]
        
        # 辞書にデータを追加
        if core not in data:
            data[core] = {}
        data[core][n] = mean_execution_time

# グラフをプロット
plt.figure(figsize=(10, 6))

# コアごとのデータをプロット
print(data)
for core, n_data in data.items():
    n_values = sorted(n_data.keys())
    execution_times = [n_data[logn] for logn in n_values]
    plt.plot(n_values, execution_times, marker='o', label=core)

plt.xticks(n_values)

# グラフの装飾
plt.xlabel('Data size', fontsize=20)
plt.ylabel('Execution Time (us)', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig("exectime.png", dpi=500)