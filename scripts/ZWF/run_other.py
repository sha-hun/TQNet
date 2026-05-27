import subprocess
from pathlib import Path

# -----------------------------
# 配置参数
# -----------------------------
model_name = "ZWF修改loss版"

root_path_name = "./dataset/"
data_name = "ETTh1_missing_20"
data_path_name = f"{data_name}.csv"
model_id_name = '修改loss版'

seq_len = 96
# pred_lens = [96, 192, 336, 720]
pred_lens = [96]
random_seeds = [2024]

# 当前文件路径
current_dir = Path.cwd()

# 回退两个目录
run_dir = current_dir.parents[1]  # 父目录的父目录

# -----------------------------
# 循环运行
# -----------------------------
for pred_len in pred_lens:
    for random_seed in random_seeds:
        model_id_full = f"{model_id_name}_{seq_len}_{pred_len}"

        cmd = [
            "python", "-u", "run.py",
            "--is_training", "1",
            "--root_path", root_path_name,
            "--data_path", data_path_name,
            "--model_id", model_id_full,
            "--model", model_name,
            "--data", data_name,
            "--features", "M",
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--enc_in", "7",
            "--cycle", "0",
            "--train_epochs", "30",
            "--patience", "5",
            "--dropout", "0.5",
            "--itr", "1",
            "--batch_size", "256",
            "--learning_rate", "0.001",
            "--random_seed", str(random_seed),
            "--num_seasonal_components" , "4",
            "--max_season_length" , "168", # 根据数据的周期性特征设置季节性组件的长度，例如168代表一周的小时级数据
            "--trend_length" , "720",  # 趋势组件的长度，可以根据数据的长期趋势特征设置，例如720代表一个月的小时级数据
        ]

        print(f"Running command in {run_dir}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=run_dir)