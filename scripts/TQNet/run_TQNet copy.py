import subprocess
from pathlib import Path

# -----------------------------
# 配置参数
# -----------------------------
model_name = "TQNet"

root_path_name = "./dataset/"

# data_name_list1 = ["ETTh1_missing_0", "ETTh1_missing_5", "ETTh1_missing_10","ETTh1_missing_20"]
data_name_list2 = ["ETTh1_missing_30", "ETTh1_missing_50", "ETTh1_missing_80"]
model_id_name = "5-14-TQNet"

seq_len = 96
pred_lens = [96, 192, 336, 720]
# pred_lens = [96]
random_seeds = [2024]

# 当前文件路径
current_dir = Path.cwd()

# 回退两个目录
run_dir = current_dir.parents[1]  # 父目录的父目录

# -----------------------------
# 循环运行
# -----------------------------
for data_name in data_name_list2:
    data_path_name = f"{data_name}.csv"

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
                "--cycle", "24",
                "--train_epochs", "30",
                "--patience", "5",
                "--dropout", "0.5",
                "--itr", "1",
                "--batch_size", "256",
                "--learning_rate", "0.001",
                "--random_seed", str(random_seed)
            ]

            print(f"Running command in {run_dir}: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=run_dir)