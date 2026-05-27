import subprocess
from pathlib import Path


# -----------------------------
# 1. 核心运行配置
# -----------------------------
model_name = "TQNet"
model_id_name = "5-16-TQNet"
root_path_name = "./dataset/"

base_datasets = [ "exchange_rate", "national_illness", "weather","traffic"]  #"ETTh2","ETTm1","ETTm2","traffic", [] # 如果要跑多个，可以写成 ["ETTh1", "ETTm1", "traffic"]
missing_rates = [0, 5, 10, 20, 30, 50, 80] 

seq_len = 96
pred_lens = [96, 192, 336, 720]
random_seeds = [2026]

# 💡 特有参数字典 (ZWF 模型特有的参数，会智能覆盖基础参数)
special_args = {
    "num_seasonal_components": 4,
    "max_season_length": 24, # 代表一周的小时级数据等
    "trend_length": 720,     # 代表一个月的小时级数据等
    "hidden_dim": 64,
    "hidden_dim_mask": 64,
    "use_miss": "True",     # 这个参数在 run.py 中是 action='store_true' 的形式，所以只要写了这个参数就是 True，不写就是 False,与赋予值无关
    "d_model": 1024
}

# 当前文件路径与运行目录
current_dir = Path.cwd()
run_dir = current_dir.parents[1]  # 父目录的父目录

# -----------------------------
# 2. 动态 enc_in 映射表
# -----------------------------
DATASET_ENC_IN = {
    "electricity": 321, # 最后一列是OT
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "exchange_rate": 8,
    "national_illness": 7,
    "traffic": 862,
    "weather": 21
}


# -----------------------------
# 3. 自动化多重循环
# -----------------------------
for base_dataset in base_datasets:
    # 安全校验：自动匹配 enc_in 维度
    if base_dataset not in DATASET_ENC_IN:
        raise ValueError(f"❌ 数据集错误: 未知数据集 '{base_dataset}'，请在 DATASET_ENC_IN 中配置维度！")
    dynamic_enc_in = DATASET_ENC_IN[base_dataset]
    batch_size = 64 if base_dataset in ["traffic"] else 256 
    print(f"\n==============================================")
    print(f"📊 开始处理数据集: {base_dataset} | 自动匹配 enc_in = {dynamic_enc_in}")
    print(f"==============================================")

    for missing_rate in missing_rates:
        # 动态拼接数据集名称和路径
        data_name = f"{base_dataset}_missing_{missing_rate}"
        data_path_name = f"{data_name}.csv"
        
        for pred_len in pred_lens:
            for random_seed in random_seeds:
                model_id_full = f"{model_id_name}_{seq_len}_{pred_len}"

                # 💡 基础通用参数 (字典格式)
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
                    "--enc_in", str(dynamic_enc_in),  # 使用动态获取的维度
                    "--cycle", "24",
                    "--train_epochs", "30",
                    "--patience", "5",
                    "--dropout", "0.5",
                    "--itr", "1",
                    "--batch_size", str(batch_size),
                    "--learning_rate", "0.001",
                    "--random_seed", str(random_seed)
                ]

                # 💡 冲突检测与特有参数追加
                for key, value in special_args.items():
                    arg_key = f"--{key}"
                    
                    # 检查特有参数是否已经存在于基础 cmd 中
                    if arg_key in cmd:
                        raise ValueError(f"❌ 参数冲突错误: 特有参数 '{arg_key}' 已经在基础 cmd 列表中定义过了！请从 special_args 或基础命令中移除一个。")
        
                    cmd.extend([arg_key, str(value)])

                print(f"🚀 Running command in {run_dir}:\n{' '.join(cmd)}\n")
                subprocess.run(cmd, cwd=run_dir)