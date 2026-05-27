import subprocess
from pathlib import Path


# -----------------------------
# 1. 配置参数
# -----------------------------
model_name = "DUET"

root_path_name = "./dataset/"
model_id_name = "5-16-DUET"

# [ "ETTh1","ETTh2","ETTm1","ETTm2","exchange_rate", "national_illness", "weather","electricity","traffic"] 
base_datasets = [ "ETTh1","ETTh2","ETTm1","ETTm2","exchange_rate", "national_illness", "weather","electricity","traffic"] 
missing_rates = [0, 5, 10, 20, 30, 50, 80] 

seq_len = 96
pred_lens = [96, 192, 336, 720]
random_seeds = [2026]

# 特有参数字典
special_args = {
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "parallel_strategy": "DP"
}

# 当前文件路径
current_dir = Path.cwd()
run_dir = current_dir.parents[1]

# -----------------------------
# 2. 动态 enc_in 映射表
# -----------------------------
# 根据常见的时间序列公开数据集特征维度整理
DATASET_ENC_IN = {
    "electricity": 321,
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "exchange_rate": 8,
    "national_illness": 7,
    "traffic": 862,
    "weather": 21
}
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

                # 基础通用参数
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
                    "--enc_in", str(dynamic_enc_in),  # <--- 使用动态获取的维度
                    "--cycle", "24",
                    "--train_epochs", "30",
                    "--patience", "5",
                    "--itr", "1",
                    "--batch_size", str(batch_size),
                    "--learning_rate", "0.001",
                    "--random_seed", str(random_seed),
                    # "--use_miss", 只要写了这个参数就是 True，不写就是 False
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