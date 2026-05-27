import subprocess
from pathlib import Path
import sys
import os
import json
import time
sys.path.append("..") 
from cmd_builder import get_cmd, init_cmd, load_progress, save_progress, get_pred_lens

# -----------------------------
# 1. 配置参数
# -----------------------------
model_name = "DUET"
model_id_name = "5-28-DUET"

# 特有参数字典
batch_size_short = 8
special_args_init = {
    "gpu" : 0,
    "batch_size": 128,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 256,
    "d_ff": 512,
    "hidden_size": 128,
    "freq": "h",
    "factor": 1,
    "n_heads": 4,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 32,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 20,
    "train_epochs":20,
    "loss": "huber",
    "num_experts": 4,
    "noisy_gating": 1,# True
    "k": 1,
    "CI": 1,# True
    "parallel_strategy": "DP"
}

# -----------------------------
# 2. 进度追踪系统配置
# -----------------------------
# 用一个 json 文件作为进度数据库
PROGRESS_FILE = "training_progress.json"

progress = load_progress(PROGRESS_FILE)
current_dir = Path.cwd()
run_dir = current_dir.parents[1]

base_datasets, missing_rates, seq_len, random_seeds,short_length_data = init_cmd()
# -----------------------------
# 3. 自动化多重循环
# -----------------------------
for base_dataset in base_datasets:
    print(f"\n==============================================")
    print(f"📊 开始处理数据集: {base_dataset}")
    print(f"==============================================")
    for missing_rate in missing_rates:
            
        pred_lens = get_pred_lens(base_dataset)
        for pred_len in pred_lens:
            for random_seed in random_seeds:
                special_args = special_args_init.copy()  # 每次循环都从初始值开始，避免前一个实验的修改影响后续实验
                special_args['batch_size'] = batch_size_short if base_dataset in short_length_data else special_args['batch_size']
                
                # -----------------------------
                # ✨ 新增：检查该实验是否已经跑过
                # -----------------------------
                # 定义这个实验独一无二的名字，用来在字典里查找
                task_id = f"{model_id_name}_{base_dataset}_mr{missing_rate}_pl{pred_len}_seed{random_seed}"
                
                # 如果这个任务存在，且状态是 Success，直接跳过！
                if progress.get(task_id) == "Success":
                    print(f"⏭️  [已完成，跳过] {task_id}")
                    continue

                cmd = get_cmd(
                    base_dataset=base_dataset,
                    missing_rate=missing_rate,
                    pred_len=pred_len,
                    random_seed=random_seed,
                    special_args=special_args,
                    model_name=model_name,
                    model_id_name=model_id_name,
                    seq_len=seq_len,
                    use_sh=0,
                )
                
                print(f"🚀 Running: {' '.join(cmd)}\n")
                
                # 执行并捕获状态
                try:
                    # 注意这里还是 cwd=run_dir，保持你原来的路径设定不变
                    result = subprocess.run(cmd, cwd=run_dir)
                    
                    if result.returncode == 0:
                        progress[task_id] = "Success"
                        print(f"✅ [成功] {task_id}")
                    else:
                        progress[task_id] = "Failed"
                        print(f"❌ [失败退出, 错误码 {result.returncode}] {task_id}")
                        
                except KeyboardInterrupt:
                    print("\n🛑 [强制停止] 用户按下了 Ctrl+C，调度终止。")
                    # 保存后立刻退出整个 python 脚本
                    save_progress(progress,PROGRESS_FILE)
                    sys.exit(0)
                except Exception as e:
                    progress[task_id] = "Failed"
                    print(f"⚠️ [系统执行错误] {task_id}: {str(e)}")


                save_progress(progress,PROGRESS_FILE)
                time.sleep(1)

print("\n🎉 所有实验调度结束！您可以打开 training_progress.json 查看具体结果。")