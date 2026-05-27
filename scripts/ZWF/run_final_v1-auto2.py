import subprocess
from pathlib import Path
import sys
import os
import json
import time

sys.path.append("..") 
from cmd_builder import get_cmd


# -----------------------------
# 1. 核心运行配置
# -----------------------------
# Fredformer 对应ETT的四个数据集,DistDF-TimeBridge对应ECL、Traffic、Weather、PEMS03、PEMS08
model_name = "ZWF_final_v1"
model_id_name = "5-16-v1"
root_path_name = "./dataset/"
# [ "ETTh1","ETTh2","ETTm1","ETTm2","exchange_rate", "national_illness", "weather","electricity","traffic","PEMS03","PEMS04","PEMS07","PEMS08"] 
base_datasets = ["Wike2000"]# ,"FRED-MD" ,"FRED-MD","Covid-19","NASDAQ","NN5","NYSE"
long_pred_len = [96, 192, 336, 720]
short = ["solar_AL", "PEMS03","PEMS04","PEMS07","PEMS08"]
short_pred_len = [12, 24, 48]
mid = ["FRED-MD","NASDAQ","NYSE","NN5","ILI","Covid-19", "Wike2000"]
mid_pred_len = [24, 36, 48 , 60]

missing_rates = [0, 5, 10, 20, 30, 50, 80] 

seq_len = 96
random_seeds = [2026]

# 💡 特有参数字典
special_args = {
    "use_miss": True,     # 这个参数在 run.py 中是 action='store_true' 的形式，所以只要写了这个参数就是 True，不写就是 False,与赋予值无关
    "batch_size": 8,
    "gpu": 2,
}


# -----------------------------
# 2. 进度追踪系统配置
# -----------------------------
# 用一个 json 文件作为进度数据库，比 CSV 在 Python 里读写更方便和安全
PROGRESS_FILE = "training_progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_progress(progress_dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_dict, f, indent=4)

# 读取现有的进度（如果文件不存在，就是一个空字典 {}）
progress = load_progress()

current_dir = Path.cwd()
run_dir = current_dir.parents[1]  


# -----------------------------
# 3. 自动化多重循环
# -----------------------------
for base_dataset in base_datasets:
    print(f"\n==============================================")
    print(f"📊 开始处理数据集: {base_dataset}")
    print(f"==============================================")
    if base_dataset in short:
        pred_lens = short_pred_len
    elif base_dataset in mid:
        pred_lens = mid_pred_len
    else:
        pred_lens = long_pred_len
        
    for missing_rate in missing_rates:
        for pred_len in pred_lens:
            for random_seed in random_seeds:
                
                # -----------------------------
                # ✨ 新增：检查该实验是否已经跑过
                # -----------------------------
                # 定义这个实验独一无二的名字，用来在字典里查找
                task_id = f"{model_id_name}_{base_dataset}_mr{missing_rate}_pl{pred_len}_seed{random_seed}"
                
                # 如果这个任务存在，且状态是 Success，直接跳过！
                if progress.get(task_id) == "Success":
                    print(f"⏭️  [已完成，跳过] {task_id}")
                    continue
                
                # 如果是 Failed 或者没跑过，往下正常执行
                
                # 💡 拿到组装好的完整命令
                cmd = get_cmd(
                    base_dataset=base_dataset,
                    missing_rate=missing_rate,
                    pred_len=pred_len,
                    random_seed=random_seed,
                    special_args=special_args,
                    model_name=model_name,
                    model_id_name=model_id_name,
                    seq_len=seq_len,
                )
                
                print(f"🚀 Running: {' '.join(cmd)}\n")
                
                # 执行并捕获状态
                try:
                    # 注意这里还是 cwd=run_dir，保持你原来的路径设定不变
                    result = subprocess.run(cmd, cwd=run_dir)
                    
                    # 检查 returncode！只有 0 才是代码在没有 raise 任何错误的情况下正常结束的
                    if result.returncode == 0:
                        progress[task_id] = "Success"
                        print(f"✅ [成功] {task_id}")
                    else:
                        progress[task_id] = "Failed"
                        print(f"❌ [失败退出, 错误码 {result.returncode}] {task_id}")
                        
                except KeyboardInterrupt:
                    print("\n🛑 [强制停止] 用户按下了 Ctrl+C，调度终止。")
                    # 保存后立刻退出整个 python 脚本
                    save_progress(progress)
                    sys.exit(0)
                except Exception as e:
                    progress[task_id] = "Failed"
                    print(f"⚠️ [系统执行错误] {task_id}: {str(e)}")

                # 每跑完（或失败）一个实验，立刻保存进度字典！
                save_progress(progress)
                
                # (可选) 每跑完释放一下显卡缓冲时间，可根据需要删除
                time.sleep(1)

print("\n🎉 所有实验调度结束！您可以打开 training_progress.json 查看具体结果。")