import os
import re
import json
import shutil
from pathlib import Path


root_path_name = "./dataset/"
base_datasets = ["FRED-MD","Covid-19","NASDAQ","NN5","NYSE","Wike2000" , "national_illness","exchange_rate", "ETTh1","ETTh2","ETTm1","ETTm2","weather","electricity","PEMS08" ]
long_pred_len = [96, 192, 336, 720]

special_pred_data = ["solar_AL", "PEMS03","PEMS04","PEMS07","PEMS08"]
special_pred_len = [12, 24, 48]

short_length_data = ["FRED-MD","NASDAQ","NYSE","NN5","national_illness","Covid-19", "Wike2000","exchange_rate"]
short_length_data_pred_len = [24, 36, 48 , 60]
missing_rates = [50] 
# missing_rates = [0, 5, 10, 20, 30, 50, 80] 
seq_len = 96
random_seeds = [2026]

def init_cmd():
    return base_datasets, missing_rates,seq_len, random_seeds ,short_length_data

def get_pred_lens(base_dataset):
    if base_dataset in special_pred_data:
        return special_pred_len
    elif base_dataset in short_length_data:
        return short_length_data_pred_len
    else:
        return long_pred_len




def load_progress(PROGRESS_FILE):
    progress_path = Path(PROGRESS_FILE)
    backup_path = progress_path.with_suffix(progress_path.suffix + ".bak")

    # 先尝试读取正式文件
    if progress_path.exists() and progress_path.stat().st_size > 0:
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 正式进度文件读取失败: {e}")

    # 正式文件不存在、为空、损坏时，尝试读取备份
    if backup_path.exists() and backup_path.stat().st_size > 0:
        try:
            with open(backup_path, "r", encoding="utf-8") as f:
                print(f"✅ 已从备份文件恢复进度: {backup_path}")
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 备份进度文件读取失败: {e}")

    print("ℹ️ 没有可用进度文件，从空进度开始。")
    return {}


def save_progress(progress_dict, PROGRESS_FILE):
    progress_path = Path(PROGRESS_FILE)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    backup_path = progress_path.with_suffix(progress_path.suffix + ".bak")

    try:
        # 如果原文件存在且非空，先备份
        if progress_path.exists() and progress_path.stat().st_size > 0:
            shutil.copy2(progress_path, backup_path)

        # 先写入临时文件，不直接碰正式文件
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(progress_dict, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # 临时文件完整写完后，再替换正式文件
        os.replace(tmp_path, progress_path)

    except Exception as e:
        print(f"❌ 保存进度失败: {e}")

        # 清理残留 tmp 文件
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# 💡 全局缓存：防止在四层嵌套循环中重复读取同一个 .sh 文件
_SH_CACHE = {}
dataset_enc_map = {
    "electricity": 321,
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "exchange_rate": 8,
    "national_illness": 7,
    "traffic": 862,
    "weather": 21,
    "solar_AL": 137,
    "PEMS03": 358,
    "PEMS04": 307,
    "PEMS07": 883,
    "PEMS08": 170,
    "solar_AL": 137,
    "FRED-MD": 126, # 原始127去除date
    "Covid-19": 948,
    "Wike2000":2000,
    "NN5":111,
    "NYSE":5,
    "NASDAQ":5,
}
def _parse_sh_hyperparams(dataset_name, sh_dir="./"):

    """内部函数：V3 终极版解析器（支持文件名忽略大小写匹配）"""
    global dataset_enc_map
    # 1. 获取目录下所有文件
    all_files = os.listdir(sh_dir)
    
    # 2. 尝试寻找匹配的文件名（忽略大小写）
    target_filename = f"{dataset_name}.sh".lower()
    matched_file = None
    
    for f in all_files:
        if f.lower() == target_filename:
            matched_file = f
            break
    
    # 3. 确定最终路径
    if matched_file:
        sh_file_path = os.path.join(sh_dir, matched_file)
    else:
        # 4. 如果没找到，尝试保底的 etth1.sh (同样忽略大小写)
        print(f"⚠️ 警告: 未找到 {dataset_name}.sh（忽略大小写）！")
        if dataset_enc_map[dataset_name] <= 30:
            print("尝试回退使用 etth1.sh")
            matched_file = next((f for f in all_files if f.lower() == "etth1.sh"), None)
        else:
            print("尝试回退使用 weather.sh")
            matched_file = next((f for f in all_files if f.lower() == "weather.sh"), None)
            if not matched_file:
                print("⚠️ 警告: 未找到回退配置 weather.sh！尝试使用 etth1.sh")
                matched_file = next((f for f in all_files if f.lower() == "etth1.sh"), None)

        if not matched_file:
            print("❌ 致命错误: 未能找到回退配置！返回空配置。")
            return {}
        sh_file_path = os.path.join(sh_dir, matched_file)

    params_by_pl = {}




    with open(sh_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"✅ 成功加载配置文件: {sh_file_path}")

    variables = {}
    for key, val in re.findall(r'^([a-zA-Z0-9_]+)=([^\s]+)$', content, re.MULTILINE):
        variables[key] = val

    def extract_kwargs(cmd_text):
        kwargs = {}
        cmd_str = cmd_text.replace('\\\n', ' ').replace('\n', ' ')
        ignore_keys = [
            'is_training', 'root_path', 'data_path', 'model_id', 
            'model', 'data', 'pred_len', 'seq_len', 'enc_in', 
            'batch_size', 'gpu', 'CUDA_VISIBLE_DEVICES'
        ]
        
        # 核心改动 1：正则改为匹配到下一个 '--' 或字符串结尾之前的所有内容
        # (.*?) 非贪婪匹配，(?=\s+--|$) 向前断言找下一个参数或结尾
        for key, val in re.findall(r'--([a-zA-Z0-9_]+)\s+(.*?)(?=\s+--|$)', cmd_str):
            if key in ignore_keys: 
                continue
                
            val = val.strip() # 去除首尾可能多余的空格
            
            if val.startswith('$'):
                # 注意：这里的 variables 需要确保在你的作用域内定义了
                val = variables.get(val.replace('$', '').strip("'\""), val)
                
            # 核心改动 2：如果捕获的值里包含空格，说明是 nargs='+' 的列表参数
            if ' ' in val:
                parsed_list = []
                for v in val.split():
                    try:
                        parsed_list.append(float(v) if '.' in v else int(v))
                    except ValueError:
                        parsed_list.append(v)
                val = parsed_list
            else:
                # 原始的单值转换逻辑
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
                    
            kwargs[key] = val
            
        return kwargs

    blocks = re.findall(r'for\s+pred_len\s+in\s+([\d\s]+).*?do(.*?)done', content, re.DOTALL)
    if blocks:
        for pl_str, cmd_text in blocks:
            for pl in [int(p) for p in pl_str.split()]:
                params_by_pl[pl] = extract_kwargs(cmd_text).copy()
    else:
        params_by_pl['default'] = extract_kwargs(content)
    # print(f"🔍 解析结果: {params_by_pl}")
    return params_by_pl


def get_cmd(
    base_dataset, 
    missing_rate, 
    pred_len, 
    random_seed,
    model_name,
    model_id_name,
    seq_len,
    root_path="./dataset/",
    sh_dir="./",
    special_args=None,
    use_sh = 1,
):
    """
    暴露给外部调用的核心接口：生成一条完整的命令行 list
    """
    global _SH_CACHE, dataset_enc_map
    if special_args is None:
        special_args = {}

    # 1. 检查维度匹配
    if base_dataset not in dataset_enc_map:
        raise ValueError(f"❌ 数据集错误: 未知数据集 '{base_dataset}'，请配置 enc_in！")

    dynamic_enc_in = dataset_enc_map[base_dataset]
    current_sh_args = {}
    # 2. 从缓存读取或解析 .sh 配置
    if use_sh == 1:
        if base_dataset not in _SH_CACHE:
            _SH_CACHE[base_dataset] = _parse_sh_hyperparams(base_dataset, sh_dir)

        sh_args_dict = _SH_CACHE[base_dataset]

        # 3. 提取该 pred_len 对应的参数
        if pred_len in sh_args_dict:
            current_sh_args = sh_args_dict[pred_len]
        elif 'default' in sh_args_dict:
            current_sh_args = sh_args_dict['default']
        else:
            print(f"❌ 致命错误: 无法获取 {base_dataset} pred_len={pred_len} 的参数！")
    
    # 4. 合并你的覆盖参数
    final_extra_args = current_sh_args.copy()
    final_extra_args.update(special_args)

    # 处理布尔值参数（如 use_revin），确保它们被正确转换为 0/1
    revin_val = final_extra_args.get('use_revin')
    if revin_val in [1, '1', True, 'True', 'true']:
        final_extra_args['use_revin'] = 1
    elif revin_val in [0, '0', False, 'False', 'false']:
        final_extra_args['use_revin'] = 0

    # 5. 动态构建字符串
    data_name = f"{base_dataset}_missing_{missing_rate}"
    data_path_name = f"{data_name}.csv"
    model_id_full = f"{model_id_name}_{seq_len}_{pred_len}"

    # 6. 生成极简版基础 cmd（只保留外部变量控制的核心路径和 ID）
    cmd = [
        "python", "-u", "run.py",
        "--is_training", "1",
        "--root_path", root_path,
        "--data_path", data_path_name,
        "--model_id", model_id_full,
        "--model", model_name,
        "--data", data_name,
        "--pred_len", str(pred_len),
        "--random_seed", str(random_seed),
        "--enc_in", str(dynamic_enc_in),
        "--dec_in", str(dynamic_enc_in),
        "--c_out", str(dynamic_enc_in),
        "--seq_len", str(seq_len),
    ]
    
    # 7. 追加特有参数（如有冲突，以 final_extra_args 为准，并打印提示）
    for key, value in final_extra_args.items():
        arg_key = f"--{key}"
        
        # 🌟 新增：检测冲突并处理
        if arg_key in cmd:
            try:
                idx = cmd.index(arg_key)
                # 提取命令行已有的旧值用于提示
                old_val = cmd[idx + 1] if idx + 1 < len(cmd) and not cmd[idx + 1].startswith('--') else "True"
                
                # 提示：命令行参数优先级更高，忽略配置文件/覆盖参数
                print(f"⚠️ 提示: 参数冲突！保留命令行输入，忽略配置文件修改: {arg_key} (保持为 {old_val}，丢弃新值 {value})")
                
                # 【关键】因为要保留命令行自带的参数，这里直接 continue 跳过，不执行后面添加新值的逻辑
                continue 
                
            except ValueError:
                pass

        # if isinstance(value, bool):
        #     if value is True: 
        #         cmd.append(arg_key)
        # else:
        #     cmd.extend([arg_key, str(value)])

        # 只有当 cmd 中没有这个参数时，才会把配置文件的值（value）加进去
        if isinstance(value, bool):
            if value is True: 
                cmd.append(arg_key)
        elif isinstance(value, list):
            # 如果值是列表，将列表展开为多个独立的字符串参数
            cmd.append(arg_key)
            cmd.extend([str(v) for v in value])
        else:
            cmd.extend([arg_key, str(value)])

    return cmd