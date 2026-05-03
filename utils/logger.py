import logging
import os

def get_logger(args):
    """
    根据 args 配置 logger，自动生成日志文件路径。
    """
    # 构建目录
    log_dir = os.path.join("result_log", args.model, args.data)
    os.makedirs(log_dir, exist_ok=True)

    # 构建日志文件名
    log_file_name = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.pred_len,
    args.cycle,
    args.fix_seed)
    log_file_name = f"{log_file_name}.txt"
    log_file = os.path.join(log_dir, log_file_name)  # 完整文件路径
    if not os.path.exists(log_file):
        with open(log_file, 'w', encoding='utf-8') as f:
            pass

    # 创建 logger
    logger = logging.getLogger(f"logger_{args.logger_uique_id}")
    logger.setLevel(logging.INFO)

    # 防止重复添加 Handler
    if not logger.handlers:
        # 文件输出 handler
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)

        # 控制台输出 handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger