import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')
    parser = argparse.ArgumentParser(description='Time Series Forecasting', allow_abbrev=False)
    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='TQNet',
                        help='model name, options: [TQNet, Informer, Autoformer, ...]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')  #fixed
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # TQNet & CycleNet
    parser.add_argument('--cycle', type=int, default=24, help='cycle length')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1, help='1: use revin or 0: no revin')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # SegRNN
    parser.add_argument('--rnn_type', default='gru', help='rnn_type')
    parser.add_argument('--dec_way', default='pmf', help='decode way')
    parser.add_argument('--seg_len', type=int, default=48, help='segment length')
    parser.add_argument('--channel_id', type=int, default=1, help='Whether to enable channel position encoding')

    # SEED
    parser.add_argument('--enable_env', type=int, default=0, help='enable env weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')
    parser.add_argument('--use_dropout', type=float, default=0.0, help='Use dropout')

    #DUET
    parser.add_argument('--k', type=int, default=1, help='DUET cluster k')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    # parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--enc_in', type=int, required=True, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')



    parser.add_argument('--max_norm', type=int, default=5, help='maximum gradient norm for clipping')
    parser.add_argument('--logger_uique_id', type=int, default=1, help='unique id for logger to avoid conflicts when using multiple loggers')


    # action='store_true' 的意思是：只要写了这个参数，就是 True；不写，默认就是 False
    parser.add_argument('--use_miss', action='store_true', help='use missing value features')

    # args = parser.parse_args()
    # 解析已知参数，未知参数放在 unknown
    args, unknown = parser.parse_known_args()
    # print(f"Parsed unknown args: {unknown}")
    # 处理 unknown 参数，转换成字典
    def unknown_to_namespace(unknown_list):
        it = iter(unknown_list)
        d = {}
        for key in it:
            if key.startswith('--'):
                value = next(it)
                # 尝试转换成 int 或 float，如果失败就保留字符串
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                d[key[2:]] = value
        return argparse.Namespace(**d)

    unknown_ns = unknown_to_namespace(unknown)
    # 合并两个 Namespace 对象
    for key, value in vars(unknown_ns).items():
        setattr(args, key, value)

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.fix_seed = fix_seed
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        print('Multiple GPU is on, use gpu: {}'.format(args.devices))
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.use_gpu and not args.use_multi_gpu:
        # 检测当前环境下 PyTorch 真正能看到的 GPU 数量
        available_gpus = torch.cuda.device_count()
        # 如果当前被设置成了单卡环境（比如被环境变量限制了），但你指定的 gpu 索引超出了范围
        if args.gpu >= available_gpus:
            print(f"⚠️ 警告: 你指定了 --gpu {args.gpu}，但当前环境仅有 {available_gpus} 块可用GPU。")
            assert f"🔄 自动将 args.gpu 重置为 0，以防止模型加载时发生反序列化崩溃。"
            
    # print('Args in experiment:')
    # print(args)

    from utils.logger import get_logger
    logger = get_logger(args)

    Exp = Exp_Main


    if args.is_training:
        for ii in range(args.itr):# experiments times, default 1

            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.cycle,
                fix_seed)

            exp = Exp(args)  # set experiments
            # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            logger.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            
            exp.train(setting)

            # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

            exp.test(setting)

            if args.do_predict:
                # print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                logger.info('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.cycle,
            fix_seed)

        exp = Exp(args)  # set experiments
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        logger.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
