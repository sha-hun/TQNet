from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar, Dataset_PEMS, Dataset_Miss
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'Miss': Dataset_Miss
}


def data_provider(args, flag, use_miss=False):
    if args.data not in data_dict:
        if '_missing_' in args.data:
            Data = Dataset_Miss
            print('使用 Dataset_Miss 处理缺失数据集')
        else:
            raise ValueError(f"数据集 {args.data} 不在预定义的数据字典中，请检查数据集名称或添加到 data_dict 中。")
    else:
        Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1

        # Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # print(f"Preparing {flag}  features是{args.features}，freq是{freq}，cycle是{args.cycle} target是{args.target}")

    # data_set = Data(
    #     root_path=args.root_path,
    #     data_path=args.data_path,
    #     flag=flag,
    #     size=[args.seq_len, args.label_len, args.pred_len],
    #     timeenc=timeenc,
    #     args= args,
    # )
    data_set = Data(
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        args = args,
    )
    print("use_miss=",args.use_miss," flag:", flag," len:", len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
