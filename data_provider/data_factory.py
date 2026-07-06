from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, Dataset_Climate
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'Climate': Dataset_Climate,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    # embed has been removed; default timeenc to 0 (dummy timestamps returned anyway)
    timeenc = 0

    shuffle_flag = False if flag == 'test' else True
    drop_last = False
    batch_size = args.batch_size
    # freq has been removed; default to 'h'
    freq = 'h'

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        # label_len has been removed; pass 0 as placeholder
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=None
    )
    
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader