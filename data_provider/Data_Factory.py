from data_provider.Data_Loader import Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    # 確保 args 中有 train_only 屬性，避免報錯
    train_only = args.train_only if hasattr(args, 'train_only') else False

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    stride = args.stride_train if flag == 'train' else args.stride_eval
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        stride = stride,
        input_col=args.input_col,
        segment_col=args.segment_col,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(f"[{flag} set] 總筆數: {len(data_set)}")
    
    # 注意：在 Windows 環境下，DataLoader 的 num_workers 如果大於 0 有時會卡死
    # 建議之後在設定 args 時，先將 args.num_workers 設為 0 來測試
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        
    return data_set, data_loader