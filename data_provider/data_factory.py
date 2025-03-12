import numpy as np
import torch

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_PEMS, Dataset_Solar, Dataset_Crypto
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    # Filter out any problematic samples
    valid_samples = []
    for sample in batch:
        if all(isinstance(x, (np.ndarray, torch.Tensor)) and x.size > 0 for x in sample):
            valid_samples.append(sample)

    # Return empty batch with correct shapes if no valid samples
    if len(valid_samples) == 0:
        return (
            torch.zeros((0, 24, 44)),  # Empty batch_x with correct features
            torch.zeros((0, 49, 1)),  # Empty batch_y with correct shape
            torch.zeros((0, 24, 4)),  # Empty batch_x_mark (typical 4 time features)
            torch.zeros((0, 49, 4))  # Empty batch_y_mark
        )

    # Standardize dimensions
    normalized_batch = []
    for seq_x, seq_y, seq_x_mark, seq_y_mark in valid_samples:
        # Convert to tensors
        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x).float()
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y).float()
        if isinstance(seq_x_mark, np.ndarray):
            seq_x_mark = torch.from_numpy(seq_x_mark).float()
        if isinstance(seq_y_mark, np.ndarray):
            seq_y_mark = torch.from_numpy(seq_y_mark).float()

        # Ensure correct dimensions
        if seq_y.dim() == 1:
            seq_y = seq_y.unsqueeze(-1)

        normalized_batch.append((seq_x, seq_y, seq_x_mark, seq_y_mark))

    return torch.utils.data.dataloader.default_collate(normalized_batch)


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
    'Solar': Dataset_Solar,
    'crypto': Dataset_Crypto,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
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
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))

    if len(data_set) < batch_size:
        print(f"WARNING: Dataset size ({len(data_set)}) is smaller than batch size ({batch_size})!")
        if len(data_set) > 0:
            # Adjust batch size to match dataset size
            print(f"Adjusting batch size from {batch_size} to {len(data_set)}")
            batch_size = len(data_set)
        else:
            print(f"Dataset is empty! This will cause errors in training.")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn
    )

    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    return data_set, data_loader
