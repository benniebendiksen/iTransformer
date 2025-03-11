import numpy as np
import torch

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_PEMS, Dataset_Solar, Dataset_Crypto
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    # Filter out any samples that don't have the right dimensions
    valid_samples = []

    for sample in batch:
        try:
            # Check if all tensors in the sample have proper dimensions
            if all(isinstance(x, np.ndarray) and x.size > 0 for x in sample):
                valid_samples.append(sample)
        except:
            continue

    # If we have no valid samples, return empty batch with proper structure
    if len(valid_samples) == 0:
        return (torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor())

    # Use default collate for valid samples
    return torch.utils.data.dataloader.default_collate(valid_samples)


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

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=custom_collate_fn  # Add this line
    )

    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=batch_size,
    #     shuffle=shuffle_flag,
    #     num_workers=args.num_workers,
    #     drop_last=drop_last)
    return data_set, data_loader
