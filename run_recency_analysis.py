#!/usr/bin/env python
# run_recency_analysis.py

import argparse
import torch
import numpy as np
from experiments.exp_crypto_forecasting import Exp_Crypto_Forecast
import os
import random

if __name__ == '__main__':
    # Set fixed random seed
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Run recency effect analysis on a trained iTransformer model')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint directory')
    parser.add_argument('--setting', type=str, required=True,
                        help='Model setting string (same as used during training)')

    # Optional arguments (matching those used during training)
    parser.add_argument('--data', type=str, default='crypto', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/logits/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='btcusdt_pca_components_6h_50.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task')
    parser.add_argument('--target', type=str, default='close', help='target feature')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=50, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=50, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--exp_name', type=str, default='crypto', help='experiment name')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model', type=str, default='iTransformer', help='model name')
    parser.add_argument('--is_shorting', type=int, default=1, help='whether shorting is enabled')
    parser.add_argument('--precision_factor', type=float, default=2.0, help='factor to adjust precision weighting')
    parser.add_argument('--auto_weight', type=int, default=1, help='automatically adjust weighting if set to 1')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--use_norm', type=int, default=1, help='use normalization')

    args = parser.parse_args()

    # Ensure model_path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        exit(1)

    # Set checkpoint path based on provided model_path
    args.checkpoints = os.path.dirname(args.model_path)

    # Setup device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Initialize experiment
    print(f"Initializing experiment with model: {args.model}")
    exp = Exp_Crypto_Forecast(args)

    # Load model and run analysis
    print(f"Running recency effect analysis for model: {args.setting}")
    exp.test(args.setting, test=1)

    print("Analysis complete!")