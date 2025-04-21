#!/usr/bin/env python
import argparse
import torch
import random
import numpy as np
import os
import sys
import subprocess
from pathlib import Path


def check_environment():
    """Print environment information similar to the bash script"""
    print(f"Python path: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed")


def main():
    # Fix seed for reproducibility
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Print environment info
    check_environment()

    # Define configuration parameters (from bash script)
    model_name = "iTransformer"
    data_path = "pca_components_btcusdt_12h_45_reduced_lance_seed.csv"
    seq_len = 96
    pred_len = 1
    enc_in = 50
    d_model = 512

    # Extract filename without extension for model_id
    data_file = Path(data_path).stem
    model_id = f"{data_file}_{seq_len}_{pred_len}_{enc_in}"

    # Create argument parser similar to run.py
    parser = argparse.ArgumentParser(description='iTransformer')

    # Add all the arguments from run.py
    # Basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default=model_id, help='model id')
    parser.add_argument('--model', type=str, default=model_name,
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # Data loader
    parser.add_argument('--data', type=str, default='logits', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/logits/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=data_path, help='data csv file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='12h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=pred_len, help='prediction sequence length')

    # Model define
    parser.add_argument('--enc_in', type=int, default=enc_in, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=enc_in, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=d_model, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # Automate the timeenc value based on the embed value during parsing
    args, unknown_args = parser.parse_known_args()
    timeenc_value = 0 if args.embed != 'timeF' else 1
    parser.add_argument('--timeenc', type=int, default=timeenc_value,
                        help='automated time encoding based on embed value (0 or 1)')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Logits', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, default='logits',
                        help='experiment name, options:[MTSF, partial_train, logits]')
    parser.add_argument('--channel_independence', type=bool, default=False,
                        help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/',
                        help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--partial_start_index', type=int, default=0,
                        help='the start index of variates for partial training')
    parser.add_argument('--is_shorting', type=int, default=1,
                        help='whether shorting is enabled (1 for true, 0 for false)')
    parser.add_argument('--precision_factor', type=float, default=2.0, help='factor to adjust precision weighting')
    parser.add_argument('--auto_weight', type=int, default=1, help='automatically adjust weighting if set to 1')

    # Adaptive testing
    parser.add_argument('--adaptive_test', default=False, help='whether to perform adaptive fine-tuning per test sample')
    parser.add_argument('--adaptive_top_n', type=int, default=10,
                        help='number of similar samples for adaptive fine-tuning')
    parser.add_argument('--adaptive_epochs', type=int, default=10, help='number of epochs for adaptive fine-tuning')
    parser.add_argument('--adaptive_lr', type=float, default=0.0001, help='learning rate for adaptive fine-tuning')

    # Parse arguments
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Import experiment classes dynamically (as in run.py)
    if args.exp_name == 'logits':
        from experiments.exp_logits_forecasting import Exp_Logits_Forecast as Exp
    elif args.exp_name == 'partial_train':
        from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial as Exp
    else:  # MTSF: multivariate time series forecasting
        from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast as Exp

    if args.is_training:
        for ii in range(args.itr):
            # Setting for the experiment
            setting = f"{args.data_path}_{args.class_strategy}_{ii}"

            exp = Exp(args)  # Set up experiment
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            if args.adaptive_test:
                print(f'>>>>>>>adaptive testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.adaptive_test(
                    setting,
                    test=1,
                    top_n=args.adaptive_top_n,
                    epochs=args.adaptive_epochs,
                    learning_rate=args.adaptive_lr
                )

            if args.do_predict:
                print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f"{args.data_path}_{args.class_strategy}_{ii}"

        exp = Exp(args)  # Set up experiment
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test_with_direct_sample_processing(setting)
        exp.verify_test_labels(setting)

        if args.adaptive_test:
            print(f'>>>>>>>adaptive testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.adaptive_test(
                setting,
                test=1,
                top_n=args.adaptive_top_n,
                epochs=args.adaptive_epochs,
                learning_rate=args.adaptive_lr
            )

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
