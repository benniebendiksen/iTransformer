import argparse
import torch
import os
import pandas as pd
import numpy as np
from model.iTransformer import Model
from experiments.exp_logits_forecasting import Exp_Logits_Forecast
from data_provider.data_factory import data_provider
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='iTransformer Inference Script')

    # Basic config (required from bash)
    parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
    parser.add_argument('--model_id', type=str, default='1_btcusdt_pca_components_12h_60_07_05_96_1_65', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer', help='model name')

    # Data loader
    parser.add_argument('--data', type=str, default='logits', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/logits/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='btcusdt_pca_components_12h_60_07_05.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='12h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default=None, help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length (not used in iTransformer)')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # Model definition
    parser.add_argument('--enc_in', type=int, default=65, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=10, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='moving average window')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    parser.add_argument('--output_attention', action='store_true', help='output attention weights')
    parser.add_argument('--do_predict', action='store_true', help='predict unseen data')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--des', type=str, default='Logits', help='experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='lr schedule type')
    parser.add_argument('--use_amp', action='store_true', default=False, help='mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0', help='GPU devices')

    # iTransformer specific
    parser.add_argument('--exp_name', type=str, default='logits', help='experiment name')
    parser.add_argument('--channel_independence', type=bool, default=False, help='channel independence mechanism')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='target data root')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='target data file')
    parser.add_argument('--efficient_training', type=bool, default=False, help='partial training mode')
    parser.add_argument('--use_norm', type=int, default=1, help='use norm/denorm')
    parser.add_argument('--partial_start_index', type=int, default=0, help='start index for partial training')

    parser.add_argument('--is_shorting', type=int, default=0, help='enable shorting')
    parser.add_argument('--precision_factor', type=float, default=2.0, help='precision weighting factor')
    parser.add_argument('--auto_weight', type=int, default=0, help='auto weighting logic')
    parser.add_argument('--output_path', type=str, default='./outputs/', help='path to save inference results')

    # Set timeenc based on embed
    args, _ = parser.parse_known_args()
    timeenc_value = 0 if args.embed != 'timeF' else 1
    parser.add_argument('--timeenc', type=int, default=timeenc_value, help='time encoding')

    # Final parse
    args = parser.parse_args()

    # GPU setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    print('Args:')
    print(args)
    return args



def setup_experiment(args):
    """Instantiate experiment class. The controller for the iTransformer model use"""
    exp = Exp_Logits_Forecast(args)
    return exp


# def load_model(exp, args, setting):
#     """Load the trained model from the checkpoints dir"""
#     if args.checkpoints:
#         checkpoint_path = args.checkpoints
#     else:
#         checkpoint_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
#
#     print(f'Loading model from {checkpoint_path}')
#     exp.model.load_state_dict(torch.load(checkpoint_path))
#     return exp.model

def load_model(exp, args, setting):
    """Load the trained model from the checkpoints dir"""
    # Use the hardcoded checkpoint path
    if args.checkpoints:
        checkpoint_path = args.checkpoints
    else:
        checkpoint_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')

    print(f'Loading model from {checkpoint_path}')

    # Check if file exists before loading
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint file not found at {checkpoint_path}")
        print("Please check that the file exists and the path is correct")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # IMPORTANT: Always force CPU loading since the model was trained on GPU
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    exp.model.load_state_dict(state_dict)
    return exp.model


def get_test_data(args):
    """Get test dataset and dataloader"""
    test_data, test_loader = data_provider(args, 'test')
    print(f'Test dataset size: {len(test_data)}')
    return test_data, test_loader


def run_inference(model, test_data, test_loader, args, device):
    """Generate predictions on test data"""
    model.eval()
    preds = []
    trues = []
    probs = []
    timestamps = []
    original_indices = []

    print('Running inference on test dataset...')
    with torch.no_grad():
        # Process one sample at a time to ensure correct label tracking
        for i in range(len(test_data)):
            # Get sample
            batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[i]

            # Add batch dimension and convert to tensor
            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(device)
            batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(device)
            batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # Generate prediction
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Process outputs for binary classification
            f_dim = -1 if args.features == 'MS' else 0
            outputs_last = outputs[:, -1, f_dim:]
            batch_y_last = batch_y[:, -1, f_dim:].to(device)

            # Get prediction probability and binary prediction
            output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
            output_binary = (output_prob > 0.5).astype(np.float32)
            true_label = batch_y_last.detach().cpu().numpy()[0, 0]

            # Store results
            preds.append(output_binary)
            trues.append(true_label)
            probs.append(output_prob)

            # Store metadata if available
            if hasattr(test_data, 'active_indices') and i < len(test_data.active_indices):
                orig_idx = test_data.active_indices[i]
                original_indices.append(orig_idx)

                # Get timestamp if available
                timestamp = None
                meta = None
                if hasattr(test_data, 'sequence_indices') and i in test_data.sequence_indices:
                    meta = test_data.sequence_indices[i]
                elif hasattr(test_data, 'sample_metadata') and i < len(test_data.sample_metadata):
                    meta = test_data.sample_metadata[i]

                if meta and 'timestamp' in meta:
                    timestamp = meta['timestamp']
                timestamps.append(timestamp)
            else:
                original_indices.append(i)
                timestamps.append(None)

            # Progress indication
            if (i + 1) % 100 == 0 or i == len(test_data) - 1:
                print(f'Processed {i + 1}/{len(test_data)} samples')

    # Convert to numpy arrays
    preds = np.array(preds)
    trues = np.array(trues)
    probs = np.array(probs)

    return preds, trues, probs, timestamps, original_indices


def calculate_metrics(preds, trues, is_shorting=True):
    """Calculate performance metrics"""
    # Classification metrics
    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, zero_division=0)
    recall = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)

    # Confusion matrix elements
    TP = np.logical_and(preds == 1, trues == 1).sum()
    TN = np.logical_and(preds == 0, trues == 0).sum()
    FP = np.logical_and(preds == 1, trues == 0).sum()
    FN = np.logical_and(preds == 0, trues == 1).sum()

    # Trading metrics
    if is_shorting:
        # For shorting strategy: we profit from both correct predictions
        profitable_trades = TP + TN
        unprofitable_trades = FP + FN
    else:
        # For no-shorting strategy: we only profit from correct positive predictions
        profitable_trades = TP
        unprofitable_trades = FP

    total_trades = profitable_trades + unprofitable_trades
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0
    profit_factor = profitable_trades / unprofitable_trades if unprofitable_trades > 0 else float('inf')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN)
        },
        'trading': {
            'profitable_trades': int(profitable_trades),
            'unprofitable_trades': int(unprofitable_trades),
            'total_trades': int(total_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    }


def calculate_trading_returns(preds, trues, probs, is_shorting=True):
    """Calculate trading returns for different strategies"""
    # Placeholder for returns per trade
    # In a real implementation, these would be calculated from actual price data
    # Here we use a fixed percentage for demonstration
    uptrend_return = 0.01  # 1% return for correct uptrend prediction
    downtrend_return = 0.01  # 1% return for correct downtrend prediction

    # Initialize return arrays
    returns = np.zeros(len(preds))

    for i in range(len(preds)):
        pred = preds[i]
        true = trues[i]

        if is_shorting:
            # Shorting strategy
            if (pred == 1 and true == 1):  # Correct uptrend prediction
                returns[i] = uptrend_return
            elif (pred == 0 and true == 0):  # Correct downtrend prediction
                returns[i] = downtrend_return
            elif (pred == 1 and true == 0):  # Incorrect uptrend prediction
                returns[i] = -uptrend_return
            else:  # Incorrect downtrend prediction
                returns[i] = -downtrend_return
        else:
            # No-shorting strategy
            if pred == 1:  # We take a position
                if true == 1:  # Correct uptrend prediction
                    returns[i] = uptrend_return
                else:  # Incorrect uptrend prediction
                    returns[i] = -uptrend_return
            # else: No position taken for predicted downtrends

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns) - 1

    return {
        'per_trade': returns,
        'cumulative': cumulative_returns,
        'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    }


def save_results(preds, trues, probs, timestamps, original_indices, metrics, returns, args):
    """Save inference results to output directory"""
    os.makedirs(args.output_path, exist_ok=True)

    # Create results dataframe
    results_df = pd.DataFrame({
        'original_index': original_indices,
        'prediction': preds,
        'true_label': trues,
        'probability': probs,
        'timestamp': timestamps
    })

    # Save results to CSV
    output_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Save metrics to JSON
    import json
    metrics_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump({
            'model': args.model,
            'model_id': args.model_id,
            'data_path': args.data_path,
            'metrics': metrics,
            'trading_returns': {
                'total_return': returns['total_return']
            },
            'args': vars(args)
        }, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # Generate summary report
    report_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Inference Report for {args.model} (ID: {args.model_id})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Model Configuration:\n")
        f.write(f"- Data: {args.data_path}\n")
        f.write(f"- Sequence Length: {args.seq_len}\n")
        f.write(f"- Prediction Length: {args.pred_len}\n")
        f.write(f"- Features: {args.features}\n")
        f.write(f"- Trading Strategy: {'Shorting Enabled' if args.is_shorting else 'No Shorting'}\n\n")

        f.write("Classification Metrics:\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n")
        f.write(f"- Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)\n")
        f.write(f"- Recall: {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)\n")
        f.write(f"- F1 Score: {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"- True Positives: {metrics['confusion_matrix']['TP']}\n")
        f.write(f"- True Negatives: {metrics['confusion_matrix']['TN']}\n")
        f.write(f"- False Positives: {metrics['confusion_matrix']['FP']}\n")
        f.write(f"- False Negatives: {metrics['confusion_matrix']['FN']}\n\n")

        f.write("Trading Performance:\n")
        f.write(f"- Total Trades: {metrics['trading']['total_trades']}\n")
        f.write(f"- Profitable Trades: {metrics['trading']['profitable_trades']}\n")
        f.write(f"- Unprofitable Trades: {metrics['trading']['unprofitable_trades']}\n")
        f.write(f"- Win Rate: {metrics['trading']['win_rate']:.4f} ({metrics['trading']['win_rate'] * 100:.2f}%)\n")
        f.write(f"- Profit Factor: {metrics['trading']['profit_factor']:.4f}\n")
        f.write(f"- Cumulative Return: {returns['total_return']:.4f} ({returns['total_return'] * 100:.2f}%)\n")

    print(f"Report saved to {report_file}")

    return output_file, metrics_file, report_file


def main():
    # Add model configuration args to the parser, here


    # Parse arguments
    args = parse_args()

    # Force CPU mode if CUDA is not available
    args.use_gpu = False

    # Determine device
    device = torch.device('cuda:{}'.format(args.gpu) if args.use_gpu else 'cpu')
    print(f'Using device: {device}')

    # Create experiment and model setting
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_0'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        'Logits',
        args.class_strategy
    )

    # Setup experiment
    exp = setup_experiment(args)

    # Load model
    model = load_model(exp, args, setting)
    model.to(device)

    # Get test data
    test_data, test_loader = get_test_data(args)

    # Run inference
    preds, trues, probs, timestamps, original_indices = run_inference(model, test_data, test_loader, args, device)

    # Calculate metrics
    metrics = calculate_metrics(preds, trues, bool(args.is_shorting))

    # Calculate trading returns
    returns = calculate_trading_returns(preds, trues, probs, bool(args.is_shorting))

    # Save results
    save_results(preds, trues, probs, timestamps, original_indices, metrics, returns, args)

    # Print summary
    print("\nInference Summary:")
    print(f"Total samples: {len(preds)}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Win rate: {metrics['trading']['win_rate'] * 100:.2f}%")
    print(f"Total return: {returns['total_return'] * 100:.2f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
