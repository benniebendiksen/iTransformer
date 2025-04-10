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
    # parser.add_argument('--model_id', type=str, default='1_btcusd_pca_components_lightboost_12h_4h_reduced_60_7_5_1_2_1_old_96_1_65',
    #                     help='model id')
    parser.add_argument('--model_id', type=str,
                        default='1_btcusd_pca_components_lightboost_12h_4h_reduced_70_7_5_1_2_1_old_96_1_75',
                        help='model id')
    parser.add_argument('--projection_idx', type=str, default='4', help='projection identifier (0, 1, 2, 3, 4)')
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
    parser.add_argument('--use_gpu', type=bool, help='use GPU')
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
    args.use_gpu = True if torch.cuda.is_available() else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(id_) for id_ in args.devices.split(',')]
        args.gpu = args.device_ids[0]
        print(f"using multiple GPUs, device ids: {args.device_ids}")
    print(f"using GPU: {args.gpu == 1}")
    print('Args:')
    print(args)
    return args


def setup_experiment(args):
    """Instantiate experiment class. The controller for the iTransformer model use"""
    exp = Exp_Logits_Forecast(args)
    return exp


def load_model(exp, args, setting):
    """Load the trained model from the checkpoints dir"""
    # Use the hardcoded checkpoint path
    if args.checkpoints:
        checkpoint_path = args.checkpoints
    else:
        checkpoint_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')

    print(f'Loading model from path: {checkpoint_path}')

    # Check if file exists before loading
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint file not found at {checkpoint_path}")
        print("Please check that the file exists and the path is correct")
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    exp.model.load_state_dict(state_dict)
    return exp.model


def get_test_data(args):
    """Get test dataset and dataloader"""
    test_data, test_loader = data_provider(args, 'test')
    print(f'Test dataset size: {len(test_data)}')
    return test_data, test_loader


def run_inference(model, test_data, test_loader, args, device):
    """Generate predictions on test data with improved timestamp and prediction index tracking"""
    print(f"Running inference on {len(test_data)} test samples...")
    model.eval()
    preds = []
    trues = []
    probs = []
    timestamps = []
    original_indices = []
    prediction_indices = []
    prices = []

    # Load original data to get timestamps if not available in metadata
    try:
        original_df = pd.read_csv(os.path.join(args.root_path, args.data_path))
        if 'date' in original_df.columns:
            if pd.api.types.is_numeric_dtype(original_df['date']):
                original_df['date'] = pd.to_datetime(original_df['date'], unit='s')
            else:
                original_df['date'] = pd.to_datetime(original_df['date'])
    except Exception as e:
        print(f"Warning: Could not load original data for timestamp extraction: {e}")
        original_df = None

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

            # Get timestamp, prediction index and metadata with better priority order
            timestamp = None
            price = None
            orig_idx = None
            pred_idx = None

            # First try sequence_indices which has the most detailed info
            if hasattr(test_data, 'sequence_indices') and i in test_data.sequence_indices:
                meta = test_data.sequence_indices[i]
                orig_idx = meta.get('orig_start_idx')
                pred_idx = meta.get('pred_idx')
                price = meta.get('pred_price')

                # Get timestamp if available in original_df
                if original_df is not None and pred_idx is not None and pred_idx < len(original_df):
                    timestamp = original_df.iloc[pred_idx]['date'] if 'date' in original_df.columns else None

            # Then try sample_metadata
            if (timestamp is None or pred_idx is None) and hasattr(test_data, 'sample_metadata') and i < len(
                    test_data.sample_metadata):
                meta = test_data.sample_metadata[i]
                timestamp = meta.get('timestamp') if timestamp is None else timestamp
                if orig_idx is None:
                    orig_idx = meta.get('orig_idx')
                if pred_idx is None:
                    pred_idx = meta.get('pred_idx')
                if price is None:
                    price = meta.get('pred_price')

            # Finally try active_indices
            if orig_idx is None and hasattr(test_data, 'active_indices') and i < len(test_data.active_indices):
                orig_idx = test_data.active_indices[i]

            # Calculate prediction index from original index if not directly available
            if pred_idx is None and orig_idx is not None and hasattr(test_data, 'seq_len'):
                pred_idx = orig_idx + test_data.seq_len

                # Try to get timestamp and price from original data
                if timestamp is None and original_df is not None and pred_idx < len(original_df):
                    timestamp = original_df.iloc[pred_idx]['date'] if 'date' in original_df.columns else None
                    price = original_df.iloc[pred_idx][args.target] if args.target in original_df.columns else None

            # Store data with fallbacks
            original_indices.append(orig_idx if orig_idx is not None else i)
            prediction_indices.append(
                pred_idx if pred_idx is not None else (orig_idx + args.seq_len if orig_idx is not None else i))
            timestamps.append(timestamp)
            prices.append(price)

            # Progress indication
            if (i + 1) % 100 == 0 or i == len(test_data) - 1:
                print(f'Processed {i + 1}/{len(test_data)} samples')

    # Convert to numpy arrays
    preds = np.array(preds)
    trues = np.array(trues)
    probs = np.array(probs)

    return preds, trues, probs, timestamps, original_indices, prediction_indices, prices


def calculate_metrics(preds, trues, is_shorting=True):
    """Calculate performance metrics"""
    print(f"Calculating model metrics...")
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


def calculate_returns(preds, trues, probs, is_shorting=True, actual_changes=None):
    """
    Calculate returns with improved accuracy using actual price changes when available

    Parameters:
    -----------
    preds : numpy.ndarray
        Binary predictions (0 or 1)
    trues : numpy.ndarray
        Actual binary labels (0 or 1)
    probs : numpy.ndarray
        Prediction probabilities
    is_shorting : bool
        Whether shorting is enabled in the strategy
    actual_changes : numpy.ndarray, optional
        Actual percentage price changes in decimal form. If None, will be estimated.

    Returns:
    --------
    dict
        Dictionary containing return metrics and arrays
    """
    print(f"Calculating returns...")

    # If actual changes not provided, estimate based on true labels
    if actual_changes is None:
        actual_changes = np.zeros(len(trues))
        for i in range(len(trues)):
            if trues[i] == 1:  # Actually went up
                actual_changes[i] = 0.02  # Assume 2% increase
            else:  # Actually went down
                actual_changes[i] = -0.015  # Assume 1.5% decrease

    # Calculate returns for different strategies

    # 1. Long-only strategy (take long positions only when predicting price increases)
    long_only_returns = np.zeros(len(preds))
    for i in range(len(preds)):
        if preds[i] == 1:  # Predicted price increase, take long position
            long_only_returns[i] = actual_changes[i]  # Gain/lose based on actual price change

    # 2. Short-only strategy (take short positions only when predicting price decreases)
    short_only_returns = np.zeros(len(preds))
    for i in range(len(preds)):
        if preds[i] == 0:  # Predicted price decrease, take short position
            short_only_returns[i] = -actual_changes[i]  # Gain when price falls, lose when price rises

    # 3. Combined strategy (take long for predicted increases, short for predicted decreases)
    combined_returns = np.zeros(len(preds))
    for i in range(len(preds)):
        if preds[i] == 1:  # Predicted price increase, take long position
            combined_returns[i] = actual_changes[i]
        else:  # Predicted price decrease, take short position
            combined_returns[i] = -actual_changes[i]

    # Choose which strategy to use based on shorting parameter
    if is_shorting:
        strategy_returns = combined_returns
    else:
        strategy_returns = long_only_returns

    # Calculate cumulative returns
    long_cum_returns = np.cumprod(1 + long_only_returns) - 1
    short_cum_returns = np.cumprod(1 + short_only_returns) - 1
    combined_cum_returns = np.cumprod(1 + combined_returns) - 1

    # Calculate additional metrics
    def calculate_metrics(returns):
        if len(returns) == 0:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'unprofitable_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'total_return': 0
            }

        total_trades = np.sum(returns != 0)
        profitable_trades = np.sum(returns > 0)
        unprofitable_trades = np.sum(returns < 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_return = np.mean(returns[returns != 0]) if np.sum(returns != 0) > 0 else 0
        total_return = np.cumprod(1 + returns)[-1] - 1 if len(returns) > 0 else 0

        return {
            'total_trades': int(total_trades),
            'profitable_trades': int(profitable_trades),
            'unprofitable_trades': int(unprofitable_trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return
        }

    # Calculate metrics for all strategies
    long_metrics = calculate_metrics(long_only_returns)
    short_metrics = calculate_metrics(short_only_returns)
    combined_metrics = calculate_metrics(combined_returns)

    return {
        'per_trade': strategy_returns,  # Based on shorting parameter
        'cumulative': np.cumprod(1 + strategy_returns) - 1 if len(strategy_returns) > 0 else np.array([0]),
        'total_return': np.cumprod(1 + strategy_returns)[-1] - 1 if len(strategy_returns) > 0 else 0,
        'strategies': {
            'long_only': {
                'returns': long_only_returns,
                'cumulative': long_cum_returns,
                'metrics': long_metrics
            },
            'short_only': {
                'returns': short_only_returns,
                'cumulative': short_cum_returns,
                'metrics': short_metrics
            },
            'combined': {
                'returns': combined_returns,
                'cumulative': combined_cum_returns,
                'metrics': combined_metrics
            }
        }
    }

def print_detailed_analysis(preds, trues, probs, timestamps, prices, actual_changes, returns):
    """
    Prints a detailed analysis table showing prediction results and returns

    Parameters:
    -----------
    preds : numpy.ndarray
        Binary predictions (0 or 1)
    trues : numpy.ndarray
        Actual binary labels (0 or 1)
    probs : numpy.ndarray
        Prediction probabilities
    timestamps : list
        Timestamps for each prediction
    prices : list
        Current prices at prediction time
    actual_changes : numpy.ndarray
        Actual percentage price changes in decimal form
    returns : dict
        Dictionary containing return data
    """
    # Print detailed trading results for the combined strategy
    print("\nDetailed Analysis (Combined):")
    print("-" * 120)
    print(
        f"{'Sample':<8} | {'Timestamp':<20} | {'Price':<12} | {'Pred':<5} | {'True':<5} | {'Prob':<8} | {'Actual Chg':<10} | {'Percent Change':>8} | {'Cum Percent Change':>12}")
    print("-" * 120)

    # Get combined returns and cumulative returns
    combined_returns = returns['strategies']['combined']['returns']
    cum_returns = returns['strategies']['combined']['cumulative']

    for i in range(len(preds)):
        # Format timestamp for display
        if timestamps[i] is None:
            ts_str = f"sample_{i}"
        elif isinstance(timestamps[i], pd.Timestamp):
            ts_str = timestamps[i].strftime('%Y-%m-%d %H:%M:%S')
        else:
            ts_str = str(timestamps[i])

        # Format price (handle None values)
        price_str = f"{prices[i]:<12.2f}" if prices[i] is not None else "N/A"

        # Print row
        print(
            f"{i:<8} | {ts_str:<20} | {price_str} | {preds[i]:<5.0f} | "
            f"{trues[i]:<5.0f} | {probs[i]:<8.4f} | {actual_changes[i] * 100:>10.2f}% | {combined_returns[i]:>8.4%} | {cum_returns[i]:>12.4%}")


def save_results(preds, trues, probs, timestamps, original_indices, prediction_indices, prices, metrics, returns, args,
                 actual_changes):
    """Save inference results to output directory with improved column formatting"""
    print(f"Saving results to directory: {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)

    # Process timestamps to ensure we have both unix and human-readable formats
    unix_timestamps = []
    human_timestamps = []

    for ts in timestamps:
        # Handle different timestamp formats
        if isinstance(ts, pd.Timestamp):
            # Convert pandas timestamp to unix timestamp (seconds)
            unix_ts = int(ts.timestamp())
            human_ts = ts.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(ts, (int, float)) and ts > 1000000000:  # Looks like a unix timestamp
            # Already a unix timestamp
            unix_ts = int(ts)
            human_ts = pd.Timestamp(unix_ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(ts, str):
            try:
                # Try to parse string to timestamp
                dt = pd.to_datetime(ts)
                unix_ts = int(dt.timestamp())
                human_ts = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                # If parsing fails, use None
                unix_ts = None
                human_ts = ts
        else:
            unix_ts = None
            human_ts = "Unknown" if ts is not None else None

        unix_timestamps.append(unix_ts)
        human_timestamps.append(human_ts)

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

    # Save the detailed analysis table to a CSV
    results_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_inference_results_table.csv")

    detailed_df = pd.DataFrame({
        'sample': range(len(preds)),
        'timestamp': human_timestamps,
        'price': prices,
        'prediction': preds,
        'true_label': trues,
        'probability': probs,
        'actual_change_pct': [x * 100 for x in actual_changes],
        'trade_return_pct': [x * 100 for x in returns['strategies']['combined']['returns']],
        'cumulative_return_pct': [x * 100 for x in returns['strategies']['combined']['cumulative']]
    })

    detailed_df.to_csv(results_file, index=False)
    print(f"Results table saved to {results_file}")

    return metrics_file, results_file
def extract_actual_price_changes(prediction_indices, args):
    """
    Extract actual price changes from the original dataset

    Parameters:
    -----------
    prediction_indices : list
        List of prediction indices in the dataset
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    numpy.ndarray
        Array of actual price changes in decimal form
    """
    try:
        original_df = pd.read_csv(os.path.join(args.root_path, args.data_path))

        # If 'date' column exists, convert it to datetime
        if 'date' in original_df.columns:
            if pd.api.types.is_numeric_dtype(original_df['date']):
                original_df['date'] = pd.to_datetime(original_df['date'], unit='s')
            else:
                original_df['date'] = pd.to_datetime(original_df['date'])

        # Calculate actual price changes
        actual_changes = []
        for pred_idx in prediction_indices:
            if pred_idx is not None and pred_idx < len(original_df) and pred_idx + args.pred_len < len(original_df):
                current_price = original_df.iloc[pred_idx][args.target]
                future_price = original_df.iloc[pred_idx + args.pred_len][args.target]

                if current_price > 0:  # Avoid division by zero
                    change = (future_price - current_price) / current_price  # Decimal form (e.g., 0.05 for 5%)
                else:
                    change = 0.0

                actual_changes.append(change)
            else:
                actual_changes.append(0.0)

        return np.array(actual_changes)

    except Exception as e:
        print(f"Warning: Could not extract actual price changes: {e}")
        return None

def main():
    try:
        print("Starting inference script...")
        # Parse arguments
        args = parse_args()
        print("Arguments parsed successfully")

        # Determine device
        device = torch.device('cuda:{}'.format(args.gpu) if args.use_gpu else 'cpu')
        print(f'Using device: {device}')

        # Create experiment and model setting
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
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
            args.class_strategy,
            args.projection_idx
        )

        # Override checkpoints path if specific checkpoint not provided
        if not args.checkpoints:
            checkpoint_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                args.checkpoints = checkpoint_path
                print(f"Using checkpoint: {checkpoint_path}")

        # Setup experiment
        print("Setting up experiment...")
        exp = setup_experiment(args)

        # Load model
        print("Loading model...")
        model = load_model(exp, args, setting)
        model.to(device)

        # Get test data
        print("Getting test data...")
        test_data, test_loader = get_test_data(args)

        # Run inference with improved timestamp and prediction index tracking
        print("Running inference...")
        preds, trues, probs, timestamps, original_indices, prediction_indices, prices = run_inference(
            model, test_data, test_loader, args, device)

        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_metrics(preds, trues, bool(args.is_shorting))

        # Extract actual price changes from the original dataset
        print("Extracting actual price changes...")
        actual_changes = extract_actual_price_changes(prediction_indices, args)

        # If actual changes couldn't be extracted, estimate based on true labels
        if actual_changes is None:
            print("Using estimated price changes based on true labels...")
            actual_changes = np.zeros(len(trues))
            for i in range(len(trues)):
                if trues[i] == 1:  # Actually went up
                    actual_changes[i] = 0.02  # Assume 2% increase
                else:  # Actually went down
                    actual_changes[i] = -0.015  # Assume 1.5% decrease

        # Calculate trading returns with actual price changes
        print("Calculating trading returns...")
        returns = calculate_returns(preds, trues, probs, bool(args.is_shorting), actual_changes)

        # Print detailed analysis table
        print_detailed_analysis(preds, trues, probs, timestamps, prices, actual_changes, returns)

        # Get the final return value
        final_return = returns['strategies']['combined']['metrics']['total_return']

        # Print summary
        print("\nInference Summary:")
        print(f"Total samples: {len(preds)}")
        print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision: {metrics['precision'] * 100:.2f}%")
        print(f"Win rate: {metrics['trading']['win_rate'] * 100:.2f}%")
        print(f"Total cumulative return: {final_return * 100:.2f}%")

        # Save enhanced results
        print("Saving results...")
        metrics_file, results_file = save_results(
            preds, trues, probs, timestamps, original_indices, prediction_indices,
            prices, metrics, returns, args, actual_changes)

        print("\nDone!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


# Make sure this is at the end of your script
if __name__ == "__main__":
    main()