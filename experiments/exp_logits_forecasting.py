import pandas as pd

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our custom trading loss
from utils.trading_loss import TradingBCELoss

warnings.filterwarnings('ignore')


class Exp_Logits_Forecast(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Logits_Forecast, self).__init__(args)
        # Add trading strategy parameters
        self.is_shorting = getattr(args, 'is_shorting', True)
        self.precision_factor = getattr(args, 'precision_factor', 2.0)
        self.auto_weight = getattr(args, 'auto_weight', True)

        # Calculate and store class distribution once at initialization
        self.class_distribution = None
        if self.args.is_training:
            self.calculate_class_distribution()

    def analyze_recency_effect(self, setting, test=0):
        """
        Analyzes how model performance changes with temporal distance from validation set
        using only print statements (no visualizations)
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # Storage for all predictions and ground truth
        all_preds = []
        all_trues = []

        print("\n===== RECENCY EFFECT ANALYSIS =====")
        print("Collecting test set predictions...")

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if batch_x.size(0) == 0:
                    continue

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs for binary classification
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Convert logits to probabilities with sigmoid
                output_probs = torch.sigmoid(outputs_last).detach().cpu().numpy()

                # Get binary predictions (threshold at 0.5)
                output_binary = (output_probs > 0.5).astype(np.float32)

                # Get true labels
                true_labels = batch_y_last.detach().cpu().numpy()

                # Store individual predictions
                for idx in range(len(output_binary)):
                    all_preds.append(output_binary[idx][0])  # Extract scalar value
                    all_trues.append(true_labels[idx][0])  # Extract scalar value

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        # Verify we collected predictions for the whole test set
        print(f"Collected {len(all_preds)} predictions from test set")
        print(f"trues: {all_trues}")
        print(f"preds: {all_preds}")

        # Define percentile segments to analyze
        percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Create results table
        print("\nRecency Effect Analysis - Increasing Percentiles from Validation Boundary:")
        print("-" * 80)
        print(
            f"{'Percentile':^15} | {'Samples':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1':^10} | {'Win Rate':^10}")
        print("-" * 80)

        # Analyze each percentile segment
        for p in percentiles:
            # Calculate number of samples to include
            n_samples = int(len(all_preds) * p)
            if n_samples == 0:
                continue

            # Select earliest n_samples (closest to validation set)
            segment_preds = all_preds[:n_samples]
            segment_trues = all_trues[:n_samples]

            # Calculate metrics
            accuracy = accuracy_score(segment_trues, segment_preds)

            # Handle cases where there might be only one class in predictions
            try:
                precision = precision_score(segment_trues, segment_preds)
            except:
                precision = 0.0

            try:
                recall = recall_score(segment_trues, segment_preds)
            except:
                recall = 0.0

            try:
                f1 = f1_score(segment_trues, segment_preds)
            except:
                f1 = 0.0

            # Calculate trading metrics
            if self.is_shorting:
                # For shorting strategy: we profit from both correct predictions
                correct_positives = np.logical_and(segment_preds == 1, segment_trues == 1)
                correct_negatives = np.logical_and(segment_preds == 0, segment_trues == 0)

                # Profit comes from correct predictions in both directions
                profitable_trades = correct_positives.sum() + correct_negatives.sum()
                total_trades = len(segment_preds)
            else:
                # For no-shorting strategy: we only profit from correct positive predictions
                profitable_trades = np.logical_and(segment_preds == 1, segment_trues == 1).sum()
                # We lose on false positives but not on true negatives or false negatives
                unprofitable_trades = np.logical_and(segment_preds == 1, segment_trues == 0).sum()
                total_trades = profitable_trades + unprofitable_trades

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            # Print results for this segment
            print(
                f"{p * 100:^15.1f}% | {n_samples:^10d} | {accuracy * 100:^10.2f}% | {precision * 100:^10.2f}% | {recall * 100:^10.2f}% | {f1 * 100:^10.2f}% | {win_rate * 100:^10.2f}%")

        # Equal segment analysis: split into equal-sized portions
        print("\nEqual Segment Analysis (from validation boundary to test end):")
        print("-" * 80)
        print(
            f"{'Segment':^15} | {'Samples':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1':^10} | {'Win Rate':^10}")
        print("-" * 80)

        # Define number of segments (e.g., quartiles)
        num_segments = 4
        total_samples = len(all_preds)
        segment_size = total_samples // num_segments

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else total_samples

            segment_preds = all_preds[start_idx:end_idx]
            segment_trues = all_trues[start_idx:end_idx]

            # Skip empty segments
            if len(segment_preds) == 0:
                continue

            # Calculate metrics
            accuracy = accuracy_score(segment_trues, segment_preds)

            # Handle cases where there might be only one class in predictions
            try:
                precision = precision_score(segment_trues, segment_preds)
            except:
                precision = 0.0

            try:
                recall = recall_score(segment_trues, segment_preds)
            except:
                recall = 0.0

            try:
                f1 = f1_score(segment_trues, segment_preds)
            except:
                f1 = 0.0

            # Calculate trading metrics
            if self.is_shorting:
                # For shorting strategy: we profit from both correct predictions
                correct_positives = np.logical_and(segment_preds == 1, segment_trues == 1)
                correct_negatives = np.logical_and(segment_preds == 0, segment_trues == 0)

                # Profit comes from correct predictions in both directions
                profitable_trades = correct_positives.sum() + correct_negatives.sum()
                total_trades = len(segment_preds)
            else:
                # For no-shorting strategy: we only profit from correct positive predictions
                profitable_trades = np.logical_and(segment_preds == 1, segment_trues == 1).sum()
                # We lose on false positives but not on true negatives or false negatives
                unprofitable_trades = np.logical_and(segment_preds == 1, segment_trues == 0).sum()
                total_trades = profitable_trades + unprofitable_trades

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            # Print results for this segment
            segment_name = f"{i + 1}/{num_segments}"
            print(
                f"{segment_name:^15} | {len(segment_preds):^10d} | {accuracy * 100:^10.2f}% | {precision * 100:^10.2f}% | {recall * 100:^10.2f}% | {f1 * 100:^10.2f}% | {win_rate * 100:^10.2f}%")

        # Statistical analysis with minimal libraries
        # Calculate correlation between position and correctness
        positions = np.arange(len(all_preds))
        correctness = (all_preds == all_trues).astype(int)

        # Manual correlation calculation if scipy not available
        try:
            from scipy.stats import pearsonr
            correlation, p_value = pearsonr(positions, correctness)
            significance = "Significant" if p_value < 0.05 else "Not significant"

            print("\nStatistical Analysis:")
            print(f"Correlation between distance and accuracy: {correlation:.4f}")
            print(f"P-value: {p_value:.6f} ({significance})")

            if abs(correlation) < 0.1:
                effect_strength = "No significant"
            elif abs(correlation) < 0.3:
                effect_strength = "Weak"
            elif abs(correlation) < 0.5:
                effect_strength = "Moderate"
            else:
                effect_strength = "Strong"

            effect_direction = "positive" if correlation > 0 else "negative"

            print(f"\nRECENCY EFFECT CONCLUSION: {effect_strength} {effect_direction} recency effect detected.")

            if correlation < -0.1 and p_value < 0.05:
                print("As predictions move further from the validation set, accuracy tends to DECREASE.")
                print("This suggests your model is experiencing a recency effect.")
            elif correlation > 0.1 and p_value < 0.05:
                print("Interestingly, as predictions move further from the validation set, accuracy tends to INCREASE.")
                print("This is unusual and suggests the model may be better at predicting later periods.")
            else:
                print("No significant recency effect detected. The model's performance is relatively")
                print("consistent regardless of temporal distance from the validation set.")
        except ImportError:
            # Fallback to simple manual correlation
            mean_pos = np.mean(positions)
            mean_corr = np.mean(correctness)
            numerator = np.sum((positions - mean_pos) * (correctness - mean_corr))
            denominator = np.sqrt(np.sum((positions - mean_pos) ** 2) * np.sum((correctness - mean_corr) ** 2))
            correlation = numerator / denominator if denominator != 0 else 0

            print("\nBasic Statistical Analysis:")
            print(f"Correlation between distance and accuracy: {correlation:.4f}")

            if correlation < -0.1:
                print("Evidence of potential recency effect: accuracy decreases with distance from validation set")
            elif correlation > 0.1:
                print("Unusual pattern: accuracy increases with distance from validation set")
            else:
                print("No significant recency effect detected")

        return

    def calculate_returns(self, setting, test=0):
        """
        Calculates theoretical returns from trading signals and prints detailed results
        for manual verification, including timestamps, price data, predictions, and returns.
        """
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # Load original dataset to get timestamps and prices
        try:
            print(f"Loading original data from {os.path.join(test_data.root_path, test_data.data_path)}")
            original_df = pd.read_csv(os.path.join(test_data.root_path, test_data.data_path))

            # Filter to test data
            if 'split' in original_df.columns:
                test_df = original_df[original_df['split'] == 'test'].reset_index(drop=True)
                print(f"Found {len(test_df)} test samples in the dataset")
            else:
                # Fallback to default splitting
                test_df = None

            # Convert timestamp
            if test_df is not None and 'date' in test_df.columns:
                if pd.api.types.is_integer_dtype(test_df['date']):
                    test_df['date'] = pd.to_datetime(test_df['date'], unit='s')
        except Exception as e:
            print(f"Error loading original CSV: {e}")
            test_df = None

        # Collect model predictions
        all_preds = []
        all_trues = []
        all_probs = []

        print("\n===== RETURN CALCULATION ANALYSIS =====")
        print("Collecting test set predictions...")

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if batch_x.size(0) == 0:
                    continue


                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs for binary classification
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Convert logits to probabilities with sigmoid
                output_probs = torch.sigmoid(outputs_last).detach().cpu().numpy()

                # Get binary predictions (threshold at 0.5)
                output_binary = (output_probs > 0.5).astype(np.float32)

                print(f"DEBUG: For sample {i}, output_binary={output_binary}, batch_y_last={batch_y_last}")

                # Get true labels
                true_labels = batch_y_last.detach().cpu().numpy()

                # Store predictions
                for idx in range(len(output_binary)):
                    all_preds.append(output_binary[idx][0])
                    all_trues.append(true_labels[idx][0])
                    # Add in calculate_returns where the "True" column values are set:
                    print(f"DEBUG: For sample {i}, batch_y_last={batch_y_last}, converted to True={all_trues[i]}")
                    all_probs.append(output_probs[idx][0])

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        all_probs = np.array(all_probs)

        # In calculate_returns, after collecting all_preds, all_trues:
        if hasattr(test_data, 'sample_metadata'):
            print("\nLabel Verification with Detailed Metadata:")
            print("-" * 140)
            print(
                f"{'Sample':<8} | {'Orig Idx':<8} | {'Pred Idx':<8} | {'Pred Price':<12} | {'Future Price':<12} | {'Change':<10} | {'Stored Label':<12} | {'Model True':<10} | {'Match':<5}")
            print("-" * 140)

            for i, meta in enumerate(test_data.sample_metadata):
                if i >= len(all_trues):
                    break

                price_change = ((meta['future_price'] - meta['pred_price']) / meta['pred_price']) * 100.0
                calculated_direction = "up" if meta['future_price'] > meta['pred_price'] else "down/same"
                stored_label = meta['label']
                model_true = all_trues[i]
                match = "✓" if stored_label == model_true else "✗"

                print(
                    f"{i:<8} | {meta['orig_idx']:<8} | {meta['pred_idx']:<8} | {meta['pred_price']:<12.2f} | {meta['future_price']:<12.2f} | {price_change:<10.2f}% | {stored_label:<12.1f} | {model_true:<10.1f} | {match:<5}")

            # Show summary statistics
            matches = sum(
                1 for i, meta in enumerate(test_data.sample_metadata[:len(all_trues)]) if meta['label'] == all_trues[i])
            total = min(len(test_data.sample_metadata), len(all_trues))
            print(f"\nLabel Match Rate: {matches}/{total} ({matches / total * 100:.2f}%)")

        print(f"Collected {len(all_preds)} predictions from test set")



        # Now match these predictions with timestamps and prices
        seq_len = test_data.seq_len
        pred_len = test_data.pred_len

        # Add near the beginning of calculate_returns, after loading test_df
        print("\n===== DATASET INDEX MAPPING =====")
        print(f"test_data object type: {type(test_data)}")
        print(f"test_data.seq_len: {test_data.seq_len}, test_data.pred_len: {test_data.pred_len}")

        if hasattr(test_data, 'active_indices'):
            print(f"Does test_data have active_indices? Yes, length: {len(test_data.active_indices)}")
            print(f"First 5 active indices: {test_data.active_indices[:5]}")
        else:
            print("test_data does not have active_indices attribute")

        # Add after collecting predictions
        print("\n===== RAW PREDICTIONS SUMMARY =====")
        print(f"Collected {len(all_preds)} predictions:")
        for i in range(min(5, len(all_preds))):
            print(f"Prediction {i}: Pred={all_preds[i]}, True={all_trues[i]}, Prob={all_probs[i]:.4f}")
        # Add detailed debugging for the price/timestamp mapping
        print("\n===== PRICE MAPPING DETAILS =====")
        print(f"test_df has {len(test_df) if test_df is not None else 0} rows")
        if test_df is not None:
            for i in range(min(5, len(all_preds))):
                base_idx = i
                current_idx = base_idx + seq_len

                print(f"\nSample {i} mapping:")
                print(f"  Base index in test dataset: {base_idx}")
                print(f"  Current index (base + seq_len={seq_len}): {current_idx}")

                if current_idx < len(test_df):
                    timestamp = test_df.iloc[current_idx]['date']
                    current_price = test_df.iloc[current_idx]['close']

                    # Map this back to original dataset if possible
                    if hasattr(test_data, 'active_indices') and base_idx < len(test_data.active_indices):
                        orig_idx = test_data.active_indices[base_idx]
                        orig_prediction_idx = orig_idx + seq_len
                        print(f"  Mapped to original dataset: Base={orig_idx}, Prediction point={orig_prediction_idx}")

                    print(f"  Timestamp: {timestamp}")
                    print(f"  Current price: {current_price}")

                    if current_idx + pred_len < len(test_df):
                        future_price = test_df.iloc[current_idx + pred_len]['close']
                        print(f"  Future price (+{pred_len} steps): {future_price}")
                        print(f"  Price change: {((future_price - current_price) / current_price):.2%}")
                        print(f"  True label: {all_trues[i]} (should be 1 if price went up)")
                    else:
                        print("  Future price: Not available (beyond dataset)")

        ####
        all_timestamps = []
        all_close_prices = []
        all_future_prices = []

        # For each prediction point, adjust the index mapping
        for i in range(len(all_preds)):
            base_idx = i
            # Use the active_indices attribute to map to original dataset if available
            if hasattr(test_data, 'active_indices') and base_idx < len(test_data.active_indices):
                orig_idx = test_data.active_indices[base_idx]
                current_idx = orig_idx + seq_len
            else:
                current_idx = base_idx + seq_len

            # Get timestamp and price data using the correct indices
            if current_idx < len(test_df):
                timestamp = test_df.iloc[current_idx]['date']
                current_price = test_df.iloc[current_idx]['close']

                # Get future price correctly
                if current_idx + pred_len < len(test_df):
                    future_price = test_df.iloc[current_idx + pred_len]['close']
                    # Verify alignment
                    expected_label = 1.0 if future_price > current_price else 0.0
                    # Print verification
                    print(
                        f"Idx {i}: True label={all_trues[i]}, Calculated label={expected_label}, Current={current_price}, Future={future_price}")

        # For each prediction point, get the corresponding data from the test dataset
        if test_df is not None:
            for i in range(len(all_preds)):
                base_idx = i  # Sample index in the dataset
                current_idx = base_idx + seq_len  # Index where prediction is made

                if current_idx < len(test_df):
                    # Get timestamp and current price
                    timestamp = test_df.iloc[current_idx]['date']
                    current_price = test_df.iloc[current_idx]['close']

                    # Get future price if available (for validation only)
                    if current_idx + pred_len < len(test_df):
                        future_price = test_df.iloc[current_idx + pred_len]['close']
                    else:
                        future_price = np.nan

                    all_timestamps.append(timestamp)
                    all_close_prices.append(current_price)
                    all_future_prices.append(future_price)
                else:
                    all_timestamps.append(f"sample_{i}")
                    all_close_prices.append(np.nan)
                    all_future_prices.append(np.nan)
        else:
            # If no test_df, use placeholders
            all_timestamps = [f"sample_{i}" for i in range(len(all_preds))]
            all_close_prices = [np.nan] * len(all_preds)
            all_future_prices = [np.nan] * len(all_preds)

        # Calculate returns using the TRUE LABELS from the model
        # Do NOT recalculate from prices, as they might not match with how labels were generated
        returns_per_trade = np.zeros(len(all_preds))

        for i in range(len(all_preds)):
            pred = all_preds[i]
            true = all_trues[i]  # Use the true label that the model was trained with

            # Calculate return (use a fixed placeholder percentage if prices not available)
            return_pct = 0.01 if true == 1 else -0.01

            # If we have actual prices, use them for display but not for return calculation
            if i < len(all_future_prices) and not np.isnan(all_close_prices[i]) and not np.isnan(all_future_prices[i]):
                # Just for validation - don't use this for returns calculation
                actual_change = (all_future_prices[i] - all_close_prices[i]) / all_close_prices[i]
                # Uncomment to debug:
                # print(f"Index {i}: Label={true}, Actual change: {actual_change:.2%}")

            if self.is_shorting:
                # For shorting strategy
                if (pred == 1 and true == 1) or (pred == 0 and true == 0):
                    # Correct prediction
                    returns_per_trade[i] = abs(return_pct)
                else:
                    # Incorrect prediction
                    returns_per_trade[i] = -abs(return_pct)
            else:
                # For long-only strategy
                if pred == 1:
                    returns_per_trade[i] = return_pct
                else:
                    returns_per_trade[i] = 0.0  # No trade

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns_per_trade) - 1
        uncompounded_returns = np.cumsum(returns_per_trade)

        # Modify the existing print section to add original index information
        print("\nDetailed Trading Results with Original Indices:")
        print("-" * 120)
        print(
            f"{'Timestamp':<20} | {'Unix Time':<12} | {'Orig Idx':<8} | {'Close Price':<12} | {'Pred':<5} | {'True':<5} | {'Prob':<8} | {'Return':>8} | {'Cum Return':>12}")
        print("-" * 120)

        for i in range(len(all_preds)):
            # Format timestamp for display
            if isinstance(all_timestamps[i], pd.Timestamp):
                # ts_str = all_timestamps[i].strftime('%Y-%m-%d %H:%M')
                ts_str = str(all_timestamps[i])
                # Get original unix timestamp if possible
                unix_time = int(all_timestamps[i].timestamp()) if hasattr(all_timestamps[i], 'timestamp') else 'N/A'
            else:
                ts_str = str(all_timestamps[i])
                unix_time = 'N/A'

            # Get original index if available
            orig_idx = 'N/A'
            if hasattr(test_data, 'active_indices') and (i + seq_len) < len(test_data.active_indices):
                orig_idx = test_data.active_indices[i] + seq_len

            print(
                f"{ts_str:<20} | {unix_time:<12} | {orig_idx:<8} | {all_close_prices[i]:<12.2f} | {all_preds[i]:<5.0f} | "
                f"{all_trues[i]:<5.0f} | {all_probs[i]:<8.4f} | {returns_per_trade[i]:>8.2%} | {cumulative_returns[i]:>12.2%}")

            # Validate labels against original data if possible
            if i < 10 and test_df is not None and orig_idx != 'N/A' and orig_idx - test_data.pred_len >= 0:
                try:
                    base_idx = orig_idx - seq_len
                    if base_idx < len(test_df) and (base_idx + test_data.pred_len) < len(test_df):
                        current_price = test_df.iloc[base_idx]['close']
                        future_price = test_df.iloc[base_idx + test_data.pred_len]['close']
                        expected_label = 1.0 if future_price > current_price else 0.0
                        print(f"    VALIDATION: Base idx={base_idx}, Current price={current_price}, "
                              f"Future price(+{test_data.pred_len})={future_price}, "
                              f"Expected label={expected_label}, Actual label={all_trues[i]}")
                        if expected_label != all_trues[i]:
                            print(f"    *** LABEL MISMATCH DETECTED ***")
                except Exception as e:
                    print(f"    Error validating label: {e}")
        # Print trading performance summary
        print("\nTrading Performance Summary:")
        print("-" * 50)

        # Count trades (non-zero returns)
        trades = returns_per_trade != 0
        trade_count = np.sum(trades)

        print(f"Total trades: {trade_count}")
        print(f"Profitable trades: {np.sum(returns_per_trade > 0)}")
        print(f"Unprofitable trades: {np.sum(returns_per_trade < 0)}")

        if trade_count > 0:
            win_rate = np.sum(returns_per_trade > 0) / trade_count
            print(f"Win rate: {win_rate:.2%}")
            print(f"Average return per trade: {np.mean(returns_per_trade[trades]):.2%}")
        else:
            print("Win rate: N/A (no trades)")
            print("Average return per trade: N/A (no trades)")

        print(f"Final cumulative return (compounded): {cumulative_returns[-1]:.2%}")
        print(f"Final cumulative return (uncompounded): {uncompounded_returns[-1]:.2%}")

        # Calculate additional metrics
        if trade_count > 0:
            sharpe_ratio = np.mean(returns_per_trade[trades]) / (np.std(returns_per_trade[trades]) + 1e-10) * np.sqrt(
                252)  # Annualized
            max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)

            print(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
            print(f"Maximum Drawdown: {max_drawdown:.2%}")

        # Return the results for further analysis
        return {
            'predictions': all_preds,
            'actual': all_trues,
            'probabilities': all_probs,
            'close_prices': all_close_prices,
            'timestamps': all_timestamps,
            'returns_per_trade': returns_per_trade,
            'cumulative_returns': cumulative_returns,
            'uncompounded_returns': uncompounded_returns
        }

    def calculate_class_distribution(self):
        """Calculate and store the class distribution from the training data"""
        train_data, _ = self._get_data(flag='train')

        # Extract all binary labels from the dataset
        all_labels = []
        for i in range(len(train_data)):
            _, batch_y, _, _ = train_data[i]
            # Get only the last timestep prediction (for 4 steps ahead)
            label = batch_y[-1, 0]  # Shape is [seq_len+pred_len, 1]
            all_labels.append(label)

        all_labels = np.array(all_labels)
        positive_ratio = np.mean(all_labels)
        negative_ratio = 1 - positive_ratio

        self.class_distribution = {
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'pos_weight': negative_ratio / positive_ratio if positive_ratio > 0 else 1.0
        }

        print(f"\nClass distribution in training data:")
        print(f"Positive examples (price increases): {positive_ratio:.2%}")
        print(f"Negative examples (price decreases or stays): {negative_ratio:.2%}")
        print(f"Pos_weight for BCEWithLogitsLoss: {self.class_distribution['pos_weight']:.4f}")
        print(f"Trading strategy: {'Shorting enabled' if self.is_shorting else 'No shorting (holding only)'}")
        if not self.is_shorting:
            print(f"Precision factor for non-shorting strategy: {self.precision_factor}")
        print()

    def _select_criterion(self):
        # Use our custom trading loss
        if self.class_distribution is not None and not self.auto_weight:
            # Use pre-calculated class weights if available
            pos_weight = self.class_distribution['pos_weight']
        else:
            pos_weight = None

        criterion = TradingBCELoss(
            is_shorting=self.is_shorting,
            precision_factor=self.precision_factor,
            auto_weight=self.auto_weight,
            manual_pos_weight=pos_weight
        )
        return criterion

    def _process_outputs(self, outputs, batch_y):
        # Process outputs and targets for binary classification
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # For binary classification, we only care about the last timestep prediction
        # The last prediction corresponds to the relative price change 4 timesteps ahead
        outputs_last = outputs[:, -1, :]
        batch_y_last = batch_y[:, -1, :]

        # Debug output processing
        # if outputs.shape[0] < 5:  # Only for small batches to avoid clutter
        #     print("\n===== OUTPUT PROCESSING DEBUG =====")
        #     print(f"Original batch_y shape: {batch_y.shape}")
        #     print(
        #         f"After slicing: batch_y[:, -self.args.pred_len:, f_dim:] shape: {batch_y[:, -self.args.pred_len:, f_dim:].shape}")
        #     print(f"Final batch_y_last shape: {batch_y_last.shape}")
        #
        #     # Show values for first few samples
        #     for i in range(min(3, outputs.shape[0])):
        #         print(f"Sample {i}: outputs_last={outputs_last[i].detach().cpu().numpy()}, "
        #               f"batch_y_last={batch_y_last[i].detach().cpu().numpy()}")

        return outputs_last, batch_y_last, outputs, batch_y

    def _calculate_accuracy(self, outputs, targets):
        """Calculate accuracy for binary predictions in a simpler way"""
        with torch.no_grad():
            # For binary classification with logits:
            # - If logit > 0, prediction is 1
            # - If logit <= 0, prediction is 0
            predictions = (outputs > 0).float()
            correct = (predictions == targets).float().sum()
            total = targets.size(0)
            accuracy = correct / total
        return accuracy.item()

    def _calculate_metrics(self, outputs, targets):
        """Calculate multiple metrics for binary predictions"""
        with torch.no_grad():
            # Convert to binary predictions
            predictions = (outputs > 0).float()
            # Calculate metrics
            correct = (predictions == targets).float().sum()
            total = targets.size(0)

            # Convert to numpy for sklearn metrics
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()

            # Calculate basic metrics
            accuracy = correct / total

            # Handle case where all predictions are one class
            if len(np.unique(pred_np)) == 1 or len(np.unique(target_np)) == 1:
                precision = 0.0 if np.sum(pred_np) > 0 else 1.0
                recall = 0.0 if np.sum(target_np) > 0 else 1.0
                f1 = 0.0
            else:
                precision = precision_score(target_np, pred_np)
                recall = recall_score(target_np, pred_np)
                f1 = f1_score(target_np, pred_np)

        return {
            'accuracy': accuracy.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Skip empty batches
                if batch_x.size(0) == 0:
                    continue

                # Move tensors to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Reshape batch_y to ensure it has 3 dimensions [batch, seq, feature]
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)

                # Now we can safely index with 3 dimensions
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs for binary classification
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Calculate loss (binary cross-entropy)
                loss = criterion(outputs_last, batch_y_last)
                total_loss.append(loss.item())

                # Calculate metrics
                batch_metrics = self._calculate_metrics(outputs_last, batch_y_last)
                for k, v in batch_metrics.items():
                    total_metrics[k].append(v)

        avg_loss = np.average(total_loss)
        avg_metrics = {k: np.average(v) for k, v in total_metrics.items()}

        self.model.train()
        return avg_loss, avg_metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Skip empty batches
                if batch_x.size(0) == 0:
                    continue

                # Move tensors to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Reshape batch_y to ensure it has 3 dimensions [batch, seq, feature]
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)

                # Now we can safely index with 3 dimensions
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # Process outputs for binary classification
                        outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                        # Calculate loss
                        loss = criterion(outputs_last, batch_y_last)
                        train_loss.append(loss.item())

                        # Calculate metrics for batch
                        batch_metrics = self._calculate_metrics(outputs_last, batch_y_last)
                        for k, v in batch_metrics.items():
                            train_metrics[k].append(v)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # Process outputs for binary classification
                    outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                    # Calculate loss
                    loss = criterion(outputs_last, batch_y_last)
                    train_loss.append(loss.item())

                    # Calculate metrics for batch
                    batch_metrics = self._calculate_metrics(outputs_last, batch_y_last)
                    for k, v in batch_metrics.items():
                        train_metrics[k].append(v)

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}, accuracy: {3:.2f}%, precision: {4:.2f}%, recall: {5:.2f}%".format(
                            i + 1, epoch + 1, loss.item(),
                            batch_metrics['accuracy'] * 100,
                            batch_metrics['precision'] * 100,
                            batch_metrics['recall'] * 100))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_metrics = {k: np.average(v) * 100 for k, v in train_metrics.items()}  # Convert to percentages

            vali_loss, vali_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)

            vali_metrics = {k: v * 100 for k, v in vali_metrics.items()}  # Convert to percentages
            test_metrics = {k: v * 100 for k, v in test_metrics.items()}  # Convert to percentages

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}, Train Acc: {3:.2f}%, P: {4:.2f}%, R: {5:.2f}%, F1: {6:.2f}% | "
                "Vali Loss: {7:.7f}, Vali Acc: {8:.2f}%, P: {9:.2f}%, R: {10:.2f}%, F1: {11:.2f}% | "
                "Test Loss: {12:.7f}, Test Acc: {13:.2f}%, P: {14:.2f}%, R: {15:.2f}%, F1: {16:.2f}%".format(
                    epoch + 1, train_steps, train_loss,
                    train_metrics['accuracy'], train_metrics['precision'], train_metrics['recall'], train_metrics['f1'],
                    vali_loss,
                    vali_metrics['accuracy'], vali_metrics['precision'], vali_metrics['recall'], vali_metrics['f1'],
                    test_loss,
                    test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1']))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # Directly check the raw label values in test_data
        print("\n==== DIRECT LABEL CHECK ====")
        for i in range(min(20, len(test_data))):
            sample_idx = i
            orig_idx = test_data.active_indices[i] if hasattr(test_data, 'active_indices') else None
            pred_idx = orig_idx + test_data.seq_len if orig_idx is not None else None
            raw_label = test_data.data_y[i][0] if i < len(test_data.data_y) else None

            expected_label = None
            if orig_idx is not None and pred_idx is not None:
                # Calculate expected label from raw data
                try:
                    original_df = pd.read_csv(os.path.join(test_data.root_path, test_data.data_path))
                    pred_price = original_df.iloc[pred_idx][test_data.target]
                    future_price = original_df.iloc[pred_idx + test_data.pred_len][test_data.target]
                    expected_label = 1.0 if future_price > pred_price else 0.0
                except Exception as e:
                    expected_label = f"Error: {e}"

            print(
                f"Sample {i}: orig_idx={orig_idx}, pred_idx={pred_idx}, raw_label={raw_label}, expected={expected_label}")




        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        probs = []  # Store probabilities for ROC curve

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs for binary classification - get only the final prediction
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Convert logits to probabilities with sigmoid
                output_probs = torch.sigmoid(outputs_last).detach().cpu().numpy()

                # Get binary predictions (threshold at 0.5)
                output_binary = (output_probs > 0.5).astype(np.float32)

                # Get true labels
                true_labels = batch_y_last.detach().cpu().numpy()

                preds.append(output_binary)
                probs.append(output_probs)
                trues.append(true_labels)

        # Concatenate all batches
        preds = np.concatenate(preds, axis=0)
        probs = np.concatenate(probs, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Create and visualize confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(trues, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Down', 'Predicted Up'],
                    yticklabels=['Actual Down', 'Actual Up'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix - ' + ('Shorting Enabled' if self.is_shorting else 'No Shorting (Long Only)'))
        plt.savefig(folder_path + 'confusion_matrix.png')
        plt.close()

        # Extract confusion matrix components
        TN, FP = cm[0, 0], cm[0, 1]  # True Negative, False Positive
        FN, TP = cm[1, 0], cm[1, 1]  # False Negative, True Positive

        # Calculate trading metrics based on confusion matrix
        if self.is_shorting:
            profitable_trades = TP + TN
            unprofitable_trades = FP + FN
        else:
            profitable_trades = TP
            unprofitable_trades = FP
            # TN and FN are ignored in no-shorting strategy as we don't trade on these signals

        # Print detailed confusion matrix breakdown
        print('\nConfusion Matrix Breakdown:')
        print(f'  True Positives (correctly predicted price increases): {TP}')
        print(f'  True Negatives (correctly predicted price decreases): {TN}')
        print(f'  False Positives (incorrectly predicted price increases): {FP}')
        print(f'  False Negatives (incorrectly predicted price decreases): {FN}')

        print('\nTrading Interpretation:')
        if self.is_shorting:
            print(f'  Profitable Long Trades: {TP} (from true positives)')
            print(f'  Profitable Short Trades: {TN} (from true negatives)')
            print(f'  Unprofitable Long Trades: {FP} (from false positives)')
            print(f'  Unprofitable Short Trades: {FN} (from false negatives)')
            print(f'  Total Trades: {TP + TN + FP + FN}')
        else:
            print(f'  Profitable Long Trades: {TP} (from true positives)')
            print(f'  Unprofitable Long Trades: {FP} (from false positives)')
            print(f'  Ignored Signals (no position taken): {TN + FN} (all negative predictions)')
            print(f'  Total Trades: {TP + FP}')

        # Calculate trading performance metrics
        if self.is_shorting:
            # For shorting strategy: we profit from both correct predictions
            correct_positives = np.logical_and(preds == 1, trues == 1)
            correct_negatives = np.logical_and(preds == 0, trues == 0)

            incorrect_positives = np.logical_and(preds == 1, trues == 0)
            incorrect_negatives = np.logical_and(preds == 0, trues == 1)

            # Profit comes from correct predictions in both directions
            profitable_trades = correct_positives.sum() + correct_negatives.sum()
            unprofitable_trades = incorrect_positives.sum() + incorrect_negatives.sum()
        else:
            # For no-shorting strategy: we only profit from correct positive predictions
            # and avoid losses from incorrect negative predictions
            profitable_trades = np.logical_and(preds == 1, trues == 1).sum()
            # We lose on false positives but not on true negatives or false negatives
            unprofitable_trades = np.logical_and(preds == 1, trues == 0).sum()

        # Calculate win rate and profit metrics
        total_trades = profitable_trades + unprofitable_trades
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        profit_factor = profitable_trades / unprofitable_trades if unprofitable_trades > 0 else float('inf')

        # Calculate standard classification metrics
        accuracy = accuracy_score(trues, preds)
        precision = precision_score(trues, preds, zero_division=0)
        recall = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds, zero_division=0)

        # Print detailed performance metrics
        print('\nTest metrics:')
        print('Classification Performance:')
        print('  Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
            accuracy * 100, precision * 100, recall * 100, f1 * 100))

        print('\nTrading Performance:')
        print('  Strategy: {}'.format('Shorting enabled' if self.is_shorting else 'No shorting (holding only)'))
        print('  Profitable Trades: {}, Unprofitable Trades: {}, Total Trades: {}'.format(
            profitable_trades, unprofitable_trades, total_trades))
        print('  Win Rate: {:.2f}%'.format(win_rate * 100))
        print('  Profit Factor: {:.2f}'.format(profit_factor))

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save all metrics in a structured format
        metrics = {
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'trading': {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'profitable_trades': profitable_trades,
                'unprofitable_trades': unprofitable_trades,
                'total_trades': total_trades,
                'is_shorting': self.is_shorting
            }
        }

        np.save(folder_path + 'metrics.npy', metrics)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'prob.npy', probs)
        np.save(folder_path + 'true.npy', trues)

        # Write results to file
        f = open("result_logits_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write(
            'Trading Strategy: {}\n'.format('Shorting enabled' if self.is_shorting else 'No shorting (holding only)'))
        f.write('Classification - Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%\n'.format(
            accuracy * 100, precision * 100, recall * 100, f1 * 100))
        f.write('Trading - Win Rate: {:.2f}%, Profit Factor: {:.2f}\n'.format(win_rate * 100, profit_factor))
        f.write('\n\n')
        f.close()

        print("\nPerforming recency effect analysis...")
        self.analyze_recency_effect(setting, test=test)

        print("\nCalculating trading returns...")
        self.calculate_returns(setting, test=test)

        return

    def verify_test_labels(self, setting, test=0):
        """
        Directly verifies the labels in the test dataset against the original data
        """
        test_data, _ = self._get_data(flag='test')
        print("\n===== DIRECT LABEL VERIFICATION =====")

        # Load original dataset
        try:
            original_df = pd.read_csv(os.path.join(test_data.root_path, test_data.data_path))
            print(f"Loaded original dataset: {len(original_df)} rows")

            # Test set filtering
            if 'split' in original_df.columns:
                test_df = original_df[original_df['split'] == 'test'].reset_index(drop=True)
                print(f"Test data: {len(test_df)} rows")
            else:
                test_df = None
                print("No 'split' column found in dataset")

            # Access binary labels from test_data if available
            if hasattr(test_data, 'data_y') and hasattr(test_data, 'active_indices'):
                print("\nVerifying first 20 test samples:")
                print(
                    f"{'Index':<8} | {'Orig Idx':<8} | {'Base+Seq':<8} | {'Timestamp':<20} | {'Current Price':<14} | {'Future Price':<14} | {'Expected Label':<14} | {'Actual Label':<12} | {'Match':<5}")
                print("-" * 120)

                # Add this to verify_test_labels:
                for i in range(min(10, len(test_data.data_y))):
                    print(f"Test sample {i}: Label in dataset = {test_data.data_y[i][0]}")

                for i in range(min(20, len(test_data.active_indices))):
                    # Get original index
                    orig_idx = test_data.active_indices[i]

                    # Get prediction point index
                    pred_idx = orig_idx + test_data.seq_len

                    # Get actual label from test_data
                    actual_label = test_data.data_y[i][0] if i < len(test_data.data_y) else 'N/A'


                    if test_df is not None and orig_idx < len(test_df) and orig_idx + test_data.pred_len < len(test_df):
                        # Get timestamp
                        timestamp = test_df.iloc[orig_idx]['date']
                        if pd.api.types.is_integer_dtype(type(timestamp)):
                            timestamp = pd.to_datetime(timestamp, unit='s')
                        ts_str = timestamp.strftime('%Y-%m-%d %H:%M') if isinstance(timestamp, pd.Timestamp) else str(
                            timestamp)

                        # Get prices
                        current_price = test_df.iloc[orig_idx]['close']
                        future_price = test_df.iloc[orig_idx + test_data.pred_len]['close']

                        # Calculate expected label
                        expected_label = 1.0 if future_price > current_price else 0.0

                        # Check match
                        match = "✓" if expected_label == actual_label else "✗"

                        print(
                            f"{i:<8} | {orig_idx:<8} | {pred_idx:<8} | {ts_str:<20} | {current_price:<14.2f} | {future_price:<14.2f} | {expected_label:<14.1f} | {actual_label:<12} | {match:<5}")
                    else:
                        print(
                            f"{i:<8} | {orig_idx:<8} | {pred_idx:<8} | {'N/A':<20} | {'N/A':<14} | {'N/A':<14} | {'N/A':<14} | {actual_label:<12} | {'N/A':<5}")

        except Exception as e:
            print(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()

        return

    def verify_labels(self, setting):
        """
        Verify that labels in test data match the actual price movements from prediction point
        """
        test_data, _ = self._get_data(flag='test')

        print("\n===== LABEL VERIFICATION =====")
        # Load original dataset
        try:
            original_df = pd.read_csv(os.path.join(test_data.root_path, test_data.data_path))
            print(f"Loaded original dataset with {len(original_df)} rows")

            # Test set filtering
            if 'split' in original_df.columns:
                test_df = original_df[original_df['split'] == 'test'].reset_index(drop=True)
                print(f"Test data: {len(test_df)} rows")
            else:
                test_df = None
                print("No 'split' column found in dataset")

            # Verify first 10 test samples
            print("\nVerifying first 10 test labels:")
            print("-" * 120)
            print(
                f"{'Sample':<10} | {'Base Idx':<10} | {'Pred Idx':<10} | {'Pred Price':<12} | {'Future Price':<12} | {'Change':<10} | {'Label':<8} | {'Correct':<8}")
            print("-" * 120)

            for i in range(min(10, len(test_data.active_indices))):
                base_idx = test_data.active_indices[i]
                pred_idx = base_idx + test_data.seq_len

                if pred_idx < len(original_df) and pred_idx + test_data.pred_len < len(original_df):
                    pred_price = original_df.iloc[pred_idx][test_data.target]
                    future_price = original_df.iloc[pred_idx + test_data.pred_len][test_data.target]
                    price_change = (future_price - pred_price) / pred_price * 100.0
                    expected_label = 1.0 if future_price > pred_price else 0.0

                    # Get actual label from dataset
                    actual_label = test_data.data_y[i][0] if i < len(test_data.data_y) else 'N/A'

                    correct = "✓" if expected_label == actual_label else "✗"
                    print(
                        f"{i:<10} | {base_idx:<10} | {pred_idx:<10} | {pred_price:<12.2f} | {future_price:<12.2f} | {price_change:<10.2f}% | {actual_label:<8} | {correct:<8}")
                else:
                    print(
                        f"{i:<10} | {base_idx:<10} | {pred_idx:<10} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10} | {'N/A':<8} | {'N/A':<8}")

        except Exception as e:
            print(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()

