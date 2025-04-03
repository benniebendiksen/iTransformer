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

        # Debug info about test_data
        print("\n===== TEST DATA INSPECTION =====")
        print(f"test_data type: {type(test_data)}")
        print(f"Has 'active_indices': {hasattr(test_data, 'active_indices')}")
        print(f"Has 'sequence_indices': {hasattr(test_data, 'sequence_indices')}")
        if hasattr(test_data, 'sequence_indices'):
            print(f"sequence_indices length: {len(test_data.sequence_indices)}")
        # Prepare for collecting predictions
        all_preds = []
        all_trues = []
        all_probs = []
        all_sample_indices = []
        all_metadata = []

        # print("\n===== COLLECTING PREDICTIONS WITH SEQUENCE TRACKING =====")
        self.model.eval()
        with torch.no_grad():
            # Process each sample separately to ensure correct indexing
            for i in range(len(test_data)):
                # Get a single sample
                batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[i]

                # Add batch dimension
                batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(self.device)
                batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(self.device)
                batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(self.device)
                batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Make prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Get prediction
                output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
                output_binary = (output_prob > 0.5).astype(np.float32)
                true_label = batch_y_last.detach().cpu().numpy()[0, 0]

                # Store results
                all_preds.append(output_binary)
                all_trues.append(true_label)
                all_probs.append(output_prob)
                all_sample_indices.append(i)

                # Store metadata if available
                meta = None
                if hasattr(test_data, 'sequence_indices') and i in test_data.sequence_indices:
                    meta = test_data.sequence_indices[i]
                elif hasattr(test_data, 'sample_metadata') and i < len(test_data.sample_metadata):
                    meta = test_data.sample_metadata[i]
                all_metadata.append(meta)

                # Debug output for first few samples
                #TODO: HANDLE THIS DEBUG CASE

                # if i < 10:
                #     print(f"Sample {i}: Pred={output_binary:.1f}, True={true_label:.1f}, Prob={output_prob:.4f}")
                #     if meta:
                #         orig_idx = meta.get('orig_start_idx') or meta.get('orig_idx')
                #         label = meta.get('label')
                #         print(f"  Metadata: orig_idx={orig_idx}, stored_label={label}")

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        all_probs = np.array(all_probs)

        print(f"Collected {len(all_preds)} predictions from test set")

        # Verify we collected predictions for the whole test set
        print(f"Collected {len(all_preds)} predictions from test set")
        # print(f"trues: {all_trues}")
        # print(f"preds: {all_preds}")

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

        This enhanced version calculates returns for multiple trading strategies:
        - Long-only: Only take long positions when predicting price increase
        - Short-only: Only take short positions when predicting price decrease
        - Both: Take long positions for predicted increases and short positions for predicted decreases

        Returns are calculated based on actual price changes, not placeholders.
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

        # Debug info about test_data
        print("\n===== TEST DATA INSPECTION =====")
        print(f"test_data type: {type(test_data)}")
        print(f"Has 'active_indices': {hasattr(test_data, 'active_indices')}")
        print(f"Has 'sequence_indices': {hasattr(test_data, 'sequence_indices')}")
        if hasattr(test_data, 'sequence_indices'):
            print(f"sequence_indices length: {len(test_data.sequence_indices)}")
        if hasattr(test_data, 'active_indices'):
            print(f"active_indices length: {len(test_data.active_indices)}")
            print(f"First 5 active indices: {test_data.active_indices[:5]}")

        # Add integrity check
        if hasattr(test_data, 'verify_indices_integrity'):
            print("\n===== VERIFYING TEST DATA INDICES INTEGRITY =====")
            indices_valid = test_data.verify_indices_integrity()
            print(f"Test data indices integrity: {'Valid' if indices_valid else 'COMPROMISED'}")

        # Prepare for collecting predictions
        all_preds = []
        all_trues = []
        all_probs = []
        all_metadata = []

        print("\n===== COLLECTING PREDICTIONS WITH SEQUENCE TRACKING =====")
        self.model.eval()
        with torch.no_grad():
            # Process each sample separately to ensure correct indexing
            for i in range(len(test_data)):
                # Get a single sample
                batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[i]

                # Add batch dimension
                batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(self.device)
                batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(self.device)
                batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(self.device)
                batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Make prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Get prediction
                output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
                output_binary = (output_prob > 0.5).astype(np.float32)
                true_label = batch_y_last.detach().cpu().numpy()[0, 0]

                # Store results
                all_preds.append(output_binary)
                all_trues.append(true_label)
                all_probs.append(output_prob)

                # Store metadata if available
                meta = None
                if hasattr(test_data, 'sequence_indices') and i in test_data.sequence_indices:
                    meta = test_data.sequence_indices[i]
                all_metadata.append(meta)

                # Debug output for first few samples
                if i < 10:
                    print(f"Sample {i}: Pred={output_binary:.1f}, True={true_label:.1f}, Prob={output_prob:.4f}")
                    if meta:
                        print(
                            f"  Metadata: orig_idx={meta['orig_start_idx']}, pred_price={meta['pred_price']:.2f}, future_price={meta['future_price']:.2f}, label={meta['label']}")

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        all_probs = np.array(all_probs)

        print(f"Collected {len(all_preds)} predictions from test set")

        # Show detailed label verification
        # TODO: handle this debug case
        # print("\nLabel Verification with Detailed Metadata:")
        # print("-" * 140)
        # print(
        #     f"{'Sample':<8} | {'Orig Idx':<8} | {'Pred Idx':<8} | {'Pred Price':<12} | {'Future Price':<12} | {'Change':<10} | {'Stored Label':<12} | {'Model True':<10} | {'Match':<5}")
        # print("-" * 140)
        #
        # for i, meta in enumerate(all_metadata):
        #     if i >= len(all_trues) or meta is None:
        #         continue
        #
        #     orig_idx = meta['orig_start_idx']
        #     pred_idx = meta['pred_idx']
        #     pred_price = meta['pred_price']
        #     future_price = meta['future_price']
        #     price_change = meta['price_change']
        #     stored_label = meta['label']
        #     model_true = all_trues[i]
        #     match = "✓" if abs(stored_label - model_true) < 0.01 else "✗"
        #
        #     print(
        #         f"{i:<8} | {orig_idx:<8} | {pred_idx:<8} | {pred_price:<12.2f} | {future_price:<12.2f} | {price_change:<10.2f}% | {stored_label:<12.1f} | {model_true:<10.1f} | {match:<5}")

        # Show summary statistics
        matches = sum(
            1 for i, meta in enumerate(all_metadata)
            if meta is not None and i < len(all_trues) and
            abs(meta['label'] - all_trues[i]) < 0.01)
        total = len([meta for meta in all_metadata if meta is not None])
        print(f"\nLabel Match Rate: {matches}/{total} ({matches / total * 100:.2f}%)")

        # Extract actual price changes from metadata
        actual_changes = []
        timestamps = []
        prices = []

        for i, meta in enumerate(all_metadata):
            if meta is not None:
                # Extract actual price change as decimal (not percentage)
                actual_changes.append(meta['price_change'] / 100.0)
                # Extract timestamp if available from test_df
                orig_idx = meta['orig_start_idx']
                pred_idx = meta['pred_idx']
                if test_df is not None and pred_idx < len(test_df):
                    timestamp = test_df.iloc[pred_idx]['date']
                else:
                    timestamp = f"sample_{i}"
                timestamps.append(timestamp)
                prices.append(meta['pred_price'])
            else:
                actual_changes.append(0.0)
                timestamps.append(f"sample_{i}")
                prices.append(np.nan)

        # Convert to numpy array
        actual_changes = np.array(actual_changes)

        # Calculate returns for different trading strategies
        # 1. Long-only strategy (take long positions only when predicting price increases)
        long_only_returns = np.zeros(len(all_preds))
        for i in range(len(all_preds)):
            if all_preds[i] == 1:  # Predicted price increase, take long position
                long_only_returns[i] = actual_changes[i]  # Gain/lose based on actual price change

        # 2. Short-only strategy (take short positions only when predicting price decreases)
        short_only_returns = np.zeros(len(all_preds))
        for i in range(len(all_preds)):
            if all_preds[i] == 0:  # Predicted price decrease, take short position
                short_only_returns[i] = -actual_changes[i]  # Gain when price falls, lose when price rises

        # 3. Combined strategy (take long for predicted increases, short for predicted decreases)
        combined_returns = np.zeros(len(all_preds))
        for i in range(len(all_preds)):
            if all_preds[i] == 1:  # Predicted price increase, take long position
                combined_returns[i] = actual_changes[i]
            else:  # Predicted price decrease, take short position
                combined_returns[i] = -actual_changes[i]

        # Calculate metrics for each strategy
        strategy_names = ["Long-Only", "Short-Only", "Combined (Long+Short)"]
        strategy_returns = [long_only_returns, short_only_returns, combined_returns]

        # Helper function to calculate trading metrics
        def calculate_trading_metrics(returns):
            if len(returns) == 0:
                return {
                    'trades': 0,
                    'profitable_trades': 0,
                    'unprofitable_trades': 0,
                    'win_rate': 0.0,
                    'avg_return': 0.0,
                    'cumulative_return': 0.0,
                    'uncompounded_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                }

            # Calculate standard metrics
            trades = np.sum(returns != 0)
            profitable_trades = np.sum(returns > 0)
            unprofitable_trades = np.sum(returns < 0)
            win_rate = profitable_trades / trades if trades > 0 else 0.0
            avg_return = np.mean(returns[returns != 0]) if trades > 0 else 0.0

            # Calculate cumulative (compounded) returns
            cumulative_returns = np.cumprod(1 + returns) - 1
            final_cumulative_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0

            # Calculate uncompounded returns
            uncompounded_return = np.sum(returns)

            # Calculate Sharpe Ratio (annualized)
            # Assumes returns are per-period (e.g., hourly or daily)
            # For annualization, adjust based on your data frequency
            ann_factor = 252  # Typical trading days in a year for daily data
            if trades > 1:
                returns_std = np.std(returns, ddof=1)
                sharpe_ratio = (np.mean(returns) / (returns_std + 1e-10)) * np.sqrt(ann_factor)
            else:
                sharpe_ratio = 0.0

            # Calculate Maximum Drawdown
            if len(cumulative_returns) > 0:
                peak = np.maximum.accumulate(cumulative_returns)
                drawdown = peak - cumulative_returns
                max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            else:
                max_drawdown = 0.0

            return {
                'trades': trades,
                'profitable_trades': profitable_trades,
                'unprofitable_trades': unprofitable_trades,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'cumulative_return': final_cumulative_return,
                'uncompounded_return': uncompounded_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }

        # Calculate metrics for each strategy
        strategy_metrics = [calculate_trading_metrics(returns) for returns in strategy_returns]

        # Print results for each strategy
        print("\n===== COMPARING TRADING STRATEGIES =====")
        for name, returns, metrics in zip(strategy_names, strategy_returns, strategy_metrics):
            print(f"\n{name} Strategy Results:")
            print("-" * 50)
            print(f"Total trades: {metrics['trades']}")
            print(f"Profitable trades: {metrics['profitable_trades']}")
            print(f"Unprofitable trades: {metrics['unprofitable_trades']}")
            print(f"Win rate: {metrics['win_rate']:.2%}")
            print(f"Average return per trade: {metrics['avg_return']:.4%}")
            print(f"Final cumulative return (compounded): {metrics['cumulative_return']:.4%}")
            print(f"Final total return (uncompounded): {metrics['uncompounded_return']:.4%}")
            print(f"Sharpe Ratio (annualized): {metrics['sharpe_ratio']:.4f}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.4%}")

        # Print detailed trading results for the combined strategy
        print("\nDetailed Trading Results (Combined Strategy):")
        print("-" * 120)
        print(
            f"{'Sample':<8} | {'Timestamp':<20} | {'Price':<12} | {'Pred':<5} | {'True':<5} | {'Prob':<8} | {'Actual Chg':<10} | {'Return':>8} | {'Cum Return':>12}")
        print("-" * 120)

        # Calculate cumulative returns for display
        cum_returns = np.cumprod(1 + combined_returns) - 1

        for i in range(len(all_preds)):
            # Format timestamp for display
            if isinstance(timestamps[i], pd.Timestamp):
                ts_str = str(timestamps[i])
            else:
                ts_str = str(timestamps[i])

            print(
                f"{i:<8} | {ts_str:<20} | {prices[i]:<12.2f} | {all_preds[i]:<5.0f} | "
                f"{all_trues[i]:<5.0f} | {all_probs[i]:<8.4f} | {actual_changes[i] * 100:>10.2f}% | {combined_returns[i]:>8.4%} | {cum_returns[i]:>12.4%}")

        # Return the results for further analysis
        return {
            'predictions': all_preds,
            'actual': all_trues,
            'probabilities': all_probs,
            'actual_changes': actual_changes,
            'timestamps': timestamps,
            'prices': prices,
            'strategies': {
                'long_only': {
                    'returns': long_only_returns,
                    'cumulative_returns': np.cumprod(1 + long_only_returns) - 1,
                    'metrics': strategy_metrics[0]
                },
                'short_only': {
                    'returns': short_only_returns,
                    'cumulative_returns': np.cumprod(1 + short_only_returns) - 1,
                    'metrics': strategy_metrics[1]
                },
                'combined': {
                    'returns': combined_returns,
                    'cumulative_returns': cum_returns,
                    'metrics': strategy_metrics[2]
                }
            }
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

        # print test label distribution
        test_data, _ = self._get_data(flag='test')
        test_labels = []
        for i in range(len(test_data)):
            _, batch_y, _, _ = test_data[i]
            label = batch_y[-1, 0]
            test_labels.append(label)
        test_labels = np.array(test_labels)
        pos_ratio = np.mean(test_labels)
        print(f"Test set - Positive examples: {pos_ratio:.2%}, Negative: {(1 - pos_ratio):.2%}")

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
        # outputs shape: torch.Size([32, 1, 1])
        # batch_y shape: torch.Size([32, 2, 1])
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        # batch_y shape: torch.Size([32, 1, 1])
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # For binary classification, we only care about the last timestep prediction
        # outputs_last shape: torch.Size([32, 1])
        outputs_last = outputs[:, -1, :]
        # batch_y_last shape: torch.Size([32, 1]). Values of form [0.] or [1.]
        batch_y_last = batch_y[:, -1, :]

        return outputs_last, batch_y_last, outputs, batch_y

    def _calculate_accuracy(self, outputs, targets):
        """Calculate accuracy for binary predictions in a simple way"""
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
                #TODO: trace dec_inp across the model
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            print(f"Outputs if shape: {outputs.shape}")
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            print(f"Outputs else shape: {outputs.shape}")

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

            # Pass both validation loss and training loss to early stopping
            # early_stopping(vali_loss, self.model, path, train_loss)
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
        #TODO HANDLE COMMENTED OUt PRNT STMTS
        # print("\n==== DIRECT LABEL CHECK ====")
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

            # print(
            #     f"Sample {i}: orig_idx={orig_idx}, pred_idx={pred_idx}, raw_label={raw_label}, expected={expected_label}")




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

    def test_with_direct_sample_processing(self, setting, test=0):
        """
        Modified test method that processes each sample individually to ensure correct labels.
        This avoids batch-related indexing issues with DataLoader.
        """
        test_data, _ = self._get_data(flag='test')  # We won't use the loader

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        probs = []

        self.model.eval()
        with torch.no_grad():
            # Process one sample at a time
            for i in range(len(test_data)):
                # Get the sample
                batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[i]

                # Add batch dimension
                batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(self.device)
                batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(self.device)
                batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(self.device)
                batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Make prediction
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Process outputs
                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                # Get prediction
                output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()
                output_binary = (output_prob > 0.5).astype(np.float32)
                true_label = batch_y_last.detach().cpu().numpy()

                # Store results
                preds.append(output_binary)
                trues.append(true_label)
                probs.append(output_prob)

        # Concatenate all samples
        preds = np.concatenate(preds, axis=0)
        probs = np.concatenate(probs, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Continue with existing metrics calculation...
        # Rest of the test method would be the same

        return preds, trues, probs

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

