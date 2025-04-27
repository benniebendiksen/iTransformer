import sys

import pandas as pd

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Import our custom trading loss
from utils.trading_loss import TradingBCELoss

warnings.filterwarnings('ignore')


# Define a simple FFN model
class SimilarityPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=8, output_size=1):
        super(SimilarityPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


# Function to train the model
def train_consensus_model(features, labels, epochs=100, lr=0.01, batch_size=32):
    # Convert to PyTorch tensors
    X = torch.FloatTensor(features)
    y = torch.FloatTensor(labels).view(-1, 1)

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SimilarityPredictor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    return model


# Modified code for your original snippet with the FFN model
def process_confusion_matrix(TP, TN, FP, FN, output_binary_std, output_prob_std, true_label_sim, idx=None,
                             model=None, features_history=None, labels_history=None, is_test_sample=False):
    total_cases = TP + TN + FP + FN
    total_pos = TP + FP
    total_prop_pos = total_pos / total_cases
    prop_pos_acc = TP / (TP + FP) if (TP + FP) > 0 else 0
    total_neg = TN + FN
    total_prop_neg = total_neg / total_cases
    prop_neg_acc = TN / (TN + FN) if (TN + FN) > 0 else 0

    # Calculate the simple expected value as before
    simple_exp_val = total_prop_pos * prop_pos_acc - total_prop_neg * prop_neg_acc
    simple_prediction = 1 if simple_exp_val > 0 else 0

    # Display basic information
    if idx is not None:
        print(
            f"Prediction for sample {idx}: {output_binary_std}, True Label: {true_label_sim}, Probability: {output_prob_std:.4f}")

    print(f'\nConfusion Matrix:')
    print(f'  True Positives: {TP}')
    print(f'  True Negatives: {TN}')
    print(f'  False Positives: {FP}')
    print(f'  False Negatives: {FN}')
    print(f"Proportion of Accurate Positive Predictions: {prop_pos_acc:.2f}")
    print(f"Proportion of Accurate Negative Predictions: {prop_neg_acc:.2f}")
    print(f'  Total Similarity Cases: {total_cases}')

    # Store the features and labels for model training (only for training data)
    if features_history is not None and labels_history is not None and not is_test_sample:
        features = [output_binary_std, output_prob_std, TP, TN, FP, FN]
        features_history.append(features)
        labels_history.append(true_label_sim)

    # Make prediction using the model if available
    if model is not None:
        # Prepare features
        input_features = torch.FloatTensor([output_binary_std, output_prob_std, TP, TN, FP, FN])

        # Get model prediction
        with torch.no_grad():
            model_pred_prob = model(input_features).item()
            model_pred = 1 if model_pred_prob >= 0.5 else 0

        sample_type = "TEST SAMPLE" if is_test_sample else "TRAINING SAMPLE"
        print(f"[{sample_type}]")
        print(f"Simple Expected Value: {simple_exp_val:.4f} → Prediction: {simple_prediction}")
        print(f"FFN Model Prediction: {model_pred} (Probability: {model_pred_prob:.4f})")

        return model_pred, model_pred_prob
    else:
        # Fall back to simple expected value if no model is available
        print(f"Simple Expected Value of Similarity Cases: {simple_exp_val:.4f}")
        if simple_exp_val > 0:
            print(f"Expected Value Prediction: 1")
        else:
            print(f"Expected Value Prediction: 0")

        return simple_prediction, abs(simple_exp_val)


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

        # # Show detailed label verification
        # print("\nLabel Verification with Detailed Metadata:")
        # print("-" * 140)
        # print(
        #     f"{'Sample':<8} | {'Orig Idx':<8} | {'Pred Idx':<8} | {'Pred Price':<12} | {'Future Price':<12} | {'Change':<10} | {'Stored Label':<12} | {'Model True':<10} | {'Match':<5}")
        # print("-" * 140)

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
        #     timestamp = meta['timestamp']
        #
        #     print(
        #         f"{i:<8} | {orig_idx:<8} | {pred_idx:<8} | {pred_price:<12.2f} | {future_price:<12.2f} | {price_change:<10.2f}% | {stored_label:<12.1f} | {model_true:<10.1f} | {match:<5} | {timestamp:<5}")

        # Show summary statistics
        matches = sum(
            1 for i, meta in enumerate(all_metadata)
            if meta is not None and i < len(all_trues) and
            abs(meta['label'] - all_trues[i]) < 0.01)
        total = len([meta for meta in all_metadata if meta is not None])
        print(f"\nLabel Match Rate: {matches}/{total} ({matches / total * 100:.2f}%)")
        # exit program if match rate is less than 100%
        if matches / total < 1.0:
            print(f"Label match rate is less than 100%. Exiting program: {matches / total}")
            sys.exit(1)

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
                timestamps.append(meta['timestamp'])
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
        print("\n===== COMPARISONS =====")
        for name, returns, metrics in zip(strategy_names, strategy_returns, strategy_metrics):
            print(f"\n{name} Strategy Results:")
            print("-" * 50)
            print(f"Total predictions: {metrics['trades']}")
            print(f"Successful prediction: {metrics['profitable_trades']}")
            print(f"Unsuccessful predictions: {metrics['unprofitable_trades']}")
            print(f"Success rate: {metrics['win_rate']:.2%}")
            print(f"Average percent change per prediction: {metrics['avg_return']:.4%}")
            print(f"Final cumulative percent change (compounded): {metrics['cumulative_return']:.4%}")
            print(f"Final total percent change (uncompounded): {metrics['uncompounded_return']:.4%}")
            print(f"Sharpe Ratio (annualized): {metrics['sharpe_ratio']:.4f}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.4%}")

        # Print detailed trading results for the combined strategy
        print("\nDetailed Analysis (Combined):")
        print("-" * 120)
        print(
            f"{'Sample':<8} | {'Timestamp':<20} | {'Price':<12} | {'Pred':<5} | {'True':<5} | {'Prob':<8} | {'Actual Chg':<10} | {'Percent Change':>8} | {'Cum Percent Change':>12}")
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

        print(f"all preds: {', '.join(str(int(p)) for p in all_preds)}")
        print()
        print(f"all trues: {', '.join(str(int(t)) for t in all_trues)}")

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

        # print(f"\nClass distribution in training data:")
        # print(f"Positive examples (price increases): {positive_ratio:.2%}")
        # print(f"Negative examples (price decreases or stays): {negative_ratio:.2%}")
        # print(f"Pos_weight for BCEWithLogitsLoss: {self.class_distribution['pos_weight']:.4f}")
        # # print(f"Strategy: {'Shorting enabled' if self.is_shorting else 'No shorting (holding only)'}")
        # # if not self.is_shorting:
        # #     print(f"Precision factor for non-shorting strategy: {self.precision_factor}")
        # print()

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

                iter_count += 1

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

                # if (i + 1) % 100 == 0:
                #     print(
                #         "\titers: {0}, epoch: {1} | loss: {2:.7f}, accuracy: {3:.2f}%, precision: {4:.2f}%, recall: {5:.2f}%".format(
                #             i + 1, epoch + 1, loss.item(),
                #             batch_metrics['accuracy'] * 100,
                #             batch_metrics['precision'] * 100,
                #             batch_metrics['recall'] * 100))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

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
        print(f'  True Positives: {TP}')
        print(f'  True Negatives: {TN}')
        print(f'  False Positives: {FP}')
        print(f'  False Negatives: {FN}')
        print(f'  Total Predictions: {TP + TN + FP + FN}')

        # Calculate domain specific performance metrics
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

        print('\nPerformance Results:')
        print('  Strategy: {}'.format('Short enabled' if self.is_shorting else 'No shorting (holding only)'))
        print('  Successful Predictions: {}, Unsuccessful Predictions: {}, Total Predictions: {}'.format(
            profitable_trades, unprofitable_trades, total_trades))
        print('  Accuracy: {:.2f}%'.format(win_rate * 100))
        print('  P Factor: {:.2f}'.format(profit_factor))

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
        # f = open("result_logits_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write(
        #     'Trading Strategy: {}\n'.format('Shorting enabled' if self.is_shorting else 'No shorting (holding only)'))
        # f.write('Classification - Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%\n'.format(
        #     accuracy * 100, precision * 100, recall * 100, f1 * 100))
        # f.write('Trading - Win Rate: {:.2f}%, Profit Factor: {:.2f}\n'.format(win_rate * 100, profit_factor))
        # f.write('\n\n')
        # f.close()

        print("\nPerforming recency effect analysis...")
        self.analyze_recency_effect(setting, test=test)

        print("\nAnalyzing performance...")
        self.calculate_returns(setting, test=test)

        return

    # Adaptive fine-tuning method and helper methods
    def adaptive_test(self, setting, test=0, top_n=50, epochs=5, learning_rate=0.0001, batch_size=4):
        """
        Test method with per-sample adaptive fine-tuning.

        Args:
            setting: Experiment setting name
            test: Whether to load model from checkpoint
            top_n: Number of similar training samples to use for fine-tuning
            epochs: Number of epochs for fine-tuning
            learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for fine-tuning

        Returns:
            Dictionary containing performance metrics for both methods
        """
        print("\n============= ADAPTIVE TEST WITH PER-SAMPLE FINE-TUNING =============")

        # Step 0: Load model from checkpoint if needed
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        # Load data
        train_data, _ = self._get_data(flag='train')
        val_data, _ = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Initialize empty lists to store features and labels for similarity results across test samples, then to train
        features_history = []
        labels_history = []
        consensus_model = None

        # Prepare for storing results
        standard_preds = []
        standard_trues = []
        adaptive_preds = []
        adaptive_trues = []
        adaptive_probs = []
        standard_probs = []

        # Set model to evaluation mode
        self.model.eval()

        # Step 1: Extract embeddings from all samples
        print("\nExtracting embeddings from training and validation samples...")
        train_embeddings = self._extract_embeddings(train_data)
        val_embeddings = self._extract_embeddings(val_data)

        # Create a new model with the same architecture
        # Only pass the arguments that the model constructor expects
        model_args = {}
        # Extract the model configuration from self.args
        # Adjust this list based on what your model's __init__ method actually accepts
        model_arg_names = [
            'seq_len', 'pred_len', 'output_attention', 'use_norm',
            'embed', 'freq', 'dropout', 'class_strategy', 'd_model',
            'n_heads', 'e_layers', 'd_layers', 'd_ff', 'activation',
            'channel_independence', 'enc_in', 'dec_in', 'c_out', 'distil'
        ]

        for arg_name in model_arg_names:
            if hasattr(self.args, arg_name):
                model_args[arg_name] = getattr(self.args, arg_name)

        #adaptive_model = type(self.model)(**model_args)
        adaptive_model = self.model
        adaptive_model.to(self.device)

        print(f"\nProcessing {len(test_data)} test samples with adaptive fine-tuning...")

        self.model.eval()
        # get test samples' predictions
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

                outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)
                output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
                output_binary = (output_prob > 0.5).astype(np.float32)
                true_label = batch_y_last.detach().cpu().numpy()[0, 0]

                # Store standard prediction
                standard_preds.append(output_binary)
                standard_trues.append(true_label)
                standard_probs.append(output_prob)

        # Process each test sample individually
        for idx in range(len(test_data)):
            # Get current test sample
            batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[idx]

            # Add batch dimension
            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(self.device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(self.device)
            batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(self.device)
            batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(self.device)

            # Create decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # # Step 2: Extract embedding for test samples
            test_embed = self._extract_embedding_single(batch_x, batch_x_mark)
            test_embeddings = self._extract_embeddings(test_data, idx)

            # Step 3: Find similar samples in train and validation set
            similar_indices = self._find_similar_samples(
                test_embed, train_embeddings, val_embeddings, test_embeddings
            )
            # get mean probability of similar samples once inferred
            mean_probs_train = []
            mean_false_probs_train = []
            mean_sim_labels_train = []
            true_sim_train_pred_counter = 0
            false_sim_train_pred_counter = 0
            total_counter = 0
            trues = []
            preds = []
            # Get top similar train samples across inaccurate and accurate predictions
            for sim_count, sim_sample in enumerate(similar_indices):
                if sim_sample[0] == "train":
                    # get the probability of the sample
                    batch_x_sim, batch_y_sim, batch_x_mark_sim, batch_y_mark_sim = train_data[sim_sample[1]]
                    batch_x_sim = torch.tensor(batch_x_sim).unsqueeze(0).float().to(self.device)
                    batch_y_sim = torch.tensor(batch_y_sim).unsqueeze(0).float().to(self.device)
                    batch_x_mark_sim = torch.tensor(batch_x_mark_sim).unsqueeze(0).float().to(self.device)
                    batch_y_mark_sim = torch.tensor(batch_y_mark_sim).unsqueeze(0).float().to(self.device)

                    # Create decoder input for similar samples
                    dec_inp_sim = torch.zeros_like(batch_y_sim[:, -self.args.pred_len:, :]).float()
                    dec_inp_sim = torch.cat([batch_y_sim[:, :self.args.label_len, :], dec_inp_sim], dim=1).float().to(
                        self.device)
                    # Get the probability of the sample
                    if self.args.output_attention:
                        outputs_sim = self.model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)[0]
                    else:
                        outputs_sim = self.model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)
                    outputs_last_sim, batch_y_last_sim, _, _ = self._process_outputs(outputs_sim, batch_y_sim)
                    output_prob_sim = torch.sigmoid(outputs_last_sim).detach().cpu().numpy()[0, 0]
                    # check if the prediction is correct
                    output_binary_sim = (output_prob_sim > 0.5).astype(np.float32)
                    true_label_sim = batch_y_last_sim.detach().cpu().numpy()[0, 0]
                    # Store standard prediction
                    preds.append(output_binary_sim)
                    trues.append(true_label_sim)
                    if total_counter == 10:
                        break
                    if output_binary_sim == true_label_sim:
                        true_sim_train_pred_counter += 1
                        total_counter += 1
                    else:
                        false_sim_train_pred_counter += 1
                        mean_false_probs_train.append(output_prob_sim)
                        total_counter += 1
                        continue
                    mean_probs_train.append(output_prob_sim)
                    mean_sim_labels_train.append(true_label_sim)

            train_prop_sim_accurate = true_sim_train_pred_counter / (true_sim_train_pred_counter + false_sim_train_pred_counter)
            train_prop_sim_negative = false_sim_train_pred_counter / (true_sim_train_pred_counter + false_sim_train_pred_counter)
            cm = confusion_matrix(trues, preds)
            TN, FP = cm[0, 0], cm[0, 1]  # True Negative, False Positive
            FN, TP = cm[1, 0], cm[1, 1]  # False Negative, True Positive



            mean_probs_val = []
            mean_sim_labels_val = []
            false_sim_val_pred_counter = 0
            trues_vals = []
            preds_vals = []
            total_counter = 0
            for sim_count, sim_sample in enumerate(similar_indices):
                if sim_sample[0] == "val":
                    # get the probability of the sample
                    batch_x_sim, batch_y_sim, batch_x_mark_sim, batch_y_mark_sim = train_data[sim_sample[1]]
                    batch_x_sim = torch.tensor(batch_x_sim).unsqueeze(0).float().to(self.device)
                    batch_y_sim = torch.tensor(batch_y_sim).unsqueeze(0).float().to(self.device)
                    batch_x_mark_sim = torch.tensor(batch_x_mark_sim).unsqueeze(0).float().to(self.device)
                    batch_y_mark_sim = torch.tensor(batch_y_mark_sim).unsqueeze(0).float().to(self.device)

                    # Create decoder input for similar samples
                    dec_inp_sim = torch.zeros_like(batch_y_sim[:, -self.args.pred_len:, :]).float()
                    dec_inp_sim = torch.cat([batch_y_sim[:, :self.args.label_len, :], dec_inp_sim], dim=1).float().to(
                        self.device)
                    # Get the probability of the sample
                    if self.args.output_attention:
                        outputs_sim = self.model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)[0]
                    else:
                        outputs_sim = self.model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)
                    outputs_last_sim, batch_y_last_sim, _, _ = self._process_outputs(outputs_sim, batch_y_sim)
                    output_prob_val = torch.sigmoid(outputs_last_sim).detach().cpu().numpy()[0, 0]
                    # check if the prediction is correct
                    output_binary_val = (output_prob_val > 0.5).astype(np.float32)
                    true_label_val = batch_y_last_sim.detach().cpu().numpy()[0, 0]
                    preds_vals.append(output_binary_val)
                    trues_vals.append(true_label_val)
                    if total_counter == 10:
                        break
                    total_counter += 1
                    if output_binary_val != true_label_val:
                        false_sim_val_pred_counter += 1
                        continue
                    mean_probs_val.append(output_prob_val)
                    mean_sim_labels_val.append(true_label_val)
                    if len(mean_probs_val) == 15:
                        break

            cm_vals = confusion_matrix(trues_vals, preds_vals)
            TN_VAL, FP_VAL = cm_vals[0, 0], cm_vals[0, 1]  # True Negative, False Positive
            FN_VAL, TP_VAL = cm_vals[1, 0], cm_vals[1, 1]  # False Negative, True Positive

            mean_probs_test = []
            mean_sim_labels_test = []
            false_sim_test_pred_counter = 0
            # Get top 15 similar test samples that are accurate
            for sim_count, sim_sample in enumerate(similar_indices):
                if sim_sample[0] == "test":
                    # get the probability of the sample
                    batch_x_sim, batch_y_sim, batch_x_mark_sim, batch_y_mark_sim = train_data[sim_sample[1]]
                    batch_x_sim = torch.tensor(batch_x_sim).unsqueeze(0).float().to(self.device)
                    batch_y_sim = torch.tensor(batch_y_sim).unsqueeze(0).float().to(self.device)
                    batch_x_mark_sim = torch.tensor(batch_x_mark_sim).unsqueeze(0).float().to(self.device)
                    batch_y_mark_sim = torch.tensor(batch_y_mark_sim).unsqueeze(0).float().to(self.device)

                    # Create decoder input for similar samples
                    dec_inp_sim = torch.zeros_like(batch_y_sim[:, -self.args.pred_len:, :]).float()
                    dec_inp_sim = torch.cat([batch_y_sim[:, :self.args.label_len, :], dec_inp_sim], dim=1).float().to(
                        self.device)
                    # Get the probability of the sample
                    if self.args.output_attention:
                        outputs_sim = self.model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)[0]
                    else:
                        outputs_sim = self.model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)
                    outputs_last_test, batch_y_last_test, _, _ = self._process_outputs(outputs_sim, batch_y_sim)
                    output_prob_test = torch.sigmoid(outputs_last_test).detach().cpu().numpy()[0, 0]
                    # check if the prediction is correct
                    output_binary_test = (output_prob_test > 0.5).astype(np.float32)
                    true_label_test = batch_y_last_test.detach().cpu().numpy()[0, 0]
                    if output_binary_test != true_label_test:
                        false_sim_test_pred_counter += 1
                        continue
                    mean_probs_test.append(output_prob_test)
                    mean_sim_labels_test.append(true_label_test)
                    if len(mean_probs_test) == 15:
                        break

            # # Step 4: Fine-tune on similar samples
            # # load the trained model from the checkpoints dir
            # checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            # state_dict = torch.load(checkpoint_path)
            # exp = Exp_Logits_Forecast(self.args)
            # exp.model.load_state_dict(state_dict)
            # per_sample_model = exp.model
            # per_sample_model.to(self.device)
            #
            # # Fine-tune adaptive model
            # self._fine_tune_model(
            #     per_sample_model,
            #     train_data,
            #     val_data,
            #     similar_indices,
            #     epochs=epochs,
            #     lr=learning_rate,
            #     batch_size=batch_size
            # )
            #
            # # Get standard prediction from current model
            # if self.args.output_attention:
            #     outputs = per_sample_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            # else:
            #     outputs = per_sample_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            #
            # outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)
            # output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
            # output_binary = (output_prob > 0.5).astype(np.float32)
            # true_label = batch_y_last.detach().cpu().numpy()[0, 0]
            #
            # # Store adaptive prediction
            # adaptive_preds.append(output_binary)
            # adaptive_trues.append(true_label)
            # adaptive_probs.append(output_prob)
            # print(f"Adaptive prediction for sample {i + 1}: {output_binary}, True Label: {true_label}, Probability: {output_prob}")

            # print mean probabilities of similar samples
            # get mean probability of similar samples
            # mean_adaptive_probs = []
            # for i in range(len(train_data)):
            #     for sim_sample in similar_indices:
            #         if sim_sample[0] == "train":
            #             if sim_sample[1] == i:
            #                 # get the probability of the sample
            #                 batch_x_sim, batch_y_sim, batch_x_mark_sim, batch_y_mark_sim = train_data[i]
            #                 batch_x_sim = torch.tensor(batch_x_sim).unsqueeze(0).float().to(self.device)
            #                 batch_y_sim = torch.tensor(batch_y_sim).unsqueeze(0).float().to(self.device)
            #                 batch_x_mark_sim = torch.tensor(batch_x_mark_sim).unsqueeze(0).float().to(self.device)
            #                 batch_y_mark_sim = torch.tensor(batch_y_mark_sim).unsqueeze(0).float().to(self.device)
            #
            #                 # Create decoder input for similar samples
            #                 dec_inp_sim = torch.zeros_like(batch_y_sim[:, -self.args.pred_len:, :]).float()
            #                 dec_inp_sim = torch.cat([batch_y_sim[:, :self.args.label_len, :], dec_inp_sim],
            #                                         dim=1).float().to(
            #                     self.device)
            #                 # Get the probability of the sample
            #                 if self.args.output_attention:
            #                     outputs_sim = per_sample_model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)[
            #                         0]
            #                 else:
            #                     outputs_sim = per_sample_model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)
            #                 outputs_last_sim, batch_y_last_sim, _, _ = self._process_outputs(outputs_sim, batch_y_sim)
            #                 output_prob_sim = torch.sigmoid(outputs_last_sim).detach().cpu().numpy()[0, 0]
            #                 mean_adaptive_probs.append(output_prob_sim)

            # for i in range(len(test_data)):
            #     for sim_sample in similar_indices:
            #         if sim_sample[0] == "val":
            #             if sim_sample[1] == i:
            #                 # get the probability of the sample
            #                 batch_x_sim, batch_y_sim, batch_x_mark_sim, batch_y_mark_sim = train_data[i]
            #                 batch_x_sim = torch.tensor(batch_x_sim).unsqueeze(0).float().to(self.device)
            #                 batch_y_sim = torch.tensor(batch_y_sim).unsqueeze(0).float().to(self.device)
            #                 batch_x_mark_sim = torch.tensor(batch_x_mark_sim).unsqueeze(0).float().to(self.device)
            #                 batch_y_mark_sim = torch.tensor(batch_y_mark_sim).unsqueeze(0).float().to(self.device)
            #
            #                 # Create decoder input for similar samples
            #                 dec_inp_sim = torch.zeros_like(batch_y_sim[:, -self.args.pred_len:, :]).float()
            #                 dec_inp_sim = torch.cat([batch_y_sim[:, :self.args.label_len, :], dec_inp_sim],
            #                                         dim=1).float().to(
            #                     self.device)
            #                 # Get the probability of the sample
            #                 if self.args.output_attention:
            #                     outputs_sim = per_sample_model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)[
            #                         0]
            #                 else:
            #                     outputs_sim = per_sample_model(batch_x_sim, batch_x_mark_sim, dec_inp_sim, batch_y_mark_sim)
            #                 outputs_last_sim, batch_y_last_sim, _, _ = self._process_outputs(outputs_sim, batch_y_sim)
            #                 output_prob_sim = torch.sigmoid(outputs_last_sim).detach().cpu().numpy()[0, 0]
            #                 mean_adaptive_probs.append(output_prob_sim)
            # print(f"Mean probabilities of adaptive similar samples: {sum(mean_adaptive_probs) / len(mean_adaptive_probs)}")

            # standard model single test sample prediction
            if self.args.output_attention:
                outputs_std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs_std = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs_last_std, batch_y_last_std, _, _ = self._process_outputs(outputs_std, batch_y)
            true_label_std = batch_y_last_std.detach().cpu().numpy()[0, 0]
            output_prob_std = torch.sigmoid(outputs_last_std).detach().cpu().numpy()[0, 0]
            output_binary_std = (output_prob_std > 0.5).astype(np.float32)

            total_cases = TP + TN + FP + FN
            total_pos = TP + FP
            total_prop_pos = total_pos / total_cases
            prop_pos_acc = TP / (TP + FP)
            total_neg = TN + FN
            total_prop_neg = total_neg / total_cases
            prop_neg_acc = TN / (TN + FN)

            # Process and collect data
            process_confusion_matrix(TP, TN, FP, FN, output_binary_std, output_prob_std, true_label_std,
                                     idx=i, features_history=features_history, labels_history=labels_history)


            # total_cases = TP_VAL + TN_VAL + FP_VAL + FN_VAL
            # total_pos = TP_VAL + FP_VAL
            # total_prop_pos = total_pos / total_cases
            # prop_pos_acc = TP_VAL / (TP_VAL + FP_VAL)
            # total_neg = TN_VAL + FN_VAL
            # total_prop_neg = total_neg / total_cases
            # prop_neg_acc = TN_VAL / (TN_VAL + FN_VAL)
            # print(f'\nVal Confusion Matrix:')
            # print(f'  True Positives: {TP_VAL}')
            # print(f'  True Negatives: {TN_VAL}')
            # print(f'  False Positives: {FP_VAL}')
            # print(f'  False Negatives: {FN_VAL}')
            # print(f"Proportion of Accurate Positive Predictions: {prop_pos_acc:.2f}")
            # print(f"Proportion of Accurate Negative Predictions: {prop_neg_acc:.2f}")
            # print(f'  Total Similarity Cases: {total_cases}')
            # # print expected value across positive and negative predictions
            # exp_val = total_prop_pos * prop_pos_acc - total_prop_neg * prop_neg_acc
            # if exp_val > 0:
            #     print(f"Expected Value of Similarity Cases: 1")
            # else:
            #     print(f"Expected Value of Similarity Cases: 0")
            #
            # # print(f"Proportion of accurate predictions from top 25 similar training samples: {train_prop_sim_accurate}")
            # # print(f"Train Mean False Probs: {sum(mean_false_probs_train) / len(mean_false_probs_train)}")
            # # print(f"Train Mean Label: {sum(mean_sim_labels_train) / len(mean_sim_labels_train)}, Mean Probs: {sum(mean_probs_train) / len(mean_probs_train)}, skipped: {false_sim_train_pred_counter}")
            # print(f"Val Mean Label: {sum(mean_sim_labels_val) / len(mean_sim_labels_val)}, Mean Probs: {sum(mean_probs_val) / len(mean_probs_val)}, skipped: {false_sim_val_pred_counter}")
            # print(f"Test Mean Label: {sum(mean_sim_labels_test) / len(mean_sim_labels_test)}, Mean Probs: {sum(mean_probs_test) / len(mean_probs_test)}, skipped: {false_sim_test_pred_counter}")
            # print(f"Combined Mean Label: {(sum(mean_sim_labels_train) + sum(mean_sim_labels_val) + sum(mean_sim_labels_test)) / (len(mean_sim_labels_train) + len(mean_sim_labels_val) + len(mean_sim_labels_test))}, Mean Probs: {(sum(mean_probs_train) + sum(mean_probs_val) + sum(mean_probs_test)) / (len(mean_probs_train) + len(mean_probs_val) + len(mean_probs_test))}, skipped: {false_sim_train_pred_counter + false_sim_val_pred_counter + false_sim_test_pred_counter}")
            # print()

        # Phase 2: Split data into training and testing sets
        print("\nPhase 2: Preparing training and testing data...")
        features_array = np.array(features_history)
        labels_array = np.array(labels_history)

        # Randomly select 20% of the data as test set
        test_size = 0.2
        indices = np.arange(len(features_array))
        np.random.shuffle(indices)
        test_split_idx = int(len(indices) * test_size)
        test_indices = indices[:test_split_idx]
        train_indices = indices[test_split_idx:]

        # Create train and test sets
        X_train = features_array[train_indices]
        y_train = labels_array[train_indices]
        X_test = features_array[test_indices]
        y_test = labels_array[test_indices]

        print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")

        # Train the model
        print("\nPhase 3: Training consensus model...")
        consensus_model = train_consensus_model(X_train, y_train, epochs=100)
        print("Model training complete!")

        # Phase 4: Evaluate on test set
        # Phase 4: Evaluate on test set
        print("\nPhase 4: Evaluating model on test set...")
        correct_predictions_original = 0
        correct_predictions_ffn = 0
        total_test_samples = len(X_test)

        print("\n----- TEST SET EVALUATION -----")
        print("Original | FFN | True | Match?")
        print("-------------------------------")

        for i in range(total_test_samples):
            # Extract features for this test sample
            TP, TN, FP, FN = int(X_test[i][2]), int(X_test[i][3]), int(X_test[i][4]), int(X_test[i][5])
            output_binary = int(X_test[i][0])
            output_prob = float(X_test[i][1])
            true_label = int(y_test[i])

            print(f"\nTest Sample {i + 1}:")

            # Process and get FFN prediction
            pred, prob = process_confusion_matrix(
                TP, TN, FP, FN, output_binary, output_prob, true_label,
                model=consensus_model, is_test_sample=True
            )

            # Check if original prediction was correct
            if output_binary == true_label:
                correct_predictions_original += 1
                original_result = "✓"
            else:
                original_result = "✗"

            # Check if FFN prediction was correct
            if pred == true_label:
                correct_predictions_ffn += 1
                ffn_result = "✓"
            else:
                ffn_result = "✗"

            # Print summary line for this sample
            print(
                f"  {output_binary} {original_result} |  {pred} {ffn_result}  |  {true_label}  | {'Yes' if pred == true_label else 'No'}")

        # Calculate and report accuracy metrics
        original_accuracy = correct_predictions_original / total_test_samples
        ffn_accuracy = correct_predictions_ffn / total_test_samples

        print("\n----- ACCURACY SUMMARY -----")
        print(f"Original Model Accuracy: {original_accuracy:.4f} ({correct_predictions_original}/{total_test_samples})")
        print(f"FFN Model Accuracy: {ffn_accuracy:.4f} ({correct_predictions_ffn}/{total_test_samples})")
        print(f"Improvement: {(ffn_accuracy - original_accuracy) * 100:.2f}%")

        # Convert results to numpy arrays
        # standard_preds = np.array(standard_preds)
        # standard_trues = np.array(standard_trues)
        # standard_probs = np.array(standard_probs)
        # adaptive_preds = np.array(adaptive_preds)
        # adaptive_trues = np.array(adaptive_trues)
        # adaptive_probs = np.array(adaptive_probs)

        # Calculate metrics for both methods
        # standard_metrics = self._calculate_test_metrics(standard_trues, standard_preds, standard_probs)
        # adaptive_metrics = self._calculate_test_metrics(adaptive_trues, adaptive_preds, adaptive_probs)

        # Compare the two methods
        # print("\n============= COMPARISON OF METHODS =============")
        # print("Standard Method:")
        # self._print_test_metrics(standard_metrics)

        # print("\nAdaptive Fine-tuning Method:")
        # self._print_test_metrics(adaptive_metrics)

        # Sample-by-sample comparison
        # correct_standard = (standard_preds == standard_trues).sum()
        # correct_adaptive = (adaptive_preds == adaptive_trues).sum()
        # improved = ((standard_preds != standard_trues) & (adaptive_preds == adaptive_trues)).sum()
        # degraded = ((standard_preds == standard_trues) & (adaptive_preds != adaptive_trues)).sum()

        # print("\nSample-by-sample comparison:")
        # print(f"Samples improved by adaptive method: {improved} ({improved / len(test_data):.2%})")
        # print(f"Samples degraded by adaptive method: {degraded} ({degraded / len(test_data):.2%})")
        # print(f"Net improvement: {improved - degraded} samples ({(improved - degraded) / len(test_data):.2%})")

        # Save results
        # folder_path = './results/' + setting + '/adaptive/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # np.save(folder_path + 'standard_metrics.npy', standard_metrics)
        # np.save(folder_path + 'adaptive_metrics.npy', adaptive_metrics)
        # np.save(folder_path + 'standard_preds.npy', standard_preds)
        # np.save(folder_path + 'adaptive_preds.npy', adaptive_preds)
        #
        # return {
        #     'standard': standard_metrics,
        #     'adaptive': adaptive_metrics,
        # }

    def _extract_embeddings(self, dataset, index=None):
        """
        Extract embeddings for all samples in a dataset

        Args:
            dataset: Dataset to extract embeddings from

        Returns:
            Dictionary mapping indices to embeddings
        """
        embeddings = {}

        self.model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                if index is not None and i == index:
                    continue
                batch_x, batch_y, batch_x_mark, batch_y_mark = dataset[i]

                # Add batch dimension
                batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(self.device)
                batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(self.device)

                # Extract embedding
                embedding = self._extract_embedding_single(batch_x, batch_x_mark)
                embeddings[i] = embedding

        return embeddings

    def _extract_embedding_single(self, batch_x, batch_x_mark):
        """
        Extract embedding for a single sample

        Args:
            batch_x: Input tensor
            batch_x_mark: Input timestamp tensor

        Returns:
            Embedding tensor
        """
        with torch.no_grad():
            # Access encoder embedding from the model
            # This is model-specific and needs to be adapted to the actual model architecture
            try:
                if hasattr(self.model, 'enc_embedding'):
                    # For iTransformer model
                    embedding = self.model.enc_embedding(batch_x, batch_x_mark)
                    embedding, _attns = self.model.encoder(embedding, attn_mask=None)
                    # print(f"Embedding shape orig: {embedding.shape}") # [1, 75, 512]

                    # Flatten embedding for easier distance calculation
                    embedding = embedding.squeeze(0).cpu().numpy()  # [75, 512]
                    # print(f"Embedding shape after mean: {embedding.shape}")

                elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'attn_layers'):
                    # Another common architecture pattern
                    # First get the embedding
                    if hasattr(self.model, 'enc_embedding'):
                        print("Using alternative enc_embedding to extract embedding")
                        enc_out = self.model.enc_embedding(batch_x, batch_x_mark)
                    else:
                        print("Using batch_x directly to extract embedding")
                        # If no embedding layer, use input directly
                        enc_out = batch_x.transpose(1, 2)  # [B, N, L]

                    # Run through first encoder layer to get more semantic representation
                    enc_out = self.model.encoder.attn_layers[0](enc_out)
                    embedding = enc_out.mean(dim=1).cpu().numpy()
                else:
                    raise ValueError("Model does not have enc_embedding or encoder with attn_layers")
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                # Fall back to using raw features
                embedding = batch_x.mean(dim=1).cpu().numpy()
        return embedding

    def _find_similar_samples(self, test_embedding, train_embeddings, val_embeddings, test_embeddings, similarity='cosine'):
        """
        Find the most similar samples to the test sample using per-timestep similarity.

        Args:
            test_embedding: [seq_len, embed_dim]
            train_embeddings / val_embeddings: dict of {idx: embedding [seq_len, embed_dim]}
        """

        def avg_cosine_similarity(seq1, seq2):
            # Assumes shape [T, D]
            cos_sims = [
                np.dot(seq1[t], seq2[t]) / (np.linalg.norm(seq1[t]) * np.linalg.norm(seq2[t]) + 1e-8)
                for t in range(seq1.shape[0])
            ]
            return np.mean(cos_sims)

        similarities = []

        for idx, embed in train_embeddings.items():
            if similarity == 'cosine':
                sim = avg_cosine_similarity(test_embedding, embed)
                similarities.append(('train', idx, sim))
            else:
                dist = np.linalg.norm(test_embedding - embed)
                similarities.append(('train', idx, -dist))

        for idx, embed in val_embeddings.items():
            if similarity == 'cosine':
                sim = avg_cosine_similarity(test_embedding, embed)
                similarities.append(('val', idx, sim))
            else:
                dist = np.linalg.norm(test_embedding - embed)
                similarities.append(('val', idx, -dist))

        for idx, embed in test_embeddings.items():
            if similarity == 'cosine':
                sim = avg_cosine_similarity(test_embedding, embed)
                similarities.append(('test', idx, sim))
            else:
                dist = np.linalg.norm(test_embedding - embed)
                similarities.append(('test', idx, -dist))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities

    # def _find_similar_samples(self, test_embedding, train_embeddings, val_embeddings, top_n=10, similarity='cosine'):
    #     """
    #     Find the most similar samples to the test sample
    #
    #     Args:
    #         test_embedding: Embedding of test sample
    #         train_embeddings: Dictionary of train embeddings
    #         val_embeddings: Dictionary of validation embeddings
    #         top_n: Number of similar samples to find
    #         similarity: Similarity metric to use ('cosine' or 'euclidean')
    #
    #     Returns:
    #         List of tuples (dataset, index) for the most similar samples
    #     """
    #     similarities = []
    #
    #     # Calculate similarity for training samples
    #     for idx, embed in train_embeddings.items():
    #         if similarity == 'cosine':
    #             # Cosine similarity (higher is more similar)
    #             sim = np.dot(test_embedding.flatten(), embed.flatten()) / (
    #                     np.linalg.norm(test_embedding) * np.linalg.norm(embed)
    #             )
    #             similarities.append(('train', idx, sim))
    #         else:
    #             # Euclidean distance (lower is more similar)
    #             dist = np.linalg.norm(test_embedding - embed)
    #             similarities.append(('train', idx, -dist))  # Negate so higher is more similar
    #
    #     # Calculate similarity for validation samples
    #     for idx, embed in val_embeddings.items():
    #         if similarity == 'cosine':
    #             sim = np.dot(test_embedding.flatten(), embed.flatten()) / (
    #                     np.linalg.norm(test_embedding) * np.linalg.norm(embed)
    #             )
    #             similarities.append(('val', idx, sim))
    #         else:
    #             dist = np.linalg.norm(test_embedding - embed)
    #             similarities.append(('val', idx, -dist))
    #
    #     # Sort by similarity (descending)
    #     similarities.sort(key=lambda x: x[2], reverse=True)
    #
    #     # Return top N most similar
    #     return similarities[:top_n]

    def _fine_tune_model(self, model, train_data, val_data, similar_indices, epochs=5, lr=0.0001, batch_size=4):
        """
        Fine-tune model on similar samples

        Args:
            model: Model to fine-tune
            train_data: Training dataset
            val_data: Validation dataset
            similar_indices: List of (dataset, index) tuples for similar samples
            epochs: Number of epochs
            lr: Learning rate
            batch_size: Batch size
        """
        # Set model to training mode
        model.train()

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create criterion
        criterion = self._select_criterion()

        # Collect similar samples
        similar_samples = []
        for dataset_name, idx, _ in similar_indices:
            try:
                if dataset_name == 'train':
                    if idx >= len(train_data):
                        print(f"Warning: Train index {idx} out of range ({len(train_data)})")
                        continue
                    similar_samples.append(train_data[idx])
                else:  # 'val'
                    if idx >= len(val_data):
                        print(f"Warning: Val index {idx} out of range ({len(val_data)})")
                        continue
                    similar_samples.append(val_data[idx])
            except Exception as e:
                print(f"Error accessing {dataset_name} sample at index {idx}: {e}")
                continue

        # Check if we have enough samples to proceed
        if len(similar_samples) == 0:
            print("No similar samples could be collected for fine-tuning")
            return

        # Fine-tune for specified epochs
        for epoch in range(epochs):
            epoch_loss = 0
            samples_processed = 0

            # Process samples in mini-batches
            for i in range(0, len(similar_samples), batch_size):
                batch_samples = similar_samples[i:i + batch_size]

                # Prepare batch data
                batch_x_list = []
                batch_y_list = []
                batch_x_mark_list = []
                batch_y_mark_list = []

                # Filter out any problematic samples
                valid_samples = []
                for j, sample in enumerate(batch_samples):
                    try:
                        if len(sample) != 4:
                            print(f"Warning: Sample {i + j} has unexpected length {len(sample)}, expected 4")
                            continue

                        seq_x, seq_y, seq_x_mark, seq_y_mark = sample

                        # Check for None values
                        if seq_x is None or seq_y is None or seq_x_mark is None or seq_y_mark is None:
                            print(f"Warning: Sample {i + j} contains None values")
                            continue

                        # Add to valid samples
                        valid_samples.append(sample)
                        batch_x_list.append(seq_x)
                        batch_y_list.append(seq_y)
                        batch_x_mark_list.append(seq_x_mark)
                        batch_y_mark_list.append(seq_y_mark)
                    except Exception as e:
                        print(f"Error processing sample {i + j}: {e}")
                        continue

                # Skip this batch if no valid samples
                if len(valid_samples) == 0:
                    continue

                try:
                    # Make sure all data is in the same format (numpy arrays) before stacking
                    batch_x_numpy = []
                    batch_y_numpy = []
                    batch_x_mark_numpy = []
                    batch_y_mark_numpy = []

                    for x in batch_x_list:
                        if isinstance(x, torch.Tensor):
                            batch_x_numpy.append(x.detach().cpu().numpy())
                        else:
                            batch_x_numpy.append(x)

                    for y in batch_y_list:
                        if isinstance(y, torch.Tensor):
                            batch_y_numpy.append(y.detach().cpu().numpy())
                        else:
                            batch_y_numpy.append(y)

                    for x_mark in batch_x_mark_list:
                        if isinstance(x_mark, torch.Tensor):
                            batch_x_mark_numpy.append(x_mark.detach().cpu().numpy())
                        else:
                            batch_x_mark_numpy.append(x_mark)

                    for y_mark in batch_y_mark_list:
                        if isinstance(y_mark, torch.Tensor):
                            batch_y_mark_numpy.append(y_mark.detach().cpu().numpy())
                        else:
                            batch_y_mark_numpy.append(y_mark)

                    # Convert to tensors and move to device
                    batch_x = torch.tensor(np.stack(batch_x_numpy)).float().to(self.device)
                    batch_y = torch.tensor(np.stack(batch_y_numpy)).float().to(self.device)
                    batch_x_mark = torch.tensor(np.stack(batch_x_mark_numpy)).float().to(self.device)
                    batch_y_mark = torch.tensor(np.stack(batch_y_mark_numpy)).float().to(self.device)

                    # Create decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    if self.args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # Process outputs
                    outputs_last, batch_y_last, _, _ = self._process_outputs(outputs, batch_y)

                    # Calculate loss
                    loss = criterion(outputs_last, batch_y_last)
                    epoch_loss += loss.item()
                    samples_processed += len(valid_samples)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"Error during batch processing: {e}")
                    continue

            # Print progress (only if samples were processed)
            if samples_processed > 0:
                avg_loss = epoch_loss / samples_processed
                if epoch == epochs - 1 or epoch % 2 == 0:  # Print first, last and every other epoch
                    print(
                        f"Fine-tuning epoch {epoch + 1}/{epochs}, avg loss: {avg_loss:.6f}, samples: {samples_processed}")

        # Set model back to evaluation mode
        model.eval()

    def _calculate_test_metrics(self, trues, preds, probs):
        """
        Calculate test metrics for both classification and trading performance

        Args:
            trues: True labels
            preds: Predicted labels
            probs: Predicted probabilities

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # Calculate classification metrics
        accuracy = accuracy_score(trues, preds)

        # Handle cases where classes might be missing
        try:
            precision = precision_score(trues, preds)
        except:
            precision = 0.0

        try:
            recall = recall_score(trues, preds)
        except:
            recall = 0.0

        try:
            f1 = f1_score(trues, preds)
        except:
            f1 = 0.0

        # Create confusion matrix
        cm = confusion_matrix(trues, preds)

        # Calculate trading metrics
        if self.is_shorting:
            # For shorting strategy: profit from both correct predictions
            correct_positives = np.logical_and(preds == 1, trues == 1)
            correct_negatives = np.logical_and(preds == 0, trues == 0)

            # Profit comes from correct predictions in both directions
            profitable_trades = correct_positives.sum() + correct_negatives.sum()
            total_trades = len(preds)
        else:
            # For no-shorting strategy: only profit from correct positive predictions
            profitable_trades = np.logical_and(preds == 1, trues == 1).sum()
            # We lose on false positives but not on true negatives or false negatives
            unprofitable_trades = np.logical_and(preds == 1, trues == 0).sum()
            total_trades = profitable_trades + unprofitable_trades

        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        profit_factor = profitable_trades / (len(preds) - profitable_trades) if (
                                                                                            len(preds) - profitable_trades) > 0 else float(
            'inf')

        return {
            'classification': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            },
            'trading': {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'profitable_trades': profitable_trades,
                'total_trades': total_trades,
                'is_shorting': self.is_shorting
            }
        }

    def _print_test_metrics(self, metrics):
        """Print test metrics in a formatted way"""
        cm = metrics['classification']['confusion_matrix']

        # Extract confusion matrix components if available
        if cm.shape == (2, 2):
            TN, FP = cm[0, 0], cm[0, 1]
            FN, TP = cm[1, 0], cm[1, 1]

            print('\nConfusion Matrix Breakdown:')
            print(f'  True Positives: {TP}')
            print(f'  True Negatives: {TN}')
            print(f'  False Positives: {FP}')
            print(f'  False Negatives: {FN}')
            print(f'  Total Predictions: {TP + TN + FP + FN}')

        cls = metrics['classification']
        trading = metrics['trading']

        print('\nClassification Performance:')
        print('  Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
            cls['accuracy'] * 100, cls['precision'] * 100, cls['recall'] * 100, cls['f1'] * 100))

        print('\nTrading Performance:')
        # print('  Strategy: {}'.format('Short enabled' if trading['is_shorting'] else 'No shorting (holding only)'))
        print('  Profitable Trades: {}, Total Trades: {}'.format(
            trading['profitable_trades'], trading['total_trades']))
        print('  Win Rate: {:.2f}%'.format(trading['win_rate'] * 100))
        print('  Profit Factor: {:.2f}'.format(trading['profit_factor']))

    # adaptive testing and helper methods end here
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

