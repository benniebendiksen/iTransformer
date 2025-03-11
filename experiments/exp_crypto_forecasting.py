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


class Exp_Crypto_Forecast(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Crypto_Forecast, self).__init__(args)
        # Add trading strategy parameters
        self.is_shorting = getattr(args, 'is_shorting', True)
        self.precision_factor = getattr(args, 'precision_factor', 2.0)
        self.auto_weight = getattr(args, 'auto_weight', True)

        # Calculate and store class distribution once at initialization
        self.class_distribution = None
        if self.args.is_training:
            self.calculate_class_distribution()

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
        f = open("result_crypto_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write(
            'Trading Strategy: {}\n'.format('Shorting enabled' if self.is_shorting else 'No shorting (holding only)'))
        f.write('Classification - Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%\n'.format(
            accuracy * 100, precision * 100, recall * 100, f1 * 100))
        f.write('Trading - Win Rate: {:.2f}%, Profit Factor: {:.2f}\n'.format(win_rate * 100, profit_factor))
        f.write('\n\n')
        f.close()

        return