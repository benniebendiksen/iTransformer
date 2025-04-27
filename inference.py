import argparse
import torch
import os
import pandas as pd
import numpy as np
from model.iTransformer import Model
from experiments.exp_logits_forecasting import Exp_Logits_Forecast
from data_provider.data_factory import data_provider
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network that preserves the temporal structure
    of the input embeddings while learning temporal patterns
    """

    def __init__(self, seq_len, embed_dim, num_channels=[64, 32, 16], kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = embed_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2

            # Temporal convolution block
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=padding, dilation=dilation_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Conv1d(out_channels, out_channels, kernel_size,
                          padding=padding, dilation=dilation_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # Residual connection
            if in_channels != out_channels:
                residual = nn.Conv1d(in_channels, out_channels, 1)
            else:
                residual = nn.Identity()

            layers.append((conv_block, residual))

        self.network = nn.ModuleList([nn.ModuleList([conv, res]) for conv, res in layers])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Convert to [batch_size, embed_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)

        for conv_block, residual in self.network:
            res = residual(x)
            out = conv_block(x)
            x = F.relu(out + res)

        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return self.sigmoid(x)


class TransformerEmbeddingModel(nn.Module):
    """
    Transformer-based model that directly processes the sequential embeddings
    using self-attention to capture temporal dependencies
    """

    def __init__(self, seq_len, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerEmbeddingModel, self).__init__()

        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        # Use the last token's embedding for classification
        x = x[:, -1, :]
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNLSTMEmbeddingModel(nn.Module):
    """
    Combined CNN-LSTM model that uses convolution to extract local patterns
    and LSTM to capture long-term dependencies
    """

    def __init__(self, seq_len, embed_dim, cnn_channels=64, lstm_hidden=64, num_layers=2, dropout=0.2):
        super(CNNLSTMEmbeddingModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Convert to [batch_size, embed_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # Convert back to [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Use the last output
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output)


class AttentionPoolingEmbeddingModel(nn.Module):
    """
    Model that uses attention-based pooling instead of flattening embeddings
    to preserve important temporal information
    """

    def __init__(self, seq_len, embed_dim, hidden_size=128, num_heads=4, dropout=0.2):
        super(AttentionPoolingEmbeddingModel, self).__init__()

        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # Learnable query token for pooling
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        batch_size = x.size(0)

        # Expand query token for batch
        query = self.query_token.expand(batch_size, -1, -1)

        # Apply attention pooling
        # Need to transpose for MultiheadAttention: [seq_len, batch_size, embed_dim]
        x_transposed = x.transpose(0, 1)
        query_transposed = query.transpose(0, 1)

        attended_output, _ = self.attention_pool(query_transposed, x_transposed, x_transposed)

        # Back to [batch_size, 1, embed_dim] and squeeze
        pooled_output = attended_output.transpose(0, 1).squeeze(1)

        return self.classifier(pooled_output)


def create_embedding_model(model_type, seq_len, embed_dim, **kwargs):
    """
    Factory function to create embedding models
    """
    if model_type == "tcn":
        return TemporalConvNet(seq_len, embed_dim, **kwargs)
    elif model_type == "transformer":
        return TransformerEmbeddingModel(seq_len, embed_dim, **kwargs)
    elif model_type == "cnn_lstm":
        return CNNLSTMEmbeddingModel(seq_len, embed_dim, **kwargs)
    elif model_type == "attention_pooling":
        return AttentionPoolingEmbeddingModel(seq_len, embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Enhanced training function with accuracy tracking
def train_embedding_model(model, train_embeddings, train_labels, val_embeddings=None, val_labels=None,
                          epochs=50, lr=0.001, batch_size=32, device='cuda'):
    """
    Enhanced training function with validation and accuracy tracking
    """
    # Convert to tensors if they're not already
    if not isinstance(train_embeddings, torch.Tensor):
        train_embeddings = torch.FloatTensor(train_embeddings)
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.FloatTensor(train_labels).view(-1, 1)

    # Create data loaders
    dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup training
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Validation
        if val_embeddings is not None and val_labels is not None:
            model.eval()
            with torch.no_grad():
                val_embeddings_tensor = torch.FloatTensor(val_embeddings).to(device)
                val_labels_tensor = torch.FloatTensor(val_labels).view(-1, 1).to(device)

                val_outputs = model(val_embeddings_tensor)
                val_predicted = (val_outputs > 0.5).float()
                val_accuracy = 100 * (val_predicted == val_labels_tensor).sum().item() / len(val_labels)
                val_accuracies.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%', end='')
            if val_embeddings is not None:
                print(f', Val Acc: {val_accuracy:.2f}%')
            else:
                print()

    return model, train_accuracies, val_accuracies


# Function to evaluate and compare models
def evaluate_models(models, test_embeddings, test_labels, device='cuda'):
    """
    Evaluate and compare different embedding models
    """
    results = {}

    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            test_embeddings_tensor = torch.FloatTensor(test_embeddings).to(device)
            test_labels_tensor = torch.FloatTensor(test_labels).view(-1, 1).to(device)

            outputs = model(test_embeddings_tensor)
            predicted = (outputs > 0.5).float()
            accuracy = 100 * (predicted == test_labels_tensor).sum().item() / len(test_labels)

            results[model_name] = {
                'accuracy': accuracy,
                'outputs': outputs.cpu().numpy(),
                'predicted': predicted.cpu().numpy()
            }

    return results


# Function to analyze similar samples accuracy
def analyze_similar_samples_accuracy(test_embeddings, test_labels, train_embeddings, train_labels,
                                     model, device='cuda', top_n=50):
    """
    Analyze the accuracy of predictions on similar samples
    """
    similar_samples_accuracies = []
    original_model_accuracies = []

    for i in range(len(test_embeddings)):
        test_embedding = test_embeddings[i]
        test_label = test_labels[i]

        # Find similar samples (using simple cosine similarity for brevity)
        similarities = []
        for j, train_embedding in enumerate(train_embeddings):
            sim = F.cosine_similarity(
                torch.tensor(test_embedding).flatten(),
                torch.tensor(train_embedding).flatten(),
                dim=0
            ).item()
            similarities.append((j, sim))

        # Get top_n similar samples
        similar_indices = [idx for idx, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]]
        similar_embeddings = [train_embeddings[idx] for idx in similar_indices]
        similar_labels = [train_labels[idx] for idx in similar_indices]

        # Train model on similar samples
        similar_model = create_embedding_model("tcn", seq_len=test_embedding.shape[0],
                                               embed_dim=test_embedding.shape[1])
        similar_model, _, _ = train_embedding_model(
            similar_model, similar_embeddings, similar_labels,
            epochs=50, device=device
        )

        # Evaluate on current test sample
        similar_model.eval()
        with torch.no_grad():
            test_input = torch.FloatTensor(test_embedding).unsqueeze(0).to(device)
            similar_pred = (similar_model(test_input) > 0.5).float().item()
            similar_correct = (similar_pred == test_label)
            similar_samples_accuracies.append(similar_correct)

            # Also evaluate original model on this sample
            original_pred = (model(test_input) > 0.5).float().item()
            original_correct = (original_pred == test_label)
            original_model_accuracies.append(original_correct)

    return {
        'similar_samples_accuracy': np.mean(similar_samples_accuracies) * 100,
        'original_model_accuracy': np.mean(original_model_accuracies) * 100
    }


class EmbeddingFFN(nn.Module):
    """
    Neural network that takes flattened embeddings as input and predicts binary labels
    """

    def __init__(self, embedding_size, hidden_size=64, output_size=1):
        super(EmbeddingFFN, self).__init__()
        self.layer1 = nn.Linear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


# Define the FFN Similarity Predictor model
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


# Function to train the consensus model
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

    return model, losses


def parse_args():
    parser = argparse.ArgumentParser(description='iTransformer Inference Script')

    # Basic config (required from bash)
    parser.add_argument('--is_training', type=int, required=False, default=0, help='status')
    # parser.add_argument('--model_id', type=str,
    #                     default='pca_components_btcusdt_12h_45_reduced_lance_seed_2_96_1_50',
    #                     help='model id')
    # parser.add_argument('--model_id', type=str,
    #                     default='pca_components_btcusdt_12h_45_reduced_lance_seed_april_15_96_1_50',
    #                     help='model id')
    parser.add_argument('--model_id', type=str,
                        default='pca_components_btcusdt_4h_48_lance_seed_march_9_2020',
                        help='model id')

    parser.add_argument('--projection_idx', type=str, default='2', help='projection identifier (0, 1, 2, 3, 4)')
    parser.add_argument('--model', type=str, default='iTransformer', help='model name')
    # Data loader
    parser.add_argument('--data', type=str, default='logits', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/logits/', help='root path of the data file')
    #parser.add_argument('--data_path', type=str, default='pca_components_btcusdt_12h_45_reduced_lance_seed_2.csv', help='data file')
    # parser.add_argument('--data_path', type=str, default='pca_components_btcusdt_12h_45_reduced_lance_seed_april_15.csv',
    #                     help='data file')
    parser.add_argument('--data_path', type=str,
                        default='pca_components_btcusdt_4h_48_lance_seed_march_9_2020.csv',
                        help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='12h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default=None, help='location of model checkpoints')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length (not used in iTransformer)')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # Model definition
    parser.add_argument('--enc_in', type=int, default=50, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=50, help='decoder input size')
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

    # Add the FFN consensus model parameters
    parser.add_argument('--use_consensus_model', type=int, default=1, help='whether to use FFN consensus model')
    parser.add_argument('--ffn_test_size', type=float, default=0.2, help='proportion of data to use for FFN testing')
    parser.add_argument('--ffn_epochs', type=int, default=150, help='epochs for FFN consensus model training')
    parser.add_argument('--ffn_learning_rate', type=float, default=0.01, help='learning rate for FFN consensus model')
    parser.add_argument('--ffn_batch_size', type=int, default=32, help='batch size for FFN consensus model training')

    # Add these arguments to your parse_args function
    parser.add_argument('--use_embedding_approach', type=int, default=1,
                        help='whether to use embedding-based approach')
    parser.add_argument('--similar_samples', type=int, default=25,
                        help='number of similar samples for embedding-based approach')
    parser.add_argument('--embedding_ffn_epochs', type=int, default=50,
                        help='number of epochs for embedding-based FFN training')
    parser.add_argument('--embedding_ffn_lr', type=float, default=0.001,
                        help='learning rate for embedding-based FFN training')

    # Add the new trading approach arguments
    parser.add_argument('--use_conditional_approach', type=int, default=1,
                        help='whether to use conditional trading approach')
    parser.add_argument('--use_ensemble_approach', type=int, default=1,
                        help='whether to use ensemble trading approach')
    parser.add_argument('--ensemble_method', type=str, default='weighted',
                        help='ensemble method: weighted, confidence_based, or boosted')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='confidence threshold for boosted ensemble method')

    # Set timeenc based on embed
    args, _ = parser.parse_known_args()
    timeenc_value = 0 if args.embed != 'timeF' else 1
    parser.add_argument('--timeenc', type=int, default=timeenc_value, help='time encoding')

    # Final parse
    args = parser.parse_args()

    # GPU setup
    if args.use_gpu:
        args.use_gpu = True if torch.cuda.is_available() else False
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            args.device_ids = [int(id_) for id_ in args.devices.split(',')]
            args.gpu = args.device_ids[0]
            print(f"using multiple GPUs, device ids: {args.device_ids}")
    print(f"using GPU: {args.use_gpu == 1}")
    print('Args:')
    print(args)
    return args


def extract_embeddings_for_test_samples(model, test_data, device):
    """
    Extract embeddings for all test samples, preserving temporal information

    Parameters:
    -----------
    model : torch.nn.Module
        iTransformer model
    test_data : Dataset
        Test dataset
    device : torch.device
        Device to run model on

    Returns:
    --------
    dict
        Dictionary of {idx: embedding [seq_len, embed_dim]} for test samples
    """
    print("Extracting embeddings for test samples...")
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for i in range(len(test_data)):
            # Get sample
            batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[i]

            # Add batch dimension
            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(device)
            batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(device)

            # Get embedding using the specified approach
            if hasattr(model, 'enc_embedding'):
                # For iTransformer model
                embedding = model.enc_embedding(batch_x, batch_x_mark)
                embedding, _attns = model.encoder(embedding, attn_mask=None)

                # Preserve temporal information - keep shape [seq_len, embed_dim]
                embedding = embedding.squeeze(0).detach().cpu().numpy()  # [seq_len, d_model]
            else:
                # Fallback approach if enc_embedding isn't available
                print("Warning: Model doesn't have enc_embedding attribute, using alternative method")
                embedding = model.encoder(batch_x, batch_x_mark).detach().cpu().numpy()
                embedding = embedding.squeeze(0)  # Ensure [seq_len, d_model]

            embeddings[i] = embedding

            # Progress indication
            if (i + 1) % 20 == 0 or i == len(test_data) - 1:
                print(f'Processed {i + 1}/{len(test_data)} test samples')

    return embeddings


def extract_embeddings_for_train_samples(model, train_data, device):
    """
    Extract embeddings for all training samples, preserving temporal information

    Parameters:
    -----------
    model : torch.nn.Module
        iTransformer model
    train_data : Dataset
        Training dataset
    device : torch.device
        Device to run model on

    Returns:
    --------
    tuple
        (embeddings dict, labels array) for training samples
    """
    print("Extracting embeddings for training samples...")
    model.eval()
    embeddings = {}
    labels = []

    with torch.no_grad():
        for i in range(len(train_data)):
            # Get sample
            batch_x, batch_y, batch_x_mark, batch_y_mark = train_data[i]

            # Add batch dimension
            batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(device)
            batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(device)
            batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(device)

            # Get embedding using the specified approach
            if hasattr(model, 'enc_embedding'):
                # For iTransformer model
                embedding = model.enc_embedding(batch_x, batch_x_mark)
                embedding, _attns = model.encoder(embedding, attn_mask=None)

                # Preserve temporal information - keep shape [seq_len, embed_dim]
                embedding = embedding.squeeze(0).detach().cpu().numpy()  # [seq_len, d_model]
            else:
                # Fallback approach if enc_embedding isn't available
                print("Warning: Model doesn't have enc_embedding attribute, using alternative method")
                embedding = model.encoder(batch_x, batch_x_mark).detach().cpu().numpy()
                embedding = embedding.squeeze(0)  # Ensure [seq_len, d_model]

            # Get label (last time step, last feature)
            label = batch_y[:, -1, -1].detach().cpu().numpy()[0]

            embeddings[i] = embedding
            labels.append(label)

            # Progress indication
            if (i + 1) % 100 == 0 or i == len(train_data) - 1:
                print(f'Processed {i + 1}/{len(train_data)} training samples')

    return embeddings, np.array(labels)


def extract_single_embedding(model, batch_x, batch_x_mark, device):
    """
    Extract embedding for a single sample

    Parameters:
    -----------
    model : torch.nn.Module
        iTransformer model
    batch_x : torch.Tensor
        Input tensor
    batch_x_mark : torch.Tensor
        Input mark tensor
    device : torch.device
        Device to run model on

    Returns:
    --------
    numpy.ndarray
        Embedding with shape [seq_len, embed_dim]
    """
    model.eval()
    with torch.no_grad():
        # Get embedding using the specified approach
        if hasattr(model, 'enc_embedding'):
            # For iTransformer model
            embedding = model.enc_embedding(batch_x, batch_x_mark)
            embedding, _attns = model.encoder(embedding, attn_mask=None)

            # Preserve temporal information - keep shape [seq_len, embed_dim]
            embedding = embedding.squeeze(0).detach().cpu().numpy()  # [seq_len, d_model]
        else:
            # Fallback approach if enc_embedding isn't available
            print("Warning: Model doesn't have enc_embedding attribute, using alternative method")
            embedding = model.encoder(batch_x, batch_x_mark).detach().cpu().numpy()
            embedding = embedding.squeeze(0)  # Ensure [seq_len, d_model]

    return embedding


def find_similar_samples(test_embedding, train_embeddings, val_embeddings=None, test_embeddings=None, top_n=50,
                         similarity='euclidean'):
    """
    Find the most similar samples to the test sample using per-timestep similarity.

    Parameters:
    -----------
    test_embedding : numpy.ndarray
        Embedding of test sample [seq_len, embed_dim]
    train_embeddings : dict
        Dictionary of {idx: embedding [seq_len, embed_dim]} for training samples
    val_embeddings : dict, optional
        Dictionary of {idx: embedding [seq_len, embed_dim]} for validation samples
    test_embeddings : dict, optional
        Dictionary of {idx: embedding [seq_len, embed_dim]} for other test samples
    top_n : int
        Number of similar samples to return
    similarity : str
        Similarity metric to use ('cosine' or 'euclidean')

    Returns:
    --------
    list
        List of tuples (split, idx, similarity) for most similar samples
    """

    def avg_cosine_similarity(seq1, seq2):
        # Assumes shape [T, D]
        cos_sims = [
            np.dot(seq1[t], seq2[t]) / (np.linalg.norm(seq1[t]) * np.linalg.norm(seq2[t]) + 1e-8)
            for t in range(min(seq1.shape[0], seq2.shape[0]))
        ]
        return np.mean(cos_sims)

    similarities = []

    # Process training embeddings
    if train_embeddings is not None:
        for idx, embed in train_embeddings.items():
            if similarity == 'cosine':
                sim = avg_cosine_similarity(test_embedding, embed)
                similarities.append(('train', idx, sim))
            else:
                dist = np.linalg.norm(test_embedding - embed)
                similarities.append(('train', idx, -dist))

    # Process validation embeddings if provided
    if val_embeddings is not None:
        for idx, embed in val_embeddings.items():
            if similarity == 'cosine':
                sim = avg_cosine_similarity(test_embedding, embed)
                similarities.append(('val', idx, sim))
            else:
                dist = np.linalg.norm(test_embedding - embed)
                similarities.append(('val', idx, -dist))

    # Process other test embeddings if provided
    if test_embeddings is not None:
        for idx, embed in test_embeddings.items():
            if similarity == 'cosine':
                sim = avg_cosine_similarity(test_embedding, embed)
                similarities.append(('test', idx, sim))
            else:
                dist = np.linalg.norm(test_embedding - embed)
                similarities.append(('test', idx, -dist))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Return top_n most similar samples
    return similarities[:top_n]


# def apply_enhanced_embedding_approach(model, train_data, val_data, test_data, device, args,
#                                       top_n=20, model_type='tcn',
#                                       ffn_epochs=50, ffn_lr=0.001):
#     """
#     Apply enhanced embedding-based approach with better architecture choices,
#     GPU optimization, and multi-policy trading decisions
#
#     Parameters:
#     -----------
#     model_type : str
#         Type of embedding model to use: 'tcn', 'transformer', 'cnn_lstm', or 'attention_pooling'
#     device : torch.device
#         Device to use for computation (CPU or GPU)
#     """
#     print(f"Applying enhanced embedding-based approach with {model_type} model...")
#
#     # Print device information
#     print(f"Using device: {device}")
#     if str(device).startswith('cuda'):
#         print(f"GPU: {torch.cuda.get_device_name(device)}")
#         print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
#         print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
#
#     # Extract embeddings for all datasets
#     train_embeddings, train_labels_array = extract_embeddings_for_train_samples(model, train_data, device)
#
#     # Extract validation embeddings if available
#     val_embeddings = None
#     val_labels = None
#     if val_data is not None:
#         val_embeddings, val_labels = extract_embeddings_for_train_samples(model, val_data, device)
#
#     # Initialize results arrays
#     embedding_preds = []
#     embedding_probs = []
#     trues = []
#     original_preds = []
#     original_probs = []
#
#     # Different trading policies
#     trade_decisions_original = []
#     trade_decisions_threshold = []
#     trade_decisions_weighted = []
#     trade_decisions_confidence_gap = []
#     trade_decisions_combined = []
#
#     # Track accuracy metrics
#     similar_sample_orig_accuracies = []
#     trained_model_accuracies = []
#     confusion_matrices = []
#
#     # Process each test sample
#     for idx in range(len(test_data)):
#         print(f"\nProcessing test sample {idx + 1}/{len(test_data)}...")
#
#         # Get test sample
#         batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[idx]
#         true_label = batch_y[-1, -1]
#         trues.append(true_label)
#
#         # Add batch dimension and ensure tensors are on the correct device
#         batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(device)
#         batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(device)
#         batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(device)
#         batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(device)
#
#         # Generate original model prediction
#         dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
#         dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
#
#         # Generate prediction
#         if args.output_attention:
#             outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#         else:
#             outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#
#         # Process outputs for binary classification
#         f_dim = -1 if args.features == 'MS' else 0
#         outputs_last = outputs[:, -1, f_dim:]
#         batch_y_last = batch_y[:, -1, f_dim:].to(device)
#
#         # Get prediction probability and binary prediction
#         orig_output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
#         orig_output_binary = (orig_output_prob > 0.5).astype(np.float32)
#
#         # Save original predictions
#         original_preds.append(orig_output_binary)
#         original_probs.append(orig_output_prob)
#
#         # Extract embedding for this test sample
#         test_embedding = extract_single_embedding(model, batch_x, batch_x_mark, device)
#
#         # Find similar samples
#         similar_samples = find_similar_samples(
#             test_embedding,
#             train_embeddings,
#             None,
#             None,
#             top_n=top_n
#         )
#
#         # Sort similar samples by similarity (descending)
#         similar_samples.sort(key=lambda x: x[2], reverse=True)
#
#         # Get embeddings and labels for similar samples
#         similar_train_indices = [idx for split, idx, _ in similar_samples if split == 'train']
#         similar_val_indices = [idx for split, idx, _ in similar_samples if split == 'val']
#
#         # Initialize confusion matrix for this test sample
#         TP, TN, FP, FN = 0, 0, 0, 0
#
#         # Track predictions of similar cases for confusion matrix
#         similar_labels = []
#         similar_predictions = []
#
#         # Process all similar samples to build confusion matrix
#         for split, similarity_idx, _ in similar_samples:
#             # Get the original data for this similar sample
#             if split == 'train':
#                 similar_batch_x, similar_batch_y, similar_batch_x_mark, similar_batch_y_mark = train_data[
#                     similarity_idx]
#                 true_label_similar = train_labels_array[similarity_idx]
#             elif split == 'val' and val_data is not None:
#                 similar_batch_x, similar_batch_y, similar_batch_x_mark, similar_batch_y_mark = val_data[similarity_idx]
#                 true_label_similar = val_labels[similarity_idx]
#             else:
#                 continue
#
#             # Add batch dimension and convert to tensor
#             similar_batch_x = torch.tensor(similar_batch_x).unsqueeze(0).float().to(device)
#             similar_batch_y = torch.tensor(similar_batch_y).unsqueeze(0).float().to(device)
#             similar_batch_x_mark = torch.tensor(similar_batch_x_mark).unsqueeze(0).float().to(device)
#             similar_batch_y_mark = torch.tensor(similar_batch_y_mark).unsqueeze(0).float().to(device)
#
#             # Use original model for prediction
#             with torch.no_grad():
#                 # Decoder input
#                 dec_inp = torch.zeros_like(similar_batch_y[:, -args.pred_len:, :]).float()
#                 dec_inp = torch.cat([similar_batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
#
#                 # Generate prediction
#                 if args.output_attention:
#                     outputs = model(similar_batch_x, similar_batch_x_mark, dec_inp, similar_batch_y_mark)[0]
#                 else:
#                     outputs = model(similar_batch_x, similar_batch_x_mark, dec_inp, similar_batch_y_mark)
#
#                 # Process outputs for binary classification
#                 f_dim = -1 if args.features == 'MS' else 0
#                 outputs_last = outputs[:, -1, f_dim:]
#
#                 # Get prediction probability and binary prediction
#                 similar_pred_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
#                 similar_pred = 1 if similar_pred_prob >= 0.5 else 0
#
#                 # Update confusion matrix
#                 if similar_pred == 1 and true_label_similar == 1:
#                     TP += 1
#                 elif similar_pred == 0 and true_label_similar == 0:
#                     TN += 1
#                 elif similar_pred == 1 and true_label_similar == 0:
#                     FP += 1
#                 elif similar_pred == 0 and true_label_similar == 1:
#                     FN += 1
#
#                 # Store predictions and true labels
#                 similar_labels.append(true_label_similar)
#                 similar_predictions.append(similar_pred)
#
#         # Calculate accuracy metrics
#         positive_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0
#         negative_accuracy = TN / (TN + FP) if (TN + FP) > 0 else 0
#
#         # Calculate weighted accuracy (accounts for sample size)
#         positive_samples = TP + FN
#         negative_samples = TN + FP
#         total_samples = positive_samples + negative_samples
#
#         if total_samples > 0:
#             weighted_positive_accuracy = positive_accuracy * (positive_samples / total_samples)
#             weighted_negative_accuracy = negative_accuracy * (negative_samples / total_samples)
#         else:
#             weighted_positive_accuracy = weighted_negative_accuracy = 0
#
#         # Store confusion matrix information
#         confusion_matrix_info = {
#             'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
#             'positive_accuracy': positive_accuracy,
#             'negative_accuracy': negative_accuracy,
#             'weighted_positive_accuracy': weighted_positive_accuracy,
#             'weighted_negative_accuracy': weighted_negative_accuracy,
#             'positive_samples': positive_samples,
#             'negative_samples': negative_samples,
#             'total_similar_samples': len(similar_samples)
#         }
#         confusion_matrices.append(confusion_matrix_info)
#
#         print(f"Confusion matrix for similar samples:")
#         print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
#         print(f"Positive accuracy: {positive_accuracy:.4f} ({positive_samples} samples)")
#         print(f"Negative accuracy: {negative_accuracy:.4f} ({negative_samples} samples)")
#         print(f"Weighted positive accuracy: {weighted_positive_accuracy:.4f}")
#         print(f"Weighted negative accuracy: {weighted_negative_accuracy:.4f}")
#
#         # Collect similar samples for model training (preserve temporal structure)
#         ffn_train_embeddings = []
#         ffn_train_labels = []
#
#         for train_idx in similar_train_indices:
#             # Keep the temporal structure [seq_len, embed_dim]
#             embedding = train_embeddings[train_idx]
#             ffn_train_embeddings.append(embedding)
#             ffn_train_labels.append(train_labels_array[train_idx])
#
#         # Create appropriate embedding model
#         seq_len, embed_dim = ffn_train_embeddings[0].shape if ffn_train_embeddings else (0, 0)
#
#         # Check if we have enough samples for the embedding-based approach
#         if len(ffn_train_embeddings) < 10:
#             print(
#                 f"Warning: Not enough similar samples found for embedding model ({len(ffn_train_embeddings)}), using original prediction")
#             embedding_preds.append(orig_output_binary)
#             embedding_probs.append(orig_output_prob)
#         else:
#             # Create and train embedding model
#             if model_type == 'tcn':
#                 embedding_model = TemporalConvNet(seq_len, embed_dim)
#             elif model_type == 'transformer':
#                 embedding_model = TransformerEmbeddingModel(seq_len, embed_dim)
#             elif model_type == 'cnn_lstm':
#                 embedding_model = CNNLSTMEmbeddingModel(seq_len, embed_dim)
#             elif model_type == 'attention_pooling':
#                 embedding_model = AttentionPoolingEmbeddingModel(seq_len, embed_dim)
#             else:
#                 raise ValueError(f"Unknown model type: {model_type}")
#
#             # Move model to the correct device
#             embedding_model = embedding_model.to(device)
#
#             # Train on similar samples
#             print(f"Training {model_type} model on {len(ffn_train_embeddings)} similar samples on {device}...")
#
#             # Convert to PyTorch tensors and move to device
#             X = torch.FloatTensor(ffn_train_embeddings).to(device)
#             y = torch.FloatTensor(ffn_train_labels).view(-1, 1).to(device)
#
#             # Create dataset and dataloader
#             dataset = TensorDataset(X, y)
#             dataloader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)
#
#             # Define loss function and optimizer
#             criterion = nn.BCELoss()
#             optimizer = optim.Adam(embedding_model.parameters(), lr=ffn_lr)
#
#             # Training loop
#             embedding_model.train()
#             epoch_train_accuracies = []
#
#             for epoch in range(ffn_epochs):
#                 epoch_loss = 0
#                 correct = 0
#                 total = 0
#
#                 for batch_X, batch_y in dataloader:
#                     # Forward pass
#                     outputs = embedding_model(batch_X)
#                     loss = criterion(outputs, batch_y)
#
#                     # Backward and optimize
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#
#                     epoch_loss += loss.item()
#
#                     # Calculate accuracy
#                     predicted = (outputs > 0.5).float()
#                     total += batch_y.size(0)
#                     correct += (predicted == batch_y).sum().item()
#
#                 train_accuracy = correct / total
#                 epoch_train_accuracies.append(train_accuracy)
#
#                 avg_loss = epoch_loss / len(dataloader)
#                 if (epoch + 1) % 10 == 0:
#                     print(f'Epoch [{epoch + 1}/{ffn_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}')
#                     if str(device).startswith('cuda'):
#                         print(f"GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated")
#
#             # Store the training accuracy
#             trained_model_accuracies.append(np.mean(epoch_train_accuracies))
#
#             # Get prediction for current test sample
#             embedding_model.eval()
#             with torch.no_grad():
#                 test_embedding_tensor = torch.FloatTensor(test_embedding).unsqueeze(0).to(device)
#                 pred_prob = embedding_model(test_embedding_tensor).item()
#                 pred = 1 if pred_prob >= 0.5 else 0
#
#             embedding_preds.append(pred)
#             embedding_probs.append(pred_prob)
#
#         # For multi-policy trading, use the original model's prediction
#         pred = orig_output_binary
#         pred_prob = orig_output_prob
#
#         # Calculate accuracies for confusion matrix-based decisions
#         similar_accuracy = np.mean(np.array(similar_predictions) == np.array(similar_labels))
#         similar_sample_orig_accuracies.append(similar_accuracy)
#
#         # Policy 1: Original policy (higher accuracy)
#         should_trade_original = False
#         if pred == 1:
#             should_trade_original = positive_accuracy > negative_accuracy
#         else:
#             should_trade_original = negative_accuracy > positive_accuracy
#         trade_decisions_original.append(should_trade_original)
#
#         # Policy 2: Minimum accuracy threshold (0.5)
#         should_trade_threshold = False
#         if pred == 1:
#             should_trade_threshold = positive_accuracy > negative_accuracy and positive_accuracy > 0.5
#         else:
#             should_trade_threshold = negative_accuracy > positive_accuracy and negative_accuracy > 0.5
#         trade_decisions_threshold.append(should_trade_threshold)
#
#         # Policy 3: Weighted accuracy comparison
#         should_trade_weighted = False
#         if pred == 1:
#             should_trade_weighted = weighted_positive_accuracy > weighted_negative_accuracy
#         else:
#             should_trade_weighted = weighted_negative_accuracy > weighted_positive_accuracy
#         trade_decisions_weighted.append(should_trade_weighted)
#
#         # Policy 4: Confidence gap (at least 0.35 difference)
#         confidence_gap = 0.35
#         should_trade_confidence_gap = False
#         if pred == 1:
#             should_trade_confidence_gap = positive_accuracy > negative_accuracy + confidence_gap
#         else:
#             should_trade_confidence_gap = negative_accuracy > positive_accuracy + confidence_gap
#         trade_decisions_confidence_gap.append(should_trade_confidence_gap)
#
#         # Policy 5: Combined policy (must satisfy multiple criteria)
#         should_trade_combined = False
#         if pred == 1:
#             should_trade_combined = (
#                     positive_accuracy > negative_accuracy and
#                     positive_accuracy > 0.5 and
#                     positive_accuracy > negative_accuracy + 0.1 and
#                     positive_samples >= 5  # Minimum sample requirement
#             )
#         else:
#             should_trade_combined = (
#                     negative_accuracy > positive_accuracy and
#                     negative_accuracy > 0.5 and
#                     negative_accuracy > positive_accuracy + 0.1 and
#                     negative_samples >= 5  # Minimum sample requirement
#             )
#         trade_decisions_combined.append(should_trade_combined)
#
#         print(
#             f"Orig prediction: {orig_output_binary}, Embedding-based prediction: {embedding_preds[-1]}, True label: {true_label}")
#         print(f"Trade decisions:")
#         print(f"  Original policy: {'Trade' if should_trade_original else 'No Trade'}")
#         print(f"  Threshold policy: {'Trade' if should_trade_threshold else 'No Trade'}")
#         print(f"  Weighted policy: {'Trade' if should_trade_weighted else 'No Trade'}")
#         print(f"  Confidence gap policy: {'Trade' if should_trade_confidence_gap else 'No Trade'}")
#         print(f"  Combined policy: {'Trade' if should_trade_combined else 'No Trade'}")
#
#         # Clear GPU cache if using CUDA
#         if str(device).startswith('cuda'):
#             torch.cuda.empty_cache()
#
#     # Calculate metrics for the raw embedding model approach
#     accuracy = accuracy_score(trues, embedding_preds)
#
#     # Convert result arrays to numpy arrays
#     embedding_preds = np.array(embedding_preds)
#     embedding_probs = np.array(embedding_probs)
#     trues = np.array(trues)
#     original_preds = np.array(original_preds)
#     original_probs = np.array(original_probs)
#
#     # Convert trade decision arrays
#     trade_decisions_original = np.array(trade_decisions_original)
#     trade_decisions_threshold = np.array(trade_decisions_threshold)
#     trade_decisions_weighted = np.array(trade_decisions_weighted)
#     trade_decisions_confidence_gap = np.array(trade_decisions_confidence_gap)
#     trade_decisions_combined = np.array(trade_decisions_combined)
#
#     # Calculate metrics for each policy
#     def calculate_policy_metrics(trade_mask):
#         if np.sum(trade_mask) > 0:
#             # Only evaluate on samples where we would trade
#             traded_indices = np.where(trade_mask)[0]
#             accuracy = accuracy_score(trues[traded_indices], original_preds[traded_indices])
#             precision = precision_score(trues[traded_indices], original_preds[traded_indices], zero_division=0)
#             recall = recall_score(trues[traded_indices], original_preds[traded_indices], zero_division=0)
#             f1 = f1_score(trues[traded_indices], original_preds[traded_indices], zero_division=0)
#             return {
#                 'accuracy': accuracy,
#                 'precision': precision,
#                 'recall': recall,
#                 'f1': f1,
#                 'traded_count': int(np.sum(trade_mask)),
#                 'total_count': len(trues)
#             }
#         else:
#             return {
#                 'accuracy': 0.0,
#                 'precision': 0.0,
#                 'recall': 0.0,
#                 'f1': 0.0,
#                 'traded_count': 0,
#                 'total_count': len(trues)
#             }
#
#     # Calculate metrics for each policy
#     policy_metrics = {
#         'overall': {'accuracy': accuracy},
#         'original': calculate_policy_metrics(trade_decisions_original),
#         'threshold': calculate_policy_metrics(trade_decisions_threshold),
#         'weighted': calculate_policy_metrics(trade_decisions_weighted),
#         'confidence_gap': calculate_policy_metrics(trade_decisions_confidence_gap),
#         'combined': calculate_policy_metrics(trade_decisions_combined)
#     }
#
#     # Print summary of all policies
#     print(f"\nEnhanced embedding-based approach results:")
#     print(f"Total samples: {len(trues)}")
#     print(f"Embedding model accuracy: {accuracy:.4f}")
#     print(f"Average similar samples orig accuracy: {np.mean(similar_sample_orig_accuracies):.4f}")
#     print(f"Average training accuracy: {np.mean(trained_model_accuracies):.4f}")
#
#     print("\nTrading policy comparison:")
#     print(f"{'Policy':<15} {'Trades':<8} {'Skipped':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
#     print("-" * 75)
#
#     for policy_name, metrics in policy_metrics.items():
#         if policy_name == 'overall':
#             continue
#         traded = metrics['traded_count']
#         skipped = metrics['total_count'] - traded
#         acc = metrics['accuracy']
#         prec = metrics['precision']
#         rec = metrics['recall']
#         f1_score_val = metrics['f1']
#         print(
#             f"{policy_name:<15} {traded:<8} {skipped:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1_score_val:<10.4f}")
#
#     # Return both the original embedding model results and the new multi-policy trading results
#     return {
#         # Original return values for backward compatibility
#         'preds': embedding_preds,
#         'probs': embedding_probs,
#         'trues': trues,
#         'accuracy': accuracy,
#         'similar_samples_accuracy': np.mean(similar_sample_orig_accuracies),
#         'training_accuracy': np.mean(trained_model_accuracies),
#
#         # New trading policy values
#         'original_preds': original_preds,
#         'original_probs': original_probs,
#         'trade_decisions_original': trade_decisions_original,
#         'trade_decisions_threshold': trade_decisions_threshold,
#         'trade_decisions_weighted': trade_decisions_weighted,
#         'trade_decisions_confidence_gap': trade_decisions_confidence_gap,
#         'trade_decisions_combined': trade_decisions_combined,
#         'confusion_matrices': confusion_matrices,
#         'policy_metrics': policy_metrics
#     }


def apply_enhanced_embedding_approach(model, train_data, val_data, test_data, device, args,
                                      top_n=20, model_type='tcn',
                                      ffn_epochs=50, ffn_lr=0.001):
    """
    Apply enhanced embedding-based approach with better architecture choices,
    GPU optimization, and multi-policy trading decisions

    Parameters:
    -----------
    model_type : str
        Type of embedding model to use: 'tcn', 'transformer', 'cnn_lstm', or 'attention_pooling'
    device : torch.device
        Device to use for computation (CPU or GPU)
    """
    print(f"Applying enhanced embedding-based approach with {model_type} model...")

    # Print device information
    print(f"Using device: {device}")
    if str(device).startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    # Extract embeddings for all datasets
    train_embeddings, train_labels_array = extract_embeddings_for_train_samples(model, train_data, device)

    # Extract validation embeddings if available
    val_embeddings = None
    val_labels = None
    if val_data is not None:
        val_embeddings, val_labels = extract_embeddings_for_train_samples(model, val_data, device)

    # Initialize results arrays
    embedding_preds = []
    embedding_probs = []
    trues = []
    original_preds = []
    original_probs = []

    # Different trading policies
    trade_decisions_higher_acc = []
    trade_decisions_threshold = []
    trade_decisions_weighted = []
    trade_decisions_confidence_gap = []
    trade_decisions_combined = []

    # Track accuracy metrics
    similar_sample_orig_accuracies = []
    trained_model_accuracies = []
    confusion_matrices = []

    # Process each test sample
    for idx in range(len(test_data)):
        print(f"\nProcessing test sample {idx + 1}/{len(test_data)}...")

        # Get test sample
        batch_x, batch_y, batch_x_mark, batch_y_mark = test_data[idx]
        true_label = batch_y[-1, -1]
        trues.append(true_label)

        # Add batch dimension and ensure tensors are on the correct device
        batch_x = torch.tensor(batch_x).unsqueeze(0).float().to(device)
        batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0).float().to(device)
        batch_y = torch.tensor(batch_y).unsqueeze(0).float().to(device)
        batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0).float().to(device)

        # Get original model prediction
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

        # Get prediction probability and binary prediction
        orig_output_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
        orig_output_binary = (orig_output_prob > 0.5).astype(np.float32)

        # Save original predictions
        original_preds.append(orig_output_binary)
        original_probs.append(orig_output_prob)

        # Extract embedding for this test sample
        test_embedding = extract_single_embedding(model, batch_x, batch_x_mark, device)

        # Find similar samples
        similar_samples = find_similar_samples(
            test_embedding,
            train_embeddings,
            val_embeddings,
            None,
            top_n=top_n
        )

        # Sort similar samples by similarity (descending)
        similar_samples.sort(key=lambda x: x[2], reverse=True)

        # Get similar training samples for embedding model
        similar_train_indices = [idx for split, idx, _ in similar_samples if split == 'train']

        # Calculate original model confusion matrix on similar samples
        # This is just for display and comparison purposes
        orig_TP, orig_TN, orig_FP, orig_FN = 0, 0, 0, 0
        for split, similarity_idx, _ in similar_samples:
            # Get the original data for this similar sample
            if split == 'train':
                similar_batch_x, similar_batch_y, similar_batch_x_mark, similar_batch_y_mark = train_data[
                    similarity_idx]
                true_label_similar = train_labels_array[similarity_idx]
            elif split == 'val' and val_data is not None:
                similar_batch_x, similar_batch_y, similar_batch_x_mark, similar_batch_y_mark = val_data[similarity_idx]
                true_label_similar = val_labels[similarity_idx]
            else:
                continue

            # Add batch dimension and convert to tensor
            similar_batch_x = torch.tensor(similar_batch_x).unsqueeze(0).float().to(device)
            similar_batch_y = torch.tensor(similar_batch_y).unsqueeze(0).float().to(device)
            similar_batch_x_mark = torch.tensor(similar_batch_x_mark).unsqueeze(0).float().to(device)
            similar_batch_y_mark = torch.tensor(similar_batch_y_mark).unsqueeze(0).float().to(device)

            # Use original model for prediction
            with torch.no_grad():
                # Decoder input
                dec_inp = torch.zeros_like(similar_batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([similar_batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                # Generate prediction
                if args.output_attention:
                    outputs = model(similar_batch_x, similar_batch_x_mark, dec_inp, similar_batch_y_mark)[0]
                else:
                    outputs = model(similar_batch_x, similar_batch_x_mark, dec_inp, similar_batch_y_mark)

                # Process outputs for binary classification
                f_dim = -1 if args.features == 'MS' else 0
                outputs_last = outputs[:, -1, f_dim:]

                # Get prediction probability and binary prediction
                similar_pred_prob = torch.sigmoid(outputs_last).detach().cpu().numpy()[0, 0]
                similar_pred = 1 if similar_pred_prob >= 0.5 else 0

                # Update confusion matrix for original model
                if similar_pred == 1 and true_label_similar == 1:
                    orig_TP += 1
                elif similar_pred == 0 and true_label_similar == 0:
                    orig_TN += 1
                elif similar_pred == 1 and true_label_similar == 0:
                    orig_FP += 1
                elif similar_pred == 0 and true_label_similar == 1:
                    orig_FN += 1

        # Calculate accuracy metrics for original model confusion matrix (just for display)
        orig_positive_accuracy = orig_TP / (orig_TP + orig_FN) if (orig_TP + orig_FN) > 0 else 0
        orig_negative_accuracy = orig_TN / (orig_TN + orig_FP) if (orig_TN + orig_FP) > 0 else 0
        orig_positive_samples = orig_TP + orig_FN
        orig_negative_samples = orig_TN + orig_FP
        orig_total_samples = orig_positive_samples + orig_negative_samples

        if orig_total_samples > 0:
            orig_weighted_positive_accuracy = orig_positive_accuracy * (orig_positive_samples / orig_total_samples)
            orig_weighted_negative_accuracy = orig_negative_accuracy * (orig_negative_samples / orig_total_samples)
        else:
            orig_weighted_positive_accuracy = orig_weighted_negative_accuracy = 0

        # Print original model confusion matrix (for reference only)
        print(f"Confusion matrix for original model on similar samples:")
        print(f"TP: {orig_TP}, TN: {orig_TN}, FP: {orig_FP}, FN: {orig_FN}")
        print(f"Positive accuracy: {orig_positive_accuracy:.4f} ({orig_positive_samples} samples)")
        print(f"Negative accuracy: {orig_negative_accuracy:.4f} ({orig_negative_samples} samples)")
        print(f"Weighted positive accuracy: {orig_weighted_positive_accuracy:.4f}")
        print(f"Weighted negative accuracy: {orig_weighted_negative_accuracy:.4f}")

        # Collect similar samples for model training (preserve temporal structure)
        ffn_train_embeddings = []
        ffn_train_labels = []

        for train_idx in similar_train_indices:
            # Keep the temporal structure [seq_len, embed_dim]
            embedding = train_embeddings[train_idx]
            ffn_train_embeddings.append(embedding)
            ffn_train_labels.append(train_labels_array[train_idx])

        # Check if we have enough samples for the embedding-based approach
        if len(ffn_train_embeddings) < 10:
            print(
                f"Warning: Not enough similar samples found for embedding model ({len(ffn_train_embeddings)}), using original prediction")
            embedding_preds.append(orig_output_binary)
            embedding_probs.append(orig_output_prob)

            # For consistency, still build empty confusion matrix
            confusion_matrix_info = {
                'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
                'positive_accuracy': 0,
                'negative_accuracy': 0,
                'weighted_positive_accuracy': 0,
                'weighted_negative_accuracy': 0,
                'positive_samples': 0,
                'negative_samples': 0,
                'total_similar_samples': 0
            }
            confusion_matrices.append(confusion_matrix_info)

            # Skip trading policies when insufficient data
            trade_decisions_higher_acc.append(False)
            trade_decisions_threshold.append(False)
            trade_decisions_weighted.append(False)
            trade_decisions_confidence_gap.append(False)
            trade_decisions_combined.append(False)

            # Record empty accuracy values
            similar_sample_orig_accuracies.append(0)
            trained_model_accuracies.append(0)
            continue

        # Create appropriate embedding model
        seq_len, embed_dim = ffn_train_embeddings[0].shape

        if model_type == 'tcn':
            embedding_model = TemporalConvNet(seq_len, embed_dim)
        elif model_type == 'transformer':
            embedding_model = TransformerEmbeddingModel(seq_len, embed_dim)
        elif model_type == 'cnn_lstm':
            embedding_model = CNNLSTMEmbeddingModel(seq_len, embed_dim)
        elif model_type == 'attention_pooling':
            embedding_model = AttentionPoolingEmbeddingModel(seq_len, embed_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Move model to the correct device
        embedding_model = embedding_model.to(device)

        # Train on similar samples
        print(f"Training {model_type} model on {len(ffn_train_embeddings)} similar samples on {device}...")

        # Convert to PyTorch tensors and move to device
        X = torch.FloatTensor(ffn_train_embeddings).to(device)
        y = torch.FloatTensor(ffn_train_labels).view(-1, 1).to(device)

        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=min(16, len(X)), shuffle=True)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(embedding_model.parameters(), lr=ffn_lr)

        # Training loop
        embedding_model.train()
        epoch_train_accuracies = []

        for epoch in range(ffn_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = embedding_model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_accuracy = correct / total
            epoch_train_accuracies.append(train_accuracy)

            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{ffn_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}')
                if str(device).startswith('cuda'):
                    print(f"GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated")

        # Store the training accuracy
        trained_model_accuracies.append(np.mean(epoch_train_accuracies))

        # Get prediction for current test sample
        embedding_model.eval()
        with torch.no_grad():
            test_embedding_tensor = torch.FloatTensor(test_embedding).unsqueeze(0).to(device)
            pred_prob = embedding_model(test_embedding_tensor).item()
            pred = 1 if pred_prob >= 0.5 else 0

        embedding_preds.append(pred)
        embedding_probs.append(pred_prob)

        # Now build confusion matrix based on the embedding model's performance on similar samples
        # Instead of using the original model
        TP, TN, FP, FN = 0, 0, 0, 0
        similar_labels = []
        similar_predictions = []

        # Process all similar samples to evaluate the embedding model
        for split, similarity_idx, _ in similar_samples:
            # Get the data for this similar sample
            if split == 'train':
                similar_batch_x, similar_batch_y, similar_batch_x_mark, similar_batch_y_mark = train_data[
                    similarity_idx]
                true_label_similar = train_labels_array[similarity_idx]
            elif split == 'val' and val_data is not None:
                similar_batch_x, similar_batch_y, similar_batch_x_mark, similar_batch_y_mark = val_data[similarity_idx]
                true_label_similar = val_labels[similarity_idx]
            else:
                continue

            # Extract embedding for this similar sample
            similar_batch_x = torch.tensor(similar_batch_x).unsqueeze(0).float().to(device)
            similar_batch_x_mark = torch.tensor(similar_batch_x_mark).unsqueeze(0).float().to(device)
            similar_embedding = extract_single_embedding(model, similar_batch_x, similar_batch_x_mark, device)

            # Use embedding model to predict
            with torch.no_grad():
                similar_embedding_tensor = torch.FloatTensor(similar_embedding).unsqueeze(0).to(device)
                similar_pred_prob = embedding_model(similar_embedding_tensor).item()
                similar_pred = 1 if similar_pred_prob >= 0.5 else 0

                # Update confusion matrix
                if similar_pred == 1 and true_label_similar == 1:
                    TP += 1
                elif similar_pred == 0 and true_label_similar == 0:
                    TN += 1
                elif similar_pred == 1 and true_label_similar == 0:
                    FP += 1
                elif similar_pred == 0 and true_label_similar == 1:
                    FN += 1

                similar_labels.append(true_label_similar)
                similar_predictions.append(similar_pred)

        # Calculate accuracy metrics for embedding model confusion matrix
        positive_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0
        negative_accuracy = TN / (TN + FP) if (TN + FP) > 0 else 0

        # Calculate weighted accuracy (accounts for sample size)
        positive_samples = TP + FN
        negative_samples = TN + FP
        total_samples = positive_samples + negative_samples

        if total_samples > 0:
            weighted_positive_accuracy = positive_accuracy * (positive_samples / total_samples)
            weighted_negative_accuracy = negative_accuracy * (negative_samples / total_samples)
        else:
            weighted_positive_accuracy = weighted_negative_accuracy = 0

        # Store confusion matrix information
        confusion_matrix_info = {
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'positive_accuracy': positive_accuracy,
            'negative_accuracy': negative_accuracy,
            'weighted_positive_accuracy': weighted_positive_accuracy,
            'weighted_negative_accuracy': weighted_negative_accuracy,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'total_similar_samples': len(similar_samples)
        }
        confusion_matrices.append(confusion_matrix_info)

        # Calculate similar sample accuracy
        similar_accuracy = np.mean(np.array(similar_predictions) == np.array(similar_labels))
        similar_sample_orig_accuracies.append(similar_accuracy)

        print(f"Embedding model confusion matrix on similar samples:")
        print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        print(f"Positive accuracy: {positive_accuracy:.4f} ({positive_samples} samples)")
        print(f"Negative accuracy: {negative_accuracy:.4f} ({negative_samples} samples)")
        print(f"Weighted positive accuracy: {weighted_positive_accuracy:.4f}")
        print(f"Weighted negative accuracy: {weighted_negative_accuracy:.4f}")

        # Apply multi-policy trading logic based on EMBEDDING model's confusion matrix
        # Note: using the embedding model's prediction for decision making
        # instead of the original model's prediction
        embedding_pred = pred

        # Policy 1: Original policy (higher accuracy)
        should_trade_higher_acc = False
        if embedding_pred == 1:
            should_trade_higher_acc = positive_accuracy > negative_accuracy
        else:
            should_trade_higher_acc = negative_accuracy > positive_accuracy
        trade_decisions_higher_acc.append(should_trade_higher_acc)

        # Policy 2: Minimum accuracy threshold (0.5)
        should_trade_threshold = False
        if embedding_pred == 1:
            should_trade_threshold = positive_accuracy > negative_accuracy and positive_accuracy > 0.5
        else:
            should_trade_threshold = negative_accuracy > positive_accuracy and negative_accuracy > 0.5
        trade_decisions_threshold.append(should_trade_threshold)

        # Policy 3: Weighted accuracy comparison
        should_trade_weighted = False
        if embedding_pred == 1:
            should_trade_weighted = weighted_positive_accuracy > weighted_negative_accuracy
        else:
            should_trade_weighted = weighted_negative_accuracy > weighted_positive_accuracy
        trade_decisions_weighted.append(should_trade_weighted)

        # Policy 4: Confidence gap
        confidence_gap = 0.2
        should_trade_confidence_gap = False
        if embedding_pred == 1:
            should_trade_confidence_gap = positive_accuracy > negative_accuracy + confidence_gap
        else:
            should_trade_confidence_gap = negative_accuracy > positive_accuracy + confidence_gap
        trade_decisions_confidence_gap.append(should_trade_confidence_gap)

        # Policy 5: Combined policy (must satisfy multiple criteria)
        should_trade_combined = False
        if embedding_pred == 1:
            should_trade_combined = (
                    positive_accuracy > negative_accuracy and
                    positive_accuracy > 0.5 and
                    positive_accuracy > negative_accuracy + 0.1 and
                    positive_samples >= 5  # Minimum sample requirement
            )
        else:
            should_trade_combined = (
                    negative_accuracy > positive_accuracy and
                    negative_accuracy > 0.5 and
                    negative_accuracy > positive_accuracy + 0.1 and
                    negative_samples >= 5  # Minimum sample requirement
            )
        trade_decisions_combined.append(should_trade_combined)

        # Print results
        print(f"Orig prediction: {orig_output_binary}, Embedding-based prediction: {pred}, True label: {true_label}")
        print(f"Trade decisions (based on embedding model performance):")
        print(f"  Higher acc policy: {'Trade' if should_trade_higher_acc else 'No Trade'}")
        print(f"  Threshold policy: {'Trade' if should_trade_threshold else 'No Trade'}")
        print(f"  Weighted policy: {'Trade' if should_trade_weighted else 'No Trade'}")
        print(f"  Confidence gap policy: {'Trade' if should_trade_confidence_gap else 'No Trade'}")
        print(f"  Combined policy: {'Trade' if should_trade_combined else 'No Trade'}")

        # Clear GPU cache if using CUDA
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    # Calculate metrics for the embedding model approach
    accuracy = accuracy_score(trues, embedding_preds)

    # Convert result arrays to numpy arrays
    embedding_preds = np.array(embedding_preds)
    embedding_probs = np.array(embedding_probs)
    trues = np.array(trues)
    original_preds = np.array(original_preds)
    original_probs = np.array(original_probs)

    # Convert trade decision arrays
    trade_decisions_higher_acc = np.array(trade_decisions_higher_acc)
    trade_decisions_threshold = np.array(trade_decisions_threshold)
    trade_decisions_weighted = np.array(trade_decisions_weighted)
    trade_decisions_confidence_gap = np.array(trade_decisions_confidence_gap)
    trade_decisions_combined = np.array(trade_decisions_combined)

    # Calculate metrics for each policy using embedding predictions for evaluation
    def calculate_policy_metrics(trade_mask):
        if np.sum(trade_mask) > 0:
            # Only evaluate on samples where we would trade
            traded_indices = np.where(trade_mask)[0]
            accuracy = accuracy_score(trues[traded_indices], embedding_preds[traded_indices])
            precision = precision_score(trues[traded_indices], embedding_preds[traded_indices], zero_division=0)
            recall = recall_score(trues[traded_indices], embedding_preds[traded_indices], zero_division=0)
            f1 = f1_score(trues[traded_indices], embedding_preds[traded_indices], zero_division=0)
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'traded_count': int(np.sum(trade_mask)),
                'total_count': len(trues)
            }
        else:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'traded_count': 0,
                'total_count': len(trues)
            }

    # Calculate metrics for each policy
    policy_metrics = {
        'overall': {'accuracy': accuracy},
        'original': calculate_policy_metrics(trade_decisions_higher_acc),
        'threshold': calculate_policy_metrics(trade_decisions_threshold),
        'weighted': calculate_policy_metrics(trade_decisions_weighted),
        'confidence_gap': calculate_policy_metrics(trade_decisions_confidence_gap),
        'combined': calculate_policy_metrics(trade_decisions_combined)
    }

    # Print summary of all policies
    print(f"\nEnhanced embedding-based approach results:")
    print(f"Total samples: {len(trues)}")
    print(f"Embedding model accuracy: {accuracy:.4f}")
    print(f"Average similar samples orig accuracy: {np.mean(similar_sample_orig_accuracies):.4f}")
    print(f"Average training accuracy: {np.mean(trained_model_accuracies):.4f}")

    print("\nTrading policy comparison:")
    print(f"{'Policy':<15} {'Trades':<8} {'Skipped':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 75)

    for policy_name, metrics in policy_metrics.items():
        if policy_name == 'overall':
            continue
        traded = metrics['traded_count']
        skipped = metrics['total_count'] - traded
        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        f1_score_val = metrics['f1']
        print(
            f"{policy_name:<15} {traded:<8} {skipped:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1_score_val:<10.4f}")

    # Return both the original embedding model results and the new multi-policy trading results
    return {
        # Original return values for backward compatibility
        'preds': embedding_preds,
        'probs': embedding_probs,
        'trues': trues,
        'accuracy': accuracy,
        'similar_samples_accuracy': np.mean(similar_sample_orig_accuracies),
        'training_accuracy': np.mean(trained_model_accuracies),

        # New trading policy values
        'original_preds': original_preds,
        'original_probs': original_probs,
        'trade_decisions_original': trade_decisions_higher_acc,
        'trade_decisions_threshold': trade_decisions_threshold,
        'trade_decisions_weighted': trade_decisions_weighted,
        'trade_decisions_confidence_gap': trade_decisions_confidence_gap,
        'trade_decisions_combined': trade_decisions_combined,
        'confusion_matrices': confusion_matrices,
        'policy_metrics': policy_metrics
    }

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


def calculate_and_display_metrics(model, train_data, val_data, test_data, args, device):
    """
    Calculate and display metrics for train, validation, and test datasets
    similar to exp_logits_forecasting.py

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    train_data, val_data, test_data : Dataset
        Datasets for evaluation
    args : argparse.Namespace
        Command line arguments
    device : torch.device
        Device to run model on
    """
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Set model to evaluation mode
    model.eval()

    # Function to evaluate model on a dataset
    def evaluate_dataset(dataset, name):
        print(f"Evaluating model on {name} dataset ({len(dataset)} samples)...")

        all_preds = []
        all_trues = []
        all_probs = []

        with torch.no_grad():
            # Process each sample individually to avoid batch issues
            for i in range(len(dataset)):
                # Get sample
                batch_x, batch_y, batch_x_mark, batch_y_mark = dataset[i]

                # Add batch dimension
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

                all_preds.append(output_binary)
                all_trues.append(true_label)
                all_probs.append(output_prob)

                # Progress indication for larger datasets
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} samples...")

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_trues, all_preds)

        # Calculate precision, recall, and F1 with error handling
        try:
            precision = precision_score(all_trues, all_preds)
        except:
            precision = 0.0

        try:
            recall = recall_score(all_trues, all_preds)
        except:
            recall = 0.0

        try:
            f1 = f1_score(all_trues, all_preds)
        except:
            f1 = 0.0

        # Calculate confusion matrix
        cm = confusion_matrix(all_trues, all_preds)

        # Extract values from confusion matrix
        if cm.shape == (2, 2):
            TN, FP = cm[0, 0], cm[0, 1]
            FN, TP = cm[1, 0], cm[1, 1]
        else:
            TN, FP, FN, TP = 0, 0, 0, 0
            print(f"  Warning: Confusion matrix not 2x2, actual shape: {cm.shape}")

        # Print detailed results
        print(f"\n{name} Set Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({int(accuracy * len(all_trues))}/{len(all_trues)})")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        print(f"\n{name} Set Confusion Matrix:")
        print(f"  True Positives (TP): {TP}")
        print(f"  True Negatives (TN): {TN}")
        print(f"  False Positives (FP): {FP}")
        print(f"  False Negatives (FN): {FN}")

        # Calculate class distribution
        pos_count = np.sum(all_trues == 1)
        neg_count = np.sum(all_trues == 0)
        pos_ratio = pos_count / len(all_trues) if len(all_trues) > 0 else 0

        print(f"\n{name} Set Class Distribution:")
        print(f"  Positive samples (1): {pos_count} ({pos_ratio:.2%})")
        print(f"  Negative samples (0): {neg_count} ({1 - pos_ratio:.2%})")

        # Calculate trading metrics
        if args.is_shorting:
            # For shorting strategy: profit from both correct predictions
            profitable_trades = np.logical_and(all_preds == 1, all_trues == 1).sum() + np.logical_and(all_preds == 0,
                                                                                                      all_trues == 0).sum()
            total_trades = len(all_preds)
        else:
            # For no-shorting strategy: only profit from correct positive predictions
            profitable_trades = np.logical_and(all_preds == 1, all_trues == 1).sum()
            unprofitable_trades = np.logical_and(all_preds == 1, all_trues == 0).sum()
            total_trades = profitable_trades + unprofitable_trades

        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        print(f"\n{name} Set Trading Performance:")
        print(f"  Win Rate: {win_rate:.4f} ({profitable_trades}/{total_trades})")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'win_rate': win_rate,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }

    # Evaluate datasets
    train_metrics = evaluate_dataset(train_data, "Training") if train_data is not None else None
    val_metrics = evaluate_dataset(val_data, "Validation") if val_data is not None else None
    test_metrics = evaluate_dataset(test_data, "Test") if test_data is not None else None

    # Print comparative summary similar to exp_logits_forecasting.py
    print("\nComparative Performance Summary:")
    print("-" * 80)
    print(
        f"{'Dataset':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Win Rate':<10}")
    print("-" * 80)

    if train_metrics:
        print(f"{'Train':<10} | {train_metrics['accuracy']:<10.4f} | {train_metrics['precision']:<10.4f} | "
              f"{train_metrics['recall']:<10.4f} | {train_metrics['f1']:<10.4f} | {train_metrics['win_rate']:<10.4f}")

    if val_metrics:
        print(f"{'Val':<10} | {val_metrics['accuracy']:<10.4f} | {val_metrics['precision']:<10.4f} | "
              f"{val_metrics['recall']:<10.4f} | {val_metrics['f1']:<10.4f} | {val_metrics['win_rate']:<10.4f}")

    if test_metrics:
        print(f"{'Test':<10} | {test_metrics['accuracy']:<10.4f} | {test_metrics['precision']:<10.4f} | "
              f"{test_metrics['recall']:<10.4f} | {test_metrics['f1']:<10.4f} | {test_metrics['win_rate']:<10.4f}")

    print("=" * 80 + "\n")

    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

def run_inference_for_training(model, train_data, args, device):
    """Generate predictions and confusion matrices for training data"""
    print(f"Running inference on {len(train_data)} training samples...")
    model.eval()
    train_confusion_matrices = []

    # Initialize running confusion matrix for tracking metrics
    TP, TN, FP, FN = 0, 0, 0, 0

    with torch.no_grad():
        for i in range(len(train_data)):
            # Get sample
            batch_x, batch_y, batch_x_mark, batch_y_mark = train_data[i]

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

            # Update confusion matrix values
            if output_binary == 1 and true_label == 1:
                TP += 1
            elif output_binary == 0 and true_label == 0:
                TN += 1
            elif output_binary == 1 and true_label == 0:
                FP += 1
            elif output_binary == 0 and true_label == 1:
                FN += 1

            # Store confusion matrix for this prediction
            train_confusion_matrices.append({
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'output_binary': output_binary,
                'output_prob': output_prob,
                'true_label': true_label
            })

            # Progress indication
            if (i + 1) % 100 == 0 or i == len(train_data) - 1:
                print(f'Processed {i + 1}/{len(train_data)} training samples')

    return train_confusion_matrices


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

    # For confusion matrix tracking
    confusion_matrices = []

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

    # Initialize running confusion matrix for tracking metrics
    TP, TN, FP, FN = 0, 0, 0, 0

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

            # Update confusion matrix values
            if output_binary == 1 and true_label == 1:
                TP += 1
            elif output_binary == 0 and true_label == 0:
                TN += 1
            elif output_binary == 1 and true_label == 0:
                FP += 1
            elif output_binary == 0 and true_label == 1:
                FN += 1

            # Store confusion matrix for this prediction
            confusion_matrices.append({
                'TP': TP,
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'output_binary': output_binary,
                'output_prob': output_prob,
                'true_label': true_label
            })

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

    return preds, trues, probs, timestamps, original_indices, prediction_indices, prices, confusion_matrices


def apply_ffn_consensus_model(confusion_matrices, train_confusion_matrices, args):
    """
    Train and apply the FFN consensus model using only training data

    Parameters:
    -----------
    confusion_matrices : list
        List of dictionaries containing confusion matrix values and predictions for test data
    train_confusion_matrices : list
        List of dictionaries containing confusion matrix values and predictions for training data
    args : argparse.Namespace
        Command line arguments

    Returns:
    --------
    tuple
        (ffn_preds, ffn_probs, consensus_model, losses)
    """
    print("\nApplying FFN Consensus Model...")

    # Convert training data to features and labels
    train_features = []
    train_labels = []

    for cm in train_confusion_matrices:
        feature = [
            cm['output_binary'],
            cm['output_prob'],
            cm['TP'],
            cm['TN'],
            cm['FP'],
            cm['FN']
        ]
        train_features.append(feature)
        train_labels.append(cm['true_label'])

    # Convert to numpy arrays
    train_features_array = np.array(train_features)
    train_labels_array = np.array(train_labels)

    print(f"Training FFN model on {len(train_features_array)} training samples...")

    # Train the consensus model on training data only
    consensus_model, losses = train_consensus_model(
        train_features_array, train_labels_array,
        epochs=args.ffn_epochs,
        lr=args.ffn_learning_rate,
        batch_size=args.ffn_batch_size
    )

    # Generate FFN predictions for test samples
    ffn_preds = []
    ffn_probs = []

    # Convert test data to features
    test_features = []
    for cm in confusion_matrices:
        feature = [
            cm['output_binary'],
            cm['output_prob'],
            cm['TP'],
            cm['TN'],
            cm['FP'],
            cm['FN']
        ]
        test_features.append(feature)

    test_features_array = np.array(test_features)

    # Apply the model to test features
    for feature in test_features_array:
        # Convert to tensor
        input_features = torch.FloatTensor(feature)

        # Get prediction
        with torch.no_grad():
            model_pred_prob = consensus_model(input_features).item()
            model_pred = 1 if model_pred_prob >= 0.5 else 0

        ffn_preds.append(model_pred)
        ffn_probs.append(model_pred_prob)

    return np.array(ffn_preds), np.array(ffn_probs), consensus_model, losses


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


def print_consensus_comparison(preds, ffn_preds, trues, timestamps):
    """
    Print comparison between original predictions and FFN consensus model predictions

    Parameters:
    -----------
    preds : numpy.ndarray
        Original binary predictions (0 or 1)
    ffn_preds : numpy.ndarray
        FFN consensus model predictions (0 or 1)
    trues : numpy.ndarray
        Actual binary labels (0 or 1)
    timestamps : list
        Timestamps for each prediction
    """
    print("\n----- ORIGINAL vs FFN MODEL PREDICTIONS -----")
    print("Sample | Timestamp | Original | FFN | True | Match?")
    print("------------------------------------------------")

    # Calculate where predictions differ for focused analysis
    diff_indices = np.where(preds != ffn_preds)[0]

    # Show some examples where predictions differ (up to 10)
    samples_to_show = diff_indices[:min(10, len(diff_indices))]

    if len(samples_to_show) == 0:
        print("No differences in predictions between original and FFN models.")
        samples_to_show = range(min(10, len(preds)))  # Show first 10 samples instead

    for i in samples_to_show:
        # Format timestamp for display
        if timestamps[i] is None:
            ts_str = f"sample_{i}"
        elif isinstance(timestamps[i], pd.Timestamp):
            ts_str = timestamps[i].strftime('%Y-%m-%d %H:%M:%S')
        else:
            ts_str = str(timestamps[i])

        # Check if original prediction was correct
        if preds[i] == trues[i]:
            original_result = "✓"
        else:
            original_result = "✗"

        # Check if FFN prediction was correct
        if ffn_preds[i] == trues[i]:
            ffn_result = "✓"
        else:
            ffn_result = "✗"

        # Print summary line for this sample
        print(
            f"{i:<6} | {ts_str:<18} | {preds[i]:<8} {original_result} | {ffn_preds[i]:<3} {ffn_result} | {trues[i]:<4} | {'Yes' if ffn_preds[i] == trues[i] else 'No'}")

    # Calculate and print overall metrics
    original_accuracy = accuracy_score(trues, preds)
    ffn_accuracy = accuracy_score(trues, ffn_preds)
    improvement = ffn_accuracy - original_accuracy

    # Count samples where FFN improved or worsened prediction
    improved = np.sum((preds != trues) & (ffn_preds == trues))
    worsened = np.sum((preds == trues) & (ffn_preds != trues))

    print("\nOverall Metrics:")
    print(f"Original Model Accuracy: {original_accuracy:.4f}")
    print(f"FFN Model Accuracy: {ffn_accuracy:.4f}")
    print(f"Improvement: {improvement * 100:.2f}%")
    print(f"Number of predictions improved by FFN: {improved}")
    print(f"Number of predictions worsened by FFN: {worsened}")


def print_detailed_analysis(preds, trues, probs, timestamps, prices, actual_changes, returns, ffn_preds=None,
                            ffn_probs=None):
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
    ffn_preds : numpy.ndarray, optional
        FFN consensus model predictions
    ffn_probs : numpy.ndarray, optional
        FFN consensus model prediction probabilities
    """
    # Print detailed trading results for the combined strategy
    print("\nDetailed Analysis (Combined):")
    print("-" * 140)
    header = f"{'Sample':<8} | {'Timestamp':<20} | {'Price':<12} | {'Orig':<5} | "
    if ffn_preds is not None:
        header += f"{'FFN':<5} | "
    header += f"{'True':<5} | {'Prob':<8} | "
    if ffn_probs is not None:
        header += f"{'FFN Prob':<8} | "
    header += f"{'Actual Chg':<10} | {'Percent Change':>8} | {'Cum Percent Change':>12}"
    print(header)
    print("-" * 140)

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

        # Start the row with common fields
        row = f"{i:<8} | {ts_str:<20} | {price_str} | {preds[i]:<5.0f} | "

        # Add FFN prediction if available
        if ffn_preds is not None:
            row += f"{ffn_preds[i]:<5.0f} | "

        # Continue with common fields
        row += f"{trues[i]:<5.0f} | {probs[i]:<8.4f} | "

        # Add FFN probability if available
        if ffn_probs is not None:
            row += f"{ffn_probs[i]:<8.4f} | "

        # Finish with remaining fields
        row += f"{actual_changes[i] * 100:>10.2f}% | {combined_returns[i]:>8.4%} | {cum_returns[i]:>12.4%}"

        # Print the completed row
        print(row)

def save_results(preds, trues, probs, timestamps, original_indices, prediction_indices, prices, metrics, returns, args,
                 actual_changes, ffn_preds=None, ffn_probs=None, ffn_metrics=None):
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

    # Prepare metrics data
    metrics_data = {
        'model': args.model,
        'model_id': args.model_id,
        'data_path': args.data_path,
        'metrics': metrics,
        'trading_returns': {
            'total_return': returns['total_return']
        },
        'args': vars(args)
    }

    # Add FFN metrics if available
    if ffn_metrics is not None:
        metrics_data['ffn_metrics'] = ffn_metrics
        metrics_data['metrics_comparison'] = {
            'accuracy_improvement': ffn_metrics['accuracy'] - metrics['accuracy'],
            'precision_improvement': ffn_metrics['precision'] - metrics['precision'],
            'recall_improvement': ffn_metrics['recall'] - metrics['recall'],
            'f1_improvement': ffn_metrics['f1'] - metrics['f1'],
            'win_rate_improvement': ffn_metrics['trading']['win_rate'] - metrics['trading']['win_rate']
        }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # Save the detailed analysis table to a CSV
    results_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_inference_results_table.csv")

    # Create dataframe with base columns
    detailed_df = pd.DataFrame({
        'sample': range(len(preds)),
        'timestamp': human_timestamps,
        'price': prices,
        'prediction_original': preds,
        'true_label': trues,
        'probability_original': probs,
        'actual_change_pct': [x * 100 for x in actual_changes],
        'trade_return_pct': [x * 100 for x in returns['strategies']['combined']['returns']],
        'cumulative_return_pct': [x * 100 for x in returns['strategies']['combined']['cumulative']]
    })

    # Add FFN columns if available
    if ffn_preds is not None and ffn_probs is not None:
        detailed_df['prediction_ffn'] = ffn_preds
        detailed_df['probability_ffn'] = ffn_probs
        detailed_df['prediction_match'] = detailed_df['prediction_original'] == detailed_df['prediction_ffn']
        detailed_df['original_correct'] = detailed_df['prediction_original'] == detailed_df['true_label']
        detailed_df['ffn_correct'] = detailed_df['prediction_ffn'] == detailed_df['true_label']
        detailed_df['prediction_improved'] = (~detailed_df['original_correct']) & detailed_df['ffn_correct']

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


def apply_conditional_trading_approach(preds, ffn_preds, trues, probs, ffn_probs, actual_changes, is_shorting=True):
    """
    Apply a conditional trading approach that only takes positions when both models agree

    Parameters:
    -----------
    preds : numpy.ndarray
        Original binary predictions (0 or 1)
    ffn_preds : numpy.ndarray
        FFN consensus model predictions (0 or 1)
    trues : numpy.ndarray
        Actual binary labels (0 or 1)
    probs : numpy.ndarray
        Original prediction probabilities
    ffn_probs : numpy.ndarray
        FFN prediction probabilities
    actual_changes : numpy.ndarray
        Actual percentage price changes
    is_shorting : bool
        Whether shorting is enabled in the strategy

    Returns:
    --------
    dict
        Dictionary containing returns and metrics for the conditional approach
    """
    print("Applying conditional trading approach (trade only when models agree)...")

    # Initialize arrays
    conditional_preds = np.zeros_like(preds)
    conditional_returns = np.zeros_like(preds, dtype=float)

    # Only take positions when both models agree
    agreement_mask = (preds == ffn_preds)
    conditional_preds[agreement_mask] = preds[agreement_mask]

    # Calculate returns based on conditional predictions
    for i in range(len(conditional_preds)):
        if agreement_mask[i]:  # Only trade when models agree
            if conditional_preds[i] == 1:  # Both predict up
                conditional_returns[i] = actual_changes[i]  # Long position
            elif conditional_preds[i] == 0 and is_shorting:  # Both predict down
                conditional_returns[i] = -actual_changes[i]  # Short position

    # Calculate cumulative returns
    conditional_cum_returns = np.cumprod(1 + conditional_returns) - 1

    # Calculate metrics
    agreement_count = np.sum(agreement_mask)
    trade_count = np.sum(conditional_returns != 0)
    profitable_trades = np.sum(conditional_returns > 0)
    unprofitable_trades = np.sum(conditional_returns < 0)
    win_rate = profitable_trades / trade_count if trade_count > 0 else 0

    # Calculate accuracy only for samples where we made predictions
    traded_mask = agreement_mask & ((conditional_preds == 1) | (conditional_preds == 0 & is_shorting))
    if np.sum(traded_mask) > 0:
        accuracy = accuracy_score(trues[traded_mask], conditional_preds[traded_mask])
    else:
        accuracy = 0

    # Calculate final return
    total_return = conditional_cum_returns[-1] if len(conditional_cum_returns) > 0 else 0

    return {
        'preds': conditional_preds,
        'returns': conditional_returns,
        'cumulative': conditional_cum_returns,
        'total_return': total_return,
        'metrics': {
            'agreement_rate': agreement_count / len(preds),
            'trades_taken': trade_count,
            'trades_skipped': len(preds) - trade_count,
            'profitable_trades': int(profitable_trades),
            'unprofitable_trades': int(unprofitable_trades),
            'win_rate': win_rate,
            'accuracy': accuracy
        }
    }

def apply_ensemble_approach(preds, ffn_preds, trues, probs, ffn_probs, actual_changes, is_shorting=True,
                            confidence_threshold=0.7, ensemble_method='confidence_based'):
    """
    Apply an ensemble approach that combines both models based on their confidence

    Parameters:
    -----------
    preds : numpy.ndarray
        Original binary predictions (0 or 1)
    ffn_preds : numpy.ndarray
        FFN consensus model predictions (0 or 1)
    trues : numpy.ndarray
        Actual binary labels (0 or 1)
    probs : numpy.ndarray
        Original prediction probabilities
    ffn_probs : numpy.ndarray
        FFN prediction probabilities
    actual_changes : numpy.ndarray
        Actual percentage price changes
    is_shorting : bool
        Whether shorting is enabled in the strategy
    confidence_threshold : float
        Threshold for high confidence
    ensemble_method : str
        Method for ensemble ('weighted', 'confidence_based', or 'boosted')

    Returns:
    --------
    dict
        Dictionary containing returns and metrics for the ensemble approach
    """
    print(f"Applying ensemble trading approach (method: {ensemble_method})...")

    # Initialize arrays
    ensemble_preds = np.zeros_like(preds)
    ensemble_probs = np.zeros_like(probs)
    ensemble_trues = np.zeros_like(trues)
    ensemble_returns = np.zeros_like(preds, dtype=float)

    # Create ensemble predictions based on method
    if ensemble_method == 'weighted':
        # Simple weighted average of probabilities (equal weights)
        ensemble_probs = (probs + ffn_probs) / 2
        ensemble_preds = (ensemble_probs > 0.5).astype(np.float32)

    elif ensemble_method == 'confidence_based':
        # Use the model with higher confidence for each prediction
        original_confidence = np.abs(probs - 0.5) * 2  # Scale to [0, 1]
        ffn_confidence = np.abs(ffn_probs - 0.5) * 2

        for i in range(len(preds)):
            if original_confidence[i] > ffn_confidence[i]:
                ensemble_preds[i] = preds[i]
                ensemble_probs[i] = probs[i]
            else:
                ensemble_preds[i] = ffn_preds[i]
                ensemble_probs[i] = ffn_probs[i]

    elif ensemble_method == 'boosted':
        # Trade only if original model is confident enough
        original_confidence = np.abs(probs - 0.5) * 2

        for i in range(len(preds)):
            if original_confidence[i] >= confidence_threshold:
                # High confidence in original model
                ensemble_preds[i] = preds[i]
                ensemble_probs[i] = probs[i]
                ensemble_trues[i] = trues[i]
            else:
                continue

    # Calculate returns based on ensemble predictions
    for i in range(len(ensemble_preds)):
        if ensemble_preds[i] == 1:  # Predict up
            ensemble_returns[i] = actual_changes[i]  # Long position
        elif ensemble_preds[i] == 0 and is_shorting:  # Predict down
            ensemble_returns[i] = -actual_changes[i]  # Short position

    # Calculate cumulative returns
    ensemble_cum_returns = np.cumprod(1 + ensemble_returns) - 1

    # Calculate metrics
    trade_count = len(ensemble_preds)
    profitable_trades = np.sum(ensemble_returns > 0)
    unprofitable_trades = np.sum(ensemble_returns < 0)
    win_rate = profitable_trades / trade_count if trade_count > 0 else 0
    accuracy = accuracy_score(ensemble_trues, ensemble_preds)

    # Calculate final return
    total_return = ensemble_cum_returns[-1] if len(ensemble_cum_returns) > 0 else 0

    return {
        'preds': ensemble_preds,
        'probs': ensemble_probs,
        'returns': ensemble_returns,
        'cumulative': ensemble_cum_returns,
        'total_return': total_return,
        'metrics': {
            'accuracy': accuracy,
            'trades_taken': trade_count,
            'profitable_trades': int(profitable_trades),
            'unprofitable_trades': int(unprofitable_trades),
            'win_rate': win_rate
        }
    }


def print_combined_approaches_summary(original_metrics, ffn_metrics, conditional_results, ensemble_results,
                                      original_returns, ffn_returns):
    """
    Print a summary comparison of all approaches

    Parameters:
    -----------
    original_metrics : dict
        Metrics for the original model
    ffn_metrics : dict
        Metrics for the FFN model
    conditional_results : dict
        Results for the conditional approach
    ensemble_results : dict
        Results for the ensemble approach
    original_returns : dict
        Returns for the original model
    ffn_returns : dict
        Returns for the FFN model
    """
    print("\n----- TRADING APPROACHES COMPARISON -----")
    print(f"{'Approach':<20} | {'Accuracy':<10} | {'Win Rate':<10} | {'Total Return':<15} | {'Trades':<10}")
    print("-" * 75)

    # Original model
    print(f"{'Original Model':<20} | {original_metrics['accuracy'] * 100:>8.2f}% | "
          f"{original_metrics['trading']['win_rate'] * 100:>8.2f}% | "
          f"{original_returns['total_return'] * 100:>13.2f}% | "
          f"{original_metrics['trading']['total_trades']:>10}")

    # FFN model
    print(f"{'FFN Model':<20} | {ffn_metrics['accuracy'] * 100:>8.2f}% | "
          f"{ffn_metrics['trading']['win_rate'] * 100:>8.2f}% | "
          f"{ffn_returns['total_return'] * 100:>13.2f}% | "
          f"{ffn_metrics['trading']['total_trades']:>10}")

    # Conditional approach
    print(f"{'Conditional':<20} | {conditional_results['metrics']['accuracy'] * 100:>8.2f}% | "
          f"{conditional_results['metrics']['win_rate'] * 100:>8.2f}% | "
          f"{conditional_results['total_return'] * 100:>13.2f}% | "
          f"{conditional_results['metrics']['trades_taken']:>10}")

    # Ensemble approach
    print(f"{'Ensemble':<20} | {ensemble_results['metrics']['accuracy'] * 100:>8.2f}% | "
          f"{ensemble_results['metrics']['win_rate'] * 100:>8.2f}% | "
          f"{ensemble_results['total_return'] * 100:>13.2f}% | "
          f"{ensemble_results['metrics']['trades_taken']:>10}")

    print("\nConditional Approach Details:")
    print(f"Agreement Rate: {conditional_results['metrics']['agreement_rate'] * 100:.2f}%")
    print(f"Trades Skipped: {conditional_results['metrics']['trades_skipped']}")


def main():
    try:
        print("Starting inference script...")
        # Parse arguments
        args = parse_args()

        # Determine device
        device = torch.device('cuda:{}'.format(args.gpu) if args.use_gpu else 'cpu')
        print(f'Using device: {device}')

        # Create experiment and model setting
        setting = '{}_{}_{}'.format(
            args.data_path,
            args.class_strategy, args.projection_idx)

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

        # Get all datasets
        print("Getting data...")
        train_data, _ = exp._get_data(flag='train')
        val_data, _ = exp._get_data(flag='val')
        test_data, test_loader = get_test_data(args)

        # Analyze loaded model performance on all datasets
        metrics_results = calculate_and_display_metrics(model, train_data, val_data, test_data, args, device)

        print("Running inference on training data...")
        train_confusion_matrices = run_inference_for_training(model, train_data, args, device)

        # Run inference with improved timestamp and prediction index tracking
        print("Running inference with original model...")
        preds, trues, probs, timestamps, original_indices, prediction_indices, prices, confusion_matrices = run_inference(
            model, test_data, test_loader, args, device)

        # Calculate metrics for original predictions
        print("Calculating metrics...")
        metrics = calculate_metrics(preds, trues, bool(args.is_shorting))

        # explicitly show the original model's accuracy on test set
        print(f"\nOriginal model accuracy on test set: {metrics['accuracy']:.4f}")

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

        # Calculate trading returns for original model
        print("Calculating trading returns for original model...")
        original_returns = calculate_returns(preds, trues, probs, bool(args.is_shorting), actual_changes)

        # Initialize variables for additional methods
        ffn_preds = None
        ffn_probs = None
        ffn_metrics = None
        ffn_returns = None
        embedding_preds = None
        embedding_probs = None
        embedding_metrics = None
        embedding_returns = None

        # Apply FFN consensus model if enabled
        if args.use_consensus_model:
            print("\nApplying FFN consensus model...")
            ffn_preds, ffn_probs, consensus_model, losses = apply_ffn_consensus_model(
                confusion_matrices, train_confusion_matrices, args)
            # ffn_preds, ffn_probs, consensus_model, train_indices, test_indices, losses = apply_ffn_consensus_model(
            #     confusion_matrices, args)

            # Calculate metrics for FFN predictions
            ffn_metrics = calculate_metrics(ffn_preds, trues, bool(args.is_shorting))

            # Calculate returns for FFN predictions
            ffn_returns = calculate_returns(ffn_preds, trues, ffn_probs, bool(args.is_shorting), actual_changes)

            # Print comparison between original and FFN predictions
            print_consensus_comparison(preds, ffn_preds, trues, timestamps)

        # Apply embedding-based approach if enabled
        if args.use_embedding_approach:
            print("\nApplying embedding-based approach...")
            # embedding_results = apply_embedding_based_approach(
            #     args,
            #     model,
            #     train_data,
            #     val_data,
            #     test_data,
            #     device,
            #     top_n=args.similar_samples,
            #     ffn_epochs=args.embedding_ffn_epochs,
            #     ffn_lr=args.embedding_ffn_lr
            # )

            embedding_results = apply_enhanced_embedding_approach(
                model,
                train_data,
                val_data,
                test_data,
                device,
                args,
                top_n=args.similar_samples,
                ffn_epochs=args.embedding_ffn_epochs,
                ffn_lr=args.embedding_ffn_lr
            )

            # Extract results
            embedding_preds = embedding_results['preds']
            embedding_probs = embedding_results['probs']

            # Calculate metrics for embedding predictions
            embedding_metrics = calculate_metrics(embedding_preds, trues, bool(args.is_shorting))

            # Calculate returns for embedding predictions
            embedding_returns = calculate_returns(embedding_preds, trues, embedding_probs,
                                                  bool(args.is_shorting), actual_changes)

        # Create combined ensemble if all methods are available
        ensemble_preds = None
        ensemble_probs = None
        ensemble_metrics = None
        ensemble_returns = None

        if ffn_preds is not None and embedding_preds is not None:
            print("\nCreating combined ensemble of all methods...")
            # Create ensemble predictions using majority voting
            ensemble_preds = np.zeros_like(preds)
            ensemble_probs = np.zeros_like(probs)

            for i in range(len(preds)):
                # For each sample, collect votes from all methods
                votes = [preds[i], ffn_preds[i], embedding_preds[i]]
                vote_count = np.bincount(np.array(votes).astype(int))

                # Use majority vote, or original prediction in case of tie
                if len(vote_count) > 1:
                    if vote_count[0] > vote_count[1]:
                        ensemble_preds[i] = 0
                    elif vote_count[1] > vote_count[0]:
                        ensemble_preds[i] = 1
                    else:
                        # Tie - use original prediction
                        ensemble_preds[i] = preds[i]
                else:
                    # Only one value was voted for
                    ensemble_preds[i] = vote_count.argmax()

                # Average probabilities
                ensemble_probs[i] = np.mean([probs[i], ffn_probs[i], embedding_probs[i]])

            # Calculate metrics for ensemble
            ensemble_metrics = calculate_metrics(ensemble_preds, trues, bool(args.is_shorting))

            # Calculate returns for ensemble
            ensemble_returns = calculate_returns(ensemble_preds, trues, ensemble_probs, bool(args.is_shorting),
                                                 actual_changes)

        # Print comprehensive comparison of all methods
        print("\n----- COMPREHENSIVE METHOD COMPARISON -----")
        print(
            f"{'Method':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Win Rate':<10} | {'Total Return':<15}")
        print("-" * 75)

        # Original model
        print(f"{'Original Model':<20} | {metrics['accuracy'] * 100:>8.2f}% | "
              f"{metrics['precision'] * 100:>8.2f}% | "
              f"{metrics['trading']['win_rate'] * 100:>8.2f}% | "
              f"{original_returns['total_return'] * 100:>13.2f}%")

        # FFN model if available
        if ffn_metrics is not None:
            print(f"{'FFN Model':<20} | {ffn_metrics['accuracy'] * 100:>8.2f}% | "
                  f"{ffn_metrics['precision'] * 100:>8.2f}% | "
                  f"{ffn_metrics['trading']['win_rate'] * 100:>8.2f}% | "
                  f"{ffn_returns['total_return'] * 100:>13.2f}%")

        # Embedding model if available
        if embedding_metrics is not None:
            print(f"{'Embedding Model':<20} | {embedding_metrics['accuracy'] * 100:>8.2f}% | "
                  f"{embedding_metrics['precision'] * 100:>8.2f}% | "
                  f"{embedding_metrics['trading']['win_rate'] * 100:>8.2f}% | "
                  f"{embedding_returns['total_return'] * 100:>13.2f}%")

        # Ensemble if available
        if ensemble_metrics is not None:
            print(f"{'Ensemble':<20} | {ensemble_metrics['accuracy'] * 100:>8.2f}% | "
                  f"{ensemble_metrics['precision'] * 100:>8.2f}% | "
                  f"{ensemble_metrics['trading']['win_rate'] * 100:>8.2f}% | "
                  f"{ensemble_returns['total_return'] * 100:>13.2f}%")

        # Print agreement analysis if multiple methods are available
        if ffn_preds is not None or embedding_preds is not None:
            print("\n----- AGREEMENT ANALYSIS -----")

            # Calculate agreement between methods
            if ffn_preds is not None and embedding_preds is not None:
                # Agreement between all three methods
                all_agree = np.sum((preds == ffn_preds) & (preds == embedding_preds))
                all_agree_pct = all_agree / len(preds) * 100

                # Correct predictions when all agree
                correct_when_agree = np.sum(
                    ((preds == ffn_preds) & (preds == embedding_preds)) & (preds == trues))
                accuracy_when_agree = correct_when_agree / all_agree if all_agree > 0 else 0

                print(f"All three methods agree on {all_agree} samples ({all_agree_pct:.2f}%)")
                print(f"Accuracy when all agree: {accuracy_when_agree * 100:.2f}%")

                # Pairwise agreements with accuracy
                orig_ffn_agree_mask = (preds == ffn_preds)
                orig_ffn_agree = np.sum(orig_ffn_agree_mask)
                orig_ffn_agree_correct = np.sum(orig_ffn_agree_mask & (preds == trues))
                orig_ffn_agree_accuracy = orig_ffn_agree_correct / orig_ffn_agree if orig_ffn_agree > 0 else 0

                orig_emb_agree_mask = (preds == embedding_preds)
                orig_emb_agree = np.sum(orig_emb_agree_mask)
                orig_emb_agree_correct = np.sum(orig_emb_agree_mask & (preds == trues))
                orig_emb_agree_accuracy = orig_emb_agree_correct / orig_emb_agree if orig_emb_agree > 0 else 0

                ffn_emb_agree_mask = (ffn_preds == embedding_preds)
                ffn_emb_agree = np.sum(ffn_emb_agree_mask)
                ffn_emb_agree_correct = np.sum(ffn_emb_agree_mask & (ffn_preds == trues))
                ffn_emb_agree_accuracy = ffn_emb_agree_correct / ffn_emb_agree if ffn_emb_agree > 0 else 0

                print(
                    f"Original and FFN agree: {orig_ffn_agree / len(preds) * 100:.2f}% (Accuracy: {orig_ffn_agree_accuracy * 100:.2f}%)")
                print(
                    f"Original and Embedding agree: {orig_emb_agree / len(preds) * 100:.2f}% (Accuracy: {orig_emb_agree_accuracy * 100:.2f}%)")
                print(
                    f"FFN and Embedding agree: {ffn_emb_agree / len(preds) * 100:.2f}% (Accuracy: {ffn_emb_agree_accuracy * 100:.2f}%)")

                # When exactly two methods agree (but not all three)
                print("\nWhen exactly two methods agree (not all three):")

                # Original and FFN agree but not Embedding
                orig_ffn_only_mask = orig_ffn_agree_mask & ~(preds == embedding_preds)
                orig_ffn_only_count = np.sum(orig_ffn_only_mask)
                orig_ffn_only_correct = np.sum(orig_ffn_only_mask & (preds == trues))
                orig_ffn_only_accuracy = orig_ffn_only_correct / orig_ffn_only_count if orig_ffn_only_count > 0 else 0

                # Original and Embedding agree but not FFN
                orig_emb_only_mask = orig_emb_agree_mask & ~(preds == ffn_preds)
                orig_emb_only_count = np.sum(orig_emb_only_mask)
                orig_emb_only_correct = np.sum(orig_emb_only_mask & (preds == trues))
                orig_emb_only_accuracy = orig_emb_only_correct / orig_emb_only_count if orig_emb_only_count > 0 else 0

                # FFN and Embedding agree but not Original
                ffn_emb_only_mask = ffn_emb_agree_mask & ~(ffn_preds == preds)
                ffn_emb_only_count = np.sum(ffn_emb_only_mask)
                ffn_emb_only_correct = np.sum(ffn_emb_only_mask & (ffn_preds == trues))
                ffn_emb_only_accuracy = ffn_emb_only_correct / ffn_emb_only_count if ffn_emb_only_count > 0 else 0

                print(
                    f"Original and FFN only: {orig_ffn_only_count} samples ({orig_ffn_only_count / len(preds) * 100:.2f}%) | Accuracy: {orig_ffn_only_accuracy * 100:.2f}%")
                print(
                    f"Original and Embedding only: {orig_emb_only_count} samples ({orig_emb_only_count / len(preds) * 100:.2f}%) | Accuracy: {orig_emb_only_accuracy * 100:.2f}%")
                print(
                    f"FFN and Embedding only: {ffn_emb_only_count} samples ({ffn_emb_only_count / len(preds) * 100:.2f}%) | Accuracy: {ffn_emb_only_accuracy * 100:.2f}%")

                # When all three methods disagree
                all_disagree_mask = ~(
                            (preds == ffn_preds) | (preds == embedding_preds) | (ffn_preds == embedding_preds))
                all_disagree_count = np.sum(all_disagree_mask)

                if all_disagree_count > 0:
                    orig_disagree_correct = np.sum(all_disagree_mask & (preds == trues))
                    ffn_disagree_correct = np.sum(all_disagree_mask & (ffn_preds == trues))
                    emb_disagree_correct = np.sum(all_disagree_mask & (embedding_preds == trues))

                    orig_disagree_accuracy = orig_disagree_correct / all_disagree_count
                    ffn_disagree_accuracy = ffn_disagree_correct / all_disagree_count
                    emb_disagree_accuracy = emb_disagree_correct / all_disagree_count

                    print(
                        f"\nWhen all three methods disagree: {all_disagree_count} samples ({all_disagree_count / len(preds) * 100:.2f}%)")
                    print(f"Original accuracy: {orig_disagree_accuracy * 100:.2f}%")
                    print(f"FFN accuracy: {ffn_disagree_accuracy * 100:.2f}%")
                    print(f"Embedding accuracy: {emb_disagree_accuracy * 100:.2f}%")

            elif ffn_preds is not None:
                # Agreement between original and FFN
                agree = np.sum(preds == ffn_preds)
                agree_pct = agree / len(preds) * 100

                # Correct predictions when agree
                correct_when_agree = np.sum((preds == ffn_preds) & (preds == trues))
                accuracy_when_agree = correct_when_agree / agree if agree > 0 else 0

                print(f"Original and FFN agree on {agree} samples ({agree_pct:.2f}%)")
                print(f"Accuracy when both agree: {accuracy_when_agree * 100:.2f}%")

            elif embedding_preds is not None:
                # Agreement between original and embedding
                agree = np.sum(preds == embedding_preds)
                agree_pct = agree / len(preds) * 100

                # Correct predictions when agree
                correct_when_agree = np.sum((preds == embedding_preds) & (preds == trues))
                accuracy_when_agree = correct_when_agree / agree if agree > 0 else 0

                print(f"Original and Embedding agree on {agree} samples ({agree_pct:.2f}%)")
                print(f"Accuracy when both agree: {accuracy_when_agree * 100:.2f}%")

        # Print agreement analysis if multiple methods are available
        # if ffn_preds is not None or embedding_preds is not None:
        #     print("\n----- AGREEMENT ANALYSIS -----")
        #
        #     # Calculate agreement between methods
        #     if ffn_preds is not None and embedding_preds is not None:
        #         # Agreement between all three methods
        #         all_agree = np.sum((preds == ffn_preds) & (preds == embedding_preds))
        #         all_agree_pct = all_agree / len(preds) * 100
        #
        #         # Correct predictions when all agree
        #         correct_when_agree = np.sum(
        #             ((preds == ffn_preds) & (preds == embedding_preds)) & (preds == trues))
        #         accuracy_when_agree = correct_when_agree / all_agree if all_agree > 0 else 0
        #
        #         print(f"All three methods agree on {all_agree} samples ({all_agree_pct:.2f}%)")
        #         print(f"Accuracy when all agree: {accuracy_when_agree * 100:.2f}%")
        #
        #         # Pairwise agreements
        #         orig_ffn_agree = np.sum(preds == ffn_preds)
        #         orig_emb_agree = np.sum(preds == embedding_preds)
        #         ffn_emb_agree = np.sum(ffn_preds == embedding_preds)
        #
        #         print(f"Original and FFN agree: {orig_ffn_agree / len(preds) * 100:.2f}%")
        #         print(f"Original and Embedding agree: {orig_emb_agree / len(preds) * 100:.2f}%")
        #         print(f"FFN and Embedding agree: {ffn_emb_agree / len(preds) * 100:.2f}%")
        #
        #     elif ffn_preds is not None:
        #         # Agreement between original and FFN
        #         agree = np.sum(preds == ffn_preds)
        #         agree_pct = agree / len(preds) * 100
        #
        #         # Correct predictions when agree
        #         correct_when_agree = np.sum((preds == ffn_preds) & (preds == trues))
        #         accuracy_when_agree = correct_when_agree / agree if agree > 0 else 0
        #
        #         print(f"Original and FFN agree on {agree} samples ({agree_pct:.2f}%)")
        #         print(f"Accuracy when both agree: {accuracy_when_agree * 100:.2f}%")
        #
        #     elif embedding_preds is not None:
        #         # Agreement between original and embedding
        #         agree = np.sum(preds == embedding_preds)
        #         agree_pct = agree / len(preds) * 100
        #
        #         # Correct predictions when agree
        #         correct_when_agree = np.sum((preds == embedding_preds) & (preds == trues))
        #         accuracy_when_agree = correct_when_agree / agree if agree > 0 else 0
        #
        #         print(f"Original and Embedding agree on {agree} samples ({agree_pct:.2f}%)")
        #         print(f"Accuracy when both agree: {accuracy_when_agree * 100:.2f}%")

        # Print detailed analysis of samples
        print("\n----- SAMPLE ANALYSIS -----")

        # Sample the first few cases where methods disagree
        disagreement_indices = []
        if ffn_preds is not None and embedding_preds is not None:
            disagreement_indices = np.where((preds != ffn_preds) | (preds != embedding_preds))[0]
        elif ffn_preds is not None:
            disagreement_indices = np.where(preds != ffn_preds)[0]
        elif embedding_preds is not None:
            disagreement_indices = np.where(preds != embedding_preds)[0]

        if len(disagreement_indices) > 0:
            # Show the first few disagreement cases
            print("Samples where methods disagree:")
            print(
                f"{'Sample':<7} | {'Timestamp':<20} | {'Original':<10} | {'FFN':<10} | {'Embedding':<10} | {'True':<6} | {'Correct'}")
            print("-" * 85)

            for i in disagreement_indices[:min(10, len(disagreement_indices))]:
                # Format timestamp
                ts_str = "N/A"
                if timestamps[i] is not None:
                    if isinstance(timestamps[i], pd.Timestamp):
                        ts_str = timestamps[i].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        ts_str = str(timestamps[i])

                # Format predictions
                orig_pred = f"{preds[i]:.0f} ({probs[i]:.4f})"
                ffn_pred = f"{ffn_preds[i]:.0f} ({ffn_probs[i]:.4f})" if ffn_preds is not None else "N/A"
                emb_pred = f"{embedding_preds[i]:.0f} ({embedding_probs[i]:.4f})" if embedding_preds is not None else "N/A"

                # Determine which method was correct
                correct = []
                if preds[i] == trues[i]:
                    correct.append("Original")
                if ffn_preds is not None and ffn_preds[i] == trues[i]:
                    correct.append("FFN")
                if embedding_preds is not None and embedding_preds[i] == trues[i]:
                    correct.append("Embedding")

                correct_str = ", ".join(correct) if correct else "None"

                print(
                    f"{i:<7} | {ts_str:<20} | {orig_pred:<10} | {ffn_pred:<10} | {emb_pred:<10} | {trues[i]:<6.0f} | {correct_str}")

        # Save all results to CSV and JSON files
        print("\nSaving comprehensive results...")
        results_data = {
            'preds': preds,
            'trues': trues,
            'probs': probs,
            'timestamps': timestamps,
            'prices': prices,
            'metrics': metrics,
            'returns': original_returns,
            'ffn_preds': ffn_preds,
            'ffn_probs': ffn_probs,
            'ffn_metrics': ffn_metrics,
            'ffn_returns': ffn_returns,
            'embedding_preds': embedding_preds,
            'embedding_probs': embedding_probs,
            'embedding_metrics': embedding_metrics,
            'embedding_returns': embedding_returns,
            'ensemble_preds': ensemble_preds,
            'ensemble_probs': ensemble_probs,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_returns': ensemble_returns
        }

        # Save results to files
        save_comprehensive_results(results_data, args)

        print("\nDone!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def save_combined_results(results_data):
    """
    Save results from all approaches to CSV files

    Parameters:
    -----------
    results_data : dict
        Dictionary containing all results data
    """
    args = results_data['args']
    os.makedirs(args.output_path, exist_ok=True)

    # Process timestamps
    human_timestamps = []
    for ts in results_data['timestamps']:
        if isinstance(ts, pd.Timestamp):
            human_ts = ts.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(ts, (int, float)) and ts > 1000000000:
            human_ts = pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(ts, str):
            try:
                human_ts = pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')
            except:
                human_ts = ts
        else:
            human_ts = "Unknown" if ts is not None else None
        human_timestamps.append(human_ts)

    # Create base dataframe
    detailed_df = pd.DataFrame({
        'sample': range(len(results_data['preds'])),
        'timestamp': human_timestamps,
        'price': results_data['prices'],
        'original_prediction': results_data['preds'],
        'true_label': results_data['trues'],
        'original_probability': results_data['probs'],
        'actual_change_pct': [x * 100 for x in results_data['actual_changes']],
        'original_return_pct': [x * 100 for x in results_data['returns']['strategies']['combined']['returns']],
        'original_cum_return_pct': [x * 100 for x in results_data['returns']['strategies']['combined']['cumulative']]
    })

    # Add FFN model results if available
    if results_data['ffn_preds'] is not None:
        detailed_df['ffn_prediction'] = results_data['ffn_preds']
        detailed_df['ffn_probability'] = results_data['ffn_probs']
        detailed_df['original_correct'] = detailed_df['original_prediction'] == detailed_df['true_label']
        detailed_df['ffn_correct'] = detailed_df['ffn_prediction'] == detailed_df['true_label']
        detailed_df['models_agree'] = detailed_df['original_prediction'] == detailed_df['ffn_prediction']

    # Add conditional approach results if available
    if 'conditional_results' in results_data:
        detailed_df['conditional_prediction'] = results_data['conditional_results']['preds']
        detailed_df['conditional_return_pct'] = [x * 100 for x in results_data['conditional_results']['returns']]
        detailed_df['conditional_cum_return_pct'] = [x * 100 for x in results_data['conditional_results']['cumulative']]
        detailed_df['conditional_correct'] = detailed_df['conditional_prediction'] == detailed_df['true_label']

    # Add ensemble approach results if available
    if 'ensemble_results' in results_data:
        detailed_df['ensemble_prediction'] = results_data['ensemble_results']['preds']
        detailed_df['ensemble_probability'] = results_data['ensemble_results']['probs']
        detailed_df['ensemble_return_pct'] = [x * 100 for x in results_data['ensemble_results']['returns']]
        detailed_df['ensemble_cum_return_pct'] = [x * 100 for x in results_data['ensemble_results']['cumulative']]
        detailed_df['ensemble_correct'] = detailed_df['ensemble_prediction'] == detailed_df['true_label']

    # Save detailed results
    results_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_combined_results.csv")
    detailed_df.to_csv(results_file, index=False)
    print(f"Combined results saved to {results_file}")

    # Save metrics summary to JSON
    import json
    metrics_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_combined_metrics.json")

    # Helper function to convert NumPy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj

    # Create metrics data with conversion to JSON-serializable types
    metrics_data = {
        'model': args.model,
        'model_id': args.model_id,
        'data_path': args.data_path,
        'original_metrics': convert_numpy_types(results_data['metrics']),
        'original_returns': {
            'total_return': float(results_data['returns']['total_return'])
        }
    }

    # Add FFN metrics if available
    if results_data['ffn_metrics'] is not None:
        metrics_data['ffn_metrics'] = convert_numpy_types(results_data['ffn_metrics'])
        if 'ffn_returns' in results_data:
            metrics_data['ffn_returns'] = {
                'total_return': float(results_data['ffn_returns']['total_return'])
            }

    # Add conditional approach metrics if available
    if 'conditional_results' in results_data:
        metrics_data['conditional_metrics'] = convert_numpy_types(results_data['conditional_results']['metrics'])
        metrics_data['conditional_returns'] = {
            'total_return': float(results_data['conditional_results']['total_return'])
        }

    # Add ensemble approach metrics if available
    if 'ensemble_results' in results_data:
        metrics_data['ensemble_metrics'] = convert_numpy_types(results_data['ensemble_results']['metrics'])
        metrics_data['ensemble_returns'] = {
            'total_return': float(results_data['ensemble_results']['total_return'])
        }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Combined metrics saved to {metrics_file}")

    return results_file, metrics_file


def save_comprehensive_results(results_data, args):
    """
    Save comprehensive results from all methods to files

    Parameters:
    -----------
    results_data : dict
        Dictionary containing all results data
    args : argparse.Namespace
        Command line arguments
    """
    os.makedirs(args.output_path, exist_ok=True)

    # Create timestamp strings
    human_timestamps = []
    for ts in results_data['timestamps']:
        if isinstance(ts, pd.Timestamp):
            human_ts = ts.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(ts, (int, float)) and ts > 1000000000:
            human_ts = pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(ts, str):
            try:
                human_ts = pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')
            except:
                human_ts = ts
        else:
            human_ts = "Unknown" if ts is not None else None
        human_timestamps.append(human_ts)

    # Create pandas DataFrame for detailed results
    df_dict = {
        'sample': range(len(results_data['preds'])),
        'timestamp': human_timestamps,
        'price': results_data['prices'],
        'original_prediction': results_data['preds'],
        'original_probability': results_data['probs'],
        'true_label': results_data['trues']
    }

    # Add FFN results if available
    if results_data['ffn_preds'] is not None:
        df_dict['ffn_prediction'] = results_data['ffn_preds']
        df_dict['ffn_probability'] = results_data['ffn_probs']

    # Add embedding results if available
    if results_data['embedding_preds'] is not None:
        df_dict['embedding_prediction'] = results_data['embedding_preds']
        df_dict['embedding_probability'] = results_data['embedding_probs']

    # Add ensemble results if available
    if results_data['ensemble_preds'] is not None:
        df_dict['ensemble_prediction'] = results_data['ensemble_preds']
        df_dict['ensemble_probability'] = results_data['ensemble_probs']

    # Add accuracy columns
    df_dict['original_correct'] = results_data['preds'] == results_data['trues']

    if results_data['ffn_preds'] is not None:
        df_dict['ffn_correct'] = results_data['ffn_preds'] == results_data['trues']

    if results_data['embedding_preds'] is not None:
        df_dict['embedding_correct'] = results_data['embedding_preds'] == results_data['trues']

    if results_data['ensemble_preds'] is not None:
        df_dict['ensemble_correct'] = results_data['ensemble_preds'] == results_data['trues']

    # Create DataFrame
    detailed_df = pd.DataFrame(df_dict)

    # Save to CSV
    csv_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_comprehensive_results.csv")
    detailed_df.to_csv(csv_file, index=False)
    print(f"Comprehensive results saved to {csv_file}")

    # Convert metrics to JSON-serializable format
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj

    # Create metrics summary
    metrics_summary = {
        'model_id': args.model_id,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(results_data['preds']),
        'metrics': {
            'original': convert_numpy_types(results_data['metrics']),
            'original_returns': {
                'total_return': float(results_data['returns']['total_return'])
            }
        }
    }

    # Add FFN metrics if available
    if results_data['ffn_metrics'] is not None:
        metrics_summary['metrics']['ffn'] = convert_numpy_types(results_data['ffn_metrics'])
        metrics_summary['metrics']['ffn_returns'] = {
            'total_return': float(results_data['ffn_returns']['total_return'])
        }

    # Add embedding metrics if available
    if results_data['embedding_metrics'] is not None:
        metrics_summary['metrics']['embedding'] = convert_numpy_types(results_data['embedding_metrics'])
        metrics_summary['metrics']['embedding_returns'] = {
            'total_return': float(results_data['embedding_returns']['total_return'])
        }

    # Add ensemble metrics if available
    if results_data['ensemble_metrics'] is not None:
        metrics_summary['metrics']['ensemble'] = convert_numpy_types(results_data['ensemble_metrics'])
        metrics_summary['metrics']['ensemble_returns'] = {
            'total_return': float(results_data['ensemble_returns']['total_return'])
        }

    # Save metrics to JSON
    import json
    json_file = os.path.join(args.output_path, f"{args.model}_{args.model_id}_comprehensive_metrics.json")
    with open(json_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Comprehensive metrics saved to {json_file}")

    return csv_file, json_file


# Make sure this is at the end of your script
if __name__ == "__main__":
    main()
