import os
import argparse
import re

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import optuna
from optuna.samplers import TPESampler
import joblib
import json
from typing import Dict, List, Tuple, Any, Optional
import time
import random
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for the feature selection workflow"""
    parser = argparse.ArgumentParser(description='Optimized Feature Selection for Cryptocurrency Forecasting')

    # Basic arguments
    parser.add_argument('--root_path', type=str, default='./dataset/logits/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='btcusdt_12h_4h_april_15.csv',
                         help='data file')
    #parser.add_argument('--data_path', type=str, default='btcusdt_12h_4h_complete_april_15.csv',
    #                    help='data file')
    parser.add_argument('--target', type=str, default='price_direction',
                        help='target column for binary prediction')
    parser.add_argument('--price_col', type=str, default='close',
                        help='price column name for generating direction target')
    parser.add_argument('--output_dir', type=str, default='./optimized_features_light/',
                        help='directory to save results')

    # Data processing arguments
    parser.add_argument('--seq_len', type=int, default=96,
                        help='sequence length (lookback window) for prediction')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='prediction length')
    parser.add_argument('--exclude_cols', type=str,
                        default='date,timestamp,unix_timestamp,open_time,close_time,12h_unix_timestamp,4h_batch1_unix_timestamp,4h_batch2_unix_timestamp,4h_batch3_unix_timestamp',
                        help='comma-separated columns to exclude from feature selection')
    # parser.add_argument('--train_ratio', type=float, default=0.88,
    #                     help='ratio of data to use for training')
    # parser.add_argument('--val_ratio', type=float, default=0.07,
    #                     help='ratio of data to use for validation')
    parser.add_argument('--train_ratio', type=float, default=0.90,
                        help='ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='ratio of data to use for validation')

    # Optimization arguments
    # parser.add_argument('--n_trials', type=int, default=100,
    #                    help='number of hyperparameter optimization trials')
    parser.add_argument('--n_trials', type=int, default=150,
                        help='number of hyperparameter optimization trials')
    parser.add_argument('--cv_splits', type=int, default=3,
                        help='number of CV splits for time series evaluation')
    parser.add_argument('--optimize_metric', type=str, default='accuracy',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'win_rate', 'profit_factor'],
                        help='metric to optimize during hyperparameter tuning')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='timeout for hyperparameter optimization in seconds')

    # Feature selection arguments
    # parser.add_argument('--top_n_features', type=int, default=150,
    #                     help='number of top features to select')
    parser.add_argument('--top_n_features', type=int, default=150,
                        help='number of top features to select')
    parser.add_argument('--use_pca', action='store_true',
                        help='whether to apply PCA to the top features')
    parser.add_argument('--pca_components', type=int, default=44,
                        help='number of PCA components to extract')

    # GPU/Performance arguments
    parser.add_argument('--use_gpu', default=False,
                        help='use GPU for LightGBM training if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def sanitize_feature_names(X):
    """Clean feature names to make them compatible with LightGBM"""
    if isinstance(X, pd.DataFrame):
        # Replace problematic characters in column names
        X = X.copy()
        X.columns = [re.sub(r'[^\w_]+', '_', col) for col in X.columns]
    return X

def find_problematic_features(feature_names):
    """Find feature names with special JSON characters"""
    import re
    problematic = []
    for name in feature_names:
        if re.search(r'[^\w_]+', name):
            problematic.append(name)
    return problematic


def ensure_unique_feature_names(X):
    """Ensure all feature names are unique by adding a suffix to duplicates"""
    if isinstance(X, pd.DataFrame):
        # Get column names
        cols = X.columns.tolist()

        # Check for duplicates
        seen = {}
        duplicates = []
        for col in cols:
            if col in seen:
                duplicates.append(col)
            else:
                seen[col] = 1

        # If duplicates found, rename them with a suffix
        if duplicates:
            print(f"Found {len(duplicates)} duplicate column names")
            for dup in duplicates:
                # Find all occurrences of this column name
                indices = [i for i, col in enumerate(cols) if col == dup]

                # Rename all but the first occurrence
                for i, idx in enumerate(indices[1:], 1):
                    new_name = f"{dup}_{i}"
                    # Make sure the new name doesn't already exist
                    while new_name in cols:
                        i += 1
                        new_name = f"{dup}_{i}"

                    cols[idx] = new_name
                    print(f"  Renamed duplicate column '{dup}' to '{new_name}'")

            # Update DataFrame column names
            X = X.copy()
            X.columns = cols

    return X


def generate_target_column(df, price_col, seq_len, pred_len, target_col='price_direction'):
    """
    Generate binary labels for future price direction prediction.

    Args:
        df: DataFrame containing the price data
        price_col: Column name for the price data (e.g., 'close')
        seq_len: Sequence length (lookback window) - 96 for our case
        pred_len: Prediction length - 1 for our case
        target_col: Name of the target column to create

    Returns:
        DataFrame with added target column containing binary labels
    """
    logger.info(f"Generating target column '{target_col}' based on future price direction changes")

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataset!")

    # Get the price values
    close_prices = df[price_col].values

    # Initialize binary labels array
    binary_labels = np.zeros(len(close_prices))

    # For each potential sequence starting point
    for i in range(len(close_prices) - seq_len - pred_len):
        # Calculate prediction point index (end of sequence)
        pred_idx = i + seq_len

        # Get prices at prediction point and future point
        pred_price = close_prices[pred_idx]
        future_price = close_prices[pred_idx + pred_len]

        # Set label based on future price direction from prediction point
        binary_labels[i] = 1.0 if future_price > pred_price else 0.0

    # Add binary labels to the DataFrame
    df_with_target = df.copy()
    df_with_target[target_col] = binary_labels

    # Calculate some statistics for logging
    valid_indices = range(len(close_prices) - seq_len - pred_len)
    valid_labels = binary_labels[valid_indices]
    up_ratio = np.mean(valid_labels) if len(valid_labels) > 0 else 0

    logger.info(f"Generated target column '{target_col}'")
    logger.info(f"  - Valid samples: {len(valid_labels)} (after accounting for sequence windows)")
    logger.info(f"  - Upward movement ratio: {up_ratio:.4f} ({up_ratio * 100:.2f}%)")
    logger.info(f"  - Downward movement ratio: {(1 - up_ratio):.4f} ({(1 - up_ratio) * 100:.2f}%)")

    return df_with_target


def load_data(args):
    """Load the dataset and perform basic preprocessing"""
    print(f"Loading data from {os.path.join(args.root_path, args.data_path)}")
    df = pd.read_csv(os.path.join(args.root_path, args.data_path))
    logger.info(f"Loaded data shape: {df.shape}")

    # Basic data stats
    logger.info(f"Dataset columns: {len(df.columns)}")

    # Check if we need to generate the target column
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in dataset, generating it...")
        if args.price_col not in df.columns:
            raise ValueError(f"'{args.price_col}' column required for generating direction target, but not found!")
        df = generate_target_column(df, args.price_col, args.seq_len, args.pred_len, target_col=args.target)
    else:
        target_counts = df[args.target].value_counts(normalize=True)
        logger.info(f"Target distribution:\n{target_counts}")

    return df


def prepare_data_splits(df, args):
    """Prepare data splits for training, validation, and testing"""
    # Create time-based splits
    total_rows = len(df)
    train_end = int(total_rows * args.train_ratio)
    val_end = train_end + int(total_rows * args.val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"Train data: {train_df.shape}")
    logger.info(f"Validation data: {val_df.shape}")
    logger.info(f"Test data: {test_df.shape}")

    # Add split column for later use
    df['split'] = 'test'
    df.loc[:train_end, 'split'] = 'train'
    df.loc[train_end:val_end, 'split'] = 'val'

    return train_df, val_df, test_df, df


# def prepare_features_targets(df, args, exclude_cols=None):
#     """Prepare feature matrix and target vector"""
#     if exclude_cols is None:
#         exclude_cols = args.exclude_cols.split(',')
#
#     # Make sure we add the target to exclude columns
#     if args.target not in exclude_cols:
#         exclude_cols.append(args.target)
#
#     # Add 'split' to exclude columns if present
#     if 'split' in df.columns and 'split' not in exclude_cols:
#         exclude_cols.append('split')
#
#     # Get features and target
#     feature_cols = [col for col in df.columns if col not in exclude_cols]
#     X = df[feature_cols]
#     y = df[args.target].values
#
#     # Handle missing values using the non-deprecated methods
#     X = X.ffill().bfill().fillna(0)
#
#     logger.info(f"Prepared features shape: {X.shape}")
#     logger.info(f"Prepared target shape: {y.shape}")
#
#     return X, y, feature_cols

def prepare_features_targets(df, args, exclude_cols=None):
    """Prepare feature matrix and target vector"""
    if exclude_cols is None:
        exclude_cols = args.exclude_cols.split(',')

    # Make sure we add the target to exclude columns
    if args.target not in exclude_cols:
        exclude_cols.append(args.target)

    # Add 'split' to exclude columns if present
    if 'split' in df.columns and 'split' not in exclude_cols:
        exclude_cols.append('split')

    # Get features and target
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df[args.target].values

    # Handle missing values using the non-deprecated methods
    X = X.ffill().bfill().fillna(0)

    # First sanitize feature names
    X = sanitize_feature_names(X)

    # Then ensure uniqueness after sanitization
    X = ensure_unique_feature_names(X)

    # Update feature_cols to match the final column names
    feature_cols = X.columns.tolist()

    print(f"Prepared features shape: {X.shape}")
    print(f"Prepared target shape: {y.shape}")

    return X, y, feature_cols


def objective(trial, train_X, train_y, val_X, val_y, args, use_gpu=False):
    """Objective function for Optuna hyperparameter optimization"""
    # Define hyperparameter search space
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 10, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    # Add GPU-specific parameters if using GPU
    if use_gpu:
        param['device'] = 'gpu'
        param['gpu_platform_id'] = 0
        param['gpu_device_id'] = 0

    # Add boosting-type specific parameters
    if param['boosting_type'] == 'goss':
        param['top_rate'] = trial.suggest_float('top_rate', 0.1, 0.5)
        param['other_rate'] = trial.suggest_float('other_rate', 0.1, 0.5)
        # Remove subsample as it's not used with GOSS
        param.pop('subsample')

    if param['boosting_type'] == 'dart':
        param['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.5)
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.1, 0.5)

    # Create callbacks based on boosting type
    callbacks = []
    # Early stopping is not available in dart mode
    if param['boosting_type'] != 'dart':
        callbacks.append(lgb.early_stopping(50, verbose=False))
    callbacks.append(lgb.log_evaluation(0))  # 0 means silent

    # Get n_estimators and remove from params
    n_estimators = param.pop('n_estimators')

    # Train the model
    model = lgb.LGBMClassifier(**param, n_estimators=n_estimators)
    model.fit(
        train_X, train_y,
        eval_set=[(val_X, val_y)],
        eval_metric='binary_logloss',
        callbacks=callbacks
    )

    # Make predictions
    y_pred_proba = model.predict_proba(val_X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(val_y, y_pred)
    prec = precision_score(val_y, y_pred, zero_division=0)
    rec = recall_score(val_y, y_pred, zero_division=0)
    f1 = f1_score(val_y, y_pred, zero_division=0)

    # Calculate trading metrics
    cm = confusion_matrix(val_y, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    win_rate = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    profit_factor = (TP + TN) / (FP + FN) if (FP + FN) > 0 else float('inf')

    # Store all metrics in trial user attributes
    trial.set_user_attr('accuracy', float(acc))
    trial.set_user_attr('precision', float(prec))
    trial.set_user_attr('recall', float(rec))
    trial.set_user_attr('f1', float(f1))
    trial.set_user_attr('win_rate', float(win_rate))
    trial.set_user_attr('profit_factor', float(profit_factor))

    # Store best_iteration if available (not for dart mode)
    if hasattr(model, 'best_iteration_'):
        trial.set_user_attr('best_iteration', model.best_iteration_)
    else:
        trial.set_user_attr('best_iteration', n_estimators)

    # Return the metric to optimize
    metric_to_optimize = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

    return metric_to_optimize[args.optimize_metric]


def optimize_lightgbm(train_X, train_y, val_X, val_y, args):
    """Run hyperparameter optimization for LightGBM"""
    logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
    logger.info(f"Optimizing for metric: {args.optimize_metric}")

    start_time = time.time()

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=args.seed),
        study_name=f'lgbm_crypto_{args.optimize_metric}'
    )

    # Check if GPU should be used
    use_gpu = args.use_gpu and torch.cuda.is_available()
    print(f"Using GPU: {use_gpu}")
    if use_gpu:
        logger.info("Using GPU for LightGBM training")
    else:
        logger.info("Using CPU for LightGBM training")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_X, train_y, val_X, val_y, args, use_gpu),
        n_trials=args.n_trials,
        timeout=args.timeout
    )

    optimization_time = time.time() - start_time
    logger.info(f"Optimization completed in {optimization_time:.2f} seconds")

    # Log best trial information
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best {args.optimize_metric}: {study.best_value:.4f}")
    logger.info("Best parameters:")
    for param, value in study.best_params.items():
        logger.info(f"  {param}: {value}")

    return study


def train_final_model(best_params, train_X, train_y, val_X, val_y, args):
    """Train the final model with the best hyperparameters"""
    logger.info("Training final model with best parameters")

    # Use best params but remove type-specific ones if not needed
    final_params = best_params.copy()

    # Adjust params for specific boosting types
    if 'top_rate' in final_params and final_params['boosting_type'] != 'goss':
        final_params.pop('top_rate')
        final_params.pop('other_rate')

    if 'drop_rate' in final_params and final_params['boosting_type'] != 'dart':
        final_params.pop('drop_rate')
        final_params.pop('skip_drop')

    # Make sure subsample is included for gbdt and dart
    if 'subsample' not in final_params and final_params['boosting_type'] != 'goss':
        final_params['subsample'] = 0.8

    # Add GPU settings if needed
    if args.use_gpu and torch.cuda.is_available():
        final_params['device'] = 'gpu'
        final_params['gpu_platform_id'] = 0
        final_params['gpu_device_id'] = 0

    # Set up callbacks based on boosting type
    callbacks = []
    # Early stopping is not available in dart mode
    if final_params.get('boosting_type') != 'dart':
        callbacks.append(lgb.early_stopping(100, verbose=False))
    callbacks.append(lgb.log_evaluation(100))  # Log every 100 iterations

    # Get n_estimators and adjust as needed
    n_estimators = final_params.pop('n_estimators', 200)

    # Initialize and train model
    model = lgb.LGBMClassifier(
        **final_params,
        n_estimators=n_estimators,
        objective='binary',
        metric='binary_logloss',
        verbosity=-1
    )

    model.fit(
        np.vstack([train_X, val_X]),
        np.concatenate([train_y, val_y]),
        eval_set=[(val_X, val_y)],
        eval_metric='binary_logloss',
        callbacks=callbacks
    )

    # Log the number of iterations
    if hasattr(model, 'best_iteration_'):
        logger.info(f"Final model trained with {model.best_iteration_} iterations")
    else:
        logger.info(f"Final model trained with {n_estimators} iterations (dart mode doesn't use early stopping)")

    return model


def calculate_feature_importance(model, feature_names, args):
    """Calculate and save feature importance from the optimized LightGBM model"""
    logger.info("Calculating feature importance")

    # Get feature importance (split and gain)
    split_importance = model.feature_importances_
    gain_importance = model.booster_.feature_importance(importance_type='gain')

    # Create DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SplitImportance': split_importance,
        'GainImportance': gain_importance
    })

    # Normalize importance scores
    feature_importance['SplitImportanceNorm'] = feature_importance['SplitImportance'] / feature_importance[
        'SplitImportance'].sum()
    feature_importance['GainImportanceNorm'] = feature_importance['GainImportance'] / feature_importance[
        'GainImportance'].sum()

    # Sort by gain importance
    feature_importance = feature_importance.sort_values('GainImportance', ascending=False).reset_index(drop=True)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save feature importance to CSV
    output_file = os.path.join(args.output_dir, 'feature_importance_lightboost_full_binance.csv')
    feature_importance.to_csv(output_file, index=False)
    logger.info(f"Saved feature importance to {output_file}")

    # Save top N features list
    top_n = min(args.top_n_features, len(feature_importance))
    top_features = feature_importance.head(top_n)['Feature'].tolist()

    top_features_file = os.path.join(args.output_dir, 'top_features_lightboost_full_binance.txt')
    with open(top_features_file, 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    logger.info(f"Saved top {top_n} features to {top_features_file}")

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(feature_importance['Feature'][:top_n][::-1], feature_importance['GainImportanceNorm'][:top_n][::-1])
    plt.title(f'Top {top_n} Features by Importance (Gain)')
    plt.xlabel('Normalized Importance')
    plt.tight_layout()

    plot_file = os.path.join(args.output_dir, 'feature_importance_lightboost.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance plot to {plot_file}")

    return feature_importance, top_features


def create_processed_dataset(df, top_features, args):
    """Create a processed dataset with only the top features or PCA components"""
    logger.info("Creating processed dataset")

    # Ensure required columns are included
    essential_cols = ['close', 'direction']
    if args.target not in top_features:
        essential_cols.append(args.target)

    # Collect time and date columns
    time_cols = []
    for col in df.columns:
        if any(x in col.lower() for x in ['time', 'date', 'timestamp']):
            time_cols.append(col)

    # Combine essential, time, and top feature columns
    all_cols = list(set(essential_cols + time_cols + top_features))

    # Check if all columns exist in the dataframe
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"The following columns are missing in the dataset: {missing_cols}")
        all_cols = [col for col in all_cols if col in df.columns]

    # Create DataFrame with selected columns
    df_selected = df[all_cols].copy()
    logger.info(f"Created dataset with top features, shape: {df_selected.shape}")

    # Apply PCA if requested
    if args.use_pca:
        logger.info(f"Applying PCA to reduce dimensions from {len(top_features)} to {args.pca_components}")

        # Extract feature data
        X = df_selected[top_features].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Apply PCA
        n_components = min(args.pca_components, len(top_features))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)

        # Create PCA component names and DataFrame
        pca_cols = [f'pca_comp_{i + 1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols)

        # Add essential and time columns back
        for col in essential_cols + time_cols:
            if col in df.columns:
                pca_df[col] = df[col].values

        # Replace selected DataFrame with PCA results
        df_selected = pca_df
        logger.info(f"Created PCA dataset with shape: {df_selected.shape}")
        logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

        # Save PCA model
        joblib.dump(pca, os.path.join(args.output_dir, 'pca_model.joblib'))

        # Save PCA component details
        pca_info = pd.DataFrame({
            'Component': pca_cols,
            'ExplainedVariance': pca.explained_variance_ratio_,
            'CumulativeVariance': np.cumsum(pca.explained_variance_ratio_)
        })
        pca_info.to_csv(os.path.join(args.output_dir, 'pca_components.csv'), index=False)

        # Create feature contribution to PCA components
        pca_feature_mapping = pd.DataFrame(index=range(len(pca_cols)))
        pca_feature_mapping['component'] = pca_cols
        pca_feature_mapping['explained_variance'] = pca.explained_variance_ratio_

        # Add top 3 contributing features per component
        for i, comp in enumerate(pca_cols):
            loadings = pca.components_[i]
            top_indices = np.argsort(np.abs(loadings))[-3:]
            top_features_pca = [top_features[idx] for idx in top_indices]
            top_loadings = [loadings[idx] for idx in top_indices]

            for j in range(3):
                pca_feature_mapping.loc[i, f'top_feature_{j + 1}'] = top_features_pca[j]
                pca_feature_mapping.loc[i, f'loading_{j + 1}'] = top_loadings[j]

        pca_feature_mapping.to_csv(os.path.join(args.output_dir, 'pca_feature_mapping.csv'), index=False)

    return df_selected

def save_processed_dataset(df_processed, args):
    """Save the processed dataset for use in transformer models"""
    # Create filename based on processing method
    if args.use_pca:
        output_file = os.path.join(
            args.output_dir,
            f"{os.path.splitext(args.data_path)[0]}_pca_components_{args.pca_components}_light_gbm.csv"
        )
    else:
        output_file = os.path.join(
            args.output_dir,
            f"{os.path.splitext(args.data_path)[0]}_top_{args.top_n_features}_features_light_gbm.csv"
        )

    # Save to CSV
    df_processed.to_csv(output_file, index=False)
    logger.info(f"Saved processed dataset to {output_file}")

    return output_file


def evaluate_model(model, X, y, name="Test"):
    """Evaluate the optimized model on given data"""
    logger.info(f"Evaluating model on {name} data")

    # Make predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Calculate trading metrics
    cm = confusion_matrix(y, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    win_rate = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    profit_factor = (TP + TN) / (FP + FN) if (FP + FN) > 0 else float('inf')

    # Print evaluation results
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall: {rec:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Win Rate: {win_rate:.4f}")
    logger.info(f"  Profit Factor: {profit_factor:.4f}")
    logger.info(f"  Confusion Matrix: TP={TP}, TN={TN}, FP={FP}, FN={FN}")

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'confusion_matrix': {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN)
        }
    }


def save_execution_summary(args, metrics, feature_importance, output_file):
    """Save execution summary including configuration, metrics, and output locations"""
    summary = {
        'configuration': vars(args),
        'execution_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'feature_importance_file': os.path.join(args.output_dir, 'feature_importance_lightboost.csv'),
        'top_features_file': os.path.join(args.output_dir, 'top_features_lightboost_full_binance.txt'),
        'processed_dataset': output_file,
        'top_10_features': feature_importance.head(10)[['Feature', 'GainImportanceNorm']].to_dict('records')
    }

    # Save summary to JSON
    summary_file = os.path.join(args.output_dir, 'execution_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved execution summary to {summary_file}")


def main():
    """Main function for the optimized feature selection workflow"""
    start_time = time.time()

    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Step 1: Load and prepare data
    logger.info("STEP 1: Loading and preparing data")

    # Load data and ensure target column exists
    df = load_data(args)

    # Create data splits
    train_df, val_df, test_df, full_df_with_split = prepare_data_splits(df, args)

    # Prepare features and targets
    train_X, train_y, feature_cols = prepare_features_targets(train_df, args)

    problematic_features = find_problematic_features(feature_cols)
    if problematic_features:
        logger.warning(f"Found {len(problematic_features)} features with potentially problematic characters:")
        for i, feat in enumerate(problematic_features[:10]):
            print(f"  {i + 1}. {feat}")
        if len(problematic_features) > 10:
            print(f"  ... and {len(problematic_features) - 10} more")




    val_X, val_y, _ = prepare_features_targets(val_df, args)
    test_X, test_y, _ = prepare_features_targets(test_df, args)

    # Step 2: Optimize LightGBM on full dataset
    logger.info("STEP 2: Optimizing LightGBM on full dataset")
    study = optimize_lightgbm(train_X, train_y, val_X, val_y, args)

    # Step 3: Train final model with best parameters
    logger.info("STEP 3: Training final model with best hyperparameters")
    model = train_final_model(study.best_params, train_X, train_y, val_X, val_y, args)

    # Step 4: Evaluate model on test set
    logger.info("STEP 4: Evaluating model on test set")
    test_metrics = evaluate_model(model, test_X, test_y, name="Test")

    # Step 5: Calculate feature importance from optimized model
    logger.info("STEP 5: Calculating feature importance from optimized model")
    feature_importance, top_features = calculate_feature_importance(model, feature_cols, args)

    # Step 6: Create processed dataset with top features or PCA
    logger.info("STEP 6: Creating processed dataset")
    df_processed = create_processed_dataset(full_df_with_split, top_features, args)

    # Step 7: Save processed dataset
    logger.info("STEP 7: Saving processed dataset")
    output_file = save_processed_dataset(df_processed, args)

    # Step 8: Save model
    logger.info("STEP 8: Saving LightGBM model")
    model_file = os.path.join(args.output_dir, 'optimized_lightgbm_model.joblib')
    joblib.dump(model, model_file)

    # Save execution summary
    save_execution_summary(args, test_metrics, feature_importance, output_file)

    # Log completion
    total_time = time.time() - start_time
    logger.info(f"Completed optimized feature selection workflow in {total_time:.2f} seconds")
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info(f"Processed dataset ready for transformer model: {output_file}")


if __name__ == "__main__":
    main()