import os
import argparse
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
    parser.add_argument('--data_path', type=str, default='btcusdt_12h_4h_complete_2.csv',
                        help='data file')
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
    parser.add_argument('--train_ratio', type=float, default=0.88,
                        help='ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.07,
                        help='ratio of data to use for validation')

    # Optimization arguments
    parser.add_argument('--n_trials', type=int, default=100,
                        help='number of hyperparameter optimization trials')
    parser.add_argument('--cv_splits', type=int, default=3,
                        help='number of CV splits for time series evaluation')
    parser.add_argument('--optimize_metric', type=str, default='accuracy',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'win_rate', 'profit_factor'],
                        help='metric to optimize during hyperparameter tuning')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='timeout for hyperparameter optimization in seconds')

    # Feature selection arguments
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

    # "Freeze" arguments for maintaining feature selection and train/val splits given a reference dataset and new data
    # parser.add_argument('--reference_csv', type=str, default=None,
    #                     help='reference CSV file to maintain consistent feature selection and splits')
    parser.add_argument('--reference_csv', type=str, default='./dataset/logits/btcusd_pca_components_lightboost_12h_4h_reduced_70_7_5_1_2_1_old.csv',
                        help='reference CSV file to maintain consistent feature selection and splits')
    # parser.add_argument('--freeze_features', action='store_true',
    #                     help='use features from reference CSV instead of running feature selection')
    parser.add_argument('--freeze_features', default=True,
                        help='use features from reference CSV instead of running feature selection')
    # parser.add_argument('--freeze_train_val', action='store_true',
    #                     help='maintain same train/val data points as reference CSV')
    parser.add_argument('--freeze_train_val', default=True,
                        help='maintain same train/val data points as reference CSV')
    parser.add_argument('--timestamp_col', type=str, default='timestamp',
                        help='timestamp column to use for aligning data')
    parser.add_argument('--remove_future_train_val', action='store_true',
                        help='remove future timestamps from train/val that did not exist in reference data')
    parser.add_argument('--reference_timestamp_col', type=str, default='date',
                        help='timestamp column in the reference dataset (if different from timestamp_col)')
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


def load_reference_data(args):
    """Load the reference dataset for consistent feature selection and train/val splits"""
    if args.reference_csv is None:
        return None

    logger.info(f"Loading reference data from {args.reference_csv}")
    try:
        ref_df = pd.read_csv(args.reference_csv)
        logger.info(f"Reference data shape: {ref_df.shape}")
        logger.info(f"Reference data columns: {len(ref_df.columns)}")
        return ref_df
    except Exception as e:
        logger.error(f"Error loading reference data: {e}")
        return None


def identify_features_from_reference(ref_df, timestamp_col='timestamp', exclude_cols=None):
    """Identify the feature columns from the reference dataset"""
    if ref_df is None:
        return None

    logger.info("Identifying features from reference dataset")

    # Exclude timestamp columns and target columns
    if exclude_cols is None:
        exclude_cols = []

    # Always exclude the timestamp column and any unix_timestamp columns
    exclude_cols_expanded = list(exclude_cols)
    exclude_cols_expanded.append(timestamp_col)
    exclude_cols_expanded.extend([col for col in ref_df.columns if 'unix_timestamp' in col])

    # Also exclude split and target columns if they exist
    if 'split' in ref_df.columns:
        exclude_cols_expanded.append('split')
    if 'price_direction' in ref_df.columns:
        exclude_cols_expanded.append('price_direction')
    if 'direction' in ref_df.columns:
        exclude_cols_expanded.append('direction')

    feature_cols = [col for col in ref_df.columns if col not in exclude_cols_expanded]
    logger.info(f"Identified {len(feature_cols)} feature columns from reference dataset")

    return feature_cols


def align_reference_splits(new_df, ref_df, timestamp_col='timestamp', ref_timestamp_col=None, remove_future=False):
    """
    Align the train/val/test splits between new data and reference data
    based on timestamps.

    Args:
        new_df: New DataFrame to be split
        ref_df: Reference DataFrame with existing splits
        timestamp_col: Column containing timestamps in new_df for alignment
        ref_timestamp_col: Column containing timestamps in ref_df (if different from timestamp_col)
        remove_future: If True, remove timestamps from train/val that didn't exist in reference data

    Returns:
        DataFrame with aligned split column
    """
    if 'split' not in ref_df.columns:
        logger.warning("No 'split' column found in reference data, cannot align splits")
        return new_df

    logger.info("Aligning train/val/test splits with reference data")

    # If ref_timestamp_col not specified, use the same as timestamp_col
    if ref_timestamp_col is None:
        ref_timestamp_col = timestamp_col

    # Ensure timestamp columns exist in respective dataframes
    if timestamp_col not in new_df.columns:
        logger.warning(f"Timestamp column '{timestamp_col}' not found in new dataset, cannot align splits")
        return new_df

    if ref_timestamp_col not in ref_df.columns:
        logger.warning(f"Timestamp column '{ref_timestamp_col}' not found in reference dataset, cannot align splits")
        return new_df

    logger.info(f"Using column '{timestamp_col}' in new data and '{ref_timestamp_col}' in reference data for alignment")

    # Create split column in new DataFrame
    new_df['split'] = 'test'  # Default all to test

    # Get reference timestamps for each split
    ref_train_timestamps = set(ref_df[ref_df['split'] == 'train'][ref_timestamp_col])
    ref_val_timestamps = set(ref_df[ref_df['split'] == 'val'][ref_timestamp_col])

    # Check for type differences and try to handle them
    sample_new_ts = new_df[timestamp_col].iloc[0] if len(new_df) > 0 else None
    sample_ref_ts = ref_df[ref_timestamp_col].iloc[0] if len(ref_df) > 0 else None

    if sample_new_ts is not None and sample_ref_ts is not None:
        # Check if types differ (e.g., string vs datetime)
        if type(sample_new_ts) != type(sample_ref_ts):
            logger.warning(f"Type mismatch between timestamp columns: {type(sample_new_ts)} vs {type(sample_ref_ts)}")
            logger.warning("Attempting to convert for comparison...")

            # If either is a string, try to standardize format for comparison
            if isinstance(sample_new_ts, str) or isinstance(sample_ref_ts, str):
                # Convert both to strings with standardized format if possible
                try:
                    # Convert reference timestamps to strings for comparison
                    ref_train_timestamps = {str(ts) for ts in ref_train_timestamps}
                    ref_val_timestamps = {str(ts) for ts in ref_val_timestamps}

                    # Create a string version of the new timestamp column for comparison
                    new_df['_temp_ts_str'] = new_df[timestamp_col].astype(str)
                    timestamp_col_for_alignment = '_temp_ts_str'
                    logger.info("Using string representation of timestamps for alignment")
                except Exception as e:
                    logger.error(f"Failed to convert timestamps for comparison: {e}")
                    timestamp_col_for_alignment = timestamp_col
            else:
                timestamp_col_for_alignment = timestamp_col
        else:
            timestamp_col_for_alignment = timestamp_col
    else:
        timestamp_col_for_alignment = timestamp_col

    # Align based on timestamps
    train_mask = new_df[timestamp_col_for_alignment].isin(ref_train_timestamps)
    val_mask = new_df[timestamp_col_for_alignment].isin(ref_val_timestamps)

    new_df.loc[train_mask, 'split'] = 'train'
    new_df.loc[val_mask, 'split'] = 'val'

    # Clean up temporary column if created
    if '_temp_ts_str' in new_df.columns:
        new_df.drop('_temp_ts_str', axis=1, inplace=True)

    # Count the data points in each split
    train_count = len(new_df[new_df['split'] == 'train'])
    val_count = len(new_df[new_df['split'] == 'val'])
    test_count = len(new_df[new_df['split'] == 'test'])

    logger.info(f"Split alignment results:")
    logger.info(f"  - Train: {train_count} records")
    logger.info(f"  - Validation: {val_count} records")
    logger.info(f"  - Test: {test_count} records")

    # Check if there are future timestamps in train/val splits that weren't in reference data
    if remove_future:
        # Find any records in train/val that have timestamps not in the reference train/val sets
        future_records = new_df[(new_df['split'].isin(['train', 'val'])) &
                                (~new_df[timestamp_col_for_alignment].isin(
                                    ref_train_timestamps.union(ref_val_timestamps)))]

        if len(future_records) > 0:
            logger.warning(f"Found {len(future_records)} records in train/val with timestamps not in reference data")
            logger.warning("Moving these records to test split")

            # Move these records to test split
            new_df.loc[future_records.index, 'split'] = 'test'

            # Log updated counts
            train_count = len(new_df[new_df['split'] == 'train'])
            val_count = len(new_df[new_df['split'] == 'val'])
            test_count = len(new_df[new_df['split'] == 'test'])

            logger.info(f"Updated split counts after moving future records:")
            logger.info(f"  - Train: {train_count} records")
            logger.info(f"  - Validation: {val_count} records")
            logger.info(f"  - Test: {test_count} records")

    return new_df


def prepare_data_splits(df, args, ref_df=None):
    """Prepare data splits for training, validation, and testing"""
    # If we have a reference dataset and need to freeze train/val splits
    if ref_df is not None and args.freeze_train_val:
        logger.info("Using train/val splits from reference dataset")
        df = align_reference_splits(
            df,
            ref_df,
            timestamp_col=args.timestamp_col,
            ref_timestamp_col=args.reference_timestamp_col,
            remove_future=args.remove_future_train_val
        )

        # Extract each split
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()

    else:
        # Create time-based splits as in the original code
        total_rows = len(df)
        train_end = int(total_rows * args.train_ratio)
        val_end = train_end + int(total_rows * args.val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        # Add split column for later use
        df['split'] = 'test'
        df.loc[:train_end, 'split'] = 'train'
        df.loc[train_end:val_end, 'split'] = 'val'

    logger.info(f"Train data: {train_df.shape}")
    logger.info(f"Validation data: {val_df.shape}")
    logger.info(f"Test data: {test_df.shape}")

    return train_df, val_df, test_df, df


# def align_reference_splits(new_df, ref_df, timestamp_col='timestamp', remove_future=False):
#     """
#     Align the train/val/test splits between new data and reference data
#     based on timestamps.
#
#     Args:
#         new_df: New DataFrame to be split
#         ref_df: Reference DataFrame with existing splits
#         timestamp_col: Column containing timestamps for alignment
#         remove_future: If True, remove timestamps from train/val that didn't exist in reference data
#
#     Returns:
#         DataFrame with aligned split column
#     """
#     if 'split' not in ref_df.columns:
#         logger.warning("No 'split' column found in reference data, cannot align splits")
#         return new_df
#
#     logger.info("Aligning train/val/test splits with reference data")
#
#     # Ensure timestamp column exists in both dataframes
#     if timestamp_col not in new_df.columns or timestamp_col not in ref_df.columns:
#         logger.warning(f"Timestamp column '{timestamp_col}' not found in both datasets, cannot align splits")
#         return new_df
#
#     # Create split column in new DataFrame
#     new_df['split'] = 'test'  # Default all to test
#
#     # Get reference timestamps for each split
#     ref_train_timestamps = set(ref_df[ref_df['split'] == 'train'][timestamp_col])
#     ref_val_timestamps = set(ref_df[ref_df['split'] == 'val'][timestamp_col])
#
#     # Align based on timestamps
#     train_mask = new_df[timestamp_col].isin(ref_train_timestamps)
#     val_mask = new_df[timestamp_col].isin(ref_val_timestamps)
#
#     new_df.loc[train_mask, 'split'] = 'train'
#     new_df.loc[val_mask, 'split'] = 'val'
#
#     # Count the data points in each split
#     train_count = len(new_df[new_df['split'] == 'train'])
#     val_count = len(new_df[new_df['split'] == 'val'])
#     test_count = len(new_df[new_df['split'] == 'test'])
#
#     logger.info(f"Split alignment results:")
#     logger.info(f"  - Train: {train_count} records")
#     logger.info(f"  - Validation: {val_count} records")
#     logger.info(f"  - Test: {test_count} records")
#
#     # Check if there are future timestamps in train/val splits that weren't in reference data
#     if remove_future:
#         # Find the latest timestamp in reference data
#         ref_latest = ref_df[timestamp_col].max()
#
#         # Identify records in train/val that have timestamps beyond the reference data
#         future_records = new_df[(new_df['split'].isin(['train', 'val'])) &
#                                 (~new_df[timestamp_col].isin(ref_train_timestamps.union(ref_val_timestamps)))]
#
#         if len(future_records) > 0:
#             logger.warning(f"Found {len(future_records)} records in train/val with timestamps not in reference data")
#             logger.warning("Moving these records to test split")
#
#             # Move these records to test split
#             new_df.loc[future_records.index, 'split'] = 'test'
#
#             # Log updated counts
#             train_count = len(new_df[new_df['split'] == 'train'])
#             val_count = len(new_df[new_df['split'] == 'val'])
#             test_count = len(new_df[new_df['split'] == 'test'])
#
#             logger.info(f"Updated split counts after moving future records:")
#             logger.info(f"  - Train: {train_count} records")
#             logger.info(f"  - Validation: {val_count} records")
#             logger.info(f"  - Test: {test_count} records")
#
#     return new_df
#
#
# def prepare_data_splits(df, args, ref_df=None):
#     """Prepare data splits for training, validation, and testing"""
#     # If we have a reference dataset and need to freeze train/val splits
#     if ref_df is not None and args.freeze_train_val:
#         logger.info("Using train/val splits from reference dataset")
#         df = align_reference_splits(
#             df,
#             ref_df,
#             timestamp_col=args.timestamp_col,
#             remove_future=args.remove_future_train_val
#         )
#
#         # Extract each split
#         train_df = df[df['split'] == 'train'].copy()
#         val_df = df[df['split'] == 'val'].copy()
#         test_df = df[df['split'] == 'test'].copy()
#
#     else:
#         # Create time-based splits as in the original code
#         total_rows = len(df)
#         train_end = int(total_rows * args.train_ratio)
#         val_end = train_end + int(total_rows * args.val_ratio)
#
#         train_df = df.iloc[:train_end].copy()
#         val_df = df.iloc[train_end:val_end].copy()
#         test_df = df.iloc[val_end:].copy()
#
#         # Add split column for later use
#         df['split'] = 'test'
#         df.loc[:train_end, 'split'] = 'train'
#         df.loc[train_end:val_end, 'split'] = 'val'
#
#     logger.info(f"Train data: {train_df.shape}")
#     logger.info(f"Validation data: {val_df.shape}")
#     logger.info(f"Test data: {test_df.shape}")
#
#     return train_df, val_df, test_df, df


def prepare_features_targets(df, args, feature_cols=None, exclude_cols=None):
    """Prepare feature matrix and target vector with option to use predefined features"""
    if exclude_cols is None:
        exclude_cols = args.exclude_cols.split(',')

    # Make sure we add the target to exclude columns
    if args.target not in exclude_cols:
        exclude_cols.append(args.target)

    # Add 'split' to exclude columns if present
    if 'split' in df.columns and 'split' not in exclude_cols:
        exclude_cols.append('split')

    # Use predefined feature columns if provided, otherwise use all columns except excluded ones
    if feature_cols is not None:
        # Check if all feature_cols exist in the dataframe
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"The following columns from reference are missing in current data: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols]
        logger.info(f"Using {len(feature_cols)} predefined feature columns")
    else:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        logger.info(f"Using {len(feature_cols)} feature columns from current data")

    y = df[args.target].values

    # Handle missing values using the non-deprecated methods
    X = X.ffill().bfill().fillna(0)

    logger.info(f"Prepared features shape: {X.shape}")
    logger.info(f"Prepared target shape: {y.shape}")

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
    output_file = os.path.join(args.output_dir, 'feature_importance_lightboost.csv')
    feature_importance.to_csv(output_file, index=False)
    logger.info(f"Saved feature importance to {output_file}")

    # Save top N features list
    top_n = min(args.top_n_features, len(feature_importance))
    top_features = feature_importance.head(top_n)['Feature'].tolist()

    top_features_file = os.path.join(args.output_dir, 'top_features_lightboost_baseline.txt')
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


def save_processed_dataset(df_processed, args, suffix=None):
    """Save the processed dataset for use in transformer models"""
    # Create filename based on processing method and optional suffix
    base_name = os.path.splitext(args.data_path)[0]
    if suffix:
        base_name = f"{base_name}_{suffix}"

    if args.use_pca:
        output_file = os.path.join(
            args.output_dir,
            f"{base_name}_pca_components_{args.pca_components}_light_gbm.csv"
        )
    else:
        output_file = os.path.join(
            args.output_dir,
            f"{base_name}_top_{args.top_n_features}_features_light_gbm.csv"
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
    TP = cm[1, 1] if cm.shape == (2, 2) else 0
    TN = cm[0, 0] if cm.shape == (2, 2) else 0
    FP = cm[0, 1] if cm.shape == (2, 2) else 0
    FN = cm[1, 0] if cm.shape == (2, 2) else 0

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


def save_execution_summary(args, metrics, feature_importance, output_file, reference_info=None):
    """Save execution summary including configuration, metrics, and output locations"""
    summary = {
        'configuration': vars(args),
        'execution_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'feature_importance_file': os.path.join(args.output_dir, 'feature_importance_lightboost.csv'),
        'top_features_file': os.path.join(args.output_dir, 'top_features_lightboost_baseline.txt'),
        'processed_dataset': output_file,
        'top_10_features': feature_importance.head(10)[['Feature', 'GainImportanceNorm']].to_dict('records')
    }

    # Add reference information if available
    if reference_info:
        summary['reference_info'] = reference_info

    # Save summary to JSON
    summary_file = os.path.join(args.output_dir, 'execution_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    logger.info(f"Saved execution summary to {summary_file}")

    return summary_file


# def main():
#     """Main function for the optimized feature selection workflow"""
#     start_time = time.time()
#
#     # Parse arguments
#     args = parse_args()
#
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # Set random seed
#     set_seed(args.seed)
#
#     # Load reference data if specified
#     ref_df = None
#     ref_features = None
#     if args.reference_csv:
#         ref_df = load_reference_data(args)
#         if args.freeze_features:
#             ref_features = identify_features_from_reference(ref_df, args.timestamp_col, args.exclude_cols.split(','))
#
#     # Step 1: Load and prepare data
#     logger.info("STEP 1: Loading and preparing data")
#
#     # Load data and ensure target column exists
#     df = load_data(args)
#
#     # Create data splits (aligned with reference if needed)
#     train_df, val_df, test_df, full_df_with_split = prepare_data_splits(df, args, ref_df)
#
#     # Prepare features and targets (using reference features if specified)
#     train_X, train_y, feature_cols = prepare_features_targets(train_df, args, feature_cols=ref_features)
#     val_X, val_y, _ = prepare_features_targets(val_df, args, feature_cols=ref_features)
#     test_X, test_y, _ = prepare_features_targets(test_df, args, feature_cols=ref_features)
#
#     # Track reference information for summary
#     reference_info = None
#     if args.reference_csv:
#         reference_info = {
#             'reference_file': args.reference_csv,
#             'using_reference_features': args.freeze_features,
#             'using_reference_splits': args.freeze_train_val,
#             'num_reference_features': len(ref_features) if ref_features else 0,
#             'train_size': len(train_df),
#             'val_size': len(val_df),
#             'test_size': len(test_df)
#         }
#         logger.info(f"Using reference data: {args.reference_csv}")
#         if args.freeze_features:
#             logger.info(f"Using {len(ref_features)} features from reference data")
#         if args.freeze_train_val:
#             logger.info(f"Using train/val splits from reference data")
#
#     # If we're using a reference but not freezing features, we need to run optimization
#     if not args.freeze_features:
#         # Step 2: Optimize LightGBM on full dataset
#         logger.info("STEP 2: Optimizing LightGBM on full dataset")
#         study = optimize_lightgbm(train_X, train_y, val_X, val_y, args)
#
#         # Step 3: Train final model with best parameters
#         logger.info("STEP 3: Training final model with best hyperparameters")
#         model = train_final_model(study.best_params, train_X, train_y, val_X, val_y, args)
#
#         # Step 4: Evaluate model on test set
#         logger.info("STEP 4: Evaluating model on test set")
#         test_metrics = evaluate_model(model, test_X, test_y, name="Test")
#
#         # Step 5: Calculate feature importance from optimized model
#         logger.info("STEP 5: Calculating feature importance from optimized model")
#         feature_importance, top_features = calculate_feature_importance(model, feature_cols, args)
#
#         # Save model
#         logger.info("Saving LightGBM model")
#         model_file = os.path.join(args.output_dir, 'optimized_lightgbm_model.joblib')
#         joblib.dump(model, model_file)
#     else:
#         # Skip optimization and use reference features
#         logger.info("Skipping optimization and using reference features")
#         top_features = ref_features
#         # Create a dummy feature importance DataFrame
#         feature_importance = pd.DataFrame({
#             'Feature': top_features,
#             'SplitImportance': range(len(top_features), 0, -1),
#             'GainImportance': range(len(top_features), 0, -1),
#             'SplitImportanceNorm': [1.0 / len(top_features)] * len(top_features),
#             'GainImportanceNorm': [1.0 / len(top_features)] * len(top_features)
#         })
#         test_metrics = {'info': 'No model evaluation performed when using reference features'}
#
#     # Step 6: Create processed dataset with top features or PCA
#     logger.info("STEP 6: Creating processed dataset")
#     df_processed = create_processed_dataset(full_df_with_split, top_features, args)
#
#     # Step 7: Save processed dataset
#     logger.info("STEP 7: Saving processed dataset")
#     # Add a suffix to indicate this is a reference-based processing if applicable
#     suffix = "refbased" if args.reference_csv else None
#     output_file = save_processed_dataset(df_processed, args, suffix)
#
#     # Save execution summary
#     summary_file = save_execution_summary(args, test_metrics, feature_importance, output_file, reference_info)
#
#     # Log completion
#     total_time = time.time() - start_time
#     logger.info(f"Completed optimized feature selection workflow in {total_time:.2f} seconds")
#     logger.info(f"All outputs saved to: {args.output_dir}")
#     logger.info(f"Processed dataset ready for transformer model: {output_file}")
#     logger.info(f"Execution summary: {summary_file}")
#
#
# if __name__ == "__main__":
#     main()

def main():
    """Main function for the optimized feature selection workflow"""
    start_time = time.time()

    # Parse arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Set reference timestamp column if not specified
    if args.reference_timestamp_col is None:
        args.reference_timestamp_col = args.timestamp_col
        logger.info(f"Using '{args.timestamp_col}' as reference timestamp column")
    else:
        logger.info(f"Using '{args.reference_timestamp_col}' as reference timestamp column")

    # Load reference data if specified
    ref_df = None
    ref_features = None
    if args.reference_csv:
        ref_df = load_reference_data(args)
        if args.freeze_features:
            ref_features = identify_features_from_reference(ref_df, args.reference_timestamp_col,
                                                            args.exclude_cols.split(','))

    # Step 1: Load and prepare data
    logger.info("STEP 1: Loading and preparing data")

    # Load data and ensure target column exists
    df = load_data(args)

    # Create data splits (aligned with reference if needed)
    train_df, val_df, test_df, full_df_with_split = prepare_data_splits(df, args, ref_df)

    # Prepare features and targets (using reference features if specified)
    train_X, train_y, feature_cols = prepare_features_targets(train_df, args, feature_cols=ref_features)
    val_X, val_y, _ = prepare_features_targets(val_df, args, feature_cols=ref_features)
    test_X, test_y, _ = prepare_features_targets(test_df, args, feature_cols=ref_features)

    # Track reference information for summary
    reference_info = None
    if args.reference_csv:
        reference_info = {
            'reference_file': args.reference_csv,
            'using_reference_features': args.freeze_features,
            'using_reference_splits': args.freeze_train_val,
            'timestamp_column': args.timestamp_col,
            'reference_timestamp_column': args.reference_timestamp_col,
            'num_reference_features': len(ref_features) if ref_features else 0,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
        logger.info(f"Using reference data: {args.reference_csv}")
        if args.freeze_features:
            logger.info(f"Using {len(ref_features)} features from reference data")
        if args.freeze_train_val:
            logger.info(f"Using train/val splits from reference data")

    # If we're using a reference but not freezing features, we need to run optimization
    if not args.freeze_features:
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

        # Save model
        logger.info("Saving LightGBM model")
        model_file = os.path.join(args.output_dir, 'optimized_lightgbm_model.joblib')
        joblib.dump(model, model_file)
    else:
        # Skip optimization and use reference features
        logger.info("Skipping optimization and using reference features")
        top_features = ref_features
        # Create a dummy feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': top_features,
            'SplitImportance': range(len(top_features), 0, -1),
            'GainImportance': range(len(top_features), 0, -1),
            'SplitImportanceNorm': [1.0 / len(top_features)] * len(top_features),
            'GainImportanceNorm': [1.0 / len(top_features)] * len(top_features)
        })
        test_metrics = {'info': 'No model evaluation performed when using reference features'}

    # Step 6: Create processed dataset with top features or PCA
    logger.info("STEP 6: Creating processed dataset")
    df_processed = create_processed_dataset(full_df_with_split, top_features, args)

    # Step 7: Save processed dataset
    logger.info("STEP 7: Saving processed dataset")
    # Add a suffix to indicate this is a reference-based processing if applicable
    suffix = "refbased" if args.reference_csv else None
    output_file = save_processed_dataset(df_processed, args, suffix)

    # Save execution summary
    summary_file = save_execution_summary(args, test_metrics, feature_importance, output_file, reference_info)

    # Log completion
    total_time = time.time() - start_time
    logger.info(f"Completed optimized feature selection workflow in {total_time:.2f} seconds")
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info(f"Processed dataset ready for transformer model: {output_file}")
    logger.info(f"Execution summary: {summary_file}")


if __name__ == "__main__":
    main()
