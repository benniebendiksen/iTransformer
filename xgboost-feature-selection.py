import os
import argparse
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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
    parser.add_argument('--data_path', type=str, default='btcusdt_12h_4h_complete.csv',
                        help='data file')
    parser.add_argument('--target', type=str, default='price_direction',
                        help='target column for binary prediction')
    parser.add_argument('--price_col', type=str, default='close',
                        help='price column name for generating direction target')
    parser.add_argument('--output_dir', type=str, default='./optimized_features/',
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
    parser.add_argument('--optimize_metric', type=str, default='win_rate',
                        choices=['accuracy', 'precision', 'recall', 'f1', 'win_rate', 'profit_factor'],
                        help='metric to optimize during hyperparameter tuning')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='timeout for hyperparameter optimization in seconds')

    # Feature selection arguments
    parser.add_argument('--top_n_features', type=int, default=100,
                        help='number of top features to select')
    parser.add_argument('--use_pca', action='store_true',
                        help='whether to apply PCA to the top features')
    parser.add_argument('--pca_components', type=int, default=44,
                        help='number of PCA components to extract')

    # GPU/Performance arguments
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='use GPU for XGBoost training if available')
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

    logger.info(f"Prepared features shape: {X.shape}")
    logger.info(f"Prepared target shape: {y.shape}")

    return X, y, feature_cols


def objective(trial, train_X, train_y, val_X, val_y, args, use_gpu=False):
    """Objective function for Optuna hyperparameter optimization"""
    # Define hyperparameter search space for XGBoost
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0,
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': args.seed
    }

    # Add GPU-specific parameters if using GPU
    if use_gpu:
        param['tree_method'] = 'gpu_hist'
        param['gpu_id'] = 0
    else:
        param['tree_method'] = 'hist'

    # Add booster-specific parameters
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_float('rate_drop', 0.01, 0.5)
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.01, 0.5)

    # Get n_estimators and remove from params (will be set in fit method)
    n_estimators = param.pop('n_estimators')

    # Create early stopping parameters
    early_stopping_rounds = 50

    # Create evaluation dataset
    eval_set = [(val_X, val_y)]

    # Train the model
    model = xgb.XGBClassifier(**param, n_estimators=n_estimators)
    model.fit(
        train_X, train_y,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False
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

    # Store best iteration if available
    if hasattr(model, 'best_iteration'):
        trial.set_user_attr('best_iteration', model.best_iteration)
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


def optimize_xgboost(train_X, train_y, val_X, val_y, args):
    """Run hyperparameter optimization for XGBoost"""
    logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
    logger.info(f"Optimizing for metric: {args.optimize_metric}")

    start_time = time.time()

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=args.seed),
        study_name=f'xgb_crypto_{args.optimize_metric}'
    )

    # Check if GPU should be used
    use_gpu = args.use_gpu and torch.cuda.is_available()

    # Check if XGBoost GPU support is available
    if use_gpu:
        # Try to create a simple XGBoost model with GPU to check if it's supported
        try:
            # Create a minimal dataset
            mini_X = train_X[:10]
            mini_y = train_y[:10]

            # Try to train a model with GPU
            test_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'verbosity': 0
            }
            test_model = xgb.XGBClassifier(**test_params)
            test_model.fit(mini_X, mini_y)

            logger.info("Using GPU for XGBoost training")
        except Exception as e:
            use_gpu = False
            logger.warning(f"GPU support not available in XGBoost: {str(e)}")
            logger.info("Falling back to CPU for XGBoost training")
    else:
        logger.info("Using CPU for XGBoost training")

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

    # Add booster-specific parameters if not present
    if final_params.get('booster') == 'dart':
        if 'sample_type' not in final_params:
            final_params['sample_type'] = 'uniform'
        if 'normalize_type' not in final_params:
            final_params['normalize_type'] = 'tree'
        if 'rate_drop' not in final_params:
            final_params['rate_drop'] = 0.1
        if 'skip_drop' not in final_params:
            final_params['skip_drop'] = 0.1

    # Check if GPU should be used and is available
    use_gpu = False
    if args.use_gpu and torch.cuda.is_available():
        # Try to create a simple XGBoost model with GPU to check if it's supported
        try:
            # Create a minimal dataset
            mini_X = train_X[:10]
            mini_y = train_y[:10]

            # Try to train a model with GPU
            test_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'verbosity': 0
            }
            test_model = xgb.XGBClassifier(**test_params)
            test_model.fit(mini_X, mini_y)

            # If we get here, GPU is available
            use_gpu = True
            logger.info("Using GPU for final model training")

            # Add GPU settings
            final_params['tree_method'] = 'gpu_hist'
            final_params['gpu_id'] = 0
        except Exception as e:
            logger.warning(f"GPU support not available in XGBoost: {str(e)}")
            logger.info("Falling back to CPU for final model training")
            final_params['tree_method'] = 'hist'
    else:
        logger.info("Using CPU for final model training")
        final_params['tree_method'] = 'hist'

    # Add other necessary parameters
    final_params['objective'] = 'binary:logistic'
    final_params['eval_metric'] = 'logloss'
    final_params['verbosity'] = 0
    final_params['random_state'] = args.seed

    # Set n_estimators
    n_estimators = 500  # Default to a high value, will use early stopping

    # Initialize and train model
    model = xgb.XGBClassifier(
        **final_params,
        n_estimators=n_estimators
    )

    model.fit(
        np.vstack([train_X, val_X]),
        np.concatenate([train_y, val_y]),
        eval_set=[(val_X, val_y)],
        early_stopping_rounds=100,
        verbose=False
    )

    # Log the number of iterations
    if hasattr(model, 'best_iteration'):
        logger.info(f"Final model trained with {model.best_iteration} iterations")
    else:
        logger.info(f"Final model trained with {n_estimators} iterations")

    return model


def calculate_feature_importance(model, feature_names, args):
    """Calculate and save feature importance from the optimized XGBoost model"""
    logger.info("Calculating feature importance")

    # Get feature importance
    importance_type = 'gain'  # Can be 'weight', 'gain', 'cover', 'total_gain', or 'total_cover'

    # XGBoost provides feature importance directly
    importance_values = model.get_booster().get_score(importance_type=importance_type)

    # Convert to DataFrame (handling features that might not be in the importance dict)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': [importance_values.get(f, 0) for f in feature_names]
    })

    # Handle the case where feature names in the model might be different (e.g., f0, f1, etc.)
    if all(importance_values.get(f, 0) == 0 for f in feature_names):
        # Try using feature index names (f0, f1, etc.)
        importance_values = model.get_booster().get_score(importance_type=importance_type)
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': [importance_values.get(f'f{i}', 0) for i in range(len(feature_names))]
        })

    # Normalize importance scores
    feature_importance['ImportanceNorm'] = feature_importance['Importance'] / feature_importance['Importance'].sum() if \
    feature_importance['Importance'].sum() > 0 else 0

    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save feature importance to CSV
    output_file = os.path.join(args.output_dir, 'feature_importance.csv')
    feature_importance.to_csv(output_file, index=False)
    logger.info(f"Saved feature importance to {output_file}")

    # Save top N features list
    top_n = min(args.top_n_features, len(feature_importance))
    top_features = feature_importance.head(top_n)['Feature'].tolist()

    top_features_file = os.path.join(args.output_dir, 'top_features.txt')
    with open(top_features_file, 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    logger.info(f"Saved top {top_n} features to {top_features_file}")

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(feature_importance['Feature'][:top_n][::-1], feature_importance['ImportanceNorm'][:top_n][::-1])
    plt.title(f'Top {top_n} Features by Importance (Gain)')
    plt.xlabel('Normalized Importance')
    plt.tight_layout()

    plot_file = os.path.join(args.output_dir, 'feature_importance.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved feature importance plot to {plot_file}")

    return feature_importance, top_features


def create_processed_dataset(df, top_features, args):
    """Create a processed dataset with only the top features or PCA components"""
    logger.info("Creating processed dataset")

    # Ensure required columns are included
    essential_cols = ['split']
    if args.target not in top_features:
        essential_cols.append(args.target)

    # Get time column if specified
    time_cols = []
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
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

        # Add essential columns back
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
            f"{os.path.splitext(args.data_path)[0]}_pca_components_{args.pca_components}_xgboost.csv"
        )
    else:
        output_file = os.path.join(
            args.output_dir,
            f"{os.path.splitext(args.data_path)[0]}_top_{args.top_n_features}_features_xgboost.csv"
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
        'feature_importance_file': os.path.join(args.output_dir, 'feature_importance.csv'),
        'top_features_file': os.path.join(args.output_dir, 'top_features.txt'),
        'processed_dataset': output_file,
        'top_10_features': feature_importance.head(10)[['Feature', 'ImportanceNorm']].to_dict('records')
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
    val_X, val_y, _ = prepare_features_targets(val_df, args)
    test_X, test_y, _ = prepare_features_targets(test_df, args)

    # Step 2: Optimize XGBoost on full dataset
    logger.info("STEP 2: Optimizing XGBoost on full dataset")
    study = optimize_xgboost(train_X, train_y, val_X, val_y, args)

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
    logger.info("STEP 8: Saving XGBoost model")
    model_file = os.path.join(args.output_dir, 'optimized_xgboost_model.joblib')
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