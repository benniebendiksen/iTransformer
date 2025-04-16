import pandas as pd
import numpy as np

# File paths
baseline_script_path = "/optimized_features_light/btcusdt_12h_4h_complete_top_150_features_light_gbm_baseline.csv"
current_script_path = "/optimized_features_light/btcusd_12h_4h_complete_reattempt.csv"
output_path = "/optimized_features_light/btcusdt_12h_4h_complete_reattempt_reordered.csv"

def generate_target_column(df, price_col, seq_len, pred_len, target_col='price_direction'):
    """
    Generate binary labels for future price direction prediction.
    """
    print(f"Generating target column '{target_col}' based on future price direction changes")

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataset!")

    close_prices = df[price_col].values
    binary_labels = np.zeros(len(close_prices))

    for i in range(len(close_prices) - seq_len - pred_len):
        pred_idx = i + seq_len
        pred_price = close_prices[pred_idx]
        future_price = close_prices[pred_idx + pred_len]
        binary_labels[i] = 1.0 if future_price > pred_price else 0.0

    df_with_target = df.copy()
    df_with_target[target_col] = binary_labels

    valid_indices = range(len(close_prices) - seq_len - pred_len)
    valid_labels = binary_labels[valid_indices]
    up_ratio = np.mean(valid_labels) if len(valid_labels) > 0 else 0

    print(f"Generated target column '{target_col}'")
    print(f"  - Valid samples: {len(valid_labels)} (after accounting for sequence windows)")
    print(f"  - Upward movement ratio: {up_ratio:.4f} ({up_ratio * 100:.2f}%)")
    print(f"  - Downward movement ratio: {(1 - up_ratio):.4f} ({(1 - up_ratio) * 100:.2f}%)")

    return df_with_target


# Load data
df_baseline = pd.read_csv(baseline_script_path)
df_reattempt = pd.read_csv(current_script_path)

# Ensure target column exists in reattempt
if 'price_direction' not in df_reattempt.columns:
    df_reattempt = generate_target_column(df_reattempt, price_col='close', seq_len=96, pred_len=1)

# Match and reorder columns
common_cols = df_baseline.columns.intersection(df_reattempt.columns)
df_reattempt_matched = df_reattempt[common_cols]
df_reattempt_reordered = df_reattempt_matched[df_baseline.columns]

# Save result
df_reattempt_reordered.to_csv(output_path, index=False)
print(f"Reordered and trimmed DataFrame saved to: {output_path}")

