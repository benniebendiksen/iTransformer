import pandas as pd
import numpy as np

# File paths
baseline_path = "/optimized_features_light/btcusdt_12h_4h_complete_top_150_features_light_gbm_baseline.csv"
reattempt_path = "/optimized_features_light/btcusdt_12h_4h_complete_reattempt_top_150_features_light_gbm.csv"
output_path = "/optimized_features_light/btcusdt_12h_4h_complete_reattempt_reordered_2.csv"

# Load data
df_baseline = pd.read_csv(baseline_path)
df_reattempt = pd.read_csv(reattempt_path)

# Step 1: Check column match
baseline_cols = set(df_baseline.columns)
reattempt_cols = set(df_reattempt.columns)

if baseline_cols != reattempt_cols:
    print("Column mismatch detected.")
    print("Columns only in baseline:", baseline_cols - reattempt_cols)
    print("Columns only in reattempt:", reattempt_cols - baseline_cols)
else:
    print("Columns match. Proceeding to check column order...")

    # Step 2: Check column order
    column_order_matches = list(df_baseline.columns) == list(df_reattempt.columns)

    if column_order_matches:
        print("Column order matches. Proceeding to value comparison...")
    else:
        print("Columns match but order differs. Reordering reattempt dataframe...")

    # Align column order if needed
    df_reattempt = df_reattempt[df_baseline.columns]

    # Step 3: Check values
    all_match = True
    for col in df_baseline.columns:
        base_vals = df_baseline[col]
        retry_vals = df_reattempt[col]

        if pd.api.types.is_numeric_dtype(base_vals):
            equal = np.allclose(base_vals.fillna(0), retry_vals.fillna(0), rtol=1e-5, atol=1e-8)
        else:
            equal = base_vals.fillna("NaN").equals(retry_vals.fillna("NaN"))

        if not equal:
            mismatch_indices = (base_vals != retry_vals) & ~(base_vals.isna() & retry_vals.isna())
            mismatch_indices = mismatch_indices[mismatch_indices].index

            print(f"\nMismatch in column: {col}")
            print(f"Number of mismatched rows: {len(mismatch_indices)}")
            print("First few mismatches:")
            for idx in mismatch_indices[:5]:
                print(f"  Row {idx}: baseline={base_vals.iloc[idx]} | reattempt={retry_vals.iloc[idx]}")
            all_match = False

    if all_match:
        if not column_order_matches:
            print("\nAll column values match but column order was different. Saving reordered reattempt file...")
            df_reattempt.to_csv(output_path, index=False)
            print(f"Reordered file saved to: {output_path}")
        else:
            print("\nAll column values and order match. No need to save.")
    else:
        print("\nMismatch detected in one or more columns. No file saved.")