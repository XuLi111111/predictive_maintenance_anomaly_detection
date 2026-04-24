import os
import numpy as np
import pandas as pd

# ============================================================
# SKAB Dataset Builder for DL Pipeline (Strategy A) - By David
# ============================================================
# Strategy A:
#   Train = valve1/0.csv  ~ valve1/11.csv
#   Val   = valve1/12.csv ~ valve1/15.csv
#   Test  = valve2/0.csv  ~ valve2/3.csv
#
# This script:
# 1. Processes each CSV file independently
# 2. Builds sliding-window samples with future-horizon labels
# 3. Aggregates samples into train / val / test sets
# 4. Saves the final split dataset for deep learning training
# ============================================================

# =====================
# Config
# =====================
WINDOW_SIZE = 20
HORIZON = 10

DATA_DIR = "."  # Current SKAB directory
SAVE_PATH = "../data/processed/dataset2/skab_strategyA_window20_horizon10.npz"

# Strategy A split plan
TRAIN_FILES = [("valve1", f"{i}.csv") for i in range(12)]
VAL_FILES = [("valve1", f"{i}.csv") for i in range(12, 16)]
TEST_FILES = [("valve2", f"{i}.csv") for i in range(4)]


# =====================
# Build samples from one file
# =====================
def build_samples_from_one_file(df, window_size, horizon):
    """
    Build sliding window samples from a single CSV file.

    Parameters:
        df: pandas DataFrame
        window_size: number of past time steps
        horizon: number of future time steps for early warning

    Returns:
        X: shape (N, window_size, feature_dim)
        y: shape (N,)
    """
    X = []
    y = []

    if "datetime" not in df.columns:
        raise ValueError(
            f"Missing 'datetime' column. Current columns are: {list(df.columns)}"
        )
    if "anomaly" not in df.columns:
        raise ValueError(
            f"Missing 'anomaly' column. Current columns are: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    feature_cols = [
        col for col in df.columns
        if col not in ["datetime", "anomaly", "changepoint"]
    ]

    data = df[feature_cols].values
    anomaly = df["anomaly"].values
    total_len = len(df)

    for i in range(window_size - 1, total_len - horizon):
        # Past window as input
        x_window = data[i - window_size + 1: i + 1]

        # Future horizon for label construction
        future = anomaly[i + 1: i + 1 + horizon]

        # Early warning label: 1 if any anomaly occurs in future horizon
        label = 1 if np.any(future == 1) else 0

        X.append(x_window)
        y.append(label)

    return np.array(X), np.array(y)


# =====================
# Process one split
# =====================
def process_split(split_name, file_list):
    X_split = []
    y_split = []

    print("\n" + "=" * 60)
    print(f"PROCESSING {split_name.upper()} SPLIT")
    print("=" * 60)

    for folder, file_name in file_list:
        file_path = os.path.join(DATA_DIR, folder, file_name)
        print(f"Processing: {file_path}")

        df = pd.read_csv(file_path, sep=';')
        X_file, y_file = build_samples_from_one_file(df, WINDOW_SIZE, HORIZON)

        print(f"  -> Generated samples: {len(X_file)}")
        print(f"  -> Positive: {int(np.sum(y_file == 1))}")
        print(f"  -> Negative: {int(np.sum(y_file == 0))}")

        if len(X_file) > 0:
            X_split.append(X_file)
            y_split.append(y_file)

    if not X_split:
        raise ValueError(f"No samples generated for split: {split_name}")

    X_split = np.concatenate(X_split, axis=0)
    y_split = np.concatenate(y_split, axis=0)

    print(f"\n{split_name.upper()} SUMMARY")
    print(f"X_{split_name}.shape: {X_split.shape}")
    print(f"y_{split_name}.shape: {y_split.shape}")
    print(f"Positive samples: {int(np.sum(y_split == 1))}")
    print(f"Negative samples: {int(np.sum(y_split == 0))}")

    return X_split, y_split


# =====================
# Main dataset builder
# =====================
def build_dataset_strategy_a():
    X_train, y_train = process_split("train", TRAIN_FILES)
    X_val, y_val = process_split("val", VAL_FILES)
    X_test, y_test = process_split("test", TEST_FILES)

    total_samples = len(y_train) + len(y_val) + len(y_test)

    print("\n" + "=" * 60)
    print("FINAL STRATEGY A DATASET SUMMARY")
    print("=" * 60)
    print(f"Train samples: {len(y_train)} ({len(y_train) / total_samples * 100:.2f}%)")
    print(f"Val samples:   {len(y_val)} ({len(y_val) / total_samples * 100:.2f}%)")
    print(f"Test samples:  {len(y_test)} ({len(y_test) / total_samples * 100:.2f}%)")
    print()
    print(f"Train positive ratio: {np.mean(y_train == 1) * 100:.2f}%")
    print(f"Val positive ratio:   {np.mean(y_val == 1) * 100:.2f}%")
    print(f"Test positive ratio:  {np.mean(y_test == 1) * 100:.2f}%")

    np.savez(
        SAVE_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
    )

    print(f"\nSaved Strategy A dataset to: {SAVE_PATH}")


# =====================
# Entry
# =====================
if __name__ == "__main__":
    build_dataset_strategy_a()