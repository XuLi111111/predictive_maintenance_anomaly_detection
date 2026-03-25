"""
build_skab.py  -  Preprocess the SKAB dataset for model training.

Reads from:   data/raw/SKAB/
Writes to:    data/processed/skab/
  X_train.npy, X_test.npy, y_train.npy, y_test.npy,
  X_train_normal.npy (normal-only, for unsupervised models),
  feature_names.npy
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]   # repo root
RAW_DIR    = BASE_DIR / "data" / "raw" / "SKAB"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "skab"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── The 8 raw sensor columns ────────────────────────────────────────────────
FEATURE_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


def load_labeled_csvs(folders: list) -> pd.DataFrame:
    """Load valve1, valve2, other CSVs — these already have anomaly labels."""
    frames = []
    for folder in folders:
        folder_path = RAW_DIR / folder
        if not folder_path.exists():
            print(f"[WARNING] Folder not found, skipping: {folder_path}")
            continue
        for csv_file in sorted(folder_path.glob("*.csv")):
            df = pd.read_csv(csv_file, sep=";", parse_dates=["datetime"])
            df["source"] = f"{folder}/{csv_file.stem}"
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No labeled CSV files found under {RAW_DIR}")
    return pd.concat(frames, ignore_index=True)


def load_anomaly_free() -> pd.DataFrame:
    """Load the anomaly-free baseline — inject label columns as 0."""
    path = RAW_DIR / "anomaly-free" / "anomaly-free.csv"
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    df["anomaly"]     = 0
    df["changepoint"] = 0
    df["source"]      = "anomaly-free"
    return df


def add_rolling_features(df: pd.DataFrame, windows=(5, 15)) -> pd.DataFrame:
    """Add rolling mean and std for each sensor (captures temporal patterns)."""
    df = df.sort_values("datetime").reset_index(drop=True)
    for w in windows:
        for col in FEATURE_COLS:
            if col in df.columns:
                df[f"{col}_mean{w}"] = df[col].rolling(w, min_periods=1).mean()
                df[f"{col}_std{w}"]  = df[col].rolling(w, min_periods=1).std().fillna(0)
    return df


def main():
    print("=" * 55)
    print("Step 1: Load raw data")
    print("=" * 55)

    print("  Loading anomaly-free samples...")
    df_normal = load_anomaly_free()

    print("  Loading labeled samples (valve1, valve2, other)...")
    df_labeled = load_labeled_csvs(["valve1", "valve2", "other"])

    df_all = pd.concat([df_normal, df_labeled], ignore_index=True)
    df_all = df_all.sort_values("datetime").reset_index(drop=True)
    print(f"\n  Total rows before cleaning: {len(df_all)}")

    df_all = df_all.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"  Total rows after cleaning:  {len(df_all)}")

    print("\n  Anomaly label distribution:")
    print(df_all["anomaly"].value_counts().sort_index().to_string())

    print("\n" + "=" * 55)
    print("Step 2: Add rolling features")
    print("=" * 55)
    df_all = add_rolling_features(df_all, windows=(5, 15))

    all_feat_cols = [
        c for c in df_all.columns
        if c not in ("datetime", "source", "anomaly", "changepoint")
    ]
    print(f"  Total feature columns: {len(all_feat_cols)}")
    print(f"  Features: {all_feat_cols}")

    print("\n" + "=" * 55)
    print("Step 3: Build feature matrix and labels")
    print("=" * 55)
    X = df_all[all_feat_cols].values.astype(np.float32)
    y = df_all["anomaly"].values.astype(np.int64)

    # Chronological 80/20 split — no shuffle, preserves time order
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {X_train.shape}  |  anomaly ratio: {y_train.mean():.4f}")
    print(f"  Test : {X_test.shape}   |  anomaly ratio: {y_test.mean():.4f}")

    print("\n" + "=" * 55)
    print("Step 4: Save processed arrays")
    print("=" * 55)
    np.save(OUTPUT_DIR / "X_train.npy",       X_train)
    np.save(OUTPUT_DIR / "X_test.npy",        X_test)
    np.save(OUTPUT_DIR / "y_train.npy",       y_train)
    np.save(OUTPUT_DIR / "y_test.npy",        y_test)
    np.save(OUTPUT_DIR / "feature_names.npy", np.array(all_feat_cols, dtype=object))

    # Normal-only subset used to train unsupervised models
    mask_normal = y_train == 0
    np.save(OUTPUT_DIR / "X_train_normal.npy", X_train[mask_normal])
    print(f"  Normal-only train samples: {mask_normal.sum()}")
    print(f"\n  All files saved to: {OUTPUT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()