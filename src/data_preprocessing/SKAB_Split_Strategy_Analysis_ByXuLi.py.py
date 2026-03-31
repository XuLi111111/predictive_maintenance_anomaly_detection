


import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Split Strategy Experiment for SKAB (By Xu Li)
# ============================================================
# Purpose:
# 1. Estimate the sample ratio of train / val / test under different
#    file-level split strategies.
# 2. Check whether the label distribution is reasonable.
# 3. Support final decision on the dataset split plan before training.
# ============================================================

WINDOW_SIZE = 20
HORIZON = 10
RAW_DATA_DIR = "../../data/raw/dataset2"
SUBFOLDERS = ["valve1", "valve2"]


# ------------------------------------------------------------
# Core utility: build samples from one file and return labels
# ------------------------------------------------------------

def build_labels_from_one_file(df: pd.DataFrame, window_size: int, horizon: int) -> np.ndarray:
    """
    Build only labels from a single CSV file.
    This is enough for split-ratio and label-distribution experiments.

    Label rule:
    y_t = 1 if any anomaly occurs within the next `horizon` timesteps.
    """
    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' column. Current columns: {list(df.columns)}")
    if "anomaly" not in df.columns:
        raise ValueError(f"Missing 'anomaly' column. Current columns: {list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    anomaly = df["anomaly"].values
    total_len = len(df)

    labels = []
    for i in range(window_size - 1, total_len - horizon):
        future = anomaly[i + 1: i + 1 + horizon]
        label = 1 if np.any(future == 1) else 0
        labels.append(label)

    return np.array(labels, dtype=np.int64)


# ------------------------------------------------------------
# Read all file statistics
# ------------------------------------------------------------

def collect_file_statistics() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Return a nested dict like:
    {
        'valve1': {'0.csv': y_array, '1.csv': y_array, ...},
        'valve2': {'0.csv': y_array, ...}
    }
    """
    result = {}

    for folder in SUBFOLDERS:
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        file_stats = {}

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".csv"):
                continue

            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, sep=';')
            y = build_labels_from_one_file(df, WINDOW_SIZE, HORIZON)
            file_stats[file_name] = y

        result[folder] = file_stats

    return result


# ------------------------------------------------------------
# Helpers for split experiment
# ------------------------------------------------------------

def summarize_split(split_name: str,
                    file_stats: Dict[str, Dict[str, np.ndarray]],
                    split_plan: Dict[str, List[Tuple[str, str]]]):
    """
    split_plan format example:
    {
        'train': [('valve1', '0.csv'), ('valve1', '1.csv')],
        'val':   [('valve1', '2.csv')],
        'test':  [('valve2', '0.csv')]
    }
    """
    print("\n" + "=" * 90)
    print(f"SPLIT EXPERIMENT: {split_name}")
    print("=" * 90)

    split_y = {}
    total_samples = 0

    for split_key in ["train", "val", "test"]:
        labels_list = []
        print(f"\n[{split_key.upper()} FILES]")

        for folder, file_name in split_plan[split_key]:
            y = file_stats[folder][file_name]
            labels_list.append(y)
            print(
                f"  {folder}/{file_name:<8} -> samples={len(y):<5} "
                f"pos={int(np.sum(y == 1)):<5} neg={int(np.sum(y == 0)):<5}"
            )

        if len(labels_list) == 0:
            split_y[split_key] = np.array([], dtype=np.int64)
        else:
            split_y[split_key] = np.concatenate(labels_list, axis=0)

        total_samples += len(split_y[split_key])

    print("\n" + "-" * 90)
    print("SPLIT SUMMARY")
    print("-" * 90)

    for split_key in ["train", "val", "test"]:
        y = split_y[split_key]
        n_total = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        pos_ratio = (n_pos / n_total * 100) if n_total > 0 else 0.0
        data_ratio = (n_total / total_samples * 100) if total_samples > 0 else 0.0

        print(f"{split_key.upper():<6} -> samples={n_total:<6} "
              f"data_ratio={data_ratio:>6.2f}%   "
              f"pos={n_pos:<6} neg={n_neg:<6} pos_ratio={pos_ratio:>6.2f}%")

    print("\nInterpretation:")
    print("- data_ratio shows how much data falls into each split.")
    print("- pos_ratio shows the anomaly label proportion in each split.")
    print("- A good split should avoid extreme imbalance differences across train/val/test.")


# ------------------------------------------------------------
# Candidate split strategies
# ------------------------------------------------------------

def build_strategy_a() -> Dict[str, List[Tuple[str, str]]]:
    """
    Strategy A:
    train = valve1/0-11
    val   = valve1/12-15
    test  = valve2/0-3

    Motivation:
    - train/val on valve1
    - test on valve2 to evaluate generalization across operating conditions
    """
    train_files = [("valve1", f"{i}.csv") for i in range(12)]
    val_files = [("valve1", f"{i}.csv") for i in range(12, 16)]
    test_files = [("valve2", f"{i}.csv") for i in range(4)]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }



def build_strategy_b() -> Dict[str, List[Tuple[str, str]]]:
    """
    Strategy B:
    Train/val/test all contain both valve1 and valve2.

    valve1:
      train = 0-10
      val   = 11-12
      test  = 13-15

    valve2:
      train = 0-1
      val   = 2
      test  = 3
    """
    train_files = [("valve1", f"{i}.csv") for i in range(11)] + [("valve2", "0.csv"), ("valve2", "1.csv")]
    val_files = [("valve1", "11.csv"), ("valve1", "12.csv"), ("valve2", "2.csv")]
    test_files = [("valve1", "13.csv"), ("valve1", "14.csv"), ("valve1", "15.csv"), ("valve2", "3.csv")]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def run_split_experiments():
    print("Collecting per-file label statistics...")
    file_stats = collect_file_statistics()

    print("\nAvailable files:")
    for folder in SUBFOLDERS:
        print(f"  {folder}: {list(file_stats[folder].keys())}")

    strategy_a = build_strategy_a()
    summarize_split("Strategy A (train/val on valve1, test on valve2)", file_stats, strategy_a)

    strategy_b = build_strategy_b()
    summarize_split("Strategy B (mixed valve1 + valve2 across all splits)", file_stats, strategy_b)


if __name__ == "__main__":
    run_split_experiments()