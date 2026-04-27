"""
SKAB EDA By XuLi

This script performs basic exploratory data analysis (EDA) for the SKAB dataset.
The goal is to understand data distribution, anomaly ratio, and basic time-series behaviour.

Outputs will be saved under:
results/dataset2/eda/
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ================================
# Paths
# ================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset2"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================
# Load Data
# ================================
def load_all_files():
    data = []

    for subfolder in ["valve1", "valve2"]:
        folder_path = DATASET_ROOT / subfolder

        for file in sorted(os.listdir(folder_path)):
            if file.endswith(".csv"):
                path = folder_path / file
                df = pd.read_csv(path, sep=";")
                df["datetime"] = pd.to_datetime(df["datetime"])
                data.append((subfolder, file, df))

    return data


# ================================
# Basic Summary
# ================================
def summary_analysis(data):
    rows = []

    for folder, name, df in data:
        total = len(df)
        anomaly = (df["anomaly"] == 1).sum()
        normal = (df["anomaly"] == 0).sum()

        rows.append({
            "folder": folder,
            "file": name,
            "rows": total,
            "anomaly_count": anomaly,
            "normal_count": normal,
            "anomaly_ratio": anomaly / total
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n=== Dataset Summary ===")
    print(summary_df.head())

    return summary_df


# ================================
# Missing Values
# ================================
def check_missing(data):
    combined = pd.concat([df for _, _, df in data])
    missing = combined.isna().sum()

    print("\n=== Missing Values ===")
    print(missing)

    missing.to_csv(OUTPUT_DIR / "missing.csv")


# ================================
# Plot 1: Label Distribution
# ================================
def plot_label_distribution(summary_df):
    total_normal = summary_df["normal_count"].sum()
    total_anomaly = summary_df["anomaly_count"].sum()

    plt.figure()
    plt.bar(["Normal", "Anomaly"], [total_normal, total_anomaly])
    plt.title("Label Distribution")
    plt.savefig(OUTPUT_DIR / "label_distribution.png")
    plt.close()


# ================================
# Plot 2: Sample Time Series
# ================================
def plot_sample(data):
    # Use the first CSV file as an example for time-series visualization
    folder, name, df = data[0]

    # SKAB does not use generic names like sensor_1.
    # Select the first numeric sensor column automatically.
    exclude_cols = {"datetime", "anomaly", "changepoint"}
    sensor_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(sensor_cols) == 0:
        raise ValueError("No numeric sensor column found for plotting.")

    sensor_col = sensor_cols[0]

    plt.figure(figsize=(10, 4))
    plt.plot(df["datetime"], df[sensor_col], label=sensor_col)

    anomaly_points = df[df["anomaly"] == 1]
    if len(anomaly_points) > 0:
        plt.scatter(
            anomaly_points["datetime"],
            anomaly_points[sensor_col],
            color="red",
            label="anomaly"
        )

    plt.legend()
    plt.title(f"Sample Time Series: {folder}/{name} ({sensor_col})")
    plt.xlabel("Time")
    plt.ylabel(sensor_col)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_timeseries.png")
    plt.close()


# ================================
# Main
# ================================
def run_eda():
    print("Running SKAB EDA...")

    data = load_all_files()
    summary_df = summary_analysis(data)
    check_missing(data)

    # Print overall anomaly ratio
    overall_ratio = summary_df["anomaly_count"].sum() / summary_df["rows"].sum()
    print(f"\nOverall anomaly ratio: {overall_ratio:.4f}")

    plot_label_distribution(summary_df)
    plot_sample(data)

    print("\nEDA finished. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()