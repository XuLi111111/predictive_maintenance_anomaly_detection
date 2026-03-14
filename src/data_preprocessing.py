import os
import numpy as np
import pandas as pd

# Basic experiment parameters
THRESHOLD = 30
WINDOW_SIZE = 20

# Paths to raw FD001 dataset files
TRAIN_PATH = "../raw_data/train_FD001.txt"
TEST_PATH = "../raw_data/test_FD001.txt"
RUL_PATH = "../raw_data/RUL_FD001.txt"

# Directory where processed datasets will be saved
OUTPUT_DIR = "../processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names based on the NASA turbofan dataset description
columns = [
    "engine_id", "cycle",
    "setting_1", "setting_2", "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Load train, test, and RUL datasets
train_df = pd.read_csv(TRAIN_PATH, sep=r"\s+", header=None)
test_df = pd.read_csv(TEST_PATH, sep=r"\s+", header=None)
rul_df = pd.read_csv(RUL_PATH, header=None)

# Keep only the first 26 columns (NASA txt files usually include two trailing empty columns)
if train_df.shape[1] > 26:
    train_df = train_df.iloc[:, :26]
if test_df.shape[1] > 26:
    test_df = test_df.iloc[:, :26]

# Assign column names to the dataframes
train_df.columns = columns
test_df.columns = columns

# Detect sensors with zero or near-zero variance
sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
sensor_std = train_df[sensor_columns].std()
useful_sensors = sensor_std[sensor_std > 1e-6].index.tolist()
removed_sensors = sensor_std[sensor_std <= 1e-6].index.tolist()

print("Removed useless sensor columns:", removed_sensors)
print("Remaining sensor columns:", useful_sensors)
print()

# Compute Remaining Useful Life (RUL) for the training dataset
train_max_cycle = train_df.groupby("engine_id")["cycle"].max().reset_index()
train_max_cycle.columns = ["engine_id", "max_cycle"]
train_df = train_df.merge(train_max_cycle, on="engine_id", how="left")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

# Generate binary labels based on the RUL threshold
train_df["label"] = (train_df["RUL"] <= THRESHOLD).astype(int)

# Select feature columns for modeling
# Use the 3 operating settings and the sensors with non-zero variance
feature_columns = ["setting_1", "setting_2", "setting_3"] + useful_sensors

print("Number of final feature columns:", len(feature_columns))
print("Final feature columns:", feature_columns)
print()

# Construct training dataset using sliding windows
X_all = []
y_all = []
engine_ids = []

for engine_id in train_df["engine_id"].unique():
    engine_data = train_df[train_df["engine_id"] == engine_id].reset_index(drop=True)

    # Skip engines whose sequence length is shorter than the window size
    if len(engine_data) < WINDOW_SIZE:
        continue

    for start_idx in range(0, len(engine_data) - WINDOW_SIZE + 1):
        end_idx = start_idx + WINDOW_SIZE
        window_df = engine_data.iloc[start_idx:end_idx]

        # Use a window of length WINDOW_SIZE as model input
        x_window = window_df[feature_columns].values

        # The label of the window is the label of its final timestep
        y_label = window_df.iloc[-1]["label"]

        X_all.append(x_window)
        y_all.append(y_label)
        engine_ids.append(engine_id)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int64)
engine_ids = np.array(engine_ids, dtype=np.int64)

print("Training dataset construction completed:")
print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)
print("engine_id shape:", engine_ids.shape)
print()

# Process the test dataset
# Only the final window of each engine is used for testing
X_test = []
y_test = []

for idx, engine_id in enumerate(sorted(test_df["engine_id"].unique())):
    engine_data = test_df[test_df["engine_id"] == engine_id].reset_index(drop=True)

    if len(engine_data) < WINDOW_SIZE:
        continue

    # RUL of the final timestep for this engine (from RUL file)
    final_rul = rul_df.iloc[idx, 0]

    # Generate the label using the same threshold rule
    final_label = 1 if final_rul <= THRESHOLD else 0

    # Use only the final window for each engine
    last_window = engine_data.iloc[-WINDOW_SIZE:][feature_columns].values

    X_test.append(last_window)
    y_test.append(final_label)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int64)

print("Test dataset construction completed:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print()

# Save processed datasets as NumPy arrays for efficient loading
np.save(os.path.join(OUTPUT_DIR, "X_all.npy"), X_all)
np.save(os.path.join(OUTPUT_DIR, "y_all.npy"), y_all)
np.save(os.path.join(OUTPUT_DIR, "engine_id.npy"), engine_ids)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print("Data saving completed:")
print("Saved files:")
print("- X_all.npy")
print("- y_all.npy")
print("- engine_id.npy")
print("- X_test.npy")
print("- y_test.npy")