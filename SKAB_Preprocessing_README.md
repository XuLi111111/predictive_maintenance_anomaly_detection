# SKAB Dataset Preprocessing — Documentation

**Author:** David  
**Dataset:** SKAB (Skoltech Anomaly Benchmark)  
**Task:** Early-Warning Anomaly Detection for Industrial Water Pump  
**Preprocessing Strategy:** Strategy A — Sliding Window with Future-Horizon Labels

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Raw Data Description](#3-raw-data-description)
4. [Preprocessing Logic](#4-preprocessing-logic)
5. [Split Strategy (Strategy A)](#5-split-strategy-strategy-a)
6. [Output Dataset Specification](#6-output-dataset-specification)
7. [How to Reproduce the Dataset](#7-how-to-reproduce-the-dataset)
8. [How to Use the Processed Dataset](#8-how-to-use-the-processed-dataset)
9. [Important Notes & Caveats](#9-important-notes--caveats)
10. [Classical ML Baseline Reference](#10-classical-ml-baseline-reference)

---

## 1. Project Overview

This preprocessing pipeline converts raw SKAB sensor time-series CSV files into
ready-to-use NumPy arrays for **supervised anomaly detection** training.

The goal is **early-warning prediction**: given the last 20 seconds of sensor
readings, predict whether an anomaly will occur **within the next 10 seconds**.
This is a binary classification problem (label 0 = normal, label 1 = anomaly
incoming).

---

## 2. Directory Structure

The expected layout before running preprocessing:

```
capstone/
├── SKAB/                                          ← Raw data + scripts (this folder)
│   ├── valve1/
│   │   ├── 0.csv  ~ 15.csv                        ← 16 experiment runs, valve 1
│   ├── valve2/
│   │   ├── 0.csv  ~ 3.csv                         ← 4 experiment runs, valve 2
│   ├── anomaly-free/
│   │   └── anomaly-free.csv                       ← Normal operation baseline (not used in Strategy A)
│   ├── other/
│   │   └── 9.csv ~ 23.csv                         ← Other experiments (not used in Strategy A)
│   ├── Build_Dataset_SKAB_DLpipeline_By_David.py  ← Preprocessing script (run this)
│   └── SKAB_ClassicalML_Baseline_By_David.py      ← Classical ML baseline (optional reference)
│
└── data/
    ├── raw/
    │   └── dataset2/
    │       ├── valve1/   ← same CSVs mirrored here (optional backup)
    │       └── valve2/
    └── processed/
        └── dataset2/
            ├── skab_strategyA_window20_horizon10.npz   ← OUTPUT: processed dataset
            └── skab_classical_models/                  ← OUTPUT: saved ML model weights
                ├── scaler.pkl
                ├── model_lr.pkl
                ├── model_rf.pkl
                ├── model_svm.pkl
                ├── model_et.pkl
                ├── model_gb.pkl
                ├── model_knn.pkl
                └── model_xgb.pkl
```

> **Note:** The script uses relative paths. It must be run from inside the `SKAB/` directory.

---

## 3. Raw Data Description

### Source

SKAB (Skoltech Anomaly Benchmark) — publicly available benchmark dataset for
anomaly detection in industrial water pump systems. Each CSV file represents
one continuous experiment run (~1,100–1,150 seconds, 1 Hz sampling rate).

Official repo: https://github.com/waico/SKAB

### CSV Format

Each file is **semicolon-delimited** (`;`), not comma-delimited. Columns:

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | string | Timestamp, format `YYYY-MM-DD HH:MM:SS`, 1-second interval |
| `Accelerometer1RMS` | float64 | Vibration sensor 1 (RMS) |
| `Accelerometer2RMS` | float64 | Vibration sensor 2 (RMS) |
| `Current` | float64 | Motor current (A) |
| `Pressure` | float64 | Water pressure |
| `Temperature` | float64 | Water temperature |
| `Thermocouple` | float64 | Thermocouple temperature |
| `Voltage` | float64 | Voltage (V) |
| `Volume Flow RateRMS` | float64 | Volume flow rate (RMS) |
| `anomaly` | float64 | Ground truth label: `1.0` = anomaly, `0.0` = normal |
| `changepoint` | float64 | Structural changepoint flag (not used in this pipeline) |

**Features used for training:** the 8 sensor columns above (all except
`datetime`, `anomaly`, `changepoint`).

### Typical per-file anomaly ratio

Each file has roughly 35–37% anomaly timesteps. Anomalies are contiguous
blocks toward the end of each experiment run (not random).

---

## 4. Preprocessing Logic

### Step 1 — Load & Sort

Each CSV is read with `sep=';'`, the `datetime` column is parsed and the
DataFrame is sorted chronologically. This ensures correct temporal ordering
even if rows are out of order in the original file.

### Step 2 — Feature Extraction

Drop `datetime`, `anomaly`, `changepoint`. The remaining 8 sensor columns
become the feature matrix.

```
feature_cols = [Accelerometer1RMS, Accelerometer2RMS, Current,
                Pressure, Temperature, Thermocouple, Voltage,
                Volume Flow RateRMS]
```

### Step 3 — Sliding Window Sampling

A sliding window of size `WINDOW_SIZE = 20` steps is swept over the time series
with stride 1. For each window position `i`:

```
Input  X[i] = sensor data from step (i - 19) to step i     → shape (20, 8)
Future       = anomaly labels from step (i+1) to step (i+10) → 10 values
Label  y[i]  = 1 if ANY of those 10 future steps == 1, else 0
```

This is the **early-warning** formulation: the model sees the present and
must predict whether an anomaly is *about to happen*, not whether one is
happening right now.

**Diagram:**

```
time:  [... t-19  t-18 ... t-1   t ] [ t+1  t+2 ... t+10 ]
            |<-- window (input) -->|  |<-- horizon (label) -->|
                   X[i]                        y[i]
```

### Step 4 — Aggregation

Samples from all files within a split (train / val / test) are concatenated
along axis 0. Files are processed independently — no cross-file window is
created (each file is treated as a separate experiment).

### Step 5 — Save

The final arrays are saved to a single `.npz` file using `numpy.savez`.
No normalization or scaling is applied at this stage — that is left to
downstream model pipelines.

---

## 5. Split Strategy (Strategy A)

| Split | Source Files | # Samples | Positive (y=1) | Negative (y=0) |
|-------|-------------|-----------|----------------|----------------|
| Train | `valve1/0.csv` ~ `valve1/11.csv` (12 files) | 13,245 | 4,816 (36.4%) | 8,429 (63.6%) |
| Val   | `valve1/12.csv` ~ `valve1/15.csv` (4 files)  | 4,453  | 1,637 (36.8%) | 2,816 (63.2%) |
| Test  | `valve2/0.csv` ~ `valve2/3.csv` (4 files)    | 4,196  | 1,553 (37.0%) | 2,643 (63.0%) |
| **Total** | 20 files | **21,894** | **7,006 (32.0%)** | **13,888 (68.0%)** |

**Rationale for this split:**
- Train and Val come from the same physical valve (`valve1`) to allow the model
  to learn stable in-distribution patterns.
- Test comes from a **different valve** (`valve2`) to evaluate generalization
  across hardware units — a more realistic industrial scenario.
- Class ratio (~36% positive) is consistent across all three splits, which
  prevents distribution shift between splits.

---

## 6. Output Dataset Specification

**File:** `data/processed/dataset2/skab_strategyA_window20_horizon10.npz`  
**Size:** ~27 MB

| Key | Shape | dtype | Value Range |
|-----|-------|-------|-------------|
| `X_train` | (13245, 20, 8) | float64 | [-1.26, 255.32] |
| `y_train` | (13245,) | int64 | {0, 1} |
| `X_val` | (4453, 20, 8) | float64 | [-1.26, 255.11] |
| `y_val` | (4453,) | int64 | {0, 1} |
| `X_test` | (4196, 20, 8) | float64 | [-0.93, 255.17] |
| `y_test` | (4196,) | int64 | {0, 1} |
| `window_size` | scalar | int64 | 20 |
| `horizon` | scalar | int64 | 10 |

> **No normalization has been applied.** Raw sensor values are preserved.
> Downstream pipelines (ML or DL) are responsible for their own scaling.

---

## 7. How to Reproduce the Dataset

### Requirements

```
numpy
pandas
```

Install:
```bash
pip install numpy pandas
```

### Steps

```bash
# 1. Clone or download the raw SKAB data into the correct folder structure
#    (see Section 2 above)

# 2. Navigate to the SKAB directory (IMPORTANT — script uses relative paths)
cd path/to/capstone/SKAB

# 3. Run the preprocessing script
python Build_Dataset_SKAB_DLpipeline_By_David.py
```

### Expected output

```
============================================================
PROCESSING TRAIN SPLIT
============================================================
Processing: .\valve1\0.csv
  -> Generated samples: 1119
  ...

FINAL STRATEGY A DATASET SUMMARY
============================================================
Train samples: 13245 (60.50%)
Val samples:   4453 (20.34%)
Test samples:  4196 (19.17%)

Train positive ratio: 36.36%
Val positive ratio:   36.76%
Test positive ratio:  37.01%

Saved Strategy A dataset to: ../data/processed/dataset2/skab_strategyA_window20_horizon10.npz
```

### Config parameters (top of script)

To change window size or horizon, edit the constants at the top of
`Build_Dataset_SKAB_DLpipeline_By_David.py`:

```python
WINDOW_SIZE = 20   # number of past timesteps per sample
HORIZON     = 10   # number of future timesteps used to construct label
```

---

## 8. How to Use the Processed Dataset

### Load the data

```python
import numpy as np

data = np.load("data/processed/dataset2/skab_strategyA_window20_horizon10.npz")

X_train = data["X_train"]   # (13245, 20, 8)
y_train = data["y_train"]   # (13245,)
X_val   = data["X_val"]     # (4453,  20, 8)
y_val   = data["y_val"]     # (4453,)
X_test  = data["X_test"]    # (4196,  20, 8)
y_test  = data["y_test"]    # (4196,)
```

---

### For Classical ML (sklearn, XGBoost, etc.)

Flatten the time dimension first, then standardize:

```python
from sklearn.preprocessing import StandardScaler

# Flatten: (N, 20, 8) -> (N, 160)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# Standardize (fit only on train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled   = scaler.transform(X_val_flat)
X_test_scaled  = scaler.transform(X_test_flat)

# Train your model
model.fit(X_train_scaled, y_train)
model.predict(X_test_scaled)
```

---

### For Deep Learning — LSTM / GRU (PyTorch)

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

X_tr = torch.tensor(X_train, dtype=torch.float32)  # (N, 20, 8)
y_tr = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_tr, y_tr)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Inside model forward(): input shape is (batch, seq_len=20, input_size=8)
```

---

### For Deep Learning — 1D CNN (PyTorch)

CNN expects `(batch, channels, length)`, so permute:

```python
X_tr = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
# shape: (N, 8, 20)   →   (batch, in_channels=8, length=20)
```

---

### For Deep Learning — Transformer

```python
# Transformer encoder expects (batch, seq_len, d_model)
# X_train shape (N, 20, 8) is already in the correct format
X_tr = torch.tensor(X_train, dtype=torch.float32)
# (batch, seq_len=20, d_model=8)
```

---

### Handling class imbalance

The dataset has ~36% positive samples. Recommended strategies:

```python
# Option 1: class_weight in sklearn
model = RandomForestClassifier(class_weight="balanced")

# Option 2: pos_weight in PyTorch BCEWithLogitsLoss
pos_weight = torch.tensor([neg / pos])   # ~1.75
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Option 3: XGBoost scale_pos_weight
model = XGBClassifier(scale_pos_weight=neg/pos)
```

---

## 9. Important Notes & Caveats

### No data leakage
- The `StandardScaler` (if used) must be **fit only on the training set** and
  then applied to val/test. Never fit on val or test.
- Windows do not cross file boundaries. The last window of file N and the
  first window of file N+1 are independent samples.

### Label definition
`y = 1` means "an anomaly will occur **within the next 10 timesteps**", NOT
that the current timestep is anomalous. This is intentional for early warning.
If you want current-timestep detection, you need to re-run the preprocessing
script with `HORIZON = 0` and change the label logic.

### No feature normalization in the .npz
Raw sensor values have very different scales (e.g., Voltage ~200V vs.
Accelerometer ~0.02). Always normalize before training. Use the scaler fitted
on training data only.

### Temporal structure
This is time-series data. Do **not** shuffle across files. The current
pipeline processes each file independently, so within-file ordering is
respected. Shuffling **samples** from the same file during training is fine.

### changepoint column
The `changepoint` column (marks the exact moment an anomaly begins) is
**dropped** during preprocessing and not included in features or labels.
It is metadata only.

### valve2 as test set
`valve2` experiments may have slightly different sensor characteristics than
`valve1`. Test scores are expected to be marginally lower than val scores —
this is by design (cross-hardware generalization test).

---

## 10. Classical ML Baseline Reference

A classical ML baseline is provided in `SKAB_ClassicalML_Baseline_By_David.py`.
Run it **after** generating the `.npz` file. It trains 7 models on the
flattened (N, 160) features and saves all model weights.

**Run from inside the `SKAB/` directory:**

```bash
python SKAB_ClassicalML_Baseline_By_David.py
```

**Results on test set (valve2):**

| Model | Accuracy | Anomaly Precision | Anomaly Recall | Anomaly F1 |
|-------|----------|-------------------|----------------|------------|
| Logistic Regression | 93.49% | 0.9755 | 0.8455 | 0.9058 |
| XGBoost | 92.61% | **0.9944** | 0.8049 | 0.8897 |
| Gradient Boosting | 89.87% | 0.8811 | 0.8397 | 0.8599 |
| Random Forest | 89.25% | 0.8635 | 0.8429 | 0.8530 |
| Extra Trees | 88.18% | 0.8989 | 0.7669 | 0.8277 |
| KNN | 80.62% | 0.8768 | 0.5544 | 0.6793 |
| SVM (RBF) | 75.79% | 0.9409 | 0.3690 | 0.5301 |

Saved model weights location:
```
data/processed/dataset2/skab_classical_models/
├── scaler.pkl      ← StandardScaler fitted on X_train (flattened)
├── model_lr.pkl
├── model_rf.pkl
├── model_svm.pkl
├── model_et.pkl
├── model_gb.pkl
├── model_knn.pkl
└── model_xgb.pkl
```

Load a saved model:
```python
import joblib
import numpy as np

scaler = joblib.load("data/processed/dataset2/skab_classical_models/scaler.pkl")
model  = joblib.load("data/processed/dataset2/skab_classical_models/model_xgb.pkl")

X_test_flat   = X_test.reshape(X_test.shape[0], -1)
X_test_scaled = scaler.transform(X_test_flat)
y_pred        = model.predict(X_test_scaled)
```
