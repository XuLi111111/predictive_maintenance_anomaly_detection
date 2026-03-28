# Data Preprocessing Handoff
## CITS5206 Group 14 — Water Pump Predictive Maintenance

**Prepared by:** Data Preprocessing Team
**Dataset:** Zenodo University Water Supply System (Jul 2023 – Jun 2024)
**Task:** Binary fault classification — predict pump failure 30 minutes in advance

---

## Files to Use

```
processed/
├── zenodo_pump_2d/          ← USE THIS for traditional ML models
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── zenodo_pump/             ← USE THIS for deep learning (LSTM/CNN/Transformer)
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
└── zenodo_pump/
    └── scaler.pkl           ← StandardScaler fitted on training set (for inference)
```

> **If you are training LR / SVM / KNN / Random Forest / Extra Trees / XGBoost / AdaBoost → use `zenodo_pump_2d/`**
> **If you are training LSTM / CNN / Transformer → use `zenodo_pump/` (3D)**

---

## Quick Start

### Load the data (2D — traditional ML)

```python
import numpy as np

train = np.load("processed/zenodo_pump_2d/train.npz", allow_pickle=True)
val   = np.load("processed/zenodo_pump_2d/val.npz",   allow_pickle=True)
test  = np.load("processed/zenodo_pump_2d/test.npz",  allow_pickle=True)

X_train, y_train = train["X"], train["y"]   # (854218, 240), (854218,)
X_val,   y_val   = val["X"],   val["y"]     # (256737, 240), (256737,)
X_test,  y_test  = test["X"],  test["y"]    # (613895, 240), (613895,)

feature_cols = train["feature_cols"]        # column names: pwr_p1_t0 ... water_level_t29
times_train  = train["times"]               # ISO timestamp of each window's last minute
```

### Load the data (3D — deep learning)

```python
train = np.load("processed/zenodo_pump/train.npz", allow_pickle=True)

X_train = train["X"]   # (854218, 30, 8) — (samples, time_steps, features)
y_train = train["y"]   # (854218,)
```

### Load the scaler (for inference on new data)

```python
import pickle

with open("processed/zenodo_pump/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

---

## Dataset Overview

| Property | Value |
|---|---|
| Source | Zenodo University Water Supply Sensor Dataset |
| Time range | 2023-07-01 → 2024-06-30 |
| Pumps | 4 (NeWater Pump 1 & 2, Potable Pump 1 & 2) |
| Raw files | 48 CSV files, ~1.5 GB |
| Sampling rate (raw) | ~20 seconds (irregular) |
| Sampling rate (processed) | 1 minute (resampled) |
| Task | Binary classification: 0 = Normal, 1 = Fault imminent |
| Warning horizon | 30 minutes before fault onset |

---

## Label Definition

### What do the labels mean?

| Label | Meaning |
|---|---|
| `0` | Normal operation — motor is on, no fault imminent |
| `1` | **Fault imminent** — sensor readings in the 30 minutes immediately before a fault onset |

### How were labels assigned?

The raw dataset has **no label column**. Fault information is only provided as day-level annotations in the dataset documentation. The following process was used to produce minute-level labels:

**Step 1 — Identify fault days**

Fault dates were taken directly from the dataset documentation:

- **NeWater system:** 2023-08-06; 2023-09-08 through 2023-10-26 (most days); 2024-03-13
- **Potable system:** 2023-09-06; 2023-09-09; 2023-12-27

**Step 2 — Detect fault onset time T per fault day (hybrid method)**

Since only the date of fault was known (not the exact time), a fault onset proxy **T** was detected for each fault day:

- **Primary — Water level minimum:** On PLC hardware fault days, the pump fails to activate when the tank water drops to the low-level threshold. The tank therefore drains to its daily minimum water level. `T = timestamp of minimum water level` on that fault day.
  *Why this works:* The NeWater PLC faults (Sep–Oct 2023) involve the pump never activating, so power remains low all day. The only observable anomaly is the tank draining to an abnormally low level — this is the clearest signal.

- **Fallback — Power drop:** If water level data is unavailable or the minimum is at the very start of the day (sensor artefact), `T = last timestamp where total pump power drops below 50% of the day's maximum power`. This catches mechanical shutdowns where the motor stops mid-operation.

**Step 3 — Assign labels**

| Condition | Action |
|---|---|
| Motor on + timestamp in `[T − 30 min, T)` on fault day | Label = **1** |
| Motor on + any other time | Label = **0** |
| Motor completely off (total power ≤ 5 W) | **Excluded** |
| Maintenance period (2023-08-31 to 2023-09-06) | **Excluded** |
| Post-fault period (timestamp ≥ T on fault day) | **Excluded** |
| Fault day with no detectable T signal | **Excluded** |

**Why exclude downtime?** A stopped motor produces no meaningful sensor readings — all features are near zero or noise. Including these rows would pollute the Normal class with trivially easy samples and mislead the model.

**Why exclude post-fault?** After the fault onset, the system is in an unknown or recovery state. These readings don't represent either normal operation or the pre-fault warning state, so they add noise to both classes.

**Why exclude maintenance?** The 2023-08-31 to 2023-09-06 tank washing event deliberately causes abnormal water level readings. Labelling these as Normal would confuse the model.

---

## Feature Engineering

### Final features (8 columns)

| Feature | Source sensor | Unit | Engineering |
|---|---|---|---|
| `pwr_p1` | Power Sensor channel 1 | W | 1-min mean |
| `pwr_p3` | Power Sensor channel 3 | W | 1-min mean |
| `cur_i1` | Current Sensor channel 1 | A | 1-min mean |
| `energy_delta` | Energy Sensor (all 3 channels) | kWh/min | Cumulative kWh → per-minute delta (`.diff().clip(0)`) |
| `vib_speed_rms` | Vibration Speed X, Y, Z axes | mm/s | Combined as `√((X²+Y²+Z²)/3)` |
| `vib_disp_rms` | Vibration Displacement X, Y, Z axes | µm | Combined as `√((X²+Y²+Z²)/3)` |
| `vib_temp` | Vibration sensor temperature | °C | 1-min mean |
| `water_level` | Tank water level sensor | mm | 1-min mean |

### Why these features?

- **Power (P1, P3):** Motor load directly reflects pump operation state. Anomalous power patterns (drops, spikes, irregular cycling) are primary indicators of faults.
- **Current (I1):** Correlated with motor load; captures electrical anomalies not visible in power alone.
- **Energy delta:** Converts a cumulative counter into a meaningful per-minute consumption rate. Abnormal energy consumption is a fault indicator.
- **Vibration RMS:** Mechanical wear, imbalance, and bearing degradation all manifest as increased vibration. Three axes are collapsed to RMS to reduce dimensionality while preserving magnitude.
- **Vibration temperature:** Bearing/motor heat buildup is a precursor to mechanical failure.
- **Water level:** Key for detecting PLC-type faults where the pump fails to activate and the tank drains to an abnormally low level.

### What was dropped and why?

| Dropped | Reason |
|---|---|
| `press_in` (Incoming Pressure) | 27.3% missing values. This level of NaN would cause the majority of sliding windows to be discarded, drastically reducing the training set size. |
| `press_out` (Outgoing Pressure) | Same reason — 27.3% NaN. |
| `pwr_p2`, `cur_i2`, `cur_i3` | Highly correlated with kept channels. Dropping them reduces dimensionality without meaningful information loss. |

### Resampling

Raw sensor files record at irregular intervals (~20-second granularity). All signals were resampled to **1-minute means** before merging. This ensures:
- A single, consistent timeline across all sensors
- Natural alignment with the 30-minute window and warning horizon
- Robustness to short sensor dropouts within a minute

### Missing value imputation

After resampling and merging, remaining NaN values (from sensor dropouts, communication errors) were handled in three steps:

1. **Forward-fill up to 60 minutes** — carries the last valid reading forward through short outages
2. **Backward-fill up to 60 minutes** — fills gaps at the beginning of a sensor's record
3. **Column median fill** — fills any remaining NaN with the column's global median, ensuring zero NaN in the final dataset

After imputation, residual NaN across all pumps was **0.0%**.

---

## Sliding Window Construction

### Why sliding windows?

Traditional ML models expect a fixed-size feature vector per sample. A sliding window converts the time series into supervised learning samples, each representing one 30-minute observation period ending at the prediction point.

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Window size | 30 steps = 30 minutes | Matches the 30-minute pre-fault warning horizon |
| Stride | 1 minute | Maximum data utilisation |
| Max time gap | 5 minutes | Windows spanning sensor outages > 5 min are discarded |

### How windows are labelled

- **Label = label of the last (most recent) timestep** in the window
- A window is labelled `1` if its final minute falls within the 30-minute pre-fault zone `[T − 30min, T)`
- This means: "given the past 30 minutes of readings, will the pump fail in the next moment?"

### 2D vs 3D shape

| Format | Shape | Interpretation |
|---|---|---|
| 3D (deep learning) | `(N, 30, 8)` | N samples, 30 time steps, 8 features per step |
| 2D (traditional ML) | `(N, 240)` | N samples, 240 flattened features (`pwr_p1_t0` ... `water_level_t29`) |

The 2D format simply flattens the time dimension: each feature at each time step becomes a separate column. Column names follow the pattern `{feature_name}_t{timestep}`, e.g. `pwr_p1_t0` is the power channel 1 reading 30 minutes ago, `pwr_p1_t29` is the reading at the prediction moment.

---

## Dataset Split

### Split boundaries

| Split | Period | Windows | Fault windows | Purpose |
|---|---|---|---|---|
| **Train** | 2023-07-01 → 2023-12-31 | 854,218 | 2,529 (0.30%) | Model training — contains the main Sep–Oct 2023 fault cluster |
| **Val** | 2024-01-01 → 2024-02-29 | 256,737 | 0 (0.00%) | Hyperparameter tuning — fault-free period, use to evaluate false positive rate |
| **Test** | 2024-03-01 → 2024-06-30 | 613,895 | 30 (0.005%) | Final evaluation — contains 2024-03-13 NeWater fault |

### Why chronological split (not random)?

Time series data has temporal dependency. Random splitting would leak future data into the training set, producing unrealistically optimistic evaluation metrics. A chronological split simulates real deployment: the model is trained on historical data and evaluated on future unseen events.

### Why is val fault count = 0?

January–February 2024 had no documented fault events. This is intentional — the validation set serves as a **false positive rate benchmark**. A good model should predict very few `1`s on this period. If your model shows high recall on the test set but also high false positives on val, you should tune the classification threshold.

### Scaling

A `StandardScaler` was fitted **on the training set only**, then applied to all three splits. This prevents data leakage from val/test statistics into the scaler. The fitted scaler is saved as `scaler.pkl` and must be used when running inference on new data.

---

## Class Imbalance — Important

Fault samples represent only **0.15%** of all windows. If you train without addressing this, most models will achieve ~99.85% accuracy by simply predicting 0 for every sample — this is useless for fault detection.

### Recommended solutions

**For sklearn models (LR, SVM, RF, Extra Trees, AdaBoost):**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced', n_estimators=100)
```

**For XGBoost:**
```python
from xgboost import XGBClassifier

ratio = (y_train == 0).sum() / (y_train == 1).sum()  # ≈ 337
model = XGBClassifier(scale_pos_weight=ratio)
```

**For KNN and AdaBoost (do not support class_weight):**
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
# then train on X_train_res, y_train_res
```

### Recommended evaluation metrics

Do **not** use accuracy as the primary metric. Use:

| Metric | Why |
|---|---|
| **Recall (sensitivity)** | Most important — did we catch the fault? Missing a fault is costly |
| **Precision** | How many of our fault alerts were real? |
| **F1-score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Threshold-independent ranking performance |
| **PR-AUC** | Better than ROC-AUC under severe class imbalance |

```python
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Normal", "Fault"]))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
```

---

## Full Training Template

```python
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ── Load data ──────────────────────────────────────────────
train = np.load("processed/zenodo_pump_2d/train.npz", allow_pickle=True)
val   = np.load("processed/zenodo_pump_2d/val.npz",   allow_pickle=True)
test  = np.load("processed/zenodo_pump_2d/test.npz",  allow_pickle=True)

X_train, y_train = train["X"], train["y"]
X_val,   y_val   = val["X"],   val["y"]
X_test,  y_test  = test["X"],  test["y"]

# ── Model definitions ──────────────────────────────────────
ratio = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "SVM":                 SVC(class_weight="balanced", probability=True),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),        # use SMOTE before fitting
    "Random Forest":       RandomForestClassifier(class_weight="balanced", n_estimators=100),
    "Extra Trees":         ExtraTreesClassifier(class_weight="balanced", n_estimators=100),
    "XGBoost":             XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss"),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100),       # use SMOTE before fitting
}

# ── Train & evaluate ───────────────────────────────────────
for name, model in models.items():
    print(f"\n{'='*50}\n{name}")
    model.fit(X_train, y_train)

    for split_name, Xs, ys in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        y_pred = model.predict(Xs)
        y_prob = model.predict_proba(Xs)[:, 1] if hasattr(model, "predict_proba") else None

        print(f"\n  [{split_name}]")
        print(classification_report(ys, y_pred, target_names=["Normal", "Fault"]))
        if y_prob is not None and ys.sum() > 0:
            print(f"  ROC-AUC: {roc_auc_score(ys, y_prob):.4f}")
```

---

## Summary of All Preprocessing Decisions

| Decision | Choice | Reason |
|---|---|---|
| Resampling | 1-minute mean | Aligns all sensors; natural unit for 30-min window |
| Energy feature | Per-minute delta | Cumulative kWh is non-stationary; delta captures consumption rate |
| Vibration axes | RMS across X/Y/Z | Reduces 6 columns to 2 without losing magnitude information |
| Pressure sensors | Dropped | 27.3% NaN — too high to recover without distorting signal |
| Motor-off rows | Excluded | Zero/noise readings; trivially easy for model, misleads training |
| Maintenance period | Excluded | Deliberate anomaly unrelated to pump faults |
| Post-fault period | Excluded | Unknown recovery state; not representative of either class |
| Fault onset T | Hybrid (water level min + power drop) | Power-only fails for PLC faults where motor never activates |
| Label window | [T−30min, T) | Directly targets the 30-minute advance warning goal |
| NaN imputation | ffill(60min) + median fill | Preserves windows that would otherwise be discarded |
| Split | Chronological | Prevents temporal data leakage; simulates real deployment |
| Scaler | StandardScaler fit on train only | Prevents leakage of val/test statistics into training |
| Class imbalance | `class_weight='balanced'` / `scale_pos_weight` | Fault = 0.15% of data; naive model predicts all-Normal |
