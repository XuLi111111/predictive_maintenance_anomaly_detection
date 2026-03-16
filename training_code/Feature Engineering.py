import pandas as pd
import numpy as np
from scipy.stats import linregress

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("sensor_processed.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Sensor columns (exclude timestamp, original label text column, numeric label column)
sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

# Confirm numeric label column name (mapped column during cleaning)
LABEL_COL = "machine_status"   # ← Change this if your saved column name is different

# ── 2. Parameters ───────────────────────────────────────────────────
WINDOW     = 30   # Input window (minutes)
LOOK_AHEAD_MIN = 5    # Prediction interval start
LOOK_AHEAD_MAX = 30   # Prediction interval end

# ── 3. Look-ahead label function ───────────────────────────────────────────
def lookahead_label(labels_ahead):
    """
    labels_ahead: Original label sequence (0/1/2) in the future [T+5, T+30] interval
    Priority: BROKEN(2) > RECOVERING(1) > NORMAL(0)
    """
    if 2 in labels_ahead.values:
        return 2
    elif 1 in labels_ahead.values:
        return 1
    else:
        return 0

# ── 4. Feature extraction function ───────────────────────────────────────────
def extract_features(window: pd.DataFrame) -> dict:
    feats = {}
    for col in sensor_cols:
        s = window[col].values
        feats[f"{col}_mean"] = np.mean(s)
        feats[f"{col}_std"]  = np.std(s)
        feats[f"{col}_min"]  = np.min(s)
        feats[f"{col}_max"]  = np.max(s)
        # Trend slope
        slope, *_ = linregress(np.arange(len(s)), s)
        feats[f"{col}_slope"] = slope
    return feats

# ── 5. Sliding window main loop ─────────────────────────────────────────
records = []
n = len(df)

for i in range(WINDOW, n - LOOK_AHEAD_MAX):
    # Input window: [i-30, i)
    window = df.iloc[i - WINDOW : i]

    # Look-ahead interval: [i+5, i+30]
    ahead  = df.iloc[i + LOOK_AHEAD_MIN : i + LOOK_AHEAD_MAX + 1]

    if len(ahead) < (LOOK_AHEAD_MAX - LOOK_AHEAD_MIN):
        continue  # Insufficient boundary, skip

    feats = extract_features(window)
    feats["timestamp"] = df.iloc[i]["timestamp"]   # Window end time
    feats["label"]     = lookahead_label(ahead[LABEL_COL])

    records.append(feats)

# ── 6. Output ───────────────────────────────────────────────────
result = pd.DataFrame(records)

# Put timestamp in the first column, label in the last
cols = ["timestamp"] + [c for c in result.columns if c not in ("timestamp", "label")] + ["label"]
result = result[cols]

print(f"Feature matrix shape: {result.shape}")
print(f"Label distribution:\n{result['label'].value_counts().sort_index()}")

result.to_csv("sensor_features.csv", index=False)
print("Saved as sensor_features.csv")