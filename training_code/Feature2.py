import pandas as pd
import numpy as np

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("sensor_processed.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
LABEL_COL = "label"

WINDOW         = 30
LOOK_AHEAD_MIN = 5
LOOK_AHEAD_MAX = 30

print("Start extracting features...")

# ── 2. Vectorized statistical features ─────────────────────────
feature_frames = []

for col in sensor_cols:
    s = df[col]
    feature_frames.append(s.rolling(WINDOW).mean().rename(f"{col}_mean"))
    feature_frames.append(s.rolling(WINDOW).std().rename(f"{col}_std"))
    feature_frames.append(s.rolling(WINDOW).min().rename(f"{col}_min"))
    feature_frames.append(s.rolling(WINDOW).max().rename(f"{col}_max"))
    # Slope: approximated by (last - first) / window, sufficient and fast
    feature_frames.append(
        (s - s.shift(WINDOW - 1)).divide(WINDOW - 1).rename(f"{col}_slope")
    )

features = pd.concat(feature_frames, axis=1)
features["timestamp"] = df["timestamp"]

print("Feature extraction completed, starting to generate look-ahead labels...")

# ── 3. Look-ahead labels (vectorized) ─────────────────────────
labels = df[LABEL_COL].values
n = len(labels)
lookahead_labels = np.zeros(n, dtype=int)

for i in range(n):
    end = min(i + LOOK_AHEAD_MAX + 1, n)
    start = min(i + LOOK_AHEAD_MIN, n)
    ahead = labels[start:end]
    if len(ahead) == 0:
        lookahead_labels[i] = -1  # Boundary, will be removed later
        continue
    if 2 in ahead:
        lookahead_labels[i] = 2
    elif 1 in ahead:
        lookahead_labels[i] = 1
    else:
        lookahead_labels[i] = 0

features["label"] = lookahead_labels

# ── 4. Clean up boundary rows ─────────────────────────────────
result = features.dropna().copy()
result = result[result["label"] != -1]
result = result.reset_index(drop=True)

# timestamp as first column, label as last
cols = ["timestamp"] + [c for c in result.columns if c not in ("timestamp", "label")] + ["label"]
result = result[cols]

print(f"Feature matrix shape: {result.shape}")
print(f"Label distribution:\n{result['label'].value_counts().sort_index()}")

result.to_csv("sensor_features.csv", index=False)
print("Saved as sensor_features.csv")