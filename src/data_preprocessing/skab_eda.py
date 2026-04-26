"""
SKAB (Skoltech Anomaly Benchmark) - Exploratory Data Analysis
=============================================================
Group 14 | CITS5206 Capstone | Predictive Maintenance Project
Author : David Du (PM)
Date   : April 2026

Directory layout (run this script from inside SKAB/):
    SKAB/
        anomaly-free/anomaly-free.csv   <- no label columns
        valve1/0.csv ~ 15.csv           <- anomaly + changepoint labels
        valve2/0.csv ~ 3.csv
        other/9.csv ~ 23.csv

Strategy A split (matches Build_Dataset_SKAB_DLpipeline_By_David.py):
    Train = valve1/0-11.csv
    Val   = valve1/12-15.csv
    Test  = valve2/0-3.csv

Requirements:
    pip install pandas numpy matplotlib seaborn scipy
"""

import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# Force UTF-8 output so special characters render correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────

SKAB_ROOT  = "."            # Script must be run from inside SKAB/
OUTPUT_DIR = "./eda_output" # All figures and CSVs are saved here
WINDOW_SIZE = 20            # Sliding window steps (matches DL pipeline)
HORIZON     = 10            # Forecast horizon steps (matches DL pipeline)

LABEL_COL  = "anomaly"
CHANGE_COL = "changepoint"

# Strategy A split definition
TRAIN_FILES = [os.path.join("valve1", f"{i}.csv") for i in range(12)]
VAL_FILES   = [os.path.join("valve1", f"{i}.csv") for i in range(12, 16)]
TEST_FILES  = [os.path.join("valve2", f"{i}.csv") for i in range(4)]

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


# ── Data Loading ───────────────────────────────────────────────────────────────

def _get_split(rel_path: str) -> str:
    """Map a relative file path to its Strategy A split label."""
    if rel_path in TRAIN_FILES:
        return "train"
    if rel_path in VAL_FILES:
        return "val"
    if rel_path in TEST_FILES:
        return "test"
    return "other"


def load_csv(fp: str, root: str) -> pd.DataFrame:
    """Load one SKAB CSV and attach metadata columns.

    anomaly-free.csv has no label columns, so they are filled with 0
    to keep a consistent schema across all files.
    """
    df = pd.read_csv(fp, sep=";", parse_dates=["datetime"],
                     index_col="datetime", low_memory=False)
    for col in [LABEL_COL, CHANGE_COL]:
        if col not in df.columns:
            df[col] = 0
    df["source"] = os.path.relpath(fp, root)
    df["subset"] = os.path.basename(os.path.dirname(fp))
    df["split"]  = _get_split(os.path.relpath(fp, root))
    return df


def load_skab(root: str):
    """Recursively load all CSVs under root, excluding any eda_output files."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    files = [f for f in files if "eda_output" not in f.replace("\\", "/")]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {root}")
    combined = pd.concat([load_csv(f, root) for f in files]).sort_index()
    print(f"Loaded {len(files)} files | shape = {combined.shape}")
    return combined, files


print("=" * 70)
print("  SKAB EDA — Group 14 CITS5206 Capstone")
print("=" * 70)

df_all, file_list = load_skab(SKAB_ROOT)

# Derive feature columns automatically: all numeric columns except labels
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
label_cols   = [c for c in [LABEL_COL, CHANGE_COL] if c in numeric_cols]
feature_cols = [c for c in numeric_cols if c not in label_cols]

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"Label columns: {label_cols}")


# ── Section 1: Descriptive Statistics ─────────────────────────────────────────
print("\n" + "─" * 70)
print("Section 1: Descriptive Statistics")
print("─" * 70)

print(f"\nTotal samples : {len(df_all):,}")
print(f"Time range    : {df_all.index.min()} -> {df_all.index.max()}")
print(f"File count    : {len(file_list)}")

missing_pct = df_all[feature_cols].isnull().mean().mul(100).round(2)
print("\nMissing values (%):")
print(missing_pct[missing_pct > 0].to_string() if missing_pct.any() else "  None (OK)")

print("\nDescriptive statistics (sensor features):")
desc = df_all[feature_cols].describe().T[["mean", "std", "min", "50%", "max"]]
desc.columns = ["mean", "std", "min", "median", "max"]
print(desc.round(4).to_string())
desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))

print("\nSamples per subset:")
for subset, cnt in df_all.groupby("subset").size().sort_values(ascending=False).items():
    print(f"  {subset:<20}: {cnt:>8,}")


# ── Section 2: Strategy A Split Distribution ───────────────────────────────────
print("\n" + "─" * 70)
print("Section 2: Strategy A Train / Val / Test Split")
print("─" * 70)

split_counts = (
    df_all.groupby("split")
    .agg(rows=("split", "size"), anomaly_rows=(LABEL_COL, "sum"))
    .assign(anomaly_pct=lambda x: x["anomaly_rows"] / x["rows"] * 100)
    .reindex([s for s in ["train", "val", "test", "other"] if s in df_all["split"].unique()])
)

print(f"\n{'Split':<10} {'Rows':>10} {'Anomaly Rows':>14} {'Anomaly %':>12}")
print("  " + "-" * 46)
for split, row in split_counts.iterrows():
    print(f"  {split:<10} {int(row['rows']):>10,} {int(row['anomaly_rows']):>14,} {row['anomaly_pct']:>11.2f}%")

colors_map   = {"train": "#4C9BE8", "val": "#F5A623", "test": "#7ED321", "other": "#9B9B9B"}
split_labels = split_counts.index.tolist()
split_colors = [colors_map.get(s, "#aaa") for s in split_labels]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(split_labels, split_counts["rows"], color=split_colors, edgecolor="white")
axes[0].set_title("Sample Count by Split")
axes[0].set_ylabel("Rows")
for i, (_, row) in enumerate(split_counts.iterrows()):
    axes[0].text(i, row["rows"] * 1.01, f"{int(row['rows']):,}", ha="center", fontsize=8)

axes[1].bar(split_labels, split_counts["anomaly_pct"], color=split_colors, edgecolor="white")
axes[1].set_title("Anomaly % by Split")
axes[1].set_ylabel("Anomaly Percentage (%)")
for i, (_, row) in enumerate(split_counts.iterrows()):
    axes[1].text(i, row["anomaly_pct"] + 0.3, f"{row['anomaly_pct']:.1f}%", ha="center", fontsize=8)

plt.suptitle("Strategy A: valve1/0-11=train  valve1/12-15=val  valve2/0-3=test", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_split_distribution.png"))
plt.close()
print("\n  -> Saved: fig1_split_distribution.png")


# ── Section 3: Anomaly Label Distribution ─────────────────────────────────────
print("\n" + "─" * 70)
print("Section 3: Anomaly Label Distribution")
print("─" * 70)

counts = df_all[LABEL_COL].value_counts().sort_index()
total  = len(df_all)
print(f"\n  Normal  (0): {counts.get(0, 0):>8,}  ({counts.get(0, 0) / total * 100:.2f}%)")
print(f"  Anomaly (1): {counts.get(1, 0):>8,}  ({counts.get(1, 0) / total * 100:.2f}%)")
print(f"  Imbalance ratio: {counts.get(0, 1) / max(counts.get(1, 1), 1):.1f}:1  (normal:anomaly)")

# Build per-file anomaly stats for the bar chart
file_stats = []
for fp in file_list:
    rel = os.path.relpath(fp, SKAB_ROOT)
    sub = df_all[df_all["source"] == rel]
    file_stats.append({
        "file":        os.path.join(os.path.basename(os.path.dirname(fp)), os.path.basename(fp)),
        "subset":      os.path.basename(os.path.dirname(fp)),
        "anomaly_pct": sub[LABEL_COL].mean() * 100 if len(sub) else 0,
    })
fs_df = pd.DataFrame(file_stats).sort_values(["subset", "file"])

subset_palette = {"valve1": "#4C9BE8", "valve2": "#7ED321", "other": "#F5A623", "anomaly-free": "#9B9B9B"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].pie([counts.get(0, 0), counts.get(1, 0)],
            labels=["Normal", "Anomaly"], autopct="%1.2f%%",
            colors=["#4C9BE8", "#E84C4C"], startangle=90,
            wedgeprops=dict(edgecolor="white"))
axes[0].set_title("Overall Anomaly Distribution")

bar_colors = [subset_palette.get(row["subset"], "#aaa") for _, row in fs_df.iterrows()]
axes[1].barh(fs_df["file"], fs_df["anomaly_pct"], color=bar_colors, edgecolor="white")
axes[1].set_xlabel("Anomaly Percentage (%)")
axes[1].set_title("Anomaly % per File (color = subset)")
axes[1].invert_yaxis()
legend_patches = [mpatches.Patch(facecolor=c, label=s)
                  for s, c in subset_palette.items() if s in fs_df["subset"].values]
axes[1].legend(handles=legend_patches, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_label_distribution.png"))
plt.close()
print("\n  -> Saved: fig2_label_distribution.png")


# ── Section 4: Sensor Time Series (valve1/1.csv) ───────────────────────────────
print("\n" + "─" * 70)
print("Section 4: Sensor Time Series (valve1/1.csv)")
print("─" * 70)

# Use valve1/1.csv as a representative sample; fall back to first file if missing
sample_rel = os.path.join("valve1", "1.csv")
df_sample  = df_all[df_all["source"] == sample_rel].copy()
if df_sample.empty:
    sample_rel = os.path.relpath(file_list[0], SKAB_ROOT)
    df_sample  = df_all[df_all["source"] == sample_rel].copy()
print(f"  File: {sample_rel}  (rows={len(df_sample)})")

anomaly_mask    = df_sample[LABEL_COL] == 1
changepoint_idx = df_sample.index[df_sample[CHANGE_COL] == 1]

fig, axes = plt.subplots(len(feature_cols), 1,
                         figsize=(14, 2.4 * len(feature_cols)), sharex=True)
if len(feature_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, feature_cols):
    ax.plot(df_sample.index, df_sample[col], linewidth=0.7, color="#4C9BE8", label=col)
    ymin, ymax = df_sample[col].min(), df_sample[col].max()
    pad = (ymax - ymin) * 0.05 or 0.1
    ax.set_ylim(ymin - pad, ymax + pad)
    # Shade anomaly regions in red
    if anomaly_mask.any():
        ax.fill_between(df_sample.index, ymin - pad, ymax + pad,
                        where=anomaly_mask, alpha=0.20, color="red", label="Anomaly")
    # Mark changepoints with orange dashed vertical lines
    for cp_t in changepoint_idx:
        ax.axvline(cp_t, color="orange", linewidth=0.8, alpha=0.8, linestyle="--")
    ax.set_ylabel(col, fontsize=8)
    ax.legend(loc="upper right", fontsize=7)

# Append changepoint to the first subplot's legend only when present
if len(changepoint_idx) > 0:
    cp_line = mlines.Line2D([], [], color="orange", linestyle="--",
                            linewidth=0.8, label="Changepoint")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles + [cp_line], labels=labels + ["Changepoint"],
                   loc="upper right", fontsize=7)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.suptitle(f"Sensor Time Series — {sample_rel}  (red=anomaly, dashed=changepoint)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_timeseries.png"))
plt.close()
print("  -> Saved: fig3_timeseries.png")


# ── Section 5: Feature Distribution — Normal vs Anomaly ───────────────────────
print("\n" + "─" * 70)
print("Section 5: Feature Distribution — Normal vs Anomaly")
print("─" * 70)

df_normal = df_all[df_all[LABEL_COL] == 0][feature_cols]
df_anom   = df_all[df_all[LABEL_COL] == 1][feature_cols]

cols_per_row = 4
rows = (len(feature_cols) + cols_per_row - 1) // cols_per_row
fig, _ = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 3.5))
axes = np.array(fig.axes).flatten()

for i, col in enumerate(feature_cols):
    axes[i].hist(df_normal[col].dropna(), bins=50, alpha=0.6, color="#4C9BE8",
                 label="Normal", density=True)
    axes[i].hist(df_anom[col].dropna(), bins=50, alpha=0.6, color="#E84C4C",
                 label="Anomaly", density=True)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Density")
    axes[i].legend(fontsize=7)

for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distribution: Normal vs Anomaly", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_feature_distribution.png"))
plt.close()
print("  -> Saved: fig4_feature_distribution.png")

# Mann-Whitney U test: check whether each feature differs significantly between classes
print("\nMann-Whitney U test (p < 0.05 = significant difference):")
sig_results = []
for col in feature_cols:
    a, b = df_normal[col].dropna().values, df_anom[col].dropna().values
    if len(a) > 0 and len(b) > 0:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        sig_results.append({"feature": col, "p_value": round(p, 6),
                             "significant": "Yes" if p < 0.05 else "No"})
sig_df = pd.DataFrame(sig_results).sort_values("p_value")
print(sig_df.to_string(index=False))
sig_df.to_csv(os.path.join(OUTPUT_DIR, "feature_significance.csv"), index=False)


# ── Section 6: Correlation Heatmap — All Data vs Anomaly Only ─────────────────
print("\n" + "─" * 70)
print("Section 6: Correlation Heatmap")
print("─" * 70)

corr_cols = feature_cols + [LABEL_COL]
fig, axes = plt.subplots(1, 2, figsize=(max(12, len(corr_cols) * 1.1),
                                        max(5, len(corr_cols) * 0.75)))
for ax, (title, subset_df) in zip(axes, [
    ("All Data",     df_all),
    ("Anomaly Only", df_all[df_all[LABEL_COL] == 1]),
]):
    corr = subset_df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"Correlation Matrix — {title}", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_correlation_heatmap.png"))
plt.close()
print("  -> Saved: fig5_correlation_heatmap.png")


# ── Section 7: Sensor Distribution by Subset (Boxplot) ────────────────────────
print("\n" + "─" * 70)
print("Section 7: Sensor Distribution by Subset")
print("─" * 70)

# Exclude anomaly-free since it has no real labels and skews the comparison
df_labeled = df_all[df_all["subset"].isin(["valve1", "valve2", "other"])]

cols_per_row = 4
rows = (len(feature_cols) + cols_per_row - 1) // cols_per_row
fig, _ = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 3.5))
axes = np.array(fig.axes).flatten()

for i, col in enumerate(feature_cols):
    groups = [df_labeled[df_labeled["subset"] == s][col].dropna().values
              for s in ["valve1", "valve2", "other"]]
    bp = axes[i].boxplot(groups, labels=["valve1", "valve2", "other"],
                         patch_artist=True,
                         medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], ["#4C9BE8", "#7ED321", "#F5A623"]):
        patch.set_facecolor(color)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_ylabel("Value")

for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Sensor Distribution by Subset (valve1 / valve2 / other)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_subset_boxplot.png"))
plt.close()
print("  -> Saved: fig6_subset_boxplot.png")


# ── Section 8: Sliding Window Feasibility Analysis ────────────────────────────
print("\n" + "─" * 70)
print(f"Section 8: Sliding Window Analysis  (window={WINDOW_SIZE}, horizon={HORIZON})")
print("─" * 70)

# Simulate the same windowing used in Build_Dataset_SKAB_DLpipeline_By_David.py
# Each window's label = 1 if any anomaly occurs in the next HORIZON steps
labels           = df_sample[LABEL_COL].values
window_anom_rates = []
look_ahead_labels = []

for i in range(len(labels) - WINDOW_SIZE - HORIZON):
    window_anom_rates.append(labels[i: i + WINDOW_SIZE].mean())
    future = labels[i + WINDOW_SIZE: i + WINDOW_SIZE + HORIZON]
    look_ahead_labels.append(int(future.max()) if len(future) > 0 else 0)

look_ahead_labels = np.array(look_ahead_labels)
pos_windows   = look_ahead_labels.sum()
total_windows = len(look_ahead_labels)

print(f"\n  Total windows        : {total_windows:,}")
print(f"  Future anomaly (1)   : {pos_windows:,}  ({pos_windows / total_windows * 100:.2f}%)")
print(f"  No future anomaly (0): {total_windows - pos_windows:,}  ({(total_windows - pos_windows) / total_windows * 100:.2f}%)")
print(f"\n  Binary classification: predict anomaly in next {HORIZON} steps from {WINDOW_SIZE}-step history.")
print(f"  Positive rate ~{pos_windows / total_windows * 100:.1f}% — use class_weight='balanced' or oversampling.")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

idx_range = np.arange(total_windows)
ax1.plot(idx_range, window_anom_rates, color="#4C9BE8", linewidth=0.8,
         label=f"Window anomaly rate (w={WINDOW_SIZE})")
ax1.set_ylabel("Anomaly Rate in Window")
ax1.legend(fontsize=8)
ax1.set_title(f"Sliding Window Analysis — {sample_rel}  (window={WINDOW_SIZE}, horizon={HORIZON})")

ax2.fill_between(idx_range, 0, look_ahead_labels, color="#E84C4C", alpha=0.7,
                 label=f"Future anomaly label (next {HORIZON} steps)")
ax2.set_ylabel("Future Label (0/1)")
ax2.set_xlabel("Window Index")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_sliding_window_analysis.png"))
plt.close()
print("  -> Saved: fig7_sliding_window_analysis.png")


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  EDA complete. Output files:")
for fn in sorted(os.listdir(OUTPUT_DIR)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, fn)) / 1024
    print(f"  {fn:<50} {size_kb:>7.1f} KB")
print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 70)
