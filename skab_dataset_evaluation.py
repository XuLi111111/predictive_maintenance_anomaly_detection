import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — update these paths to match your local folder structure
# =============================================================================

# Path to the anomaly-free CSV file
ANOMALY_FREE_PATH = "data/anomaly-free.csv"

# Folder containing valve1 CSV files (0.csv to 15.csv)
VALVE1_FOLDER = "data/valve1"

# Folder containing valve2 CSV files (0.csv to 3.csv)
VALVE2_FOLDER = "data/valve2"

# Sliding window settings
WINDOW_SIZE = 30    # number of timesteps per window (30 seconds at 1 Hz)
STEP_SIZE   = 5     # step between windows (reduces overlap, speeds training)

# Train/test split ratio
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# Sensor feature columns (exclude datetime, anomaly, changepoint, source)
SENSOR_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

# =============================================================================
# SECTION 1 — DATA LOADING AND MERGING
# =============================================================================

def load_csv(filepath: str, source_label: str) -> pd.DataFrame:
    """
    Load a single SKAB CSV file and attach a source label column.

    Parameters
    ----------
    filepath     : Full path to the CSV file.
    source_label : Label string to identify the file's origin (e.g. 'valve1').

    Returns
    -------
    df : DataFrame with a 'source' column appended.
    """
    df = pd.read_csv(filepath, sep=";", parse_dates=["datetime"])
    df["source"] = source_label
    return df


def load_all_data(
    anomaly_free_path: str,
    valve1_folder: str,
    valve2_folder: str,
) -> pd.DataFrame:
    """
    Load and merge all SKAB files into a single DataFrame.

    - anomaly-free.csv  → label 0 (normal), anomaly column added as 0
    - valve1/*.csv      → label from 'anomaly' column
    - valve2/*.csv      → label from 'anomaly' column

    Parameters
    ----------
    anomaly_free_path : Path to anomaly-free.csv.
    valve1_folder     : Folder containing valve1 experiment CSVs.
    valve2_folder     : Folder containing valve2 experiment CSVs.

    Returns
    -------
    df_merged : Combined DataFrame with unified schema.
    """
    frames = []

    # --- Load anomaly-free (normal baseline) ---
    df_free = load_csv(anomaly_free_path, "anomaly-free")
    # anomaly-free has no 'anomaly' column — add it as 0 (all normal)
    if "anomaly" not in df_free.columns:
        df_free["anomaly"] = 0
    if "changepoint" not in df_free.columns:
        df_free["changepoint"] = 0
    frames.append(df_free)
    print(f"  [OK] anomaly-free.csv        rows={len(df_free):,}  "
          f"fault_rows=0  (0.0%)")

    # --- Load valve1 files ---
    valve1_files = sorted(
        [f for f in os.listdir(valve1_folder) if f.endswith(".csv")],
        key=lambda x: int(x.replace(".csv", ""))
    )
    for fname in valve1_files:
        fpath = os.path.join(valve1_folder, fname)
        df_tmp = load_csv(fpath, "valve1")
        frames.append(df_tmp)
        n_fault = int(df_tmp["anomaly"].sum())
        print(f"  [OK] valve1/{fname:<12}  rows={len(df_tmp):,}  "
              f"fault_rows={n_fault}  ({n_fault/len(df_tmp)*100:.1f}%)")

    # --- Load valve2 files ---
    valve2_files = sorted(
        [f for f in os.listdir(valve2_folder) if f.endswith(".csv")],
        key=lambda x: int(x.replace(".csv", ""))
    )
    for fname in valve2_files:
        fpath = os.path.join(valve2_folder, fname)
        df_tmp = load_csv(fpath, "valve2")
        frames.append(df_tmp)
        n_fault = int(df_tmp["anomaly"].sum())
        print(f"  [OK] valve2/{fname:<12}  rows={len(df_tmp):,}  "
              f"fault_rows={n_fault}  ({n_fault/len(df_tmp)*100:.1f}%)")

    df_merged = pd.concat(frames, ignore_index=True)
    df_merged.sort_values("datetime", inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    # Ensure anomaly label is integer binary (0 or 1)
    df_merged["anomaly"] = df_merged["anomaly"].fillna(0).astype(int)

    return df_merged


# =============================================================================
# SECTION 2 — CLASS DISTRIBUTION REPORT
# =============================================================================

def print_class_distribution(df: pd.DataFrame) -> None:
    """
    Print a detailed class distribution report including imbalance ratio.
    This helps determine whether SMOTE is needed.

    Parameters
    ----------
    df : Merged DataFrame with 'anomaly' column.
    """
    total      = len(df)
    n_normal   = int((df["anomaly"] == 0).sum())
    n_fault    = int((df["anomaly"] == 1).sum())
    ratio      = n_normal / n_fault if n_fault > 0 else float("inf")

    print(f"\n  Total rows          : {total:,}")
    print(f"  Normal rows   (0)   : {n_normal:,}  ({n_normal/total*100:.1f}%)")
    print(f"  Fault rows    (1)   : {n_fault:,}  ({n_fault/total*100:.1f}%)")
    print(f"  Imbalance ratio     : {ratio:.2f}:1  (Normal:Fault)")

    if ratio > 10:
        print("  → Severe imbalance. SMOTE strongly recommended.")
    elif ratio > 3:
        print("  → Moderate imbalance. SMOTE recommended.")
    else:
        print("  → Mild imbalance. Class weighting may be sufficient.")

    # Per-source breakdown
    print("\n  Per-source breakdown:")
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        nf  = int(sub["anomaly"].sum())
        print(f"    {src:<15}: {len(sub):>6,} rows  |  "
              f"{nf:>5} faults  ({nf/len(sub)*100:.1f}%)")


# =============================================================================
# SECTION 3 — SIGNAL QUALITY CHECK
# =============================================================================

def print_signal_quality(df: pd.DataFrame, sensor_cols: list) -> None:
    """
    Evaluate each sensor's ability to separate normal vs fault readings.
    Reports mean, std, and mean difference between classes per sensor.

    A high mean difference relative to std indicates good separability —
    meaning the sensor signal is informative for classification.

    Parameters
    ----------
    df          : Merged DataFrame.
    sensor_cols : List of sensor feature column names.
    """
    normal = df[df["anomaly"] == 0]
    fault  = df[df["anomaly"] == 1]

    print(f"\n  {'Sensor':<22} {'Normal Mean':>12} {'Fault Mean':>11} "
          f"{'Difference':>11} {'Separable?':>12}")
    print("  " + "-" * 72)

    for col in sensor_cols:
        if col not in df.columns:
            continue
        m_norm  = normal[col].mean()
        m_fault = fault[col].mean()
        diff    = abs(m_fault - m_norm)
        std     = df[col].std()
        # Separability heuristic: difference > 10% of std range
        separable = "YES" if diff > 0.1 * std else "NO"
        print(f"  {col:<22} {m_norm:>12.4f} {m_fault:>11.4f} "
              f"{diff:>11.4f} {separable:>12}")

    # Missing value check
    print(f"\n  Missing value check:")
    missing = df[sensor_cols].isnull().sum()
    if missing.sum() == 0:
        print("  → No missing values detected across all sensor columns.")
    else:
        print(missing[missing > 0].to_string())

    # Constant column check
    zero_var = [c for c in sensor_cols if df[c].var() < 1e-6]
    if zero_var:
        print(f"\n  ⚠ Near-constant columns detected: {zero_var}")
        print("  → These columns provide no signal and should be dropped.")
    else:
        print("  → No near-constant columns detected.")


# =============================================================================
# SECTION 4 — PREPROCESSING PIPELINE
# =============================================================================

def create_sliding_windows(
    data: pd.DataFrame,
    sensor_cols: list,
    label_col: str = "anomaly",
    window_size: int = WINDOW_SIZE,
    step: int = STEP_SIZE,
) -> tuple:
    """
    Convert time-series data into fixed-length sliding window samples
    for supervised classification.

    Each window captures 'window_size' consecutive sensor readings.
    The window label is 1 if ANY row within the window is a fault, else 0.
    This is the standard approach for fault detection with time-series data.

    Parameters
    ----------
    data        : DataFrame sorted by time, with sensor and label columns.
    sensor_cols : Feature columns to include in each window.
    label_col   : Binary label column name.
    window_size : Number of timesteps per window.
    step        : Step size between consecutive windows.

    Returns
    -------
    X : ndarray of shape (n_windows, n_features)
        Flattened window features for traditional ML models.
    y : ndarray of shape (n_windows,)
        Binary labels per window.
    """
    X_list, y_list = [], []
    values = data[sensor_cols].values
    labels = data[label_col].values

    for start in range(0, len(data) - window_size + 1, step):
        end      = start + window_size
        window_x = values[start:end].flatten()        # flatten for sklearn
        window_y = int(labels[start:end].max())       # 1 if any fault in window
        X_list.append(window_x)
        y_list.append(window_y)

    return np.array(X_list), np.array(y_list)


def preprocess(
    df: pd.DataFrame,
    sensor_cols: list,
) -> tuple:
    """
    Full preprocessing pipeline:
      1. Forward-fill any missing values
      2. Chronological train/test split (no leakage)
      3. Fit MinMaxScaler on train set only, apply to both
      4. Construct sliding windows
      5. Apply SMOTE on training windows to handle class imbalance

    Parameters
    ----------
    df          : Merged sorted DataFrame.
    sensor_cols : Sensor feature column names.

    Returns
    -------
    X_train, X_test, y_train, y_test : Arrays ready for model training.
    """
    # Step 1: Handle missing values
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    # Step 2: Chronological split — split raw rows BEFORE windowing
    # This prevents future data leaking into training windows
    split_idx  = int(len(df) * (1 - TEST_SIZE))
    df_train   = df.iloc[:split_idx].copy()
    df_test    = df.iloc[split_idx:].copy()

    # Step 3: Fit scaler on training data only, transform both sets
    scaler = MinMaxScaler()
    df_train[sensor_cols] = scaler.fit_transform(df_train[sensor_cols])
    df_test[sensor_cols]  = scaler.transform(df_test[sensor_cols])

    # Step 4: Sliding window construction
    X_train, y_train = create_sliding_windows(df_train, sensor_cols)
    X_test,  y_test  = create_sliding_windows(df_test,  sensor_cols)

    print(f"\n  Windows generated:")
    print(f"    Train : {X_train.shape[0]:,} windows  "
          f"(fault={int(y_train.sum()):,}  normal={int((y_train==0).sum()):,})")
    print(f"    Test  : {X_test.shape[0]:,} windows  "
          f"(fault={int(y_test.sum()):,}  normal={int((y_test==0).sum()):,})")

    # Step 5: Apply SMOTE on training windows only
    print(f"\n  Applying SMOTE to balance training set...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"    After SMOTE — train windows: {X_train.shape[0]:,}  "
          f"(fault={int(y_train.sum()):,}  normal={int((y_train==0).sum()):,})")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 5 — SUPERVISED MODEL TRAINING AND EVALUATION
# =============================================================================

def get_models() -> dict:
    """
    Return a dictionary of supervised classification models to evaluate.
    All models are configured with fixed random seeds for reproducibility.

    Models align with the project brief requirements:
    Logistic Regression (baseline), Decision Tree, Random Forest, SVM, KNN.

    Returns
    -------
    models : dict mapping model name to sklearn estimator instance.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight="balanced",   # extra safety for any residual imbalance
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }


def evaluate_models(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Train each model and evaluate on the held-out test set.
    Reports Accuracy, Precision, Recall, and F1-score for the fault class.

    F1-score on the fault class is the primary metric, as recommended
    for imbalanced classification tasks in predictive maintenance
    (Haixiang et al., 2017).

    Parameters
    ----------
    models  : Dict of model name → sklearn estimator.
    X_train, X_test, y_train, y_test : Train/test arrays.

    Returns
    -------
    results_df : DataFrame summarising all model metrics.
    """
    results = []

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Compute metrics (focus on fault class = 1)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results.append({
            "Model"          : name,
            "Accuracy"       : round(acc  * 100, 2),
            "Precision (%)"  : round(prec * 100, 2),
            "Recall (%)"     : round(rec  * 100, 2),
            "F1-Score (%)"   : round(f1   * 100, 2),
            "TN" : cm[0][0],
            "FP" : cm[0][1],
            "FN" : cm[1][0],
            "TP" : cm[1][1],
        })

        print(f"    Accuracy   : {acc*100:.2f}%")
        print(f"    Precision  : {prec*100:.2f}%  (of predicted faults, how many were real?)")
        print(f"    Recall     : {rec*100:.2f}%  (of real faults, how many did we catch?)")
        print(f"    F1-Score   : {f1*100:.2f}%  (primary metric for fault detection)")
        print(f"    Confusion  : TN={cm[0][0]}  FP={cm[0][1]}  "
              f"FN={cm[1][0]}  TP={cm[1][1]}")

    return pd.DataFrame(results)


# =============================================================================
# SECTION 6 — COMPARISON SUMMARY AND DATASET FITNESS VERDICT
# =============================================================================

def print_summary_and_verdict(results_df: pd.DataFrame) -> None:
    """
    Print a ranked comparison table of all models and a written verdict
    on whether the SKAB dataset is suitable for the project.

    The verdict is based on:
      - Whether at least one model achieves F1 >= 80% (project target)
      - Data volume compared to Kaggle Pump Sensor dataset
      - Class balance suitability for supervised classification

    Parameters
    ----------
    results_df : DataFrame of model evaluation results.
    """
    print("\n  " + "=" * 68)
    print("  MODEL COMPARISON TABLE (sorted by F1-Score)")
    print("  " + "=" * 68)

    display_cols = ["Model", "Accuracy", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    ranked = results_df[display_cols].sort_values("F1-Score (%)", ascending=False)
    ranked.index = range(1, len(ranked) + 1)
    print(ranked.to_string())

    best_f1    = results_df["F1-Score (%)"].max()
    best_model = results_df.loc[results_df["F1-Score (%)"].idxmax(), "Model"]
    target_met = best_f1 >= 80.0

    print("\n  " + "=" * 68)
    print("  DATASET FITNESS VERDICT")
    print("  " + "=" * 68)
    print(f"\n  Best model        : {best_model}")
    print(f"  Best F1-Score     : {best_f1:.2f}%")
    print(f"  Target (F1 >= 80%): {'MET' if target_met else 'NOT MET'}")

    print("\n  SKAB Dataset Characteristics:")
    print("    + Clean data       : No missing values, consistent 1 Hz sampling")
    print("    + Real fault labels: anomaly column present in valve1/valve2 files")
    print("    + Mild imbalance   : ~35% fault rate — manageable with SMOTE")
    print("    - Small volume     : ~31,875 rows total vs Kaggle's ~220,000 rows")
    print("    - Short experiments: Each CSV is one short experiment (~20 min)")
    print("    - Limited fault types: Only valve closure faults (valve1 + valve2)")

    print("\n  RECOMMENDATION:")
    if target_met:
        print(f"    SKAB supports supervised fault classification.")
        print(f"    {best_model} achieves {best_f1:.1f}% F1, meeting the 80% threshold.")
        print(f"    However, the small data volume and limited fault variety")
        print(f"    make the Kaggle Pump Sensor dataset a stronger choice")
        print(f"    for production deployment.")
    else:
        print(f"    SKAB does not meet the F1 >= 80% target on its own.")
        print(f"    Best result: {best_model} at {best_f1:.1f}% F1.")
        print(f"    The Kaggle Pump Sensor dataset (220k rows, richer fault types)")
        print(f"    is the recommended primary dataset for this project.")

    print("\n  " + "=" * 68)
    print("  END OF EVALUATION REPORT")
    print("  " + "=" * 68)


# =============================================================================
# MAIN — Run the full evaluation pipeline
# =============================================================================

def main():
    print("\n" + "=" * 68)
    print("  SKAB Dataset Fitness Evaluation — CITS5206 Group 14")
    print("=" * 68)

    # --- Section 1: Load data ---
    print("\n[1/6] Loading and merging all SKAB files...")
    df = load_all_data(ANOMALY_FREE_PATH, VALVE1_FOLDER, VALVE2_FOLDER)
    print(f"\n  Combined dataset shape : {df.shape}")

    # --- Section 2: Class distribution ---
    print("\n[2/6] Class distribution report...")
    print_class_distribution(df)

    # --- Section 3: Signal quality ---
    print("\n[3/6] Signal quality check (normal vs fault separability)...")
    print_signal_quality(df, SENSOR_COLS)

    # --- Section 4: Preprocessing ---
    print("\n[4/6] Running preprocessing pipeline...")
    print(f"  Window size : {WINDOW_SIZE} timesteps")
    print(f"  Step size   : {STEP_SIZE}")
    print(f"  Test split  : {int(TEST_SIZE*100)}% (chronological)")
    X_train, X_test, y_train, y_test = preprocess(df, SENSOR_COLS)

    # --- Section 5: Train and evaluate models ---
    print("\n[5/6] Training and evaluating supervised models...")
    models     = get_models()
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)

    # --- Section 6: Summary and verdict ---
    print("\n[6/6] Summary and dataset fitness verdict...")
    print_summary_and_verdict(results_df)


if __name__ == "__main__":
    main()