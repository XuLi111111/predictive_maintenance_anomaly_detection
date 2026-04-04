import os
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "dataset1"
MODEL_NAME   = "knn"
AUTHOR       = "parinitha"

DATA_DIR     = os.path.join("processed", "zenodo_pump_2d")
RESULTS_DIR  = "results"

# KNN does not support class_weight — SMOTE is applied to balance training data
# We tune k (number of neighbours) on the validation set
K_VALUES     = [3, 5, 7, 11, 15]

# SMOTE can be slow on very large datasets — use a subsample for speed
# Set to None to use the full training set
SMOTE_SUBSAMPLE = 100000  # use 100k rows for SMOTE to keep runtime manageable

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(data_dir: str) -> tuple:
    """
    Load preprocessed train, val and test splits from .npz files.
    Temporal order is preserved — no shuffling applied.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    """
    train = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    val   = np.load(os.path.join(data_dir, "val.npz"),   allow_pickle=True)
    test  = np.load(os.path.join(data_dir, "test.npz"),  allow_pickle=True)

    X_train, y_train = train["X"], train["y"]
    X_val,   y_val   = val["X"],   val["y"]
    X_test,  y_test  = test["X"],  test["y"]
    feature_cols     = train["feature_cols"]

    print(f"  Train : {X_train.shape}  |  faults={y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"  Val   : {X_val.shape}  |  faults={y_val.sum():,} ({y_val.mean()*100:.3f}%)")
    print(f"  Test  : {X_test.shape}  |  faults={y_test.sum():,} ({y_test.mean()*100:.4f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# =============================================================================
# SMOTE BALANCING
# =============================================================================

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subsample: int = None,
) -> tuple:
    """
    Apply SMOTE oversampling to balance the training set.
    KNN does not support class_weight, so SMOTE is used instead.

    Because SMOTE is memory-intensive on 854k rows, an optional subsample
    is taken from the training set before applying SMOTE. The subsample
    preserves all fault rows and randomly selects normal rows to reach
    the subsample size.

    Parameters
    ----------
    subsample : If set, limits the training rows passed to SMOTE.

    Returns
    -------
    X_resampled, y_resampled
    """
    if subsample is not None and len(X_train) > subsample:
        print(f"  Subsampling to {subsample:,} rows before SMOTE...")
        fault_idx  = np.where(y_train == 1)[0]
        normal_idx = np.where(y_train == 0)[0]
        n_normal   = min(subsample - len(fault_idx), len(normal_idx))
        # Preserve temporal order within the subsample
        chosen_normal = np.sort(np.random.choice(normal_idx, n_normal, replace=False))
        chosen_idx    = np.sort(np.concatenate([fault_idx, chosen_normal]))
        X_sub, y_sub  = X_train[chosen_idx], y_train[chosen_idx]
        print(f"  Subsample — Normal: {(y_sub==0).sum():,}  Fault: {(y_sub==1).sum():,}")
    else:
        X_sub, y_sub = X_train, y_train

    print(f"  Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_sub, y_sub)
    print(f"  After SMOTE — Normal: {(y_res==0).sum():,}  Fault: {(y_res==1).sum():,}")

    return X_res, y_res


# =============================================================================
# HYPERPARAMETER TUNING ON VALIDATION SET
# =============================================================================

def tune_on_val(
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_values: list,
) -> tuple:
    """
    Train KNN for each k value and evaluate on validation set.
    Since val has no faults, we minimise FPR as the selection criterion.

    Parameters
    ----------
    k_values : List of k (n_neighbors) values to explore.

    Returns
    -------
    best_k     : Selected number of neighbours.
    tuning_log : List of dicts with results per k.
    """
    print(f"\n  Tuning k on validation set (fault-free — minimise FPR)...")
    print(f"  {'k':<8} {'Val FPR (%)':>12} {'Val Predicted Faults':>22}")
    print("  " + "-" * 46)

    tuning_log = []
    best_k     = k_values[0]
    best_fpr   = float("inf")

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(X_train_res, y_train_res)
        y_val_pred    = model.predict(X_val)
        predicted_pos = y_val_pred.sum()
        fpr           = predicted_pos / len(y_val) * 100

        tuning_log.append({"k": k, "val_fpr": fpr, "val_pred_faults": int(predicted_pos)})
        print(f"  {k:<8} {fpr:>11.4f}% {int(predicted_pos):>22,}")

        if fpr < best_fpr:
            best_fpr = fpr
            best_k   = k

    print(f"\n  Selected k = {best_k}  (lowest val FPR = {best_fpr:.4f}%)")
    return best_k, tuning_log


# =============================================================================
# EVALUATE ON TEST SET
# =============================================================================

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained model on the test set.

    Metrics:
      - Precision : of predicted faults, how many were real?
      - Recall    : of real faults, how many did we catch? (primary metric)
      - F1-Score  : harmonic mean of precision and recall
      - FPR       : False Positive Rate = FP / (FP + TN)
      - FNR       : False Negative Rate = FN / (FN + TP)
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "y_pred"    : y_pred,
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "fpr"       : fpr,
        "fnr"       : fnr,
        "tn"        : tn,
        "fp"        : fp,
        "fn"        : fn,
        "tp"        : tp,
        "cm"        : cm,
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results_dir: str,
    model_name: str,
    author: str,
    dataset_name: str,
    best_k: int,
    k_values: list,
    tuning_log: list,
    metrics: dict,
    y_test: np.ndarray,
    train_time: float,
    smote_subsample: int,
) -> None:
    """
    Save evaluation results to a .txt file under the results/ directory.
    """
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"{model_name}_{author}_{dataset_name}.txt")

    report = classification_report(
        y_test,
        metrics["y_pred"],
        target_names=["Normal", "Fault"],
        zero_division=0,
    )

    with open(filename, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Model          : K-Nearest Neighbours (KNN)\n")
        f.write(f"Author         : {author}\n")
        f.write(f"Dataset        : {dataset_name}\n")
        f.write(f"Script         : train_knn.py\n")
        f.write("=" * 60 + "\n\n")

        f.write("IMBALANCE HANDLING\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Method        : SMOTE oversampling\n")
        f.write(f"  Reason        : KNN does not support class_weight parameter\n")
        f.write(f"  Subsample     : {smote_subsample:,} rows used for SMOTE\n\n")

        f.write("HYPERPARAMETER TUNING\n")
        f.write("-" * 40 + "\n")
        f.write(f"  k values explored : {k_values}\n")
        f.write(f"  Selection metric  : Val False Positive Rate (minimised)\n")
        f.write(f"  Tuning results    :\n")
        for entry in tuning_log:
            f.write(f"  k={entry['k']:<6}  Val FPR={entry['val_fpr']:.4f}%  "
                    f"Predicted faults={entry['val_pred_faults']:,}\n")
        f.write(f"  Best k selected   : {best_k}\n\n")

        f.write("FINAL MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  n_neighbors   : {best_k}\n")
        f.write(f"  metric        : minkowski (default)\n")
        f.write(f"  weights       : uniform (default)\n")
        f.write(f"  imbalance     : SMOTE\n\n")

        f.write("TEST SET METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Precision           : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)\n")
        f.write(f"  Recall (Sensitivity): {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)\n")
        f.write(f"  F1-Score            : {metrics['f1']:.4f}  ({metrics['f1']*100:.2f}%)\n")
        f.write(f"  False Positive Rate : {metrics['fpr']:.4f}  ({metrics['fpr']*100:.4f}%)\n")
        f.write(f"  False Negative Rate : {metrics['fnr']:.4f}  ({metrics['fnr']*100:.2f}%)\n\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'':20} Predicted Normal  Predicted Fault\n")
        f.write(f"  {'Actual Normal':<20} {metrics['tn']:>16,}  {metrics['fp']:>15,}\n")
        f.write(f"  {'Actual Fault':<20} {metrics['fn']:>16,}  {metrics['tp']:>15,}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(report + "\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Training time : {train_time:.2f} seconds\n")
        f.write(f"  Best k        : {best_k}\n")
        f.write(f"  Faults caught : {metrics['tp']} / {int(metrics['tp'] + metrics['fn'])} "
                f"({metrics['recall']*100:.2f}% recall)\n")
        f.write(f"  False alarms  : {metrics['fp']:,} out of {int(metrics['tn'] + metrics['fp']):,} "
                f"normal windows ({metrics['fpr']*100:.4f}% FPR)\n")
        f.write(f"\n  Observations:\n")
        f.write(f"  KNN is a distance-based model that classifies each window by majority\n")
        f.write(f"  vote from its k nearest neighbours. SMOTE was used to handle imbalance\n")
        f.write(f"  as KNN does not support class_weight. KNN is slow at prediction time\n")
        f.write(f"  on large datasets due to distance computation across all training points.\n")
        f.write(f"  Performance is sensitive to the scale of features — StandardScaler\n")
        f.write(f"  applied during preprocessing ensures fair distance computation.\n")
        f.write("=" * 60 + "\n")

    print(f"\n  Results saved to: {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  K-Nearest Neighbours (KNN) — Fault Detection Training")
    print("=" * 60)

    # --- Load data ---
    print(f"\n[1/5] Loading data from {DATA_DIR}...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data(DATA_DIR)

    # --- Apply SMOTE ---
    print(f"\n[2/5] Applying SMOTE to balance training set...")
    X_train_res, y_train_res = apply_smote(X_train, y_train, SMOTE_SUBSAMPLE)

    # --- Tune k on val ---
    print(f"\n[3/5] Hyperparameter tuning...")
    best_k, tuning_log = tune_on_val(X_train_res, y_train_res, X_val, y_val, K_VALUES)

    # --- Train final model ---
    print(f"\n[4/5] Training final model with k={best_k}...")
    model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    start_time = time.time()
    model.fit(X_train_res, y_train_res)
    train_time = time.time() - start_time
    print(f"  Training complete in {train_time:.2f} seconds")

    # --- Evaluate on test set ---
    print(f"\n[5/5] Evaluating on test set...")
    metrics = evaluate(model, X_test, y_test)

    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%  ← primary metric")
    print(f"  F1-Score  : {metrics['f1']*100:.2f}%")
    print(f"  FPR       : {metrics['fpr']*100:.4f}%")
    print(f"  FNR       : {metrics['fnr']*100:.2f}%")
    print(f"  Confusion : TN={metrics['tn']:,}  FP={metrics['fp']:,}  "
          f"FN={metrics['fn']}  TP={metrics['tp']}")

    # --- Save results ---
    save_results(
        RESULTS_DIR, MODEL_NAME, AUTHOR, DATASET_NAME,
        best_k, K_VALUES, tuning_log, metrics, y_test, train_time, SMOTE_SUBSAMPLE,
    )


if __name__ == "__main__":
    main()