import os
import time
import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_NAME = "dataset2"
MODEL_NAME   = "svm"
AUTHOR       = "parinitha"

DATA_PATH    = os.path.join(
    "src", "data_preprocessing", "data", "processed",
    "dataset2", "skab_strategyA_window20_horizon10.npz"
)

RESULTS_DIR  = "results"
MODELS_DIR   = os.path.join("models", "skab_classical_ml")

# C values to explore during hyperparameter tuning on validation set
# LinearSVC is used — fast and well-suited for high-dimensional data (160 features)
C_VALUES     = [0.001, 0.01, 0.1, 1.0]

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def load_and_prepare(data_path: str) -> tuple:
    """
    Load the preprocessed SKAB .npz file and prepare data for classical ML.

    Flattens 3D input (N, 20, 8) -> (N, 160) for sklearn compatibility.
    StandardScaler fitted on train set only to prevent data leakage.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    data = np.load(data_path, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    print(f"  Raw shapes loaded:")
    print(f"  Train : {X_train.shape}  |  faults={y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Val   : {X_val.shape}  |  faults={y_val.sum():,} ({y_val.mean()*100:.2f}%)")
    print(f"  Test  : {X_test.shape}  |  faults={y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    # Flatten: (N, 20, 8) -> (N, 160)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat   = X_val.reshape(X_val.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)

    # Scale: fit on train only
    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train_flat)
    X_val_sc     = scaler.transform(X_val_flat)
    X_test_sc    = scaler.transform(X_test_flat)

    print(f"\n  After flattening and scaling:")
    print(f"  X_train : {X_train_sc.shape}")
    print(f"  X_val   : {X_val_sc.shape}")
    print(f"  X_test  : {X_test_sc.shape}")

    return X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test, scaler


# =============================================================================
# HYPERPARAMETER TUNING ON VALIDATION SET
# =============================================================================

def tune_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_values: list,
) -> tuple:
    """
    Train LinearSVC for each C value and evaluate on validation set.
    Selects the C with best F1-score on val.

    LinearSVC is used instead of SVC(kernel='rbf') because:
    - 160 features is high-dimensional — linear kernels work well
    - LinearSVC scales efficiently to thousands of samples
    - Much faster training than RBF kernel SVC

    Parameters
    ----------
    c_values : List of C regularisation values to explore.

    Returns
    -------
    best_C     : Selected C value.
    tuning_log : List of dicts with results per C.
    """
    print(f"\n  Tuning C on validation set (selecting by best F1-score)...")
    print(f"  {'C':<10} {'Val Recall (%)':>15} {'Val F1 (%)':>12} {'Val Precision (%)':>18}")
    print("  " + "-" * 60)

    tuning_log = []
    best_C     = c_values[0]
    best_f1    = -1.0

    for C in c_values:
        model = LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        recall     = recall_score(y_val, y_val_pred, zero_division=0)
        f1         = f1_score(y_val, y_val_pred, zero_division=0)
        precision  = precision_score(y_val, y_val_pred, zero_division=0)

        tuning_log.append({
            "C": C, "val_recall": recall,
            "val_f1": f1, "val_precision": precision,
        })
        print(f"  {C:<10} {recall*100:>14.2f}% {f1*100:>11.2f}% {precision*100:>17.2f}%")

        if f1 > best_f1:
            best_f1 = f1
            best_C  = C

    print(f"\n  Selected C = {best_C}  (best val F1 = {best_f1*100:.2f}%)")
    return best_C, tuning_log


# =============================================================================
# EVALUATE ON TEST SET
# =============================================================================

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate trained model on the test set.

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
    best_C: float,
    c_values: list,
    tuning_log: list,
    metrics: dict,
    y_test: np.ndarray,
    train_time: float,
) -> None:
    """
    Save evaluation results to a .txt file under results/ directory.
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
        f.write(f"Model          : Support Vector Machine (SVM)\n")
        f.write(f"Author         : {author}\n")
        f.write(f"Dataset        : SKAB — Strategy A (dataset2)\n")
        f.write(f"Script         : train_skab_svm.py\n")
        f.write("=" * 60 + "\n\n")

        f.write("DATASET INFO\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Source        : SKAB valve1 (train/val) + valve2 (test)\n")
        f.write(f"  Window size   : 20 timesteps\n")
        f.write(f"  Horizon       : 10 timesteps (early warning label)\n")
        f.write(f"  Input shape   : (N, 20, 8) -> flattened to (N, 160)\n")
        f.write(f"  Scaling       : StandardScaler fitted on train only\n\n")

        f.write("MODEL CHOICE NOTE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  LinearSVC used — faster than SVC(kernel='rbf') for 160 features.\n")
        f.write(f"  Linear kernels perform well on high-dimensional flattened windows.\n\n")

        f.write("HYPERPARAMETER TUNING\n")
        f.write("-" * 40 + "\n")
        f.write(f"  C values explored : {c_values}\n")
        f.write(f"  Selection metric  : Validation F1-score (maximised)\n")
        f.write(f"  Tuning results    :\n")
        for entry in tuning_log:
            f.write(f"  C={entry['C']:<8}  Val Recall={entry['val_recall']*100:.2f}%  "
                    f"Val F1={entry['val_f1']*100:.2f}%  "
                    f"Val Precision={entry['val_precision']*100:.2f}%\n")
        f.write(f"  Best C selected   : {best_C}\n\n")

        f.write("FINAL MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  C             : {best_C}\n")
        f.write(f"  kernel        : linear (LinearSVC)\n")
        f.write(f"  class_weight  : balanced\n")
        f.write(f"  max_iter      : 2000\n")
        f.write(f"  random_state  : 42\n\n")

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
        f.write(f"  Best C        : {best_C}\n")
        f.write(f"  Faults caught : {metrics['tp']} / {int(metrics['tp'] + metrics['fn'])} "
                f"({metrics['recall']*100:.2f}% recall)\n")
        f.write(f"  False alarms  : {metrics['fp']:,} out of {int(metrics['tn'] + metrics['fp']):,} "
                f"normal windows ({metrics['fpr']*100:.4f}% FPR)\n")
        f.write(f"\n  Observations:\n")
        f.write(f"  LinearSVC trains on full SKAB dataset without subsampling.\n")
        f.write(f"  class_weight='balanced' handles the ~36% fault imbalance.\n")
        f.write(f"  Linear kernel is well suited for 160-feature flattened windows.\n")
        f.write(f"  Training is fast due to LinearSVC's optimised linear solver.\n")
        f.write("=" * 60 + "\n")

    print(f"\n  Results saved to: {filename}")


# =============================================================================
# SAVE MODEL WEIGHTS
# =============================================================================

def save_model(model, scaler, models_dir: str) -> None:
    """
    Save trained model and scaler as pickle files for future inference.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path  = os.path.join(models_dir, "model_svm_skab.pkl")
    scaler_path = os.path.join(models_dir, "scaler_svm_skab.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  Model saved  : {model_path}")
    print(f"  Scaler saved : {scaler_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  SVM (LinearSVC) — SKAB Fault Detection Training")
    print("=" * 60)

    # --- Load and prepare data ---
    print(f"\n[1/5] Loading and preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_and_prepare(DATA_PATH)

    # --- Tune C on val ---
    print(f"\n[2/5] Hyperparameter tuning on validation set...")
    best_C, tuning_log = tune_on_val(X_train, y_train, X_val, y_val, C_VALUES)

    # --- Train final model ---
    print(f"\n[3/5] Training final model with C={best_C}...")
    model = LinearSVC(
        C=best_C,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Training complete in {train_time:.2f} seconds")

    # --- Evaluate on test set ---
    print(f"\n[4/5] Evaluating on test set...")
    metrics = evaluate(model, X_test, y_test)

    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%  <- primary metric")
    print(f"  F1-Score  : {metrics['f1']*100:.2f}%")
    print(f"  FPR       : {metrics['fpr']*100:.4f}%")
    print(f"  FNR       : {metrics['fnr']*100:.2f}%")
    print(f"  Confusion : TN={metrics['tn']:,}  FP={metrics['fp']:,}  "
          f"FN={metrics['fn']}  TP={metrics['tp']}")

    # --- Save results and model ---
    print(f"\n[5/5] Saving results and model weights...")
    save_results(
        RESULTS_DIR, MODEL_NAME, AUTHOR, DATASET_NAME,
        best_C, C_VALUES, tuning_log, metrics, y_test, train_time,
    )
    save_model(model, scaler, MODELS_DIR)


if __name__ == "__main__":
    main()