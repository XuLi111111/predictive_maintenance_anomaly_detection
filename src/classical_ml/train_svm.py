import os
import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

DATASET_NAME = "dataset1"
MODEL_NAME   = "svm"
AUTHOR       = "parinitha"
DATA_DIR     = os.path.join("processed", "zenodo_pump_2d")
RESULTS_DIR  = "results"
C_VALUES     = [0.001, 0.01, 0.1, 1.0]

def load_data(data_dir):
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

def tune_on_val(X_train, y_train, X_val, y_val, c_values):
    print(f"\n  Tuning C on validation set (fault-free — minimise FPR)...")
    print(f"  {'C':<10} {'Val FPR (%)':>12} {'Val Predicted Faults':>22}")
    print("  " + "-" * 48)
    tuning_log = []
    best_C, best_fpr = c_values[0], float("inf")
    for C in c_values:
        model = LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        y_val_pred    = model.predict(X_val)
        predicted_pos = y_val_pred.sum()
        fpr           = predicted_pos / len(y_val) * 100
        tuning_log.append({"C": C, "val_fpr": fpr, "val_pred_faults": int(predicted_pos)})
        print(f"  {C:<10} {fpr:>11.4f}% {int(predicted_pos):>22,}")
        if fpr < best_fpr:
            best_fpr, best_C = fpr, C
    print(f"\n  Selected C = {best_C}  (lowest val FPR = {best_fpr:.4f}%)")
    return best_C, tuning_log

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {"y_pred": y_pred, "precision": precision, "recall": recall,
            "f1": f1, "fpr": fpr, "fnr": fnr, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cm": cm}

def save_results(results_dir, model_name, author, dataset_name, best_C, c_values, tuning_log, metrics, y_test, train_time):
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"{model_name}_{author}_{dataset_name}.txt")
    report = classification_report(y_test, metrics["y_pred"], target_names=["Normal", "Fault"], zero_division=0)
    with open(filename, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Model          : Support Vector Machine (SVM)\n")
        f.write(f"Author         : {author}\n")
        f.write(f"Dataset        : {dataset_name}\n")
        f.write(f"Script         : train_svm.py\n")
        f.write("=" * 60 + "\n\n")
        f.write("MODEL CHOICE NOTE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  LinearSVC is used instead of SVC with RBF kernel.\n")
        f.write(f"  Reason: 240 features (high-dimensional) — linear kernels\n")
        f.write(f"  perform as well or better than RBF and scale to 800k+ rows.\n\n")
        f.write("HYPERPARAMETER TUNING\n")
        f.write("-" * 40 + "\n")
        f.write(f"  C values explored : {c_values}\n")
        f.write(f"  Kernel            : linear (LinearSVC)\n")
        f.write(f"  Selection metric  : Val False Positive Rate (minimised)\n")
        f.write(f"  Tuning results    :\n")
        for entry in tuning_log:
            f.write(f"  C={entry['C']:<8}  Val FPR={entry['val_fpr']:.4f}%  Predicted faults={entry['val_pred_faults']:,}\n")
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
        f.write(f"  Faults caught : {metrics['tp']} / {int(metrics['tp'] + metrics['fn'])} ({metrics['recall']*100:.2f}% recall)\n")
        f.write(f"  False alarms  : {metrics['fp']:,} out of {int(metrics['tn'] + metrics['fp']):,} normal windows ({metrics['fpr']*100:.4f}% FPR)\n")
        f.write(f"\n  Observations:\n")
        f.write(f"  LinearSVC trains on the full dataset without subsampling.\n")
        f.write(f"  class_weight='balanced' compensates for the severe class imbalance.\n")
        f.write(f"  Linear kernels are well-suited for high-dimensional feature spaces\n")
        f.write(f"  such as the 240-feature flattened sliding window representation.\n")
        f.write("=" * 60 + "\n")
    print(f"\n  Results saved to: {filename}")

def main():
    print("=" * 60)
    print("  Support Vector Machine (SVM) — Fault Detection Training")
    print("=" * 60)
    print(f"\n[1/4] Loading data from {DATA_DIR}...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data(DATA_DIR)
    print(f"\n[2/4] Hyperparameter tuning...")
    best_C, tuning_log = tune_on_val(X_train, y_train, X_val, y_val, C_VALUES)
    print(f"\n[3/4] Training final model with C={best_C}...")
    model = LinearSVC(C=best_C, class_weight="balanced", max_iter=2000, random_state=42)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Training complete in {train_time:.2f} seconds")
    print(f"\n[4/4] Evaluating on test set...")
    metrics = evaluate(model, X_test, y_test)
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%  <- primary metric")
    print(f"  F1-Score  : {metrics['f1']*100:.2f}%")
    print(f"  FPR       : {metrics['fpr']*100:.4f}%")
    print(f"  FNR       : {metrics['fnr']*100:.2f}%")
    print(f"  Confusion : TN={metrics['tn']:,}  FP={metrics['fp']:,}  FN={metrics['fn']}  TP={metrics['tp']}")
    save_results(RESULTS_DIR, MODEL_NAME, AUTHOR, DATASET_NAME, best_C, C_VALUES, tuning_log, metrics, y_test, train_time)

if __name__ == "__main__":
    main()
