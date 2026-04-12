import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# Evaluate trained boosting models on SKAB test set
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "processed",
    "dataset2",
    "skab_strategyA_window20_horizon10.npz"
)

MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading dataset...")
print(f"DATA_PATH: {DATA_PATH}")

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Positive ratio in test set: {np.mean(y_test) * 100:.2f}%")

def flatten(X):
    return X.reshape(X.shape[0], -1)

X_test_flat = flatten(X_test)


model_files = {
    "XGBoost": "boosting_xgboost.pkl",
    "AdaBoost": "boosting_adaboost.pkl",
    "GradientBoosting": "boosting_gradient_boosting.pkl",
    "LightGBM": "boosting_lightgbm.pkl",
}

summary_rows = []

for model_name, filename in model_files.items():
    print("\n" + "=" * 80)
    print(f"Evaluating {model_name}")
    print("=" * 80)

    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        print(f"Missing model file: {model_path}")
        continue

    model = joblib.load(model_path)
    print(f"Loaded: {model_path}")

    y_pred = model.predict(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    roc_auc = None
    pr_auc = None

    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test_flat)[:, 1]
            roc_auc = roc_auc_score(y_test, y_score)
            pr_auc = average_precision_score(y_test, y_score)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X_test_flat)
            roc_auc = roc_auc_score(y_test, y_score)
            pr_auc = average_precision_score(y_test, y_score)
        except Exception:
            pass

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR-AUC   : {pr_auc:.4f}")

    report_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.lower()}_test_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write("\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        if roc_auc is not None:
            f.write(f"ROC-AUC  : {roc_auc:.4f}\n")
        if pr_auc is not None:
            f.write(f"PR-AUC   : {pr_auc:.4f}\n")

    summary_rows.append({
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    })

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(RESULTS_DIR, "boosting_models_test_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved summary CSV:")
    print(summary_csv)
    print("\nSummary:")
    print(summary_df)
else:
    print("\nNo models were evaluated.")