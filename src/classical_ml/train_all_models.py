"""
train_all_models.py  -  Train models on SKAB.

Supervised   (4 models, use labels):
    Logistic Regression, k-NN, Decision Tree, SVM

Metrics: Precision, Recall, F1, FPR, FNR
Output:  results/skab/supervised_results.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE_DIR / "data" / "processed" / "skab"
RESULTS_DIR = BASE_DIR / "results" / "skab"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── Load data ──────────────────────────────────────────────────────────────
X_train = np.load(DATA_DIR / "X_train.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")

print("Data loaded:")
print(f"  X_train: {X_train.shape}  |  anomaly ratio: {y_train.mean():.4f}")
print(f"  X_test : {X_test.shape}   |  anomaly ratio: {y_test.mean():.4f}")


# ── Metric helper ──────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, name, model_type):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\n  [{name}]")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    FPR       : {fpr:.4f}  (false alarms)")
    print(f"    FNR       : {fnr:.4f}  (missed faults)")
    print(f"    Confusion -> TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return {
        "Model":     name,
        "Type":      model_type,
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1":        round(f1,        4),
        "FPR":       round(fpr,       4),
        "FNR":       round(fnr,       4),
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
    }


# ══════════════════════════════════════════════════════════════════════════
# SUPERVISED MODELS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("SUPERVISED MODELS  (trained on labelled data)")
print("=" * 55)

supervised_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),
}

results = []

for name, model in supervised_models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    row = compute_metrics(y_test, y_pred, name, "Supervised")
    results.append(row)

# ── Save results ───────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("Recall", ascending=False)

csv_path = RESULTS_DIR / "supervised_results.csv"
df_results.to_csv(csv_path, index=False)

print("\n\n" + "=" * 55)
print("SUPERVISED MODELS COMPARISON  (sorted by Recall)")
print("=" * 55)
print(df_results[["Model", "Precision", "Recall", "F1", "FPR", "FNR"]].to_string(index=False))
print(f"\nResults saved to:\n  {csv_path}")

# ══════════════════════════════════════════════════════════════════════════
# 2. UNSUPERVISED MODELS
# ══════════════════════════════════════════════════════════════════════════
from sklearn.ensemble import IsolationForest
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS

print("\n" + "=" * 55)
print("UNSUPERVISED MODELS  (trained on NORMAL data only)")
print("=" * 55)

X_train_normal = np.load(DATA_DIR / "X_train_normal.npy")

scaler          = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled   = scaler.transform(X_test)

# ── Isolation Forest ──────────────────────────────────────────────────────
print("\n  Training Isolation Forest...")
iso = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso.fit(X_normal_scaled)
y_pred_iso = (iso.predict(X_test_scaled) == -1).astype(int)

# ── CBLOF ─────────────────────────────────────────────────────────────────
print("\n  Training CBLOF...")
cblof = CBLOF(
    n_clusters=8,
    contamination=0.1,
    random_state=RANDOM_STATE,
)
cblof.fit(X_normal_scaled)
y_pred_cblof = cblof.predict(X_test_scaled)

# ── HBOS ──────────────────────────────────────────────────────────────────
print("\n  Training HBOS...")
hbos = HBOS(
    n_bins=20,
    contamination=0.1,
)
hbos.fit(X_normal_scaled)
y_pred_hbos = hbos.predict(X_test_scaled)

unsupervised_results = []
unsupervised_results.append(compute_metrics(y_test, y_pred_iso,   "Isolation Forest", "Unsupervised"))
unsupervised_results.append(compute_metrics(y_test, y_pred_cblof, "CBLOF",            "Unsupervised"))
unsupervised_results.append(compute_metrics(y_test, y_pred_hbos,  "HBOS",             "Unsupervised"))

df_unsupervised = pd.DataFrame(unsupervised_results)
df_unsupervised = df_unsupervised.sort_values("Recall", ascending=False)

unsup_csv = RESULTS_DIR / "unsupervised_results.csv"
df_unsupervised.to_csv(unsup_csv, index=False)

print("\n\n" + "=" * 55)
print("UNSUPERVISED MODELS COMPARISON  (sorted by Recall)")
print("=" * 55)
print(df_unsupervised[["Model", "Precision", "Recall", "F1", "FPR", "FNR"]].to_string(index=False))
print(f"\nUnsupervised results saved to:\n  {unsup_csv}")

# ── Combined comparison ────────────────────────────────────────────────────
df_all = pd.concat([df_results, df_unsupervised], ignore_index=True)
all_csv = RESULTS_DIR / "all_models_comparison.csv"
df_all.to_csv(all_csv, index=False)
print(f"\nAll models combined saved to:\n  {all_csv}")