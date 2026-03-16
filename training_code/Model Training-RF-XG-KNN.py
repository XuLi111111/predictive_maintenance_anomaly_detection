import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("sensor_features.csv")
df = df.drop(columns=["timestamp"])

X = df.drop(columns=["label"]).values
y = df["label"].values

print(f"Data shape: {X.shape}")
print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── 2. Split Train/Test (Chronological order, no random shuffle) ──
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ── 3. Standardization ────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 4. SMOTE for Imbalance (Only on training set) ─────────────
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Distribution after SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ── 5. Evaluation Function ────────────────────────────────────
def evaluate(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal","EarlyWarning","CriticalAlert"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # FPR / FNR per class
    for i, cls in enumerate(["Normal", "EarlyWarning", "CriticalAlert"]):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        print(f"  {cls}: FPR={FPR:.4f}  FNR={FNR:.4f}")

# ── 6. Model Training and Evaluation ──────────────────────────

# --- Random Forest ---
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
evaluate("Random Forest", y_test, rf.predict(X_test))

# --- XGBoost ---
print("\nTraining XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    use_label_encoder=False, eval_metric="mlogloss",
                    n_jobs=-1)
xgb.fit(X_train_sm, y_train_sm)
evaluate("XGBoost", y_test, xgb.predict(X_test))

# --- k-NN ---
print("\nTraining k-NN...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_sm, y_train_sm)
evaluate("k-NN", y_test, knn.predict(X_test))

print("\nDone!")