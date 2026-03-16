import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pyod.models.ocsvm import OCSVM

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("sensor_features.csv")
df = df.drop(columns=["timestamp"])

X = df.drop(columns=["label"]).values
y = (df["label"].values > 0).astype(int)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_normal = X_train[y_train == 0]
contamination  = y_train.mean()

print(f"Training sample size: {len(X_train_normal)}")
print(f"Anomaly ratio: {contamination:.4f}")
print("Starting OCSVM training (full set), please wait...")

# ── 2. Training ──────────────────────────────────────────────────
ocsvm_full = OCSVM(contamination=contamination)
ocsvm_full.fit(X_train_normal)

# ── 3. Evaluation ──────────────────────────────────────────────────
y_pred = ocsvm_full.predict(X_test)

print("\n" + "="*50)
print("Model: OCSVM (Full)")
print("="*50)
print(classification_report(y_test, y_pred,
                             target_names=["Normal", "Anomaly"]))
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
TP = cm[1,1]; FN = cm[1,0]; FP = cm[0,1]; TN = cm[0,0]
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
print(f"  FPR={FPR:.4f}  FNR={FNR:.4f}")

# ── 4. Saving ──────────────────────────────────────────────────
joblib.dump(ocsvm_full, "model_ocsvm_full.pkl")
joblib.dump(scaler,     "scaler_ocsvm_full.pkl")
print("\nModel saved as model_ocsvm_full.pkl")