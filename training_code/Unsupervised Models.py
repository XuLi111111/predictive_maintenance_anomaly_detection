import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("sensor_features.csv")
df = df.drop(columns=["timestamp"])

X = df.drop(columns=["label"]).values
y = (df["label"].values > 0).astype(int)  # Binary classification

# Chronological split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Anomaly ratio (used for setting contamination)
contamination = y_train.mean()
print(f"Training set anomaly ratio: {contamination:.4f}")

# ── 2. Evaluation function ───────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    TP = cm[1,1]; FN = cm[1,0]; FP = cm[0,1]; TN = cm[0,0]
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    print(f"  FPR={FPR:.4f}  FNR={FNR:.4f}")

# ── 3. Unsupervised models (trained only on normal data) ────────────────────────
# Use only normal samples from the training set
X_train_normal = X_train[y_train == 0]
print(f"Number of normal samples: {len(X_train_normal)}")

# --- Isolation Forest ---
print("\nTraining Isolation Forest...")
iforest = IForest(contamination=contamination, random_state=42, n_jobs=-1)
iforest.fit(X_train_normal)
evaluate("Isolation Forest", y_test, iforest.predict(X_test))

# --- LOF ---
print("\nTraining LOF...")
lof = LOF(contamination=contamination, n_jobs=-1)
lof.fit(X_train_normal)
evaluate("LOF", y_test, lof.predict(X_test))

# --- OCSVM (Subsampling to avoid being too slow) ---
print("\nTraining OCSVM...")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_normal), size=5000, replace=False)
ocsvm = OCSVM(contamination=contamination)
ocsvm.fit(X_train_normal[idx])
evaluate("OCSVM", y_test, ocsvm.predict(X_test))

# --- Autoencoder ---
print("\nTraining Autoencoder...")
ae = AutoEncoder(
    hidden_neuron_list=[128, 64, 32],
    epoch_num=20,
    batch_size=256,
    contamination=contamination,
    random_state=42,
    verbose=0
)
ae.fit(X_train_normal)
evaluate("Autoencoder", y_test, ae.predict(X_test))

import joblib

joblib.dump(iforest, "model_iforest.pkl")
joblib.dump(lof,     "model_lof.pkl")
joblib.dump(ocsvm,   "model_ocsvm.pkl")
joblib.dump(ae,      "model_ae.pkl")
joblib.dump(scaler,  "scaler_unsupervised.pkl")

print("\nModels saved!")
print("Done!")