import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pyod.models.iforest import IForest
from pyod.models.xgbod import XGBOD
from pyod.models.devnet import DevNet
from pyod.models.vae import VAE

# ── 1. Load data ──────────────────────────────────────────────
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

contamination = y_train.mean()
X_train_normal = X_train[y_train == 0]

print(f"Anomaly ratio: {contamination:.4f}")

# ── 2. Evaluation function ─────────────────────────────────────
def evaluate(name, y_true, y_pred):
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "Anomaly"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{cm}")
    TP = cm[1,1]; FN = cm[1,0]; FP = cm[0,1]; TN = cm[0,0]
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    print(f"  FPR={FPR:.4f}  FNR={FNR:.4f}")

# ── 3. EIF (PyOD built-in, increased n_estimators to simulate EIF effect) ─
print("\nTraining EIF...")
eif = IForest(n_estimators=200, max_samples='auto',
              contamination=contamination, random_state=42, n_jobs=-1)
eif.fit(X_train_normal)
evaluate("EIF", y_test, eif.predict(X_test))
joblib.dump(eif, "model_eif.pkl")

# ── 4. XGBOD (Semi-supervised, requires few labels) ────────────────
print("\nTraining XGBOD...")
# Use 1% labeled data
n_labeled = int(len(X_train) * 0.01)
idx = np.where(y_train == 1)[0][:n_labeled]
y_train_xgbod = np.zeros(len(X_train))
y_train_xgbod[idx] = 1

xgbod = XGBOD(random_state=42)
xgbod.fit(X_train, y_train_xgbod)
evaluate("XGBOD", y_test, xgbod.predict(X_test))
joblib.dump(xgbod, "model_xgbod.pkl")

# ── 5. DevNet (Semi-supervised) ────────────────────────────────
print("\nTraining DevNet...")
devnet = DevNet(contamination=contamination, random_state=42)
devnet.fit(X_train, y_train_xgbod)
evaluate("DevNet", y_test, devnet.predict(X_test))
joblib.dump(devnet, "model_devnet.pkl")

# ── 6. VAE (Replacing PReNet) ──────────────────────────────────
print("\nTraining VAE...")
vae = VAE(contamination=contamination, random_state=42, verbose=0)
vae.fit(X_train_normal)
evaluate("VAE", y_test, vae.predict(X_test))
joblib.dump(vae, "model_vae.pkl")

print("\nAll models saved, done!")