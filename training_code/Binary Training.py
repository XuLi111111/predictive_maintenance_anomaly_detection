import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("sensor_features.csv")
df = df.drop(columns=["timestamp"])

X = df.drop(columns=["label"]).values
y = df["label"].values

# ── 2. Convert to binary classification: 1/2 -> 1 (Anomaly) ───
y_binary = (y > 0).astype(int)
print(f"Data shape: {X.shape}")
print(f"Label distribution: Normal={np.sum(y_binary==0)}, Anomaly={np.sum(y_binary==1)}")

# ── 3. Time-based split (80% train, 20% test) ────────────────
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_binary[:split], y_binary[split:]

# ── 4. Standardization ───────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. Apply SMOTE ───────────────────────────────────────────
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: Normal={np.sum(y_train_sm==0)}, Anomaly={np.sum(y_train_sm==1)}")

# ── 6. Evaluation function ───────────────────────────────────
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

# ── 7. Training and evaluation ───────────────────────────────

# Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
evaluate("Random Forest", y_test, rf.predict(X_test))

# XGBoost
print("\nTraining XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric="logloss", n_jobs=-1)
xgb.fit(X_train_sm, y_train_sm)
evaluate("XGBoost", y_test, xgb.predict(X_test))

# k-NN
print("\nTraining k-NN...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_sm, y_train_sm)
evaluate("k-NN", y_test, knn.predict(X_test))

joblib.dump(rf,  "model_rf.pkl")
joblib.dump(xgb, "model_xgb.pkl")
joblib.dump(knn, "model_knn.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler as well

print("Models saved")

print("\nDone!")
