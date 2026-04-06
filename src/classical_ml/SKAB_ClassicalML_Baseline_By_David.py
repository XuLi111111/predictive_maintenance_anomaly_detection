import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

MODEL_SAVE_DIR = "../data/processed/dataset2/skab_classical_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# =========================
# Load dataset
# =========================
data = np.load("../data/processed/dataset2/skab_strategyA_window20_horizon10.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

# =========================
# Flatten for ML models
# =========================
def flatten(X):
    return X.reshape(X.shape[0], -1)

X_train_flat = flatten(X_train)
X_val_flat = flatten(X_val)
X_test_flat = flatten(X_test)

print("Data shapes:")
print(X_train_flat.shape, X_val_flat.shape, X_test_flat.shape)

# =========================
# Standardization (for SVM / LR)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)
X_test_scaled = scaler.transform(X_test_flat)

joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
print(f"Scaler saved to {MODEL_SAVE_DIR}/scaler.pkl")

# =========================
# Helper function
# =========================
def evaluate_model(name, model, X_tr, X_te, save_name):
    print("\n" + "=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    save_path = os.path.join(MODEL_SAVE_DIR, f"{save_name}.pkl")
    joblib.dump(model, save_path)
    print(f"Model saved: {save_path}")


# =========================
# Models
# =========================

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000)
evaluate_model("Logistic Regression", lr, X_train_scaled, X_test_scaled, "model_lr")

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
evaluate_model("Random Forest", rf, X_train_flat, X_test_flat, "model_rf")

# 3. SVM
svm = SVC(kernel='rbf')
evaluate_model("SVM", svm, X_train_scaled, X_test_scaled, "model_svm")

# 4. Extra Trees
et = ExtraTreesClassifier(n_estimators=200, n_jobs=-1)
evaluate_model("Extra Trees", et, X_train_flat, X_test_flat, "model_et")

# 5. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200)
evaluate_model("Gradient Boosting", gb, X_train_flat, X_test_flat, "model_gb")

# 6. KNN
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model("KNN", knn, X_train_scaled, X_test_scaled, "model_knn")

# 7. XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    n_jobs=-1
)
evaluate_model("XGBoost", xgb, X_train_flat, X_test_flat, "model_xgb")

print(f"\nAll models saved to: {MODEL_SAVE_DIR}")
