import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

# ============================================================
# SKAB Boosting Models Training (Strategy A) - By Nafisa
# ============================================================
# Dataset: skab_strategyA_window20_horizon10.npz
# ============================================================

# =====================
# Config
# =====================
DATA_PATH = "../../data/processed/dataset2/skab_strategyA_window20_horizon10.npz"
MODEL_SAVE_DIR = "../../models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# =====================
# Load dataset
# =====================
print("Loading dataset...")
data = np.load(DATA_PATH)

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"Train positive ratio: {np.mean(y_train == 1) * 100:.2f}%")
print(f"Val positive ratio:   {np.mean(y_val == 1) * 100:.2f}%")

# =====================
# Flatten 3D -> 2D
# =====================
def flatten(X):
    return X.reshape(X.shape[0], -1)

X_train_flat = flatten(X_train)
X_val_flat = flatten(X_val)

print(f"\nFlattened X_train shape: {X_train_flat.shape}")
print(f"Flattened X_val shape:   {X_val_flat.shape}")

# =====================
# Standardisation
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled = scaler.transform(X_val_flat)

scaler_path = os.path.join(MODEL_SAVE_DIR, "boosting_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved to: {scaler_path}")

# =====================
# XGBoost
# =====================
print("\n" + "=" * 60)
print("TRAINING: XGBoost")
print("=" * 60)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    eval_metric="logloss",
    early_stopping_rounds=20,
    n_jobs=-1,
    random_state=42
)

xgb.fit(
    X_train_flat, y_train,
    eval_set=[(X_val_flat, y_val)],
    verbose=50
)

xgb_path = os.path.join(MODEL_SAVE_DIR, "boosting_xgboost.pkl")
joblib.dump(xgb, xgb_path)
print(f"XGBoost saved to: {xgb_path}")

# =====================
# AdaBoost
# =====================
print("\n" + "=" * 60)
print("TRAINING: AdaBoost")
print("=" * 60)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

ada.fit(X_train_flat, y_train)

val_score = ada.score(X_val_flat, y_val)
print(f"Validation accuracy: {val_score * 100:.2f}%")

ada_path = os.path.join(MODEL_SAVE_DIR, "boosting_adaboost.pkl")
joblib.dump(ada, ada_path)
print(f"AdaBoost saved to: {ada_path}")

# =====================
# Gradient Boosting
# =====================
print("\n" + "=" * 60)
print("TRAINING: Gradient Boosting")
print("=" * 60)

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    random_state=42
)

gb.fit(X_train_flat, y_train)

val_score = gb.score(X_val_flat, y_val)
print(f"Validation accuracy: {val_score * 100:.2f}%")

gb_path = os.path.join(MODEL_SAVE_DIR, "boosting_gradient_boosting.pkl")
joblib.dump(gb, gb_path)
print(f"Gradient Boosting saved to: {gb_path}")

# =====================
# LightGBM
# =====================
print("\n" + "=" * 60)
print("TRAINING: LightGBM")
print("=" * 60)

lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    num_leaves=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm.fit(
    X_train_flat, y_train,
    eval_set=[(X_val_flat, y_val)]
)

X_val_df = pd.DataFrame(X_val_flat)
val_score = lgbm.score(X_val_df, y_val)
print(f"Validation accuracy: {val_score * 100:.2f}%")

lgbm_path = os.path.join(MODEL_SAVE_DIR, "boosting_lightgbm.pkl")
joblib.dump(lgbm, lgbm_path)
print(f"LightGBM saved to: {lgbm_path}")