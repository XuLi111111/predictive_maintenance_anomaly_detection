

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# basic settings
N_SPLITS = 5
RANDOM_STATE = 42

# processed data paths
X_ALL_PATH = "../processed_data/X_all.npy"
Y_ALL_PATH = "../processed_data/y_all.npy"
ENGINE_ID_PATH = "../processed_data/engine_id.npy"
X_TEST_PATH = "../processed_data/X_test.npy"
Y_TEST_PATH = "../processed_data/y_test.npy"


# load processed dataset
X_all = np.load(X_ALL_PATH)
y_all = np.load(Y_ALL_PATH)
engine_id = np.load(ENGINE_ID_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("Data loaded successfully.")
print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)
print("engine_id shape:", engine_id.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print()

# flatten 3D data into 2D for traditional ML models
X_all_flat = X_all.reshape(X_all.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# base models for 2-model stacking
base_models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
}

# meta model
meta_model_template = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)


def get_positive_prob(model, x):
    return model.predict_proba(x)[:, 1]


# 5-fold group cross validation for stacking
outer_gkf = GroupKFold(n_splits=N_SPLITS)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

fold_num = 1

for train_idx, val_idx in outer_gkf.split(X_all_flat, y_all, groups=engine_id):
    print(f"========== Outer Fold {fold_num} ==========")

    X_train_outer = X_all_flat[train_idx]
    y_train_outer = y_all[train_idx]
    groups_train_outer = engine_id[train_idx]

    X_val_outer = X_all_flat[val_idx]
    y_val_outer = y_all[val_idx]

    # build OOF predictions for meta-model training
    oof_meta_features = np.zeros((len(train_idx), len(base_models)))
    inner_gkf = GroupKFold(n_splits=N_SPLITS)

    for inner_train_idx, inner_val_idx in inner_gkf.split(
        X_train_outer, y_train_outer, groups=groups_train_outer
    ):
        X_train_inner = X_train_outer[inner_train_idx]
        y_train_inner = y_train_outer[inner_train_idx]

        X_val_inner = X_train_outer[inner_val_idx]

        for model_col, (model_name, model_template) in enumerate(base_models.items()):
            model = clone(model_template)
            model.fit(X_train_inner, y_train_inner)
            oof_meta_features[inner_val_idx, model_col] = get_positive_prob(model, X_val_inner)

    # train meta-model on OOF predictions
    meta_model = clone(meta_model_template)
    meta_model.fit(oof_meta_features, y_train_outer)

    # train base models on full outer-train fold
    val_meta_features = np.zeros((len(val_idx), len(base_models)))

    for model_col, (model_name, model_template) in enumerate(base_models.items()):
        model = clone(model_template)
        model.fit(X_train_outer, y_train_outer)
        val_meta_features[:, model_col] = get_positive_prob(model, X_val_outer)

    # final validation prediction from meta-model
    y_val_pred = meta_model.predict(val_meta_features)

    acc = accuracy_score(y_val_outer, y_val_pred)
    prec = precision_score(y_val_outer, y_val_pred, zero_division=0)
    rec = recall_score(y_val_outer, y_val_pred, zero_division=0)
    f1 = f1_score(y_val_outer, y_val_pred, zero_division=0)

    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)

    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print()

    fold_num += 1


print("========== 5-Fold Average Stacking Result ==========")
print("Average Accuracy :", round(float(np.mean(accuracy_list)), 4))
print("Average Precision:", round(float(np.mean(precision_list)), 4))
print("Average Recall   :", round(float(np.mean(recall_list)), 4))
print("Average F1-score :", round(float(np.mean(f1_list)), 4))
print()


# final training on all training data for test evaluation
full_meta_features = np.zeros((X_all_flat.shape[0], len(base_models)))
full_inner_gkf = GroupKFold(n_splits=N_SPLITS)

for inner_train_idx, inner_val_idx in full_inner_gkf.split(X_all_flat, y_all, groups=engine_id):
    X_train_inner = X_all_flat[inner_train_idx]
    y_train_inner = y_all[inner_train_idx]
    X_val_inner = X_all_flat[inner_val_idx]

    for model_col, (model_name, model_template) in enumerate(base_models.items()):
        model = clone(model_template)
        model.fit(X_train_inner, y_train_inner)
        full_meta_features[inner_val_idx, model_col] = get_positive_prob(model, X_val_inner)

final_meta_model = clone(meta_model_template)
final_meta_model.fit(full_meta_features, y_all)

# fit each base model on all training data, then build test meta features
X_test_meta = np.zeros((X_test_flat.shape[0], len(base_models)))

for model_col, (model_name, model_template) in enumerate(base_models.items()):
    model = clone(model_template)
    model.fit(X_all_flat, y_all)
    X_test_meta[:, model_col] = get_positive_prob(model, X_test_flat)

y_test_pred = final_meta_model.predict(X_test_meta)

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print("========== Final Test Result ==========")
print("Test Accuracy :", round(test_acc, 4))
print("Test Precision:", round(test_prec, 4))
print("Test Recall   :", round(test_rec, 4))
print("Test F1-score :", round(test_f1, 4))
print()

cm = confusion_matrix(y_test, y_test_pred)
print("========== Confusion Matrix ==========")
print(cm)
print()

print("========== Classification Report ==========")
print(classification_report(y_test, y_test_pred, zero_division=0))