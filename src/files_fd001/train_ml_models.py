import copy
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.neighbors import KNeighborsClassifier
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
# (samples, time_steps, features) -> (samples, time_steps * features)
X_all_flat = X_all.reshape(X_all.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


# define baseline models
models = {
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
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    ),
}


all_results = []
trained_models = {}

for model_name, base_model in models.items():
    print(f"{'=' * 12} {model_name} {'=' * 12}")

    gkf = GroupKFold(n_splits=N_SPLITS)

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    fold_num = 1

    for train_idx, val_idx in gkf.split(X_all_flat, y_all, groups=engine_id):
        print(f"---------- Fold {fold_num} ----------")

        X_train_fold = X_all_flat[train_idx]
        y_train_fold = y_all[train_idx]

        X_val_fold = X_all_flat[val_idx]
        y_val_fold = y_all[val_idx]

        model = clone(base_model)
        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)

        acc = accuracy_score(y_val_fold, y_val_pred)
        prec = precision_score(y_val_fold, y_val_pred, zero_division=0)
        rec = recall_score(y_val_fold, y_val_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)

        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        print("Recall   :", round(rec, 4))
        print("F1-score :", round(f1, 4))
        print()

        fold_num += 1

    cv_accuracy = float(np.mean(accuracy_list))
    cv_precision = float(np.mean(precision_list))
    cv_recall = float(np.mean(recall_list))
    cv_f1 = float(np.mean(f1_list))

    print("========== 5-Fold Average Result ==========")
    print("Average Accuracy :", round(cv_accuracy, 4))
    print("Average Precision:", round(cv_precision, 4))
    print("Average Recall   :", round(cv_recall, 4))
    print("Average F1-score :", round(cv_f1, 4))
    print()

    final_model = clone(base_model)
    final_model.fit(X_all_flat, y_all)
    y_test_pred = final_model.predict(X_test_flat)

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
    print()

    all_results.append(
        {
            "Model": model_name,
            "CV Accuracy": cv_accuracy,
            "CV Precision": cv_precision,
            "CV Recall": cv_recall,
            "CV F1": cv_f1,
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
            "Test F1": test_f1,
        }
    )

    trained_models[model_name] = final_model


print("========== Final Summary ==========")
all_results = sorted(all_results, key=lambda x: (x["Test Recall"], x["Test F1"]), reverse=True)
for result in all_results:
    print(
        f"{result['Model']:<20} "
        f"CV Recall={result['CV Recall']:.4f}  "
        f"CV F1={result['CV F1']:.4f}  "
        f"Test Recall={result['Test Recall']:.4f}  "
        f"Test F1={result['Test F1']:.4f}"
    )