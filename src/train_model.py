import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

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

# define model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# 5-fold group cross validation
gkf = GroupKFold(n_splits=N_SPLITS)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

fold_num = 1

for train_idx, val_idx in gkf.split(X_all_flat, y_all, groups=engine_id):
    print(f"========== Fold {fold_num} ==========")

    X_train_fold = X_all_flat[train_idx]
    y_train_fold = y_all[train_idx]

    X_val_fold = X_all_flat[val_idx]
    y_val_fold = y_all[val_idx]

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

# print average cross-validation result
print("========== 5-Fold Average Result ==========")
print("Average Accuracy :", round(np.mean(accuracy_list), 4))
print("Average Precision:", round(np.mean(precision_list), 4))
print("Average Recall   :", round(np.mean(recall_list), 4))
print("Average F1-score :", round(np.mean(f1_list), 4))
print()

# final training on all training data
# then evaluate on the test set
model.fit(X_all_flat, y_all)
y_test_pred = model.predict(X_test_flat)

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print("========== Final Test Result ==========")
print("Test Accuracy :", round(test_acc, 4))
print()

# confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

print("========== Confusion Matrix ==========")
print(cm)
print()

# classification report
print("========== Classification Report ==========")
print(classification_report(y_test, y_test_pred, zero_division=0))