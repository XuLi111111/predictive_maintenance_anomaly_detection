# Classical ML Baseline Models — Fault Detection
**CITS5206 Capstone Project | Group 14 | University of Western Australia**
**Author: Parinitha Gurram | Branch: Parinitha-branch**

---

## Overview

This folder contains baseline supervised classification models for predicting water pump faults 30 minutes in advance. The models are trained on the Zenodo University Water Supply System dataset, preprocessed by the data preprocessing team.

The goal of these baseline models is to establish a performance reference point for comparison with more advanced models developed by the team.

---

## Dataset

| Property | Value |
|---|---|
| Source | Zenodo University Water Supply Sensor Dataset |
| Time range | 2023-07-01 → 2024-06-30 |
| Features | 240 (flattened 30-step × 8-sensor sliding window) |
| Task | Binary classification: 0 = Normal, 1 = Fault imminent |
| Warning horizon | 30 minutes before fault onset |

### Data Splits

| Split | Windows | Fault Windows | Fault % | Purpose |
|---|---|---|---|---|
| Train | 854,218 | 2,529 | 0.296% | Model training |
| Val | 256,737 | 0 | 0.000% | Hyperparameter tuning (fault-free) |
| Test | 613,895 | 30 | 0.005% | Final evaluation |

### Class Imbalance

The dataset has extreme class imbalance — fault windows represent only **0.296%** of training data (approximately 337 normal windows for every 1 fault window). This makes accuracy a misleading metric. All models use `class_weight='balanced'` or SMOTE to compensate.

---

## Models Implemented

### 1. Logistic Regression (`train_lr.py`)

A linear model used as the primary baseline. Learns a weighted combination of the 240 input features to separate normal from fault windows.

**Imbalance handling:** `class_weight='balanced'`

**Hyperparameter tuned:** `C` (regularisation strength) — explored values: `[0.001, 0.01, 0.1, 1.0, 10.0]`

**Tuning criterion:** Minimise False Positive Rate on validation set (val is fault-free, so every predicted fault is a false alarm)

### 2. K-Nearest Neighbours (`train_knn.py`)

A distance-based model that classifies each window by majority vote from its k nearest neighbours in the training set.

**Imbalance handling:** SMOTE oversampling — KNN does not support `class_weight`. SMOTE applied to a 100,000-row subsample (all fault rows retained) before training.

**Hyperparameter tuned:** `k` (number of neighbours) — explored values: `[3, 5, 7, 11, 15]`

**Tuning criterion:** Minimise False Positive Rate on validation set

### 3. Support Vector Machine (`train_svm.py`)

Uses `LinearSVC` instead of `SVC` with RBF kernel. For high-dimensional data (240 features), linear kernels perform as well or better than RBF and scale efficiently to datasets of 800k+ rows without subsampling.

**Imbalance handling:** `class_weight='balanced'`

**Hyperparameter tuned:** `C` — explored values: `[0.001, 0.01, 0.1, 1.0]`

**Tuning criterion:** Minimise False Positive Rate on validation set

---

## Results

### Primary Metric

**Recall** is the primary evaluation metric for this task. Missing a fault (False Negative) is far more costly than a false alarm (False Positive) in a maintenance context. A model that catches more faults — even at the cost of more false alarms — is preferred.

### Test Set Performance

| Model | Precision | Recall | F1-Score | FPR | FNR | Faults Caught |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.00% | 3.33% | 0.00% | 22.34% | 96.67% | 1 / 30 |
| KNN | 0.00% | 0.00% | 0.00% | 2.13% | 100.00% | 0 / 30 |
| SVM (LinearSVC) | 0.00% | 0.00% | 0.00% | 21.85% | 100.00% | 0 / 30 |

### Confusion Matrices

**Logistic Regression**
```
                     Predicted Normal   Predicted Fault
Actual Normal              476,701           137,164
Actual Fault                    29                 1
```

**KNN**
```
                     Predicted Normal   Predicted Fault
Actual Normal              600,804            13,061
Actual Fault                    30                 0
```

**SVM (LinearSVC)**
```
                     Predicted Normal   Predicted Fault
Actual Normal              479,719           134,146
Actual Fault                    30                 0
```

---

## Key Observations

### Why recall is so low

The test set contains only **30 fault windows** out of 613,895 total windows. These 30 fault windows represent a single fault event (2024-03-13) — a very short, specific pattern the model has never seen in exactly this form. The training data contains faults from a different period (Sep–Oct 2023), so the models are generalising across different fault instances.

This extremely low fault density in the test set (0.005%) means even small classification thresholds shifts have a large impact on whether faults are caught.

### Logistic Regression

- Caught 1 out of 30 fault windows (3.33% recall)
- Generated 137,164 false alarms — very high FPR (22.34%)
- `class_weight='balanced'` pushes the model to predict more faults but the linear boundary cannot reliably separate the fault pattern from normal operation
- Best C selected: 0.001 (strongest regularisation minimised false alarms on val)

### KNN

- Caught 0 out of 30 fault windows (0% recall)
- Generated 13,061 false alarms — lower FPR than LR (2.13%)
- SMOTE helped reduce false alarms but the model still missed all faults
- KNN struggles with high-dimensional data (240 features) — distance metrics become less meaningful in high dimensions (curse of dimensionality)
- Best k selected: 3 (smallest neighbourhood minimised false alarms on val)

### SVM (LinearSVC)

- Caught 0 out of 30 fault windows (0% recall)
- Generated 134,146 false alarms — high FPR (21.85%), similar to LR
- LinearSVC was chosen over RBF kernel SVC as it scales to the full 854k training set without subsampling
- Best C selected: 0.001 — stronger regularisation reduced false alarms on val
- Training time: 319.98 seconds on full dataset
- Despite training on all data, the linear boundary still cannot isolate the fault signal from the extreme imbalance

### Overall baseline conclusion

Both LR and KNN establish a clear performance floor. The results confirm that simple baseline models are insufficient for this task given the extreme imbalance and small fault signal. More advanced models (Random Forest, XGBoost, deep learning) are expected to perform significantly better by learning non-linear fault patterns from the training data.

---

## How to Run

### Requirements
```
pip install numpy scikit-learn imbalanced-learn
```

### Folder Structure Required
```
processed/
└── zenodo_pump_2d/
    ├── train.npz
    ├── val.npz
    └── test.npz
results/        ← created automatically
```

### Run Each Model
```
python src/classical_ml/train_lr.py
python src/classical_ml/train_knn.py
python src/classical_ml/train_svm.py
```

### Output Files
Results are saved automatically to `results/`:
```
results/
├── lr_parinitha_dataset1.txt
├── knn_parinitha_dataset1.txt
└── svm_parinitha_dataset1.txt
```

---

## Notes

- Temporal order is preserved throughout — no random shuffling applied anywhere
- The validation set is fault-free by design — it serves as a false positive rate benchmark, not a recall benchmark
- The scaler was fitted on the training set only during preprocessing — no data leakage
- All scripts run independently and can be executed in any order