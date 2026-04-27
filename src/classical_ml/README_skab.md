# SKAB Classical ML Models — Fault Detection
**CITS5206 Capstone Project | Group 14 | University of Western Australia**
**Author: Parinitha Gurram | Branch: Parinitha-skab-classical-ml**

---

## What This Work Does

This folder contains three supervised classification models trained on the SKAB dataset to detect water pump faults **10 seconds in advance**. The models are:

- Logistic Regression (`train_skab_lr.py`)
- K-Nearest Neighbours (`train_skab_knn.py`)
- Support Vector Machine (`train_skab_svm.py`)

---

## Dataset

**Source:** SKAB (Skoltech Anomaly Benchmark) — preprocessed by David (Strategy A)

| Split | Files Used | Windows | Fault % |
|---|---|---|---|
| Train | valve1/0.csv – valve1/11.csv | 13,245 | 36.36% |
| Val | valve1/12.csv – valve1/15.csv | 4,453 | 36.76% |
| Test | valve2/0.csv – valve2/3.csv | 4,196 | 37.01% |

**Label definition:** `y=1` means a fault will occur within the next 10 timesteps (early warning). `y=0` means normal operation.

**Why valve2 as test?** It comes from a different physical valve than the training data — this tests whether the model generalises across hardware units, which is a realistic industrial scenario.

---

## How the Data Was Prepared

David's preprocessing script (`Build_Dataset_SKAB_DLpipeline_By_David.py`) generates a `.npz` file with 3D arrays of shape `(N, 20, 8)` — N windows, 20 timesteps, 8 sensor features.

For classical ML models, we apply two additional steps:

1. **Flatten:** `(N, 20, 8)` → `(N, 160)` — required by sklearn models
2. **Scale:** `StandardScaler` fitted on training set only, then applied to val and test — prevents data leakage

---

## How to Run

### Requirements
```
pip install numpy scikit-learn
```

### Step 1 — Generate preprocessed data (run once)
```
cd src/data_preprocessing/SKAB
python ../Build_Dataset_SKAB_DLpipeline_By_David.py
```

### Step 2 — Run models (from project root)
```
python src/classical_ml/train_skab_lr.py
python src/classical_ml/train_skab_knn.py
python src/classical_ml/train_skab_svm.py
```

### Output locations
```
results/
├── lr_parinitha_dataset2.txt
├── knn_parinitha_dataset2.txt
└── svm_parinitha_dataset2.txt

models/skab_classical_ml/
├── model_lr_skab.pkl    + scaler_lr_skab.pkl
├── model_knn_skab.pkl   + scaler_knn_skab.pkl
└── model_svm_skab.pkl   + scaler_svm_skab.pkl
```

---

## Results

### Test Set Performance

| Model | Precision | Recall | F1-Score | FPR | Faults Caught |
|---|---|---|---|---|---|
| Logistic Regression | 99.51% | 77.98% | 87.44% | 0.23% | 1211 / 1553 |
| KNN (k=15) | 95.48% | 54.41% | 69.32% | 1.51% | 845 / 1553 |
| **SVM (LinearSVC)** | **98.93%** | **83.39%** | **90.50%** | **0.53%** | **1295 / 1553** |

**Primary metric is Recall** — catching faults is the priority in a maintenance system.

### Confusion Matrices

**Logistic Regression**
```
                     Predicted Normal   Predicted Fault
Actual Normal               2,637                   6
Actual Fault                  342               1,211
```

**KNN**
```
                     Predicted Normal   Predicted Fault
Actual Normal               2,603                  40
Actual Fault                  708                 845
```

**SVM (LinearSVC)**
```
                     Predicted Normal   Predicted Fault
Actual Normal               2,629                  14
Actual Fault                  258               1,295
```

---

## Hyperparameter Tuning

All models were tuned on the validation set using **F1-score** as the selection criterion.

| Model | Parameter Tuned | Values Tried | Best Value |
|---|---|---|---|
| Logistic Regression | C | 0.001, 0.01, 0.1, 1.0, 10.0 | 0.001 |
| KNN | k (neighbours) | 3, 5, 7, 11, 15 | 15 |
| SVM | C | 0.001, 0.01, 0.1, 1.0 | 0.001 |

---

## Key Observations

**SVM is the best model** — highest recall (83.39%) and F1 (90.50%). It catches the most faults with very few false alarms (only 14 false positives out of 2,643 normal windows).

**LR is a strong linear baseline** — very high precision (99.51%) but misses more faults than SVM (342 missed vs 258). Fast to train (0.05 seconds).

**KNN underperforms** — recall drops to 54.41%, missing nearly half of all faults. KNN struggles with high-dimensional data (160 features) due to the curse of dimensionality.

**SKAB vs Zenodo** — on Zenodo (dataset1) all 3 models had near-zero recall due to extreme imbalance (0.3% fault rate, only 30 test faults). On SKAB (dataset2) with 37% fault rate and 1,553 test faults, all models perform dramatically better, confirming SKAB is a better fit for supervised fault classification.

---

## Conclusion

SVM (LinearSVC) is the best classical ML baseline for this task — 83.39% recall, 90.50% F1, catching 1,295 out of 1,553 fault windows with only 14 false alarms. This sets a strong benchmark for comparison with deep learning models trained on the same dataset.