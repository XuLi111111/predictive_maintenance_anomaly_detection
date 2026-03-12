# Predictive Maintenance – Anomaly Detection for Industrial Pumps

**CITS5206 Capstone Project | Group 14 | University of Western Australia**

---

## Project Overview

This project develops a machine learning framework for short-horizon predictive maintenance of industrial centrifugal pumps. Rather than estimating remaining useful life, the system aims to detect early signs of abnormal behaviour and generate failure warnings approximately 5–30 minutes in advance, giving operators actionable lead time before a fault escalates.

The project is conducted in collaboration with client Peter Whittaker (Mechanical Engineering, UWA) and industry partner Programmed.

---

## Team

| Name | Student ID | Role |
|---|---|---|
| David Du | 24074639 | |
| Xu Li | 24269773 | |
| Parinitha Gurram | 24038731 | |
| Shouvik Barua Pratik | 23869695 | |
| Nafisa Tabassum | 24627403 | |

---

## Repository Structure

```
predictive_maintenance_anomaly_detection/
├── raw_data/          # Raw datasets (not tracked by Git)
├── src/               # Source code
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Dataset

The project currently uses the [Pump Sensor Data](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) dataset from Kaggle, which contains time-series sensor readings from an industrial water pump with labelled machine status (Normal / BROKEN / RECOVERING).

The NASA Turbofan Engine Degradation Simulation Dataset (FD001) was also used in the early development stage for pipeline validation.

> Raw data files are not committed to this repository. Please download the dataset separately and place it in the `raw_data/` directory.

---

## Technical Approach

- **Labelling**: dual-threshold strategy combining fixed fault indicators and adaptive statistical thresholds (95th percentile) to assign Normal, EarlyWarning, and CriticalAlert labels
- **Feature extraction**: statistical features (mean, std, min, max, trend) extracted over sliding time windows
- **Class imbalance**: addressed using SMOTE
- **Models**: Isolation Forest, Local Outlier Factor, One-Class SVM, Random Forest, XGBoost, Autoencoder
- **Evaluation metrics**: Precision, Recall, F1-score, False Positive Rate, False Negative Rate

---

## Setup

```bash
# Clone the repository
git clone https://github.com/XuLi111111/predictive_maintenance_anomaly_detection.git
cd predictive_maintenance_anomaly_detection

# Install dependencies
pip install -r requirements.txt
```

---

## Dependencies

See `requirements.txt`. Key libraries include:

- Python 3.8+
- scikit-learn
- PyOD
- PyTorch
- pandas
- numpy
- imbalanced-learn

---

## License

This project is developed for academic purposes as part of the CITS5206 Capstone unit at UWA.
