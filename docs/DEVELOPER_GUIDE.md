# pump.detect Developer Guide

CITS5206 Information Technology Capstone Project — Group 14  
Predictive Maintenance Anomaly Detection for Industrial Pumps

## 1. Purpose

This document provides developer-focused handover notes for the `pump.detect` project. It explains the codebase structure, main modules, model integration, API responsibilities, frontend responsibilities, development workflow, testing expectations, and known technical limitations.

Setup and running instructions are provided separately in the project README.

---

## 2. Repository Overview

The repository is organised into application code, machine learning scripts, data folders, and documentation.

```text
predictive_maintenance_anomaly_detection/
├── app/
├── data/
├── docs/
├── src/
├── .gitignore
├── README.md
├── requirements.txt
└── SKAB_Preprocessing_README.md
```

| Path | Purpose |
|---|---|
| `app/` | Deployable web application containing backend, frontend, scripts, and app-level documentation |
| `data/` | Raw and processed dataset folders |
| `docs/` | Handover documentation such as user guide, developer guide, and deployment guide |
| `src/` | Offline preprocessing and model training scripts |
| `README.md` | Main project overview and workflow |
| `requirements.txt` | Python dependencies for model training and preprocessing |
| `SKAB_Preprocessing_README.md` | Notes for SKAB dataset preprocessing |

---

## 3. Application Structure

The main web application is located in the `app/` folder.

```text
app/
├── backend/
├── frontend/
├── scripts/
├── docker-compose.yml
├── README.md
└── CONTRIBUTING.md
```

| Path | Purpose |
|---|---|
| `app/backend/` | FastAPI backend for upload validation, model inference, live monitoring, streaming, and report generation |
| `app/frontend/` | React/Vite frontend for the user interface |
| `app/scripts/` | Utility scripts, including the local pump simulator |
| `app/docker-compose.yml` | Application container configuration |
| `app/README.md` | Application-level setup notes |
| `app/CONTRIBUTING.md` | Contribution and implementation notes |

---

## 4. Backend Responsibilities

The backend is responsible for the system logic behind prediction, validation, streaming, and reporting.

Main backend responsibilities:

- Validate uploaded CSV files.
- Check the required SKAB sensor schema.
- Load trained machine learning model artifacts.
- Run batch prediction on uploaded files.
- Compare prediction outputs across multiple models.
- Stream replay results to the frontend.
- Receive live or simulated sensor readings.
- Convert prediction probabilities into alert states.
- Generate downloadable PDF reports.
- Provide API endpoints for frontend communication.

The backend should keep validation, model loading, inference, live state handling, and report generation separated so that future changes can be made without rewriting the full application.

---

## 5. Frontend Responsibilities

The frontend provides the user-facing interface for the project.

Main frontend responsibilities:

- Show the landing/home page.
- Allow users to upload CSV files.
- Display clear upload success and error messages.
- Allow users to select a model.
- Allow users to adjust the anomaly threshold.
- Display anomaly probability charts.
- Show alert states such as NORMAL, WATCH, WARNING, and ALERT.
- Display model comparison results.
- Support live monitoring views.
- Allow users to download PDF reports.

Frontend features should always give clear feedback for loading, success, and error states so that non-technical users can understand what is happening.

---

## 6. Machine Learning Pipeline

The machine learning pipeline is stored mainly in the `src/` folder.

The final system uses SKAB pump sensor data as the main dataset. The ML pipeline includes:

- SKAB dataset preprocessing.
- Training and validation splits.
- Classical machine learning baselines.
- Boosting models.
- TransformerFusionLite deep learning model.
- Exporting trained artifacts for backend inference.

Models used in the final system include:

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Tree-based model |
| Extra Trees | Tree-based ensemble |
| Gradient Boosting | Boosting model |
| XGBoost | Boosting model |
| KNN | Instance-based model |
| SVM | Support vector machine |
| TransformerFusionLite | Deep learning model |

The TransformerFusionLite model achieved the strongest reported final result, with F1 = 0.9244 on the SKAB held-out test set.

---

## 7. Model Artifact Integration

The backend requires trained model artifacts to perform inference.

Expected artifact folder:

```text
app/backend/artifacts/
```

Typical model artifacts include:

```text
scaler.pkl
transformer_scaler.pkl
sample.csv
model_lr.pkl
model_rf.pkl
model_svm.pkl
model_et.pkl
model_gb.pkl
model_knn.pkl
model_xgb.pkl
model_transformer.pt
transformer_threshold.json
```

Future developers adding or replacing models should ensure that:

1. The model artifact is saved in a backend-readable format.
2. Any scaler or preprocessing object required by the model is also saved.
3. The backend model registry is updated.
4. The frontend model selector can display the new model if it is user-facing.
5. The user guide is updated if the new model changes how users interact with the system.

Generated artifacts should not be committed if they are large or excluded by `.gitignore`.

---

## 8. API Responsibilities

The frontend communicates with the backend through REST endpoints and WebSocket streams.

Common API responsibilities include:

| API area | Responsibility |
|---|---|
| Health check | Confirm backend availability |
| Model list | Return available models |
| Sample CSV | Provide an example input file |
| Upload | Validate and stage uploaded CSV data |
| Prediction | Run anomaly prediction for a selected model |
| Model comparison | Compare results across multiple models |
| Report | Generate PDF report output |
| Replay stream | Stream uploaded CSV prediction results |
| Live ingest | Receive one live sensor sample |
| Live config | Update live model or threshold settings |
| Live stream | Send live prediction updates to the frontend |

When adding or editing APIs, developers should keep responses consistent and return clear error messages that the frontend can display to users.

---

## 9. Data Schema

The final application expects SKAB-format pump sensor data.

Required columns:

```text
datetime
Accelerometer1RMS
Accelerometer2RMS
Current
Pressure
Temperature
Thermocouple
Voltage
Volume Flow RateRMS
```

Optional label column:

```text
anomaly
```

Developers should be careful when changing validation or preprocessing logic because the trained models expect the same sensor order and feature format used during training.

---

## 10. Alert State Logic

The application converts anomaly probabilities into user-friendly alert states.

| State | Meaning |
|---|---|
| NORMAL | Low anomaly risk |
| WATCH | Slight increase in anomaly probability |
| WARNING | Elevated anomaly risk |
| ALERT | High anomaly risk |

The alert state logic helps users interpret model output without needing to understand raw probability values. Future changes to thresholding or smoothing should be tested carefully because they affect the user-facing warning behaviour.

---

## 11. Development Workflow

Recommended development workflow:

1. Create a feature branch for each task.
2. Keep commits small and focused.
3. Use meaningful commit messages.
4. Test the affected backend or frontend workflow locally.
5. Open a pull request before merging.
6. Request review from at least one team member.
7. Update documentation if the change affects users, developers, models, or APIs.

Example branch names:

```text
feature/upload-validation
feature/live-monitoring-chart
feature/pdf-report-update
bugfix/model-loading-error
docs/developer-guide
```

Example commit messages:

```text
feat(upload): improve CSV validation messages
feat(results): add model comparison summary
fix(live): handle stale sensor stream
docs(handover): add developer guide
test(api): add upload validation test
```

---

## 12. Testing and Quality Assurance

Future developers should test the relevant part of the system before merging changes.

Recommended checks:

| Area | Checks |
|---|---|
| Backend | API response, validation logic, inference logic, report generation |
| Frontend | Page loading, navigation, form behaviour, charts, buttons, error messages |
| Upload workflow | Valid CSV, invalid CSV, missing columns, wrong file type |
| Prediction workflow | Model selection, threshold adjustment, prediction output |
| Live workflow | Incoming sensor ticks, alert state changes, data-quality warnings |
| Report workflow | PDF generation and download |
| Documentation | Update user/developer docs when behaviour changes |

Testing should include both expected behaviour and common failure cases.

---

## 13. Adding a New Model

To add a new model:

1. Train the model using the offline ML pipeline.
2. Save the model artifact and any required scaler/preprocessor.
3. Add the artifact to the backend artifacts folder.
4. Register the model in the backend model-loading logic.
5. Confirm the model appears in the available model list.
6. Test prediction with a valid SKAB-format CSV.
7. Update the frontend model selector if needed.
8. Update documentation if the model is part of the final user-facing system.

---

## 14. Adding a New Frontend Feature

To add a frontend feature:

1. Create a feature branch.
2. Add or update the relevant React component.
3. Connect to the backend API if required.
4. Add loading, success, and error states.
5. Test the feature in the browser.
6. Check that existing upload, results, and live pages still work.
7. Update documentation if user behaviour changes.

Frontend changes should avoid hiding important errors from the user.

---

## 15. Adding a New Backend Feature

To add a backend feature:

1. Create a feature branch.
2. Add or update the relevant backend module.
3. Validate all input data.
4. Return structured success and error responses.
5. Add tests where possible.
6. Confirm the frontend can handle the response.
7. Update documentation if the API or behaviour changes.

Backend changes should avoid tightly coupling unrelated logic. For example, upload validation, model inference, and report generation should remain separate where possible.

---

## 16. Code Quality Notes

Future development should follow these principles:

- Keep functions small and focused.
- Use clear names for files, functions, variables, and components.
- Avoid duplicating validation or preprocessing logic.
- Keep API responses consistent.
- Handle errors clearly instead of failing silently.
- Keep model-specific logic separated from general application logic where possible.
- Update documentation when implementation changes affect users or developers.

---

## 17. Known Technical Limitations

Known limitations of the current implementation:

- The system is trained and tested on SKAB data only.
- Real client pump data has not yet been used for validation.
- The prediction task is binary anomaly detection, not detailed fault diagnosis.
- The live workflow uses simulated data rather than a real SCADA or PLC connection.
- Model artifacts are not stored directly in GitHub.
- There is no user authentication or role-based access control.
- Prediction history is not persisted in a production database.
- Further testing is required before operational deployment.

---

## 18. Recommended Future Development

Recommended future improvements:

1. Validate the system using real client pump data.
2. Add model retraining or recalibration support.
3. Add explainability to show which sensors influenced predictions.
4. Add persistent storage for prediction history.
5. Add authentication if the system is made public.
6. Integrate with a real industrial data source.
7. Extend the system from binary anomaly detection to fault-type diagnosis.
8. Improve model monitoring for drift or degraded performance.

---

## 19. Handover Notes for Future Developers

Before making major changes, future developers should understand:

- The application is designed around SKAB-format pump data.
- The backend depends on model artifacts being available.
- The frontend expects clear API responses from the backend.
- The current live workflow is simulation-based.
- The system is a capstone prototype and should be validated further before real operational use.