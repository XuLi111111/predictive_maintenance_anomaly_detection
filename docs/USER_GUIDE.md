# pump.detect User Guide

CITS5206 Information Technology Capstone Project — Group 14  
Predictive Maintenance Anomaly Detection for Centrifugal Pumps

## 1. Purpose

pump.detect is a web application for detecting possible pump anomalies from sensor data. It allows users to upload SKAB-format pump sensor data, run anomaly prediction using trained machine learning models, view anomaly probabilities, monitor simulated live pump data, compare model outputs, and download a PDF report.

This guide explains how to use the application after it has already been started. Setup and deployment instructions are provided separately in the project README and Deployment Guide.

---

## 2. Main Workflows

The application supports two main workflows:

### 2.1 Batch Analysis

The batch analysis workflow is used when a user wants to upload a CSV file and analyse historical pump sensor data.

Main steps:

1. Upload a SKAB-format CSV file.
2. Select a trained prediction model.
3. Set the anomaly alert threshold.
4. Run the prediction or replay.
5. View anomaly probability and alert status.
6. Compare model outputs if required.
7. Download the generated PDF report.

### 2.2 Live Monitoring

The live monitoring workflow is used when a user wants to monitor incoming pump sensor readings in real time.

Main steps:

1. Open the live monitoring page.
2. View incoming sensor readings.
3. Monitor anomaly probability and alert state.
4. Adjust the active model or threshold if required.
5. Review data-quality warnings if they appear.

For the capstone demonstration, the live monitoring workflow uses simulated pump data rather than a real industrial pump connection.

---

## 3. Accessing the Application

Follow the setup instructions in the project README or Deployment Guide to start the application.

After the application is running, open the web interface in a browser:

```text
http://localhost:8080
```

Main pages:

| Page | Purpose |
|---|---|
| Home | Shows the project overview and navigation options |
| Upload | Allows users to upload SKAB-format CSV data and run prediction |
| Results | Displays prediction outputs, charts, summaries, and model comparison |
| Live | Shows real-time or simulated pump monitoring |

---

## 4. Required CSV Format

The upload page expects a SKAB-format CSV file.

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

If the `anomaly` column is included, the system can calculate labelled evaluation metrics such as precision, recall, and F1 score. If the `anomaly` column is not included, the system will still produce anomaly probabilities and alert states, but labelled evaluation metrics will not be available.

The system accepts CSV files using either comma `,` or semicolon `;` separators.

---

## 5. Uploading a CSV File

To upload a file:

1. Open the application.
2. Go to the **Upload** page.
3. Select a valid `.csv` file.
4. Wait for the file validation result.
5. If the file is valid, the page will show a success message.
6. If the file is invalid, the page will show an error message explaining what needs to be fixed.

Common upload issues:

| Issue | Meaning | How to fix |
|---|---|---|
| Wrong file type | The uploaded file is not a `.csv` file | Export or save the file as CSV |
| Missing columns | One or more required SKAB columns are missing | Rename or add the required columns |
| Empty file | The file has no usable data | Upload a CSV containing headers and sensor rows |
| Non-numeric values | A sensor column contains text or invalid values | Replace text placeholders with numeric readings |
| Missing sensor values | A sensor column contains blank or missing values | Fill, interpolate, or remove missing rows |
| Timestamp error | The `datetime` column cannot be parsed | Use a standard timestamp format such as `2020-03-09 15:56:30` |
| Too few rows | The file is shorter than the required prediction window | Upload a longer sequence of sensor readings |

---

## 6. Selecting a Model

After a valid CSV file is uploaded, the user can select one of the available trained models.

Available model types include:

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

The TransformerFusionLite model is the strongest reported model in the final system, with F1 = 0.9244 on the SKAB held-out test set.

---

## 7. Setting the Alert Threshold

The alert threshold controls how confident the model must be before the system marks a prediction as anomalous.

| Threshold setting | Behaviour |
|---|---|
| Lower threshold | More sensitive; may detect more anomalies but may produce more false alerts |
| Balanced threshold | Suitable for general demonstration use |
| Higher threshold | More conservative; fewer alerts but may miss weaker early signs |

Users can adjust the threshold depending on whether they want the system to prioritise early warning or reduce false alarms.

---

## 8. Running Batch Prediction

After uploading a valid file, selecting a model, and setting the threshold:

1. Click **Run Prediction** or **Start Replay**.
2. The application processes the uploaded sensor data.
3. The chart displays anomaly probability over time.
4. The status banner shows the current alert state.
5. Summary results are displayed after prediction.
6. The user can download a PDF report if required.

The system uses recent sensor readings to estimate whether an anomaly is likely.

---

## 9. Alert States

The application uses four alert states:

| State | Meaning |
|---|---|
| NORMAL | Low anomaly risk |
| WATCH | Slight increase in anomaly probability |
| WARNING | Elevated anomaly risk |
| ALERT | High anomaly risk |

These alert states help users understand the model output without needing to interpret raw probability values only.

---

## 10. Viewing Results

The results page may include:

- Anomaly probability chart
- Alert status summary
- Prediction statistics
- Model comparison results
- Anomaly timeline
- Confusion matrix if labelled data is available
- Evaluation metrics if labelled data is available
- PDF report download option

If the uploaded file does not contain the `anomaly` label column, the system will still show predictions, but ground-truth evaluation metrics will not be shown.

---

## 11. Downloading the PDF Report

After prediction is complete:

1. Go to the results section.
2. Click **Download PDF report**.
3. Save the generated PDF file locally.

The PDF report provides a shareable summary of the prediction run. It may include the anomaly timeline, model output, model comparison, and key result information.

---

## 12. Using Live Monitoring

The **Live** page displays real-time anomaly monitoring.

For the capstone demonstration, live data is produced by a local pump simulator. Setup instructions for the simulator are provided in the Deployment Guide.

Once the live data source is running, open:

```text
http://localhost:8080/live
```

The live page allows users to:

- View incoming pump sensor readings.
- See anomaly probability update over time.
- Monitor alert state changes.
- Change the active model.
- Adjust the alert threshold.
- Pause or clear the live display.
- View data-quality warnings.

Possible data-quality warnings include:

| Warning | Meaning |
|---|---|
| FROZEN_SENSOR | A sensor value appears stuck |
| OUT_OF_RANGE | A sensor value is outside the expected range |
| UNEVEN_SAMPLING | Incoming samples are not evenly spaced |
| STALE_DATA | No new sensor ticks have arrived recently |

---

## 13. Interpreting the Output

The system output should be used as a decision-support signal, not as a final maintenance decision by itself.

General interpretation:

| Output | Interpretation |
|---|---|
| Low anomaly probability | The current sensor pattern looks similar to normal SKAB pump behaviour |
| Increasing anomaly probability | The pump pattern may be moving away from normal behaviour |
| WATCH or WARNING | The user should monitor the system more closely |
| ALERT | The system has detected a high anomaly risk based on the selected model and threshold |

A high anomaly probability does not explain the exact physical fault. It indicates that the sensor pattern appears abnormal according to the trained model.

---

## 14. Current User Limitations

The current system is a capstone prototype and has the following limitations:

- The models were trained and tested on SKAB data.
- The system has not yet been validated on real client operational pump data.
- The current prediction task is binary anomaly detection, not detailed fault diagnosis.
- The live workflow uses a simulator rather than a real SCADA or PLC connection.
- Model artifacts must be available in the backend artifacts folder before prediction can run.
- The system does not currently include user authentication.
- Prediction history is not currently stored in a production database.

---

## 15. Recommended Use

pump.detect should be used as a demonstration and decision-support prototype for predictive maintenance. Before operational deployment, the system should be tested with real pump sensor data, calibrated for the client environment, and integrated with the client’s live data source.