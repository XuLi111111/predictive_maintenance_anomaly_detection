from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset1" / "newater_pump1_model_ready.csv"

OUTPUT_DIR = BASE_DIR / "results" / "dataset1" / "isolation_forest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "isolation_forest_model.pkl"
SCORES_PATH = OUTPUT_DIR / "isolation_forest_scores.csv"
TOP_ANOMALIES_PATH = OUTPUT_DIR / "isolation_forest_top_100.csv"
SUMMARY_PATH = OUTPUT_DIR / "isolation_forest_summary.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # Exclude raw calendar columns because cyclical features already exist
    excluded_cols = ["Time", "year", "month", "day", "hour", "minute"]
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].copy()

    # Isolation Forest does not require feature scaling
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,   # starting assumption: ~2% anomalies
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    # sklearn: predict -> 1 for normal, -1 for anomaly
    raw_pred = model.predict(X)
    decision_score = model.decision_function(X)

    result_df = pd.DataFrame({
        "Time": df["Time"] if "Time" in df.columns else pd.NaT,
        "anomaly_score": -decision_score,   # higher = more anomalous
        "anomaly_flag": (raw_pred == -1).astype(int)
    })

    result_df.to_csv(SCORES_PATH, index=False)

    # Save top 100 most anomalous rows with original features
    top_idx = result_df["anomaly_score"].nlargest(100).index
    top_anomalies = df.loc[top_idx].copy()
    top_anomalies["anomaly_score"] = result_df.loc[top_idx, "anomaly_score"].values
    top_anomalies["anomaly_flag"] = result_df.loc[top_idx, "anomaly_flag"].values
    top_anomalies = top_anomalies.sort_values("anomaly_score", ascending=False)
    top_anomalies.to_csv(TOP_ANOMALIES_PATH, index=False)

    joblib.dump(model, MODEL_PATH)

    summary_df = pd.DataFrame([{
        "model": "Isolation Forest",
        "input_rows": len(df),
        "feature_count": len(feature_cols),
        "predicted_anomalies": int(result_df["anomaly_flag"].sum()),
        "predicted_normals": int((result_df["anomaly_flag"] == 0).sum()),
        "contamination": 0.02,
        "n_estimators": 200
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("DONE")
    print(f"Input rows: {len(df)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Predicted anomalies: {int(result_df['anomaly_flag'].sum())}")
    print(f"Predicted normals: {int((result_df['anomaly_flag'] == 0).sum())}")
    print(f"Scores saved to: {SCORES_PATH}")
    print(f"Top anomalies saved to: {TOP_ANOMALIES_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()