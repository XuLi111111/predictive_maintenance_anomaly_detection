from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset1" / "newater_pump1_model_ready.csv"

OUTPUT_DIR = BASE_DIR / "results" / "dataset1" / "pca_reconstruction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = OUTPUT_DIR / "pca_scaler.pkl"
MODEL_PATH = OUTPUT_DIR / "pca_model.pkl"
SCORES_PATH = OUTPUT_DIR / "pca_reconstruction_scores.csv"
TOP_ANOMALIES_PATH = OUTPUT_DIR / "pca_reconstruction_top_100.csv"
SUMMARY_PATH = OUTPUT_DIR / "pca_reconstruction_summary.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    excluded_cols = ["Time", "year", "month", "day", "hour", "minute"]
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # keep 95% variance
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)

    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    # top 2% as anomalies for now
    threshold = np.quantile(reconstruction_error, 0.98)
    anomaly_flag = (reconstruction_error >= threshold).astype(int)

    result_df = pd.DataFrame({
        "Time": df["Time"] if "Time" in df.columns else pd.NaT,
        "anomaly_score": reconstruction_error,
        "anomaly_flag": anomaly_flag
    })

    result_df.to_csv(SCORES_PATH, index=False)

    top_idx = result_df["anomaly_score"].nlargest(100).index
    top_anomalies = df.loc[top_idx].copy()
    top_anomalies["anomaly_score"] = result_df.loc[top_idx, "anomaly_score"].values
    top_anomalies["anomaly_flag"] = result_df.loc[top_idx, "anomaly_flag"].values
    top_anomalies = top_anomalies.sort_values("anomaly_score", ascending=False)
    top_anomalies.to_csv(TOP_ANOMALIES_PATH, index=False)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, MODEL_PATH)

    summary_df = pd.DataFrame([{
        "model": "PCA Reconstruction Error",
        "input_rows": len(df),
        "feature_count": len(feature_cols),
        "pca_components": pca.n_components_,
        "explained_variance_ratio_sum": round(float(np.sum(pca.explained_variance_ratio_)), 6),
        "predicted_anomalies": int(result_df["anomaly_flag"].sum()),
        "predicted_normals": int((result_df["anomaly_flag"] == 0).sum()),
        "threshold": float(threshold)
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("DONE")
    print(f"Input rows: {len(df)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"PCA components kept: {pca.n_components_}")
    print(f"Explained variance kept: {np.sum(pca.explained_variance_ratio_):.6f}")
    print(f"Predicted anomalies: {int(result_df['anomaly_flag'].sum())}")
    print(f"Predicted normals: {int((result_df['anomaly_flag'] == 0).sum())}")
    print(f"Scores saved to: {SCORES_PATH}")
    print(f"Top anomalies saved to: {TOP_ANOMALIES_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()