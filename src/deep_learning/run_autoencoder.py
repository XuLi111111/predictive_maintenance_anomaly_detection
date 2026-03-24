from pathlib import Path
import os
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset1" / "newater_pump1_model_ready.csv"

OUTPUT_DIR = BASE_DIR / "results" / "dataset1" / "autoencoder"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = OUTPUT_DIR / "autoencoder_scaler.pkl"
MODEL_PATH = OUTPUT_DIR / "autoencoder_model.pt"
SCORES_PATH = OUTPUT_DIR / "autoencoder_scores.csv"
TOP_ANOMALIES_PATH = OUTPUT_DIR / "autoencoder_top_100.csv"
SUMMARY_PATH = OUTPUT_DIR / "autoencoder_summary.csv"

RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 1e-3


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def main():
    set_seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_FILE)

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    excluded_cols = ["Time", "year", "month", "day", "hour", "minute"]
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    dataset = TensorDataset(torch.tensor(X_scaled))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled)
        X_recon = model(X_tensor).numpy()

    reconstruction_error = np.mean((X_scaled - X_recon) ** 2, axis=1)

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
    torch.save(model.state_dict(), MODEL_PATH)

    summary_df = pd.DataFrame([{
        "model": "Autoencoder",
        "input_rows": len(df),
        "feature_count": len(feature_cols),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "predicted_anomalies": int(result_df["anomaly_flag"].sum()),
        "predicted_normals": int((result_df["anomaly_flag"] == 0).sum()),
        "threshold": float(threshold)
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("DONE")
    print(f"Input rows: {len(df)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Predicted anomalies: {int(result_df['anomaly_flag'].sum())}")
    print(f"Predicted normals: {int((result_df['anomaly_flag'] == 0).sum())}")
    print(f"Scores saved to: {SCORES_PATH}")
    print(f"Top anomalies saved to: {TOP_ANOMALIES_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()