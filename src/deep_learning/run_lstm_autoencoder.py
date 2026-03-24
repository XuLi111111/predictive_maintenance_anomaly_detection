from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[2]
WINDOW_DIR = BASE_DIR / "data" / "processed" / "dataset1" / "lstm_windows"

OUTPUT_DIR = BASE_DIR / "results" / "dataset1" / "lstm_autoencoder"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "lstm_autoencoder_model.pt"
SCORES_PATH = OUTPUT_DIR / "lstm_autoencoder_scores.csv"
TOP_ANOMALIES_PATH = OUTPUT_DIR / "lstm_autoencoder_top_100.csv"
SUMMARY_PATH = OUTPUT_DIR / "lstm_autoencoder_summary.csv"

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_DIM = 32
LATENT_DIM = 16


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]
        z = self.to_latent(h)

        repeated = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        out = self.output_layer(decoded)
        return out


def reconstruction_error(model, X, device):
    model.eval()
    errors = []

    with torch.no_grad():
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=256, shuffle=False)
        for batch in loader:
            x_batch = batch[0].to(device)
            x_recon = model(x_batch)
            batch_error = torch.mean((x_batch - x_recon) ** 2, dim=(1, 2))
            errors.extend(batch_error.cpu().numpy())

    return np.array(errors)


def main():
    set_seed(RANDOM_SEED)

    X_train = np.load(WINDOW_DIR / "X_train_windows.npy")
    X_val = np.load(WINDOW_DIR / "X_val_windows.npy")
    X_test = np.load(WINDOW_DIR / "X_test_windows.npy")

    train_times = pd.read_csv(WINDOW_DIR / "train_window_end_times.csv").iloc[:, 0]
    val_times = pd.read_csv(WINDOW_DIR / "val_window_end_times.csv").iloc[:, 0]
    test_times = pd.read_csv(WINDOW_DIR / "test_window_end_times.csv").iloc[:, 0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[2]
    model = LSTMAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            x_recon = model(x_batch)
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    train_error = reconstruction_error(model, X_train, device)
    val_error = reconstruction_error(model, X_val, device)
    test_error = reconstruction_error(model, X_test, device)

    threshold = np.quantile(train_error, 0.98)

    train_flag = (train_error >= threshold).astype(int)
    val_flag = (val_error >= threshold).astype(int)
    test_flag = (test_error >= threshold).astype(int)

    scores_df = pd.concat([
        pd.DataFrame({
            "split": "train",
            "window_end_time": train_times,
            "anomaly_score": train_error,
            "anomaly_flag": train_flag
        }),
        pd.DataFrame({
            "split": "validation",
            "window_end_time": val_times,
            "anomaly_score": val_error,
            "anomaly_flag": val_flag
        }),
        pd.DataFrame({
            "split": "test",
            "window_end_time": test_times,
            "anomaly_score": test_error,
            "anomaly_flag": test_flag
        })
    ], ignore_index=True)

    scores_df.to_csv(SCORES_PATH, index=False)

    top_anomalies = scores_df.sort_values("anomaly_score", ascending=False).head(100)
    top_anomalies.to_csv(TOP_ANOMALIES_PATH, index=False)

    torch.save(model.state_dict(), MODEL_PATH)

    summary_df = pd.DataFrame([{
        "model": "LSTM Autoencoder",
        "train_windows": len(X_train),
        "validation_windows": len(X_val),
        "test_windows": len(X_test),
        "feature_count": input_dim,
        "window_size": X_train.shape[1],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "threshold": float(threshold),
        "predicted_anomalies_train": int(train_flag.sum()),
        "predicted_anomalies_validation": int(val_flag.sum()),
        "predicted_anomalies_test": int(test_flag.sum())
    }])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("DONE")
    print(f"Train windows: {len(X_train)}")
    print(f"Validation windows: {len(X_val)}")
    print(f"Test windows: {len(X_test)}")
    print(f"Feature count: {input_dim}")
    print(f"Window size: {X_train.shape[1]}")
    print(f"Scores saved to: {SCORES_PATH}")
    print(f"Top anomalies saved to: {TOP_ANOMALIES_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()