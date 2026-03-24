from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset1" / "newater_pump1_model_ready.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "dataset1" / "lstm_windows"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 60
STRIDE = 10


def create_windows_with_timestamps(data_array, timestamps, window_size=60, stride=10):
    windows = []
    end_times = []

    for i in range(0, len(data_array) - window_size + 1, stride):
        windows.append(data_array[i:i + window_size])
        end_times.append(timestamps[i + window_size - 1])

    return np.array(windows, dtype=np.float32), pd.Series(end_times)


def main():
    df = pd.read_csv(INPUT_FILE)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    excluded_cols = ["Time", "year", "month", "day", "hour", "minute"]
    feature_cols = [col for col in df.columns if col not in excluded_cols]

    X = df[feature_cols].copy()
    time_col = df["Time"].copy()

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X.iloc[:train_end].copy()
    X_val = X.iloc[train_end:val_end].copy()
    X_test = X.iloc[val_end:].copy()

    time_train = time_col.iloc[:train_end].reset_index(drop=True)
    time_val = time_col.iloc[train_end:val_end].reset_index(drop=True)
    time_test = time_col.iloc[val_end:].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    train_windows, train_times = create_windows_with_timestamps(
        X_train_scaled, time_train, WINDOW_SIZE, STRIDE
    )
    val_windows, val_times = create_windows_with_timestamps(
        X_val_scaled, time_val, WINDOW_SIZE, STRIDE
    )
    test_windows, test_times = create_windows_with_timestamps(
        X_test_scaled, time_test, WINDOW_SIZE, STRIDE
    )

    np.save(OUTPUT_DIR / "X_train_windows.npy", train_windows)
    np.save(OUTPUT_DIR / "X_val_windows.npy", val_windows)
    np.save(OUTPUT_DIR / "X_test_windows.npy", test_windows)

    train_times.to_csv(OUTPUT_DIR / "train_window_end_times.csv", index=False)
    val_times.to_csv(OUTPUT_DIR / "val_window_end_times.csv", index=False)
    test_times.to_csv(OUTPUT_DIR / "test_window_end_times.csv", index=False)

    pd.DataFrame({"feature": feature_cols}).to_csv(OUTPUT_DIR / "feature_columns.csv", index=False)
    joblib.dump(scaler, OUTPUT_DIR / "lstm_window_scaler.pkl")

    summary_df = pd.DataFrame([{
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "feature_count": len(feature_cols),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "test_windows": len(test_windows),
    }])
    summary_df.to_csv(OUTPUT_DIR / "lstm_window_summary.csv", index=False)

    print("DONE")
    print(f"Train windows: {len(train_windows)}")
    print(f"Validation windows: {len(val_windows)}")
    print(f"Test windows: {len(test_windows)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()