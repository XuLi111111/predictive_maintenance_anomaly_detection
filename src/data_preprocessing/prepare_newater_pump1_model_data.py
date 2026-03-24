from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset1" / "newater_pump1_1min_merged.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "dataset1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def add_cyclical_time_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    return df


def main():
    df = pd.read_csv(INPUT_FILE)

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    # Keep a copy before filling
    raw_missing = df.isna().sum()

    sensor_cols = [col for col in df.columns if col not in ["Time", "year", "month", "day", "hour", "minute"]]

    # Interpolate time-series sensor values
    df[sensor_cols] = df[sensor_cols].interpolate(method="linear", limit_direction="both")

    # Fallback fill in case interpolation leaves edges missing
    df[sensor_cols] = df[sensor_cols].ffill().bfill()

    # Add cyclical time features
    df = add_cyclical_time_features(df)

    # Create anomaly-friendly final dataset
    final_df = df.copy()

    # Missing summary after treatment
    final_missing = final_df.isna().sum()

    missing_comparison = pd.DataFrame({
        "column": final_df.columns,
        "missing_before": [raw_missing.get(col, 0) for col in final_df.columns],
        "missing_after": [final_missing.get(col, 0) for col in final_df.columns],
    })
    missing_comparison["filled_count"] = (
        missing_comparison["missing_before"] - missing_comparison["missing_after"]
    )

    output_model = OUTPUT_DIR / "newater_pump1_model_ready.csv"
    output_missing = OUTPUT_DIR / "newater_pump1_missing_treatment_summary.csv"

    final_df.to_csv(output_model, index=False)
    missing_comparison.to_csv(output_missing, index=False)

    print("DONE")
    print(f"Model-ready dataset saved to: {output_model}")
    print(f"Missing treatment summary saved to: {output_missing}")
    print(f"Final shape: {final_df.shape}")
    print("Remaining missing values:")
    print(final_df.isna().sum()[final_df.isna().sum() > 0])


if __name__ == "__main__":
    main()