from pathlib import Path
from functools import reduce
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CLEANED_DIR = BASE_DIR / "data" / "processed" / "dataset1" / "cleaned"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "dataset1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Current Sensor - NeWater Pump 1.csv",
    "Power Sensor - NeWater Pump 1.csv",
    "Energy Sensor - NeWater Pump 1.csv",
    "Pressure Sensor - NeWater Incoming Pump 1.csv",
    "Pressure Sensor - NeWater Outgoing Pump.csv",
    "Vibration Sensor - NeWater Pump 1 Temperature.csv",
    "Vibration Sensor - NeWater Pump 1 X-Axis Displacement.csv",
    "Vibration Sensor - NeWater Pump 1 X-Axis Speed.csv",
    "Vibration Sensor - NeWater Pump 1 Y-Axis Displacement.csv",
    "Vibration Sensor - NeWater Pump 1 Y-Axis Speed.csv",
    "Vibration Sensor - NeWater Pump 1 Z-Axis Displacement.csv",
    "Vibration Sensor - NeWater Pump 1 Z-Axis Speed.csv",
    "Water Level Sensor - NeWater Tank.csv",
]


def robust_parse_time(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    parsed_default = pd.to_datetime(s, errors="coerce")
    parsed_dayfirst = pd.to_datetime(s, errors="coerce", dayfirst=True)

    numeric = pd.to_numeric(s, errors="coerce")
    parsed_epoch_s = pd.to_datetime(numeric, unit="s", errors="coerce")
    parsed_epoch_ms = pd.to_datetime(numeric, unit="ms", errors="coerce")

    candidates = {
        "default": parsed_default,
        "dayfirst": parsed_dayfirst,
        "epoch_s": parsed_epoch_s,
        "epoch_ms": parsed_epoch_ms,
    }

    best_name = max(candidates, key=lambda k: candidates[k].notna().sum())
    best = candidates[best_name]

    print(f"    [time parse mode: {best_name}, valid times: {best.notna().sum()}]")
    return best


def load_and_resample(filename: str) -> pd.DataFrame:
    path = CLEANED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing cleaned file: {path}")

    df = pd.read_csv(path)

    if "Time" not in df.columns:
        raise ValueError(f"'Time' column not found in {filename}")

    df["Time"] = robust_parse_time(df["Time"])
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != "Time"]
    if not numeric_cols:
        raise ValueError(f"No numeric columns found in {filename}")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Time"] = df["Time"].dt.floor("min")
    df = df.groupby("Time", as_index=False)[numeric_cols].mean(numeric_only=True)

    print(f"[OK] {filename} -> {df.shape}")
    return df


def merge_all(dfs):
    return reduce(lambda left, right: pd.merge(left, right, on="Time", how="outer"), dfs)


def main():
    frames = []

    for filename in FILES:
        frames.append(load_and_resample(filename))

    merged = merge_all(frames)
    merged = merged.sort_values("Time").reset_index(drop=True)

    merged["year"] = merged["Time"].dt.year
    merged["month"] = merged["Time"].dt.month
    merged["day"] = merged["Time"].dt.day
    merged["hour"] = merged["Time"].dt.hour
    merged["minute"] = merged["Time"].dt.minute

    missing_summary = pd.DataFrame({
        "column": merged.columns,
        "missing_count": merged.isna().sum().values,
        "missing_pct": (merged.isna().sum().values / len(merged) * 100).round(2),
    })

    output_csv = OUTPUT_DIR / "newater_pump1_1min_merged.csv"
    output_missing = OUTPUT_DIR / "newater_pump1_1min_missing_summary.csv"

    merged.to_csv(output_csv, index=False)
    missing_summary.to_csv(output_missing, index=False)

    print("\nDONE")
    print(f"Merged file saved to: {output_csv}")
    print(f"Missing summary saved to: {output_missing}")
    print(f"Final shape: {merged.shape}")
    print("Columns:")
    print(list(merged.columns))


if __name__ == "__main__":
    main()