from pathlib import Path
import pandas as pd
import re
import json


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "dataset1"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "dataset1"
CLEANED_DIR = PROCESSED_DIR / "cleaned"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


def make_unique(columns):
    seen = {}
    unique_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            unique_cols.append(col)
        else:
            seen[col] += 1
            unique_cols.append(f"{col}_{seen[col]}")
    return unique_cols


def clean_column_name(col):
    col = str(col).strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")
    return col if col else "unnamed"


def read_csv_flexible(path):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, errors="ignore") as f:
                first_line = f.readline().strip().lower()

            skiprows = 1 if first_line.startswith("sep=") else 0
            df = pd.read_csv(path, skiprows=skiprows, encoding=encoding, low_memory=False)
            return df
        except Exception:
            continue

    raise ValueError(f"Could not read file: {path}")


def detect_time_column(df):
    priority_names = [
        "time", "timestamp", "datetime", "date_time", "date", "recorded_at"
    ]

    lowered = {col: str(col).strip().lower() for col in df.columns}

    for col, low in lowered.items():
        if low in priority_names:
            return col

    for col, low in lowered.items():
        if any(word in low for word in ["time", "date", "stamp"]):
            return col

    return None


def parse_time_series(series):
    s = series.astype(str).str.strip()

    parsed_candidates = {}

    parsed_candidates["default"] = pd.to_datetime(s, errors="coerce")
    parsed_candidates["dayfirst"] = pd.to_datetime(s, errors="coerce", dayfirst=True)

    numeric = pd.to_numeric(s, errors="coerce")

    parsed_candidates["epoch_seconds"] = pd.to_datetime(numeric, unit="s", errors="coerce")
    parsed_candidates["epoch_milliseconds"] = pd.to_datetime(numeric, unit="ms", errors="coerce")

    # Prefer timestamps that fall into a realistic project date range
    lower_bound = pd.Timestamp("2020-01-01")
    upper_bound = pd.Timestamp("2030-12-31")

    best_name = None
    best_series = None
    best_score = -1

    for name, candidate in parsed_candidates.items():
        valid_mask = candidate.notna() & (candidate >= lower_bound) & (candidate <= upper_bound)
        score = valid_mask.sum()

        if score > best_score:
            best_score = score
            best_name = name
            best_series = candidate

    # fallback if everything scores 0
    if best_series is None or best_score == 0:
        best_name = None
        best_series = None
        best_non_null = -1

        for name, candidate in parsed_candidates.items():
            score = candidate.notna().sum()
            if score > best_non_null:
                best_non_null = score
                best_name = name
                best_series = candidate

    return best_name, best_series

def clean_numeric_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.extract(r"([-+]?\d*\.?\d+)")[0]

    return pd.to_numeric(s, errors="coerce")


def maybe_convert_numeric(df, time_col=None):
    numeric_cols = []

    for col in df.columns:
        if col == time_col:
            continue

        original_non_null = df[col].notna().sum()
        if original_non_null == 0:
            continue

        converted = clean_numeric_series(df[col])
        converted_non_null = converted.notna().sum()

        if converted_non_null / original_non_null >= 0.6:
            df[col] = converted
            numeric_cols.append(col)

    return df, numeric_cols


def process_file(path):
    df = read_csv_flexible(path)

    original_rows, original_cols = df.shape

    df.columns = [clean_column_name(col) for col in df.columns]
    df.columns = make_unique(df.columns)

    time_col = detect_time_column(df)
    time_parse_mode = None
    parsed_time_non_null = 0

    if time_col is not None:
        time_parse_mode, parsed_time = parse_time_series(df[time_col])
        parsed_time_non_null = int(parsed_time.notna().sum())
        df[time_col] = parsed_time
        if time_col != "Time":
            df = df.rename(columns={time_col: "Time"})
            time_col = "Time"

    df, numeric_cols = maybe_convert_numeric(df, time_col=time_col)

    if time_col is not None:
        df = df.sort_values(by=time_col, kind="stable")
        df = df.drop_duplicates(subset=[time_col], keep="last")

    df = df.reset_index(drop=True)

    output_file = CLEANED_DIR / path.name
    df.to_csv(output_file, index=False)

    summary = {
        "file_name": path.name,
        "rows_before": int(original_rows),
        "cols_before": int(original_cols),
        "rows_after": int(df.shape[0]),
        "cols_after": int(df.shape[1]),
        "time_column_detected": time_col if time_col is not None else "",
        "time_parse_mode": time_parse_mode if time_parse_mode is not None else "",
        "parsed_time_non_null": parsed_time_non_null,
        "numeric_columns_detected": ", ".join(numeric_cols),
        "columns_after_cleaning": ", ".join(df.columns.astype(str)),
        "output_file": str(output_file.relative_to(BASE_DIR)),
    }

    return summary


def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {RAW_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files.")
    summaries = []

    for idx, path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Processing: {path.name}")
        try:
            summary = process_file(path)
            summaries.append(summary)
        except Exception as e:
            summaries.append({
                "file_name": path.name,
                "rows_before": "",
                "cols_before": "",
                "rows_after": "",
                "cols_after": "",
                "time_column_detected": "",
                "time_parse_mode": "",
                "parsed_time_non_null": "",
                "numeric_columns_detected": "",
                "columns_after_cleaning": f"ERROR: {e}",
                "output_file": "",
            })
            print(f"ERROR in {path.name}: {e}")

    manifest_df = pd.DataFrame(summaries)
    manifest_csv = PROCESSED_DIR / "dataset1_manifest.csv"
    manifest_json = PROCESSED_DIR / "dataset1_manifest.json"

    manifest_df.to_csv(manifest_csv, index=False)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\nDone.")
    print(f"Manifest saved to: {manifest_csv}")
    print(f"JSON summary saved to: {manifest_json}")
    print(f"Cleaned files saved in: {CLEANED_DIR}")


if __name__ == "__main__":
    main()