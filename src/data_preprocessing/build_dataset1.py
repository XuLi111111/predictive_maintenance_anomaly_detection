import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 配置区
# =========================================================

# 项目根目录（predictive_maintenance_anomaly_detection）
BASE_DIR = Path(__file__).resolve().parents[2]

# 你的数据集根目录
DATA_DIR = BASE_DIR / "data" / "raw" / "dataset1"

# 输出目录
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "dataset1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 选择系统： "non-potable" -> NeWater, "potable" -> Potable
SYSTEM = "non-potable"

# 滑动窗口大小（单位：分钟，因为原始粒度是 minute）
WINDOW_SIZE = 20
STRIDE = 1

# 是否把系统维护（maintenance）也当异常
INCLUDE_MAINTENANCE_AS_ANOMALY = False

# =========================================================
# 2. 文档中给出的事件日志（根据你发给我的 md 手工整理）
# =========================================================

# Non-potable / NeWater faults
NON_POTABLE_HARDWARE_FAULT_DATES = [
    "2023-08-06",
    "2023-09-10", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-14",
    "2023-09-17", "2023-09-19", "2023-09-20", "2023-09-21", "2023-09-22",
    "2023-09-23", "2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28",
    "2023-09-29", "2023-09-30",
    "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-07",
    "2023-10-08", "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12",
    "2023-10-16", "2023-10-17", "2023-10-18", "2023-10-19", "2023-10-20",
    "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26",
]

NON_POTABLE_PUMP_FAULT_DATES = [
    "2023-09-08", "2023-09-11", "2023-09-12", "2023-09-13", "2023-09-18",
    "2023-09-20", "2023-09-21", "2023-09-22", "2023-09-23", "2023-09-25",
    "2023-09-26", "2023-09-27", "2023-09-28", "2023-09-29", "2023-09-30",
    "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-06", "2023-10-07",
    "2023-10-08", "2023-10-09", "2023-10-10", "2023-10-11", "2023-10-12",
    "2023-10-13", "2023-10-14", "2023-10-15", "2023-10-16", "2023-10-17",
    "2023-10-18", "2023-10-19", "2023-10-20", "2023-10-21", "2023-10-22",
    "2023-10-23", "2023-10-24", "2023-10-25", "2023-10-26", "2023-10-30",
    "2024-03-13",
]

# Potable faults
POTABLE_PUMP_FAULT_DATES = [
    "2023-09-06",
    "2023-09-09",
    "2023-12-27",
]

# Potable maintenance
POTABLE_MAINTENANCE_DATES = pd.date_range(
    start="2023-08-31", end="2023-09-06", freq="D"
).strftime("%Y-%m-%d").tolist()


def get_fault_dates(system: str, include_maintenance: bool = False) -> set[str]:
    if system == "non-potable":
        dates = set(NON_POTABLE_HARDWARE_FAULT_DATES) | set(NON_POTABLE_PUMP_FAULT_DATES)
    elif system == "potable":
        dates = set(POTABLE_PUMP_FAULT_DATES)
        if include_maintenance:
            dates |= set(POTABLE_MAINTENANCE_DATES)
    else:
        raise ValueError("system must be 'non-potable' or 'potable'")
    return dates


FAULT_DATES = get_fault_dates(SYSTEM, INCLUDE_MAINTENANCE_AS_ANOMALY)


# =========================================================
# 3. 选择要用的文件
#    先给你一个比较稳的组合：Current + Power + Energy + Pressure + Temperature
#    你后面可以自己加 vibration 各轴数据
# =========================================================

if SYSTEM == "non-potable":
    FILES_TO_USE = {
        "current": DATA_DIR / "Current Sensor - NeWater Pump 1.csv",
        "power": DATA_DIR / "Power Sensor - NeWater Pump 1.csv",
        "energy": DATA_DIR / "Energy Sensor - NeWater Pump 1.csv",
        "pressure_in": DATA_DIR / "Pressure Sensor - NeWater Incoming Pump 1.csv",
        "pressure_out": DATA_DIR / "Pressure Sensor - NeWater Outgoing Pump.csv",
        "temp": DATA_DIR / "Vibration Sensor - NeWater Temperature.csv",
        "water_level": DATA_DIR / "Water Level Sensor - NeWater Tank.csv",
    }
else:
    FILES_TO_USE = {
        "current": DATA_DIR / "Current Sensor - Potable Pump 1.csv",
        "power": DATA_DIR / "Power Sensor - Potable Pump 1.csv",
        "energy": DATA_DIR / "Energy Sensor - Potable Pump 1.csv",
        "pressure_in": DATA_DIR / "Pressure Sensor - Potable Incoming Pump 1.csv",
        "pressure_out": DATA_DIR / "Pressure Sensor - Potable Outgoing Pump.csv",
        "temp": DATA_DIR / "Vibration Sensor - Potable Temperature.csv",
        "water_level": DATA_DIR / "Water Level Sensor - Potable Tank.csv",
    }


# =========================================================
# 4. 工具函数
# =========================================================

def read_csv_skip_excel_sep(path: Path) -> pd.DataFrame:
    """
    Excel 导出的 csv 常常第一行是 sep=,
    所以统一 skiprows=1
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, skiprows=1)
    return df


def clean_numeric_value(series: pd.Series) -> pd.Series:
    """
    把像 '0.23 A', '12.4 W', '4.5 bar', '20 °C', '1.2 kWh'
    这种字符串清洗成 float
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip()

    # 去掉逗号分隔
    s = s.str.replace(",", "", regex=False)

    # 提取数值
    s = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(s, errors="coerce")


def standardise_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" not in df.columns:
        raise ValueError(f"'Time' column not found. Columns = {df.columns.tolist()}")

    df = df.copy()
    raw_time = df["Time"].astype(str).str.strip()

    # 打印前几条原始时间，方便调试
    print("[DEBUG] Raw Time preview:", raw_time.head(5).tolist())

    # 期望的数据范围：文档说明是 2023-07 到 2024-07
    expected_start = pd.Timestamp("2023-07-01")
    expected_end = pd.Timestamp("2024-07-31 23:59:59")

    # 候选解析方式 1：普通日期字符串
    parsed_dayfirst = pd.to_datetime(raw_time, dayfirst=True, errors="coerce")
    parsed_monthfirst = pd.to_datetime(raw_time, dayfirst=False, errors="coerce")

    # 候选解析方式 2：Unix epoch（毫秒）
    numeric_time = pd.to_numeric(raw_time, errors="coerce")

    # 只对“看起来像 epoch 毫秒时间戳”的值做转换，避免普通日期字符串转成巨大数字后溢出
    plausible_epoch_mask = numeric_time.between(1e11, 2e13, inclusive="both")
    parsed_epoch_ms = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    if plausible_epoch_mask.any():
        parsed_epoch_ms.loc[plausible_epoch_mask] = (
            pd.to_datetime(
                numeric_time.loc[plausible_epoch_mask],
                unit="ms",
                errors="coerce",
            ) + pd.Timedelta(hours=8)
        )

    def score(parsed: pd.Series) -> int:
        valid = parsed.notna()
        in_range = valid & (parsed >= expected_start) & (parsed <= expected_end)
        return int(in_range.sum())

    candidates = {
        "dayfirst=True": parsed_dayfirst,
        "dayfirst=False": parsed_monthfirst,
        "epoch_ms": parsed_epoch_ms,
    }

    scores = {name: score(parsed) for name, parsed in candidates.items()}
    chosen_mode = max(scores, key=scores.get)
    chosen = candidates[chosen_mode]

    print(
        "[DEBUG] Time parse mode selected: "
        f"{chosen_mode} ("
        + ", ".join(f"{k} score={v}" for k, v in scores.items())
        + ")"
    )

    df["Time"] = chosen
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    return df


def rename_sensor_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col != "Time":
            rename_map[col] = f"{prefix}_{col}"
    return df.rename(columns=rename_map)


def load_and_prepare_one_file(path: Path, prefix: str) -> pd.DataFrame:
    df = read_csv_skip_excel_sep(path)
    df = standardise_time_column(df)

    for col in df.columns:
        if col != "Time":
            df[col] = clean_numeric_value(df[col])

    df = rename_sensor_columns(df, prefix)
    return df


def merge_all_data(file_dict: dict[str, Path]) -> pd.DataFrame:
    merged = None
    missing_files = []

    for prefix, path in file_dict.items():
        print(f"Loading: {path.name}")

        if not path.exists():
            print(f"[WARNING] File not found, skipping: {path}")
            missing_files.append(str(path))
            continue

        df_part = load_and_prepare_one_file(path, prefix)

        if merged is None:
            merged = df_part
        else:
            merged = pd.merge(merged, df_part, on="Time", how="outer")

    if merged is None:
        raise FileNotFoundError(
            "None of the sensor files were found. Please check DATA_DIR and FILES_TO_USE."
        )

    merged = merged.sort_values("Time").reset_index(drop=True)

    # 统一到分钟级
    merged["Time"] = merged["Time"].dt.floor("min")

    # 同一分钟如果有重复，取平均
    merged = merged.groupby("Time", as_index=False).mean(numeric_only=True)

    if missing_files:
        print("\n[INFO] Missing files skipped:")
        for fp in missing_files:
            print(f" - {fp}")

    return merged


def add_labels_by_fault_dates(df: pd.DataFrame, fault_dates: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["date_str"] = df["Time"].dt.strftime("%Y-%m-%d")
    df["label"] = df["date_str"].isin(fault_dates).astype(int)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 先按时间排序
    df = df.sort_values("Time").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c not in ["Time", "date_str", "label"]]

    # 删除全为空的列（通常说明该传感器时间解析失败或完全没对齐）
    all_nan_cols = [c for c in numeric_cols if df[c].isna().all()]
    if all_nan_cols:
        print("[WARNING] Dropping all-NaN columns:", all_nan_cols)
        df = df.drop(columns=all_nan_cols)
        numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]

    # 时间序列常用填充：前向 + 后向
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 如果关键特征仍有空值，再删除对应行
    df = df.dropna(subset=numeric_cols)
    return df


def build_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int = 1,
    label_mode: str = "any_fault_in_window"
):
    """
    label_mode:
        - any_fault_in_window: 窗口里只要出现过 fault -> 1
        - label_at_last_step: 取窗口最后一个时间点的标签
    """
    feature_cols = [c for c in df.columns if c not in ["Time", "date_str", "label"]]

    X, y, times = [], [], []

    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        window = df.iloc[start:end]

        features = window[feature_cols].values

        if label_mode == "any_fault_in_window":
            label = int(window["label"].max())
        elif label_mode == "label_at_last_step":
            label = int(window["label"].iloc[-1])
        else:
            raise ValueError("Unsupported label_mode")

        X.append(features)
        y.append(label)
        times.append(window["Time"].iloc[-1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    times = np.array(times)

    return X, y, times, feature_cols


# ========== Chronological split and saving helpers ==========


def chronological_split_with_gap(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    gap: int = WINDOW_SIZE,
    min_positive_per_split: int = 50,
    adjust_step: int = 1000,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n = len(X)
    if n == 0:
        raise ValueError("Cannot split an empty dataset")

    if n <= 2 * gap + 3:
        raise ValueError("Dataset too small for split with gaps")

    target_train_end = int(n * train_ratio)
    target_val_end = target_train_end + int(n * val_ratio)

    # 先按比例切，再在边界之间留 gap，避免高重叠窗口跨集合泄漏
    train_end = max(1, min(target_train_end, n - 2 * gap - 2))
    val_start = train_end + gap
    val_end = max(val_start + 1, min(target_val_end, n - gap - 1))
    test_start = val_end + gap

    if test_start >= n:
        raise ValueError("Invalid split boundaries after applying gap")

    y_binary = (y == 1).astype(int)
    pos_cumsum = np.cumsum(y_binary)

    def positive_count(start: int, end: int) -> int:
        if end <= start:
            return 0
        return int(pos_cumsum[end - 1] - (pos_cumsum[start - 1] if start > 0 else 0))

    train_pos = positive_count(0, train_end)
    val_pos = positive_count(val_start, val_end)
    test_pos = positive_count(test_start, n)

    # 如果 test 没有足够正样本，就把 val/test 分界往前挪
    while test_pos < min_positive_per_split and val_end - adjust_step > val_start + 1:
        val_end -= adjust_step
        test_start = val_end + gap
        val_pos = positive_count(val_start, val_end)
        test_pos = positive_count(test_start, n)

    # 如果 validation 没有足够正样本，就把 train/val 分界往前挪
    while val_pos < min_positive_per_split and train_end - adjust_step > 1:
        train_end -= adjust_step
        val_start = train_end + gap
        if val_end <= val_start:
            val_end = val_start + 1
        test_start = val_end + gap
        train_pos = positive_count(0, train_end)
        val_pos = positive_count(val_start, val_end)
        test_pos = positive_count(test_start, n)

    train_split = {
        "X": X[:train_end],
        "y": y[:train_end],
        "times": times[:train_end],
    }
    val_split = {
        "X": X[val_start:val_end],
        "y": y[val_start:val_end],
        "times": times[val_start:val_end],
    }
    test_split = {
        "X": X[test_start:],
        "y": y[test_start:],
        "times": times[test_start:],
    }

    print("\n[INFO] Split with gap selected:")
    print(f"train_end index: {train_end}")
    print(f"val_start index: {val_start}")
    print(f"val_end index: {val_end}")
    print(f"test_start index: {test_start}")
    print(f"gap used: {gap}")

    if train_pos < min_positive_per_split or val_pos < min_positive_per_split or test_pos < min_positive_per_split:
        print("[WARNING] Could not fully satisfy min_positive_per_split for all splits.")
        print(f"Train positives: {train_pos}, Validation positives: {val_pos}, Test positives: {test_pos}")

    return train_split, val_split, test_split


def print_split_distribution(name: str, split: dict):
    y_split = split["y"]
    times_split = split["times"]

    print(f"\n{name} split shape: {split['X'].shape}")
    if len(y_split) == 0:
        print(f"{name} split is empty")
        return

    print(f"{name} time range: {times_split[0]} -> {times_split[-1]}")

    unique, counts = np.unique(y_split, return_counts=True)
    print(f"{name} label distribution:")
    for u, c in zip(unique, counts):
        print(f"label={u}: {c}")


def save_npz_split(split_name: str, split: dict, feature_cols: list[str]):
    np.savez_compressed(
        OUTPUT_DIR / f"{split_name}_dataset.npz",
        X=split["X"],
        y=split["y"],
        times=split["times"].astype("datetime64[s]"),
        feature_cols=np.array(feature_cols, dtype=object),
    )


def save_flat_csv_split(split_name: str, split: dict, feature_cols: list[str], window_size: int):
    X_split = split["X"]
    y_split = split["y"]
    times_split = split["times"]

    X_flat = X_split.reshape(X_split.shape[0], -1)
    flat_col_names = []
    for t in range(window_size):
        for feat in feature_cols:
            flat_col_names.append(f"{feat}_t{t}")

    df_flat = pd.DataFrame(X_flat, columns=flat_col_names)
    df_flat["label"] = y_split
    df_flat["window_end_time"] = times_split

    csv_path = OUTPUT_DIR / f"{split_name}_flat.csv"
    df_flat.to_csv(csv_path, index=False)


# =========================================================
# 5. 主流程
# =========================================================

def main():
    print("=" * 60)
    print("Step 1: Merge sensor files")
    print("=" * 60)
    df = merge_all_data(FILES_TO_USE)

    print("\nMerged shape:", df.shape)
    print("Columns:", df.columns.tolist()[:20], "..." if len(df.columns) > 20 else "")
    print(df.head())

    print("\n" + "=" * 60)
    print("Step 2: Add labels from event log")
    print("=" * 60)
    df = add_labels_by_fault_dates(df, FAULT_DATES)
    df = fill_missing_values(df)

    print("Label distribution (row level):")
    print(df["label"].value_counts(dropna=False))
    print(df[["Time", "label"]].head())

    print("\n" + "=" * 60)
    print("Step 3: Build sliding windows")
    print("=" * 60)
    X, y, times, feature_cols = build_sliding_windows(
        df=df,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        label_mode="any_fault_in_window",
    )

    print("X shape:", X.shape)   # (samples, time_steps, features)
    print("y shape:", y.shape)
    print("Feature count:", len(feature_cols))
    print("Feature columns:", feature_cols)

    print("\nWindow-level label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"label={u}: {c}")

    if len(X) == 0:
        print("\n[ERROR] No sliding windows were generated. Please check time parsing, label construction, or missing-value handling.")
        return

    print("\nExample window X[0]:")
    print(X[0])

    print("\nExample label y[0]:", y[0])
    print("Example window end time:", times[0])

    print("\n" + "=" * 60)
    print("Step 4: Save full processed dataset")
    print("=" * 60)

    full_npz_path = OUTPUT_DIR / f"{SYSTEM}_pump1_window{WINDOW_SIZE}_dataset.npz"
    np.savez_compressed(
        full_npz_path,
        X=X,
        y=y,
        times=times.astype("datetime64[s]"),
        feature_cols=np.array(feature_cols, dtype=object),
    )

    X_flat = X.reshape(X.shape[0], -1)
    flat_col_names = []
    for t in range(WINDOW_SIZE):
        for feat in feature_cols:
            flat_col_names.append(f"{feat}_t{t}")

    df_flat = pd.DataFrame(X_flat, columns=flat_col_names)
    df_flat["label"] = y
    df_flat["window_end_time"] = times

    full_csv_path = OUTPUT_DIR / f"{SYSTEM}_pump1_window{WINDOW_SIZE}_flat.csv"
    df_flat.to_csv(full_csv_path, index=False)

    print(f"Saved full NPZ to: {full_npz_path}")
    print(f"Saved full flat CSV to: {full_csv_path}")

    print("\n" + "=" * 60)
    print("Step 5: Chronological train/val/test split")
    print("=" * 60)

    train_split, val_split, test_split = chronological_split_with_gap(
        X=X,
        y=y,
        times=times,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        gap=WINDOW_SIZE,
        min_positive_per_split=50,
        adjust_step=1000,
    )

    print_split_distribution("Train", train_split)
    print_split_distribution("Validation", val_split)
    print_split_distribution("Test", test_split)

    save_npz_split("train", train_split, feature_cols)
    save_npz_split("val", val_split, feature_cols)
    save_npz_split("test", test_split, feature_cols)

    save_flat_csv_split("train", train_split, feature_cols, WINDOW_SIZE)
    save_flat_csv_split("val", val_split, feature_cols, WINDOW_SIZE)
    save_flat_csv_split("test", test_split, feature_cols, WINDOW_SIZE)

    print(f"Saved split NPZ files to: {OUTPUT_DIR}")
    print(f"Saved split flat CSV files to: {OUTPUT_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()