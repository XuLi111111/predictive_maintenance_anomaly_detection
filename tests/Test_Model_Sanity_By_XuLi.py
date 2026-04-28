"""
Test module for SKAB split strategy analysis logic.

This test file validates the data split analysis module used before the deep
learning pipeline. The tests focus on the early-warning label construction rule,
file-level split plan structure, and real SKAB split distribution statistics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset2"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import importlib.util  # noqa: E402


SPLIT_ANALYSIS_FILE = PROJECT_ROOT / "src" / "data_preprocessing" / "SKAB_Split_Strategy_Analysis_ByXuLi.py"

if not SPLIT_ANALYSIS_FILE.exists():
    SPLIT_ANALYSIS_FILE = PROJECT_ROOT / "src" / "data_preprocessing" / "SKAB_Split_Strategy_Analysis_ByXuLi.py.py"

assert SPLIT_ANALYSIS_FILE.exists(), f"Split strategy analysis file not found: {SPLIT_ANALYSIS_FILE}"

spec = importlib.util.spec_from_file_location("skab_split_strategy_analysis_by_xuli", SPLIT_ANALYSIS_FILE)
split_analysis_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_analysis_module)

# Use the absolute dataset path during tests.
# The original split-analysis script uses a relative path, which depends on the
# current working directory. Pytest/PyCharm may run from the tests directory, so
# we override it here to make the tests stable.
split_analysis_module.RAW_DATA_DIR = str(DATASET_ROOT)

WINDOW_SIZE = split_analysis_module.WINDOW_SIZE
HORIZON = split_analysis_module.HORIZON
build_labels_from_one_file = split_analysis_module.build_labels_from_one_file
collect_file_statistics = split_analysis_module.collect_file_statistics
build_strategy_a = split_analysis_module.build_strategy_a
build_strategy_b = split_analysis_module.build_strategy_b


def _summarise_split(file_stats, split_plan):
    """
    Helper used only by the tests.
    It summarises sample counts and positive-label ratios for each split.
    """
    summary = {}

    for split_key in ["train", "val", "test"]:
        labels = []

        for folder, file_name in split_plan[split_key]:
            labels.append(file_stats[folder][file_name])

        y = np.concatenate(labels, axis=0) if labels else np.array([], dtype=np.int64)
        n_total = len(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        pos_ratio = n_pos / n_total if n_total > 0 else 0.0

        summary[split_key] = {
            "samples": n_total,
            "pos": n_pos,
            "neg": n_neg,
            "pos_ratio": pos_ratio,
        }

    total_samples = sum(v["samples"] for v in summary.values())
    for split_key in ["train", "val", "test"]:
        summary[split_key]["data_ratio"] = summary[split_key]["samples"] / total_samples

    return summary


def test_build_labels_from_one_file_uses_future_horizon_only():
    """
    Test the core early-warning label rule.

    The first generated sample uses a historical window ending at index 19.
    Because anomaly=1 appears in the future horizon at index 24, its label should
    be 1. This validates that the label is based on future anomaly occurrence,
    not simply the current timestamp.
    """
    total_len = 35
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01 00:00:00", periods=total_len, freq="s"),
            "sensor_1": np.arange(total_len, dtype=float),
            "anomaly": np.zeros(total_len, dtype=int),
        }
    )

    df.loc[24, "anomaly"] = 1

    labels = build_labels_from_one_file(df, window_size=20, horizon=10)

    expected_num_samples = total_len - 20 - 10 + 1

    assert len(labels) == expected_num_samples
    assert labels[0] == 1


def test_build_labels_from_one_file_does_not_use_current_window_anomaly_as_label():
    """
    Test that anomalies inside the historical input window do not directly define
    the label if there is no anomaly in the future horizon.

    This protects the task definition from becoming current anomaly detection.
    The intended task is early-warning prediction: past window -> future anomaly.
    """
    total_len = 35
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01 00:00:00", periods=total_len, freq="s"),
            "sensor_1": np.arange(total_len, dtype=float),
            "anomaly": np.zeros(total_len, dtype=int),
        }
    )

    # This anomaly is inside the first historical window, not in its future horizon.
    df.loc[10, "anomaly"] = 1

    labels = build_labels_from_one_file(df, window_size=20, horizon=10)

    assert labels[0] == 0


def test_strategy_a_split_plan_structure():
    """
    Test that Strategy A matches the planned file-level split:
    train = valve1/0-11, val = valve1/12-15, test = valve2/0-3.
    """
    strategy_a = build_strategy_a()

    assert len(strategy_a["train"]) == 12
    assert len(strategy_a["val"]) == 4
    assert len(strategy_a["test"]) == 4

    assert strategy_a["train"][0] == ("valve1", "0.csv")
    assert strategy_a["train"][-1] == ("valve1", "11.csv")
    assert strategy_a["val"][0] == ("valve1", "12.csv")
    assert strategy_a["val"][-1] == ("valve1", "15.csv")
    assert strategy_a["test"][0] == ("valve2", "0.csv")
    assert strategy_a["test"][-1] == ("valve2", "3.csv")


def test_collect_file_statistics_loads_expected_valve_files():
    """
    Test that the split analysis module can collect labels from the expected SKAB
    valve1 and valve2 files.
    """
    file_stats = collect_file_statistics()

    assert set(file_stats.keys()) == {"valve1", "valve2"}
    assert len(file_stats["valve1"]) == 16
    assert len(file_stats["valve2"]) == 4
    assert "0.csv" in file_stats["valve1"]
    assert "3.csv" in file_stats["valve2"]
    assert len(file_stats["valve1"]["0.csv"]) > 0
    assert set(np.unique(file_stats["valve1"]["0.csv"])).issubset({0, 1})


def test_strategy_a_real_split_distribution_matches_analysis_output():
    """
    Test the real Strategy A split statistics against the analysis output.

    These values are based on the SKAB split strategy analysis result and help
    verify that the data split used for the DL pipeline has a stable label ratio
    across train, validation, and test sets.
    """
    file_stats = collect_file_statistics()
    strategy_a = build_strategy_a()
    summary = _summarise_split(file_stats, strategy_a)

    assert summary["train"]["samples"] == 13245
    assert summary["train"]["pos"] == 4816
    assert summary["train"]["neg"] == 8429

    assert summary["val"]["samples"] == 4453
    assert summary["val"]["pos"] == 1637
    assert summary["val"]["neg"] == 2816

    assert summary["test"]["samples"] == 4196
    assert summary["test"]["pos"] == 1553
    assert summary["test"]["neg"] == 2643

    assert 0.36 <= summary["train"]["pos_ratio"] <= 0.37
    assert 0.36 <= summary["val"]["pos_ratio"] <= 0.37
    assert 0.37 <= summary["test"]["pos_ratio"] <= 0.38


def test_strategy_b_real_split_distribution_matches_analysis_output():
    """
    Test the real Strategy B split statistics against the analysis output.
    This provides a second split-strategy validation baseline for comparison.
    """
    file_stats = collect_file_statistics()
    strategy_b = build_strategy_b()
    summary = _summarise_split(file_stats, strategy_b)

    assert summary["train"]["samples"] == 14263
    assert summary["train"]["pos"] == 5153
    assert summary["train"]["neg"] == 9110

    assert summary["val"]["samples"] == 3323
    assert summary["val"]["pos"] == 1220
    assert summary["val"]["neg"] == 2103

    assert summary["test"]["samples"] == 4308
    assert summary["test"]["pos"] == 1633
    assert summary["test"]["neg"] == 2675

    assert 0.36 <= summary["train"]["pos_ratio"] <= 0.37
    assert 0.36 <= summary["val"]["pos_ratio"] <= 0.37
    assert 0.37 <= summary["test"]["pos_ratio"] <= 0.38
