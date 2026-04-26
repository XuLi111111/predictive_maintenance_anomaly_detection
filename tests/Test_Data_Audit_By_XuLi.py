

"""

This test module validates the SKAB data quality audit functions used before the
Deep Learning pipeline. The goal is not to test model accuracy, but to verify that
my data-audit logic can correctly read SKAB CSV files, check timestamp integrity,
and detect overlap/label-conflict cases.
"""

import sys
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset2"

# Make sure Python can import modules from the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.SKAB_DataQualityAudit_ForDLpipeline_ByXuLi import (  # noqa: E402
    load_csv_time_info,
    compare_overlap,
)


def test_load_csv_time_info_for_real_skab_file():
    """
    Test whether load_csv_time_info() can correctly load a real SKAB CSV file
    and return the key audit information required by the DL data pipeline.
    """
    csv_file = DATASET_ROOT / "valve1" / "0.csv"

    info = load_csv_time_info(csv_file)

    assert info["file"] == str(csv_file)
    assert info["rows"] == 1148
    assert info["anomaly_count"] == 401
    assert info["start"] == pd.Timestamp("2020-03-09 10:14:33")
    assert info["end"] == pd.Timestamp("2020-03-09 10:34:32")
    assert isinstance(info["timestamps"], set)
    assert len(info["timestamps"]) == info["rows"]


def test_valve1_and_valve2_subset_has_no_overlapping_timestamps():
    """
    Test the main audit conclusion for the selected valve1 and valve2 subset.

    Based on the audit result, these files should not have overlapping timestamps,
    meaning they can be treated as independent non-overlapping time series for the
    DL pipeline.
    """
    csv_files = sorted(
        f for f in DATASET_ROOT.rglob("*.csv")
        if "valve1" in str(f) or "valve2" in str(f)
    )

    assert len(csv_files) == 20

    file_infos = [load_csv_time_info(csv_file) for csv_file in csv_files]

    for i in range(len(file_infos)):
        for j in range(i + 1, len(file_infos)):
            info_a = file_infos[i]
            info_b = file_infos[j]

            common_timestamps = info_a["timestamps"].intersection(info_b["timestamps"])

            assert len(common_timestamps) == 0, (
                f"Unexpected timestamp overlap found between {info_a['file']} "
                f"and {info_b['file']}"
            )


def test_compare_overlap_detects_conflicting_anomaly_labels(tmp_path):
    """
    Test compare_overlap() with small synthetic CSV files.

    The two files share exactly the same timestamps and sensor values, but their
    anomaly labels are different. This simulates the type of label conflict that
    the data audit is designed to detect.
    """
    file_a = tmp_path / "synthetic_a.csv"
    file_b = tmp_path / "synthetic_b.csv"

    df_a = pd.DataFrame(
        {
            "datetime": ["2020-01-01 00:00:00", "2020-01-01 00:00:01"],
            "sensor_1": [1.0, 2.0],
            "sensor_2": [3.0, 4.0],
            "anomaly": [0, 0],
            "changepoint": [0, 0],
        }
    )

    df_b = pd.DataFrame(
        {
            "datetime": ["2020-01-01 00:00:00", "2020-01-01 00:00:01"],
            "sensor_1": [1.0, 2.0],
            "sensor_2": [3.0, 4.0],
            "anomaly": [1, 1],
            "changepoint": [0, 0],
        }
    )

    df_a.to_csv(file_a, sep=";", index=False)
    df_b.to_csv(file_b, sep=";", index=False)

    common_timestamps = set(pd.to_datetime(df_a["datetime"]))
    result = compare_overlap(str(file_a), str(file_b), common_timestamps)

    assert result["sensor_equal"] is True
    assert result["label_equal"] is False
    assert result["changepoint_equal"] is True
    assert result["full_equal"] is False