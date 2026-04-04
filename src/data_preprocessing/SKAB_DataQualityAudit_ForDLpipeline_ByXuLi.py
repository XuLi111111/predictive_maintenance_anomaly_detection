import pandas as pd
from pathlib import Path
from itertools import combinations


def load_csv_time_info(csv_file: Path) -> dict:
    df = pd.read_csv(csv_file, sep=';')
    df['datetime'] = pd.to_datetime(df['datetime'])

    return {
        'file': str(csv_file),
        'start': df['datetime'].min(),
        'end': df['datetime'].max(),
        'rows': len(df),
        'anomaly_count': int((df['anomaly'] == 1).sum()) if 'anomaly' in df.columns else 0,
        'timestamps': set(df['datetime'])
    }


def compare_overlap(file_a: str, file_b: str, common_timestamps: set) -> dict:
    df_a = pd.read_csv(file_a, sep=';')
    df_b = pd.read_csv(file_b, sep=';')

    df_a['datetime'] = pd.to_datetime(df_a['datetime'])
    df_b['datetime'] = pd.to_datetime(df_b['datetime'])

    overlap_df_a = df_a[df_a['datetime'].isin(common_timestamps)].sort_values('datetime').reset_index(drop=True)
    overlap_df_b = df_b[df_b['datetime'].isin(common_timestamps)].sort_values('datetime').reset_index(drop=True)

    full_equal = overlap_df_a.equals(overlap_df_b)

    sensor_cols = [
        col for col in overlap_df_a.columns
        if col not in ['datetime', 'anomaly', 'changepoint']
    ]
    sensor_equal = overlap_df_a[sensor_cols].equals(overlap_df_b[sensor_cols])

    label_equal = True
    if 'anomaly' in overlap_df_a.columns and 'anomaly' in overlap_df_b.columns:
        label_equal = overlap_df_a['anomaly'].equals(overlap_df_b['anomaly'])

    changepoint_equal = True
    if 'changepoint' in overlap_df_a.columns and 'changepoint' in overlap_df_b.columns:
        changepoint_equal = overlap_df_a['changepoint'].equals(overlap_df_b['changepoint'])

    return {
        'full_equal': full_equal,
        'sensor_equal': sensor_equal,
        'label_equal': label_equal,
        'changepoint_equal': changepoint_equal,
        'overlap_df_a': overlap_df_a,
        'overlap_df_b': overlap_df_b
    }


def run_dataset_audit(dataset_root: str = '../../data/raw/dataset2', subset_filter=None):
    root = Path(dataset_root)
    csv_files = sorted(root.rglob('*.csv'))

    if subset_filter is not None:
        csv_files = [f for f in csv_files if subset_filter(str(f))]

    if not csv_files:
        print('No CSV files found. Please check the dataset path or filter conditions.')
        return

    print('=' * 100)
    print('SKAB DATASET AUDIT REPORT')
    print('=' * 100)
    print(f'Number of files checked: {len(csv_files)}')
    print()

    file_infos = []
    print('Section 1: Load time range and anomaly statistics for each file\n')
    for csv_file in csv_files:
        info = load_csv_time_info(csv_file)
        file_infos.append(info)

        print(f'File: {info["file"]}')
        print(f'  Start time: {info["start"]}')
        print(f'  End time: {info["end"]}')
        print(f'  Number of rows: {info["rows"]}')
        print(f'  Number of anomaly=1: {info["anomaly_count"]}\n')

    print('=' * 100)
    print('Section 2: Check for overlapping timestamps between files\n')

    overlap_cases = []

    for file_a, file_b in combinations(file_infos, 2):
        range_overlap = not (file_a['end'] < file_b['start'] or file_b['end'] < file_a['start'])
        if not range_overlap:
            continue

        common_timestamps = file_a['timestamps'].intersection(file_b['timestamps'])
        if not common_timestamps:
            continue

        overlap_result = compare_overlap(file_a['file'], file_b['file'], common_timestamps)

        case = {
            'file_a': file_a['file'],
            'file_b': file_b['file'],
            'overlap_count': len(common_timestamps),
            'sample_times': sorted(common_timestamps)[:5],
            **overlap_result
        }
        overlap_cases.append(case)

        print('Overlap detected:')
        print(f'  File A: {case["file_a"]}')
        print(f'  File B: {case["file_b"]}')
        print(f'  Number of overlapping timestamps: {case["overlap_count"]}')
        print(f'  Sample timestamps (first 5): {case["sample_times"]}')
        print(f'  Full data match: {case["full_equal"]}')
        print(f'  Sensor data match: {case["sensor_equal"]}')
        print(f'  Anomaly label match: {case["label_equal"]}')
        print(f'  Changepoint label match: {case["changepoint_equal"]}')

        if case['sensor_equal'] and not case['label_equal']:
            print('  ⚠️ Conclusion: Same signal has conflicting anomaly labels')
        elif case['full_equal']:
            print('  ✔ Conclusion: Fully duplicated overlapping segment')
        elif case['sensor_equal'] and case['label_equal']:
            print('  ✔ Conclusion: Same signal and labels, overlap due to slicing')
        else:
            print('  ⚠️ Conclusion: Both signal and labels differ')

        print()

    print('=' * 100)
    print('Section 3: Final audit conclusion\n')

    if not overlap_cases:
        print('No overlapping timestamps found between any CSV files.')
        print('Conclusion: This subset can be treated as independent, non-overlapping time series.')
    else:
        n_full_equal = sum(1 for c in overlap_cases if c['full_equal'])
        n_sensor_equal_label_conflict = sum(
            1 for c in overlap_cases if c['sensor_equal'] and not c['label_equal']
        )
        n_sensor_equal_label_equal = sum(
            1 for c in overlap_cases if c['sensor_equal'] and c['label_equal'] and not c['full_equal']
        )

        print(f'Total overlap cases: {len(overlap_cases)}')
        print(f'Fully duplicated cases: {n_full_equal}')
        print(f'Same signal but conflicting anomaly labels: {n_sensor_equal_label_conflict}')
        print(f'Same signal with consistent labels (slicing overlap): {n_sensor_equal_label_equal}')
        print()

        if n_sensor_equal_label_conflict > 0:
            print('Final Conclusion: Label conflicts exist. This subset cannot be directly merged and randomly split.')
        else:
            print('Final Conclusion: No anomaly label conflicts detected. This subset can be used as a clean data source for modeling.')

    print('=' * 100)


def audit_valve_only(dataset_root: str = '../../data/raw/dataset2'):
    print('Running audit on valve1 and valve2 subset...\n')
    run_dataset_audit(
        dataset_root=dataset_root,
        subset_filter=lambda p: ('valve1' in p or 'valve2' in p)
    )


def audit_other_only(dataset_root: str = '../../data/raw/dataset2'):
    print('Running audit on other subset...\n')
    run_dataset_audit(
        dataset_root=dataset_root,
        subset_filter=lambda p: 'other' in p
    )


def audit_all(dataset_root: str = '../../data/raw/dataset2'):
    print('Running audit on full dataset2...\n')
    run_dataset_audit(dataset_root=dataset_root)


if __name__ == '__main__':

    audit_valve_only()
    # audit_other_only()
    # audit_all()