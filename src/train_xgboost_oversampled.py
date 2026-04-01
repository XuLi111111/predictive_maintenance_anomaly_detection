from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


def load_split(data_dir: Path, split_name: str):
    split_path = data_dir / f"{split_name}.npz"
    data = np.load(split_path, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    times = data["times"]

    return X, y, times


def print_split_info(name: str, X: np.ndarray, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n=== {name.upper()} ===")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("label counts:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")


def print_time_range(name: str, times: np.ndarray) -> None:
    print(f"{name} time range: {times[0]} -> {times[-1]}")


def choose_dev_split(y_train: np.ndarray) -> int:
    candidate_fractions = [0.80, 0.85, 0.90, 0.75, 0.70, 0.95]

    for frac in candidate_fractions:
        split_idx = int(len(y_train) * frac)

        y_subtrain = y_train[:split_idx]
        y_dev = y_train[split_idx:]

        subtrain_pos = int((y_subtrain == 1).sum())
        dev_pos = int((y_dev == 1).sum())

        if subtrain_pos > 0 and dev_pos > 0:
            print(
                f"\nUsing chronological dev split at {frac:.0%} "
                f"(subtrain={split_idx}, dev={len(y_train) - split_idx})"
            )
            print(f"subtrain positives: {subtrain_pos}")
            print(f"dev positives: {dev_pos}")
            return split_idx

    raise ValueError("Could not find a chronological dev split with positives in both parts.")


def oversample_subtrain(X_subtrain: np.ndarray, y_subtrain: np.ndarray):
    print("\nApplying RandomOverSampler on subtrain only...")
    ros = RandomOverSampler(sampling_strategy=0.10, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_subtrain, y_subtrain)

    print_split_info("subtrain_resampled", X_resampled, y_resampled)
    return X_resampled, y_resampled


def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    negatives = tn + fp
    false_positive_rate = fp / negatives if negatives > 0 else 0.0

    print(f"\n=== {split_name.upper()} METRICS @ threshold={threshold:.3f} ===")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print(f"False Positive Rate: {false_positive_rate:.6f}")

    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        print(f"ROC-AUC: {roc_auc:.6f}")
        print(f"PR-AUC: {pr_auc:.6f}")
    else:
        print("ROC-AUC: not defined (only one class present in this split)")
        print("PR-AUC: not defined (only one class present in this split)")


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    best_threshold = 0.50
    best_f1 = -1.0

    print("\n=== DEV THRESHOLD SWEEP ===")
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(np.int8)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        print(
            f"threshold={threshold:.3f} | "
            f"precision={precision:.6f} | recall={recall:.6f} | f1={f1:.6f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    print(f"\nBest dev threshold: {best_threshold:.3f}")
    print(f"Best dev F1: {best_f1:.6f}")
    return best_threshold


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "processed" / "zenodo_pump_2d"

    X_train_full, y_train_full, t_train_full = load_split(data_dir, "train")
    X_val, y_val, t_val = load_split(data_dir, "val")
    X_test, y_test, t_test = load_split(data_dir, "test")

    print_split_info("train_full", X_train_full, y_train_full)
    print_split_info("val", X_val, y_val)
    print_split_info("test", X_test, y_test)

    print()
    print_time_range("train_full", t_train_full)
    print_time_range("val", t_val)
    print_time_range("test", t_test)

    split_idx = choose_dev_split(y_train_full)

    X_subtrain = X_train_full[:split_idx]
    y_subtrain = y_train_full[:split_idx]
    t_subtrain = t_train_full[:split_idx]

    X_dev = X_train_full[split_idx:]
    y_dev = y_train_full[split_idx:]
    t_dev = t_train_full[split_idx:]

    print_split_info("subtrain", X_subtrain, y_subtrain)
    print_split_info("dev", X_dev, y_dev)

    print()
    print_time_range("subtrain", t_subtrain)
    print_time_range("dev", t_dev)

    X_subtrain_resampled, y_subtrain_resampled = oversample_subtrain(X_subtrain, y_subtrain)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,
        max_delta_step=1,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\nTraining oversampled XGBoost...")
    model.fit(
        X_subtrain_resampled,
        y_subtrain_resampled,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )
    print("Training complete.")

    dev_prob = model.predict_proba(X_dev)[:, 1]
    best_threshold = find_best_threshold(y_dev, dev_prob)

    evaluate_split("dev", y_dev, dev_prob, threshold=best_threshold)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    evaluate_split("val", y_val, val_prob, threshold=best_threshold)
    evaluate_split("test", y_test, test_prob, threshold=best_threshold)


if __name__ == "__main__":
    main()