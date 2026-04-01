from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
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


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    if positives == 0:
        raise ValueError("Training set has no positive samples.")

    return negatives / positives


def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> None:
    y_pred = (y_prob >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    negatives = tn + fp
    false_positive_rate = fp / negatives if negatives > 0 else 0.0

    print(f"\n=== {split_name.upper()} METRICS @ threshold={threshold:.2f} ===")
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "processed" / "zenodo_pump_2d"

    X_train, y_train, t_train = load_split(data_dir, "train")
    X_val, y_val, t_val = load_split(data_dir, "val")
    X_test, y_test, t_test = load_split(data_dir, "test")

    print_split_info("train", X_train, y_train)
    print_split_info("val", X_val, y_val)
    print_split_info("test", X_test, y_test)

    print()
    print_time_range("train", t_train)
    print_time_range("val", t_val)
    print_time_range("test", t_test)

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"\nscale_pos_weight: {scale_pos_weight:.6f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    print("\nTraining XGBoost baseline...")
    model.fit(X_train, y_train)
    print("Training complete.")

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    evaluate_split("val", y_val, val_prob, threshold=0.5)
    evaluate_split("test", y_test, test_prob, threshold=0.5)


if __name__ == "__main__":
    main()