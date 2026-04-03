from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def load_split(data_dir: Path, split_name: str):
    split_path = data_dir / f"{split_name}.npz"
    data = np.load(split_path, allow_pickle=True)
    return data["X"], data["y"], data["times"]


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
                f"Using chronological dev split at {frac:.0%} "
                f"(subtrain={split_idx}, dev={len(y_train) - split_idx})"
            )
            print(f"subtrain positives: {subtrain_pos}")
            print(f"dev positives: {dev_pos}")
            return split_idx

    raise ValueError("Could not find a chronological dev split with positives in both parts.")


def oversample_subtrain(X_subtrain: np.ndarray, y_subtrain: np.ndarray):
    ros = RandomOverSampler(sampling_strategy=0.10, random_state=42)
    return ros.fit_resample(X_subtrain, y_subtrain)


def evaluate_inverse_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    y_pred = (y_prob <= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    negatives = tn + fp
    false_positive_rate = fp / negatives if negatives > 0 else 0.0

    print(f"\n=== {split_name.upper()} INVERSE METRICS @ prob <= {threshold:.6f} ===")
    print(f"Predicted positives: {int(y_pred.sum())}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print(f"False Positive Rate: {false_positive_rate:.6f}")

    return f1


def find_best_inverse_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = [
        0.000001,
        0.000002,
        0.000005,
        0.000010,
        0.000020,
        0.000050,
        0.000100,
        0.000200,
        0.000500,
        0.001000,
        0.002000,
        0.005000,
    ]

    best_threshold = thresholds[0]
    best_f1 = -1.0

    print("\n=== INVERSE THRESHOLD SWEEP (predict positive if prob <= threshold) ===")
    for threshold in thresholds:
        y_pred = (y_prob <= threshold).astype(np.int8)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(
            f"threshold={threshold:.6f} | "
            f"precision={precision:.6f} | recall={recall:.6f} | f1={f1:.6f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest inverse threshold: {best_threshold:.6f}")
    print(f"Best inverse dev F1: {best_f1:.6f}")
    return best_threshold


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "processed" / "zenodo_pump_2d"

    X_train_full, y_train_full, _ = load_split(data_dir, "train")
    X_val, y_val, _ = load_split(data_dir, "val")
    X_test, y_test, _ = load_split(data_dir, "test")

    split_idx = choose_dev_split(y_train_full)

    X_subtrain = X_train_full[:split_idx]
    y_subtrain = y_train_full[:split_idx]

    X_dev = X_train_full[split_idx:]
    y_dev = y_train_full[split_idx:]

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

    print("\nTraining oversampled XGBoost for inverse-score check...")
    model.fit(
        X_subtrain_resampled,
        y_subtrain_resampled,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )
    print("Training complete.")

    dev_prob = model.predict_proba(X_dev)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    best_threshold = find_best_inverse_threshold(y_dev, dev_prob)

    evaluate_inverse_split("dev", y_dev, dev_prob, best_threshold)
    evaluate_inverse_split("val", y_val, val_prob, best_threshold)
    evaluate_inverse_split("test", y_test, test_prob, best_threshold)


if __name__ == "__main__":
    main()