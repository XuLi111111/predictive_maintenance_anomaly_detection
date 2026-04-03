from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


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


def print_score_summary(split_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    print(f"\n=== {split_name.upper()} SCORE SUMMARY ===")
    print(f"overall min={y_prob.min():.6f} max={y_prob.max():.6f} mean={y_prob.mean():.6f}")

    neg_scores = y_prob[y_true == 0]
    pos_scores = y_prob[y_true == 1]

    if len(neg_scores) > 0:
        print(
            "negatives: "
            f"count={len(neg_scores)} "
            f"mean={neg_scores.mean():.6f} "
            f"p50={np.percentile(neg_scores, 50):.6f} "
            f"p90={np.percentile(neg_scores, 90):.6f} "
            f"p95={np.percentile(neg_scores, 95):.6f} "
            f"p99={np.percentile(neg_scores, 99):.6f}"
        )

    if len(pos_scores) > 0:
        print(
            "positives: "
            f"count={len(pos_scores)} "
            f"mean={pos_scores.mean():.6f} "
            f"p50={np.percentile(pos_scores, 50):.6f} "
            f"p90={np.percentile(pos_scores, 90):.6f} "
            f"p95={np.percentile(pos_scores, 95):.6f} "
            f"p99={np.percentile(pos_scores, 99):.6f}"
        )
    else:
        print("positives: none in this split")


def print_top_scores(split_name: str, y_true: np.ndarray, y_prob: np.ndarray, times: np.ndarray, top_n: int = 20) -> None:
    order = np.argsort(-y_prob)
    top_idx = order[:top_n]

    print(f"\n=== TOP {top_n} {split_name.upper()} SCORES ===")
    print("rank | prob | label | time")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank:>4} | {y_prob[idx]:.6f} | {int(y_true[idx])} | {times[idx]}")


def print_positive_scores(split_name: str, y_true: np.ndarray, y_prob: np.ndarray, times: np.ndarray, top_n: int = 20) -> None:
    pos_idx = np.where(y_true == 1)[0]

    print(f"\n=== TOP POSITIVE {split_name.upper()} SCORES ===")
    if len(pos_idx) == 0:
        print("No positive rows in this split.")
        return

    pos_probs = y_prob[pos_idx]
    order = np.argsort(-pos_probs)
    top_local = order[:top_n]
    top_global = pos_idx[top_local]

    print("rank | prob | label | time")
    for rank, idx in enumerate(top_global, start=1):
        print(f"{rank:>4} | {y_prob[idx]:.6f} | {int(y_true[idx])} | {times[idx]}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "processed" / "zenodo_pump_2d"

    X_train_full, y_train_full, t_train_full = load_split(data_dir, "train")
    X_val, y_val, t_val = load_split(data_dir, "val")
    X_test, y_test, t_test = load_split(data_dir, "test")

    split_idx = choose_dev_split(y_train_full)

    X_subtrain = X_train_full[:split_idx]
    y_subtrain = y_train_full[:split_idx]

    X_dev = X_train_full[split_idx:]
    y_dev = y_train_full[split_idx:]
    t_dev = t_train_full[split_idx:]

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

    print("\nTraining oversampled XGBoost for score inspection...")
    model.fit(
        X_subtrain_resampled,
        y_subtrain_resampled,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )
    print("Training complete.")

    dev_prob = model.predict_proba(X_dev)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    print_score_summary("dev", y_dev, dev_prob)
    print_top_scores("dev", y_dev, dev_prob, t_dev, top_n=20)
    print_positive_scores("dev", y_dev, dev_prob, t_dev, top_n=20)

    print_score_summary("test", y_test, test_prob)
    print_top_scores("test", y_test, test_prob, t_test, top_n=20)
    print_positive_scores("test", y_test, test_prob, t_test, top_n=20)


if __name__ == "__main__":
    main()