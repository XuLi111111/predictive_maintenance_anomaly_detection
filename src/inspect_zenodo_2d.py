from pathlib import Path
import numpy as np


def inspect_split(split_path: Path, split_name: str) -> None:
    data = np.load(split_path, allow_pickle=True)

    print(f"\n=== {split_name.upper()} ===")
    print(f"path: {split_path}")
    print(f"keys: {list(data.files)}")

    for key in data.files:
        value = data[key]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", type(value))
        print(f"{key}: shape={shape}, dtype={dtype}")

    if "X" in data.files and "y" in data.files:
        X = data["X"]
        y = data["y"]

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        unique, counts = np.unique(y, return_counts=True)
        print("label counts:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "processed" / "zenodo_pump_2d"

    print(f"repo root: {repo_root}")
    print(f"data dir: {data_dir}")

    inspect_split(data_dir / "train.npz", "train")
    inspect_split(data_dir / "val.npz", "val")
    inspect_split(data_dir / "test.npz", "test")


if __name__ == "__main__":
    main()