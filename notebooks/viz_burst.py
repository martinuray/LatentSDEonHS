import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.anomaly_detection import create_random_burst_mask


def _load_psm_train(data_dir: Path) -> np.ndarray:
    csv_path = data_dir / "PSM" / "raw" / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"PSM train file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
    df = df.drop(columns=ts_cols, errors="ignore")
    arr = df.to_numpy(dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0)


def _sample_window_starts(n_total: int, window_len: int, n_windows: int, seed: int) -> np.ndarray:
    max_start = n_total - window_len
    if max_start < 0:
        raise ValueError(f"window_len={window_len} exceeds sequence length={n_total}")

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=n_windows)
    return starts


def _masked_with_nans(window: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # window: (T, F), mask: (F, T)
    masked = window.copy()
    masked[~mask.T] = np.nan
    return masked


def _plot_windows(windows: list[np.ndarray], masked_windows: list[np.ndarray], starts: np.ndarray, feature_indices: np.ndarray, save_path: Path):
    n_windows = len(windows)
    n_features = len(feature_indices)

    fig, axes = plt.subplots(n_windows, n_features, figsize=(2.8 * n_features, 2.4 * n_windows), sharex=True)
    if n_windows == 1:
        axes = np.expand_dims(axes, axis=0)

    time_axis = np.arange(windows[0].shape[0])

    for row in range(n_windows):
        for col, feat_idx in enumerate(feature_indices):
            ax = axes[row, col]
            ax.plot(time_axis, windows[row][:, feat_idx], color="tab:blue", linewidth=1.0, label="original")
            ax.plot(time_axis, masked_windows[row][:, feat_idx], color="tab:red", linewidth=1.0, label="masked")
            ax.set_title(f"W{row + 1}@{starts[row]} | F{feat_idx}")
            ax.grid(alpha=0.2)

            if col == 0:
                ax.set_ylabel("value")
            if row == n_windows - 1:
                ax.set_xlabel("time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize burst masks on PSM windows.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_dir"))
    parser.add_argument("--window-len", type=int, default=100)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--masked-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=Path("out/psm_burst_windows.png"))
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data = _load_psm_train(args.data_dir)
    n_total, n_available_features = data.shape

    n_features = min(args.num_features, n_available_features)
    feature_indices = np.arange(n_features)
    starts = _sample_window_starts(n_total, args.window_len, args.num_windows, args.seed)

    windows = []
    masked_windows = []
    for start in starts:
        window = data[start:start + args.window_len]
        mask = create_random_burst_mask(
            n_features=n_features,
            x_len=args.window_len,
            masked_ratio=args.masked_ratio,
        )
        windows.append(window)
        masked_windows.append(_masked_with_nans(window, mask))

    _plot_windows(windows, masked_windows, starts, feature_indices, args.save_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
