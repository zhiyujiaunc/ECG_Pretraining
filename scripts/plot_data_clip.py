import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot a data_clips .npy file.")
    parser.add_argument("path")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--out_dir", default="visualizations/data_clips/plots")
    args = parser.parse_args()

    x = np.load(args.path, allow_pickle=False)
    if x.size == 0:
        print(f"{args.path} is empty with shape={x.shape}")
        return

    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={x.shape}")

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.max_samples, x.shape[0])
    channels = x.shape[1]

    fig, axes = plt.subplots(channels, 1, figsize=(14, max(2, channels * 2)), sharex=True)
    if channels == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(x[:n, i], linewidth=0.7)
        ax.set_ylabel(f"ch{i}", rotation=0, labelpad=18)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"{args.path} first {n} samples, shape={x.shape}, dtype={x.dtype}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    stem = os.path.splitext(os.path.basename(args.path))[0]
    out_path = os.path.join(args.out_dir, f"{stem}_first{n}_{channels}ch.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"path={args.path}")
    print(f"shape={x.shape}")
    print(f"dtype={x.dtype}")
    print(f"min={x.min(axis=0)}")
    print(f"max={x.max(axis=0)}")
    print(f"mean={x.mean(axis=0)}")
    print(out_path)


if __name__ == "__main__":
    main()
