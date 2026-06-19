import argparse
import io
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main():
    parser = argparse.ArgumentParser(description="Plot one wECGdb sample from the downloaded zip.")
    parser.add_argument("--zip_path", default="wECG_dataset.zip")
    parser.add_argument("--mat_name", default="wECG_dataset/dataset_001.mat")
    parser.add_argument("--site", default="LA_A", choices=["LA_A", "LA_V3", "LA_V5"])
    parser.add_argument("--split", default="training", choices=["training", "testing"])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out_dir", default="visualizations/wecgdb/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zip_file:
        mat = sio.loadmat(
            io.BytesIO(zip_file.read(args.mat_name)),
            squeeze_me=True,
            struct_as_record=False,
            variable_names=["info", args.site],
        )

    fs = int(mat["info"].Fs)
    site_obj = mat[args.site]
    split_obj = getattr(site_obj, args.split)
    if split_obj.data_availability != "yes":
        raise ValueError(f"{args.mat_name} {args.site}.{args.split} is not available")

    reference = np.asarray(split_obj.reference_12_lead, dtype=float)
    wrist = np.asarray(split_obj.wECG, dtype=float)
    n = min(int(args.seconds * fs), reference.shape[0], wrist.shape[0])
    t = np.arange(n) / fs

    stem = os.path.splitext(os.path.basename(args.mat_name))[0]
    prefix = f"{stem}_{args.site}_{args.split}_{args.seconds:g}s"

    fig, axes = plt.subplots(12, 1, figsize=(14, 12), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, reference[:n, i], linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=18)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"wECGdb {stem} {args.site}.{args.split}: reference 12-lead ECG")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    reference_path = os.path.join(args.out_dir, f"{prefix}_reference_12lead.png")
    fig.savefig(reference_path, dpi=180)
    plt.close(fig)

    wrist_names = ["wECG lead I", f"wECG lead {args.site.replace('LA_', 'LA-')}"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, wrist[:n, i], linewidth=0.8)
        ax.set_ylabel(wrist_names[i], rotation=0, labelpad=45)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"wECGdb {stem} {args.site}.{args.split}: wrist 2-lead ECG")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    wrist_path = os.path.join(args.out_dir, f"{prefix}_wrist_2lead.png")
    fig.savefig(wrist_path, dpi=180)
    plt.close(fig)

    print(f"sampling_rate={fs}")
    print(f"reference_shape={reference.shape}")
    print(f"wrist_shape={wrist.shape}")
    print(reference_path)
    print(wrist_path)


if __name__ == "__main__":
    main()
