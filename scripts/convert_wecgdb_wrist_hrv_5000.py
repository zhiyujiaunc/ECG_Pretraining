#!/usr/bin/env python3
import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import neurokit2 as nk
import numpy as np
import scipy.io as sio
import torch


SITES = ("LA_A", "LA_V3", "LA_V5")
SPLITS = ("training", "testing")


def zscore_channels(x):
    x = x.astype(np.float32, copy=False)
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), 1e-6)
    return (x - mean) / std


def as_bool_string(value):
    if isinstance(value, str):
        return value.lower()
    return str(value).lower()


def load_mat_from_zip(zip_file, name):
    return sio.loadmat(
        io.BytesIO(zip_file.read(name)),
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["info", *SITES],
    )


def split_files(files, val_ratio, test_ratio, seed):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_total = len(files)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    return {
        "test": set(files[:n_test]),
        "val": set(files[n_test : n_test + n_val]),
        "train": set(files[n_test + n_val :]),
    }


def label_from_window(ecg, fs, min_rr, max_rr):
    try:
        cleaned = nk.ecg_clean(ecg.astype(np.float32), sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    except Exception:
        return None

    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / float(fs)
    if np.any(rr < min_rr) or np.any(rr > max_rr):
        return None
    hr = float(np.mean(60.0 / rr))
    if not (35.0 <= hr <= 220.0):
        return None
    sdnn = float(np.std(rr, ddof=1) * 1000.0)
    rr_diff_ms = np.abs(np.diff(rr)) * 1000.0
    rmssd = float(np.sqrt(np.mean(rr_diff_ms**2))) if len(rr_diff_ms) else 0.0
    pnn50 = float(100.0 * np.mean(rr_diff_ms > 50.0)) if len(rr_diff_ms) else 0.0
    return hr, sdnn, rmssd, pnn50


def convert(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_samples = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    metadata = {"config": vars(args), "windows": []}
    counts = {
        "files_total": 0,
        "recordings_available": 0,
        "windows_total": 0,
        "windows_kept": 0,
        "windows_bad_label": 0,
    }

    with zipfile.ZipFile(args.zip_path) as zip_file:
        mat_files = sorted(name for name in zip_file.namelist() if name.endswith(".mat"))
        counts["files_total"] = len(mat_files)
        file_splits = split_files(mat_files, args.val_ratio, args.test_ratio, args.seed)

        for split_name, split_set in file_splits.items():
            for mat_name in sorted(split_set):
                mat_dict = load_mat_from_zip(zip_file, mat_name)
                fs = int(mat_dict["info"].Fs)
                for site in SITES:
                    site_obj = mat_dict[site]
                    for recording_split in SPLITS:
                        split_obj = getattr(site_obj, recording_split)
                        if as_bool_string(split_obj.data_availability) != "yes":
                            continue
                        data = np.asarray(split_obj.wECG, dtype=np.float32)
                        if data.ndim != 2 or data.shape[1] != 2:
                            continue
                        counts["recordings_available"] += 1

                        for start in range(0, data.shape[0] - args.window_size + 1, args.stride):
                            stop = start + args.window_size
                            counts["windows_total"] += 1
                            window_tc = data[start:stop]
                            label = label_from_window(
                                window_tc[:, args.label_channel],
                                fs,
                                args.min_rr,
                                args.max_rr,
                            )
                            if label is None:
                                counts["windows_bad_label"] += 1
                                continue

                            sample = torch.from_numpy(zscore_channels(window_tc.T))
                            split_samples[split_name].append(sample)
                            split_labels[split_name].append(torch.tensor(label, dtype=torch.float32))
                            counts["windows_kept"] += 1
                            metadata["windows"].append(
                                {
                                    "file": mat_name,
                                    "site": site,
                                    "recording_split": recording_split,
                                    "split": split_name,
                                    "dataset": "wECGdb Wrist",
                                    "start": start,
                                    "stop": stop,
                                    "hr_bpm": label[0],
                                    "sdnn_ms": label[1],
                                    "rmssd_ms": label[2],
                                    "pnn50_pct": label[3],
                                }
                            )

    for split_name in ("train", "val", "test"):
        if split_samples[split_name]:
            samples = torch.stack(split_samples[split_name])
            labels = torch.stack(split_labels[split_name])
        else:
            samples = torch.empty(0, 2, args.window_size)
            labels = torch.empty(0, 4)
        torch.save({"samples": samples, "labels": labels}, output_dir / f"{split_name}.pt")
        print(f"{split_name}: samples={tuple(samples.shape)} labels={tuple(labels.shape)} -> {output_dir / f'{split_name}.pt'}")

    metadata["counts"] = counts
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(counts, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Convert wECGdb wrist ECG to supervised HRV downstream data.")
    parser.add_argument("--zip_path", default="wECG_dataset.zip")
    parser.add_argument("--output_dir", default="ml-famae/data_wecg_wrist_hrv_5000")
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--label_channel", type=int, default=0)
    parser.add_argument("--min_rr", type=float, default=0.30)
    parser.add_argument("--max_rr", type=float, default=2.00)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
