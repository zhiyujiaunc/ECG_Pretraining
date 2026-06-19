#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import numpy as np
import neurokit2 as nk
import scipy.signal as signal
import torch


def zscore(x):
    x = x.astype(np.float32, copy=False)
    return (x - x.mean()) / max(float(x.std()), 1e-6)


def downsample_to_target(x, source_fs, target_fs):
    if source_fs == target_fs:
        return x.astype(np.float32, copy=False)
    gcd = np.gcd(source_fs, target_fs)
    up = target_fs // gcd
    down = source_fs // gcd
    return signal.resample_poly(x.astype(np.float32), up, down).astype(np.float32)


def detect_chest_rpeaks(chest, fs):
    cleaned = nk.ecg_clean(zscore(chest), sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def window_metrics_from_peaks(peaks, start, stop, fs, min_rr, max_rr):
    window_peaks = peaks[(peaks >= start) & (peaks < stop)]
    if len(window_peaks) < 3:
        return None

    rr = np.diff(window_peaks) / fs
    if np.any(rr < min_rr) or np.any(rr > max_rr):
        return None

    hr = float(np.mean(60.0 / rr))
    sdnn = float(np.std(rr, ddof=1) * 1000.0)
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)
    if not (35.0 <= hr <= 220.0):
        return None
    return hr, sdnn, rmssd


def subject_id(path):
    for part in path.parts:
        if part.startswith("s") and part[1:].isdigit():
            return part
    return path.parent.name


def split_subjects(subjects, val_ratio, test_ratio, seed):
    subjects = sorted(subjects)
    random.Random(seed).shuffle(subjects)
    n_test = max(1, round(len(subjects) * test_ratio)) if subjects else 0
    n_val = max(1, round(len(subjects) * val_ratio)) if len(subjects) > 2 else 0
    test = set(subjects[:n_test])
    val = set(subjects[n_test : n_test + n_val])
    train = set(subjects[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


def convert(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.rglob("*.npy"))
    subjects = {subject_id(path) for path in files}
    subject_splits = split_subjects(subjects, args.val_ratio, args.test_ratio, args.seed)

    split_samples = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    metadata = {"config": vars(args), "splits": {k: sorted(v) for k, v in subject_splits.items()}, "windows": []}
    counts = {
        "files_total": len(files),
        "files_empty": 0,
        "files_too_short": 0,
        "files_neurokit_failed": 0,
        "windows_total": 0,
        "windows_kept": 0,
        "windows_bad_label": 0,
    }

    source_window = round(args.window_size * args.source_fs / args.target_fs)
    source_stride = round(args.stride * args.source_fs / args.target_fs)

    for path in files:
        sid = subject_id(path)
        split_name = next(name for name, split_subject_set in subject_splits.items() if sid in split_subject_set)

        arr = np.load(path)
        if arr.size == 0:
            counts["files_empty"] += 1
            continue
        if arr.ndim != 2 or arr.shape[1] <= max(args.chest_channel, args.wrist_channel):
            continue
        if arr.shape[0] < source_window:
            counts["files_too_short"] += 1
            continue

        chest = downsample_to_target(arr[:, args.chest_channel].astype(np.float32), args.source_fs, args.target_fs)
        wrist = downsample_to_target(arr[:, args.wrist_channel].astype(np.float32), args.source_fs, args.target_fs)

        try:
            chest_peaks = detect_chest_rpeaks(chest, args.target_fs)
        except Exception:
            counts["files_neurokit_failed"] += 1
            continue

        for start in range(0, len(wrist) - args.window_size + 1, args.stride):
            counts["windows_total"] += 1
            stop = start + args.window_size

            metrics = window_metrics_from_peaks(
                chest_peaks,
                start,
                stop,
                args.target_fs,
                args.min_rr,
                args.max_rr,
            )
            if metrics is None:
                counts["windows_bad_label"] += 1
                continue

            wrist_window = wrist[start:stop]
            sample = torch.from_numpy(zscore(wrist_window)[None, :])
            label = torch.tensor(metrics, dtype=torch.float32)
            split_samples[split_name].append(sample)
            split_labels[split_name].append(label)
            counts["windows_kept"] += 1

            metadata["windows"].append(
                {
                    "file": str(path),
                    "subject": sid,
                    "split": split_name,
                    "source_start": round(start * args.source_fs / args.target_fs),
                    "source_stop": round(stop * args.source_fs / args.target_fs),
                    "target_start": start,
                    "target_stop": stop,
                    "hr_bpm": metrics[0],
                    "sdnn_ms": metrics[1],
                    "rmssd_ms": metrics[2],
                }
            )

    for split_name in ("train", "val", "test"):
        if split_samples[split_name]:
            samples = torch.stack(split_samples[split_name])
            labels = torch.stack(split_labels[split_name])
        else:
            samples = torch.empty(0, 1, args.window_size)
            labels = torch.empty(0, 3)

        save_path = output_dir / f"{split_name}.pt"
        torch.save({"samples": samples, "labels": labels}, save_path)
        print(f"{split_name}: samples={tuple(samples.shape)} labels={tuple(labels.shape)} -> {save_path}")

    metadata["counts"] = counts
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(counts, indent=2))
    print(f"metadata -> {output_dir / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser(description="Convert our data_clips to CHASE-style wrist ECG HR/HRV .pt files.")
    parser.add_argument("--data_dir", default="data_clips")
    parser.add_argument("--output_dir", default="ml-famae/data_our_wrist_hrv")
    parser.add_argument("--source_fs", type=int, default=2000)
    parser.add_argument("--target_fs", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--chest_channel", type=int, default=0)
    parser.add_argument("--wrist_channel", type=int, default=1)
    parser.add_argument("--min_rr", type=float, default=0.30)
    parser.add_argument("--max_rr", type=float, default=2.00)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
