import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


def activity_from_file(path):
    parts = Path(path).parts
    for name in ("gesture", "static", "freeform"):
        if name in parts:
            return name
    return "unknown"


def file_key(row):
    return row["file"]


def subject_key(row):
    return row.get("subject") or Path(row["file"]).parent.name


def load_base_dataset(data_dir):
    data_dir = Path(data_dir)
    with (data_dir / "metadata.json").open() as f:
        metadata = json.load(f)

    samples = []
    labels = []
    rows = []
    for split in ("train", "val", "test"):
        data = torch.load(data_dir / f"{split}.pt")
        split_rows = [row for row in metadata["windows"] if row["split"] == split]
        if len(split_rows) != len(data["samples"]):
            raise ValueError(f"{split} metadata/sample mismatch: {len(split_rows)} vs {len(data['samples'])}")
        samples.append(data["samples"])
        labels.append(data["labels"])
        rows.extend(split_rows)

    return torch.cat(samples), torch.cat(labels), rows, metadata


def make_file_folds(rows, n_folds, seed):
    files_by_activity = defaultdict(list)
    for row in rows:
        files_by_activity[activity_from_file(row["file"])].append(file_key(row))

    folds = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for files in files_by_activity.values():
        unique_files = sorted(set(files))
        rng.shuffle(unique_files)
        for index, path in enumerate(unique_files):
            folds[index % n_folds].append(path)
    return [set(fold) for fold in folds]


def make_intra_subject_file_folds(rows, n_folds, seed):
    files_by_subject_activity = defaultdict(list)
    for row in rows:
        group = (subject_key(row), activity_from_file(row["file"]))
        files_by_subject_activity[group].append(file_key(row))

    folds = [[] for _ in range(n_folds)]
    rng = random.Random(seed)
    for files in files_by_subject_activity.values():
        unique_files = sorted(set(files))
        rng.shuffle(unique_files)
        for index, path in enumerate(unique_files):
            folds[index % n_folds].append(path)
    return [set(fold) for fold in folds]


def make_validation_files(trainval_files, rows, val_ratio, seed, fold_idx):
    if val_ratio <= 0:
        return set()

    files_by_subject_activity = defaultdict(list)
    for row in rows:
        key = file_key(row)
        if key not in trainval_files:
            continue
        group = (subject_key(row), activity_from_file(row["file"]))
        files_by_subject_activity[group].append(key)

    val_files = set()
    rng = random.Random(seed + 1009 * (fold_idx + 1))
    for files in files_by_subject_activity.values():
        unique_files = sorted(set(files))
        rng.shuffle(unique_files)
        n_val = int(round(len(unique_files) * val_ratio))
        if len(unique_files) > 1:
            n_val = max(1, min(n_val, len(unique_files) - 1))
        else:
            n_val = 0
        val_files.update(unique_files[:n_val])
    return val_files


def save_split(output_dir, split, indices, samples, labels):
    split_samples = samples[indices] if indices else torch.empty(0, *samples.shape[1:], dtype=samples.dtype)
    split_labels = labels[indices] if indices else torch.empty(0, labels.shape[1], dtype=labels.dtype)
    torch.save({"samples": split_samples, "labels": split_labels}, output_dir / f"{split}.pt")
    return split_samples.shape, split_labels.shape


def main():
    parser = argparse.ArgumentParser(description="Create file-level CV splits for wrist HR/HRV data.")
    parser.add_argument("--data_dir", default="./data_our_wrist_hrv_3000")
    parser.add_argument("--output_root", default="./data_our_wrist_hrv_3000_5fold")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--split_mode", choices=["legacy_activity", "intra_subject"], default="legacy_activity")
    parser.add_argument("--val_ratio_of_trainval", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    samples, labels, rows, metadata = load_base_dataset(args.data_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.split_mode == "intra_subject":
        folds = make_intra_subject_file_folds(rows, args.n_folds, args.seed)
    else:
        folds = make_file_folds(rows, args.n_folds, args.seed)

    for fold_idx, test_files in enumerate(folds):
        if args.split_mode == "intra_subject":
            trainval_files = set().union(*[fold for i, fold in enumerate(folds) if i != fold_idx])
            val_files = make_validation_files(trainval_files, rows, args.val_ratio_of_trainval, args.seed, fold_idx)
            train_files = trainval_files - val_files
        else:
            val_files = folds[(fold_idx + 1) % args.n_folds]
            train_files = set().union(*[fold for i, fold in enumerate(folds) if i not in {fold_idx, (fold_idx + 1) % args.n_folds}])
        fold_dir = output_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        split_indices = {"train": [], "val": [], "test": []}
        split_rows = {"train": [], "val": [], "test": []}
        for index, row in enumerate(rows):
            key = file_key(row)
            if key in test_files:
                split = "test"
            elif key in val_files:
                split = "val"
            elif key in train_files:
                split = "train"
            else:
                raise RuntimeError(f"Unassigned file: {key}")
            new_row = dict(row)
            new_row["split"] = split
            split_indices[split].append(index)
            split_rows[split].append(new_row)

        print(f"fold {fold_idx}")
        for split in ("train", "val", "test"):
            sample_shape, label_shape = save_split(fold_dir, split, split_indices[split], samples, labels)
            subjects = len(set(subject_key(r) for r in split_rows[split]))
            print(f"  {split}: samples={tuple(sample_shape)} labels={tuple(label_shape)} files={len(set(file_key(r) for r in split_rows[split]))} subjects={subjects}")

        fold_metadata = {
            "base_data_dir": args.data_dir,
            "base_config": metadata.get("config", {}),
            "fold_index": fold_idx,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "split_mode": args.split_mode,
            "cv_protocol": "intra-subject 5-fold; each subject contributes approximately 20% files to test per fold" if args.split_mode == "intra_subject" else "legacy activity-stratified file fold",
            "val_ratio_of_trainval": args.val_ratio_of_trainval if args.split_mode == "intra_subject" else None,
            "windows": split_rows["train"] + split_rows["val"] + split_rows["test"],
        }
        with (fold_dir / "metadata.json").open("w") as f:
            json.dump(fold_metadata, f, indent=2)


if __name__ == "__main__":
    main()
