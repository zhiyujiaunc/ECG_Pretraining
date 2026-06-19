import argparse
import io
import os
import random
import zipfile

import numpy as np
import scipy.io as sio
import torch


SITES = ("LA_A", "LA_V3", "LA_V5")
SPLITS = ("training", "testing")
REFERENCE_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
WECG_LEADS = ("wrist_I", "wrist_aux")


def _as_bool_string(value):
    if isinstance(value, str):
        return value.lower()
    return str(value).lower()


def _load_mat_from_zip(zip_file, name):
    return sio.loadmat(
        io.BytesIO(zip_file.read(name)),
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["info", *SITES],
    )


def _select_channels(data, source, channel_indices, channel_names):
    if channel_names:
        names = REFERENCE_LEADS if source == "reference_12_lead" else WECG_LEADS
        name_to_index = {name: index for index, name in enumerate(names)}
        try:
            channel_indices = [name_to_index[name] for name in channel_names]
        except KeyError as exc:
            raise ValueError(f"Unknown channel name for {source}: {exc.args[0]}") from exc

    if channel_indices:
        data = data[:, channel_indices]

    return data


def _iter_recordings(mat_dict, source, channel_indices=None, channel_names=None):
    fs = int(mat_dict["info"].Fs)
    for site in SITES:
        site_obj = mat_dict[site]
        for split in SPLITS:
            split_obj = getattr(site_obj, split)
            if _as_bool_string(split_obj.data_availability) != "yes":
                continue

            if source == "reference_12_lead":
                data = np.asarray(split_obj.reference_12_lead, dtype=np.float32)
            elif source == "wecg":
                data = np.asarray(split_obj.wECG, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported source: {source}")

            if data.ndim != 2:
                raise ValueError(f"Expected 2D recording, got shape {data.shape}")

            data = _select_channels(data, source, channel_indices, channel_names)
            yield fs, site, split, data


def _windows(recording, window_size, stride, zscore):
    # Dataset stores recordings as [time, channel]; bioFAME expects [channel, time].
    for start in range(0, recording.shape[0] - window_size + 1, stride):
        segment = recording[start : start + window_size].T
        if zscore:
            mean = segment.mean(axis=1, keepdims=True)
            std = segment.std(axis=1, keepdims=True)
            segment = (segment - mean) / np.maximum(std, 1e-6)
        yield segment.astype(np.float32, copy=False)


def _split_subjects(files, val_ratio, test_ratio, seed):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_total = len(files)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    test_files = set(files[:n_test])
    val_files = set(files[n_test : n_test + n_val])
    train_files = set(files[n_test + n_val :])
    return {"train": train_files, "val": val_files, "test": test_files}


def convert(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as zip_file:
        mat_files = sorted(name for name in zip_file.namelist() if name.endswith(".mat"))
        file_splits = _split_subjects(mat_files, args.val_ratio, args.test_ratio, args.seed)

        split_samples = {"train": [], "val": [], "test": []}
        split_labels = {"train": [], "val": [], "test": []}
        fs_values = set()

        for split_name, split_files in file_splits.items():
            for subject_index, mat_name in enumerate(sorted(split_files)):
                mat_dict = _load_mat_from_zip(zip_file, mat_name)
                for fs, _site, _recording_split, recording in _iter_recordings(
                    mat_dict,
                    args.source,
                    channel_indices=args.channel_indices,
                    channel_names=args.channel_names,
                ):
                    fs_values.add(fs)
                    for segment in _windows(recording, args.window_size, args.stride, args.zscore):
                        split_samples[split_name].append(torch.from_numpy(segment))
                        split_labels[split_name].append(torch.tensor(0, dtype=torch.long))

                if args.max_subjects and subject_index + 1 >= args.max_subjects:
                    break

        for split_name in ("train", "val", "test"):
            if split_samples[split_name]:
                samples = torch.stack(split_samples[split_name], dim=0)
                labels = torch.stack(split_labels[split_name], dim=0)
            else:
                if args.channel_indices:
                    n_channels = len(args.channel_indices)
                elif args.channel_names:
                    n_channels = len(args.channel_names)
                else:
                    n_channels = 12 if args.source == "reference_12_lead" else 2
                samples = torch.empty(0, n_channels, args.window_size)
                labels = torch.empty(0, dtype=torch.long)

            save_path = os.path.join(args.output_dir, f"{split_name}.pt")
            torch.save({"samples": samples, "labels": labels}, save_path)
            print(f"{split_name}: samples={tuple(samples.shape)} labels={tuple(labels.shape)} -> {save_path}")

    print(f"Sampling rates found: {sorted(fs_values)}")


def main():
    parser = argparse.ArgumentParser(description="Convert wECGdb MAT files to bioFAME .pt files.")
    parser.add_argument("--zip_path", default="../wECG_dataset.zip")
    parser.add_argument("--output_dir", default="./data_wecg_12lead")
    parser.add_argument("--source", choices=["reference_12_lead", "wecg"], default="reference_12_lead")
    parser.add_argument("--channel_indices", nargs="+", type=int, default=None)
    parser.add_argument("--channel_names", nargs="+", default=None)
    parser.add_argument("--window_size", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=1500)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--zscore", action="store_true")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
