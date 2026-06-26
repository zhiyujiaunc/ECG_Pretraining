#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


LEADS = ("wrist_I", "wrist_aux")


def flatten_split(input_path, output_path):
    data = torch.load(input_path, map_location="cpu")
    samples = data["samples"]
    if samples.ndim != 3:
        raise ValueError(f"Expected samples [N, C, L], got {tuple(samples.shape)}")
    n_samples, n_channels, length = samples.shape
    if n_channels != len(LEADS):
        raise ValueError(f"Expected 2 channels, got {n_channels}")

    # Channel independence: fold channel into the sample axis.
    # [N, 2, L] -> [N*2, 1, L]. This does not concatenate leads along time.
    flattened = samples.contiguous().reshape(n_samples * n_channels, 1, length)
    labels = torch.zeros(flattened.shape[0], dtype=torch.long)
    torch.save({"samples": flattened, "labels": labels}, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_shape": list(samples.shape),
        "output_shape": list(flattened.shape),
        "operation": "[N, 2, L] -> [N*2, 1, L]",
        "channel_folded_into_batch": True,
        "time_concatenation": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Flatten wECGdb wrist 2-lead .pt data into one-channel ECG samples.")
    parser.add_argument("--input_dir", default="ml-famae/data_wecg_wrist_2lead_5000")
    parser.add_argument("--output_dir", default="ml-famae/data_wecg_wrist_2lead_as_1ch_5000")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(input_dir),
        "description": "Each original 2-lead wrist ECG window is flattened into 2 independent one-channel ECG windows.",
        "channel_independence_definition": "Fold channel into batch/sample dimension: [N, 2, L] -> [N*2, 1, L]. The time dimension stays length L.",
        "leads": LEADS,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        metadata["splits"][split] = flatten_split(input_dir / f"{split}.pt", output_dir / f"{split}.pt")
        info = metadata["splits"][split]
        print(f"{split}: {tuple(info['input_shape'])} -> {tuple(info['output_shape'])} -> {info['output']}")

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
