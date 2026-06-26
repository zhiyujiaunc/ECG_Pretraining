#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


TASK = "task4b-wrist2lead-as-1ch-5000-to-our-wrist1ch"
HEADS = ("unet", "cnn", "lstm", "mlp")
METRICS = (
    ("HR", "hr_bpm_mae"),
    ("SDNN", "sdnn_ms_mae"),
    ("RMSSD", "rmssd_ms_mae"),
    ("pNN50", "pnn50_pct_mae"),
)


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.mean()), float(t.std(unbiased=False))


def main():
    parser = argparse.ArgumentParser(description="Summarize wrist-2lead-as-1ch pretrain to our wrist 5-fold downstream.")
    parser.add_argument("--result_root", default="ml-famae/model_ckpt/downstream-5000-wrist2lead-as-1ch-to-our-wrist-5fold")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.result_root)
    print("=" * 128)
    print("WRIST-2LEAD-AS-1CH PRETRAIN -> OUR WRIST 1CH, 5000-SAMPLE INTRA-SUBJECT 5-FOLD")
    print("Metrics are mean+-std across folds; lower MAE is better.")
    print("=" * 128)
    print(f"{'Task':<48} {'Head':<6} {'HR MAE':>15} {'SDNN MAE':>15} {'RMSSD MAE':>15} {'pNN50 MAE':>15}")
    print("-" * 128)
    for head in HEADS:
        fold_metrics = []
        missing = []
        for fold in range(args.n_folds):
            path = root / TASK / head / f"fold_{fold}" / "metrics.json"
            if not path.exists():
                missing.append(fold)
                continue
            with path.open() as f:
                fold_metrics.append(json.load(f)["test"])
        if missing or not fold_metrics:
            print(f"{TASK:<48} {head:<6} missing folds {missing}")
            continue
        cells = []
        for _label, key in METRICS:
            mean, std = mean_std([row[key] for row in fold_metrics])
            cells.append(f"{mean:.2f}+-{std:.2f}")
        print(f"{TASK:<48} {head:<6} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} {cells[3]:>15}")
    print("-" * 128)


if __name__ == "__main__":
    main()
