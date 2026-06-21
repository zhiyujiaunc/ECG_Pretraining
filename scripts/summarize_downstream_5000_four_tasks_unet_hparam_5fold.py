#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


TASKS = (
    "task1-12lead5000-to-our-wrist1ch",
    "task2-12lead5000-to-wecgdb-wrist2ch-lead2gt",
    "task3-v2v6-5000-to-our-wrist1ch",
    "task4-wrist2lead-5000-to-our-wrist1ch",
)
CONFIGS = ("baseline", "lr1e-3", "lr1e-4", "hidden256", "dropout03", "unfreeze1")
METRICS = (
    ("HR", "hr_bpm_mae"),
    ("SDNN", "sdnn_ms_mae"),
    ("RMSSD", "rmssd_ms_mae"),
    ("pNN50", "pnn50_pct_mae"),
)


def mean_std(values):
    t = torch.tensor(values, dtype=torch.float32)
    return float(t.mean()), float(t.std(unbiased=False))


def format_cell(values):
    mean, std = mean_std(values)
    return f"{mean:.2f}+-{std:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize UNet hyperparameter sweep for 5000-sample 4-task 5-fold downstream.")
    parser.add_argument("--result_root", default="ml-famae/model_ckpt/downstream-5000-four-tasks-unet-hparam-5fold")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = parser.parse_args()

    root = Path(args.result_root)
    print("=" * 150)
    print("DOWNSTREAM 5000 FOUR-TASK UNET HYPERPARAMETER SWEEP, INTRA-SUBJECT 5-FOLD")
    print("Metrics are mean+-std across folds; lower MAE is better.")
    print("=" * 150)
    print(f"{'Task':<48} {'Config':<12} {'HR MAE':>15} {'SDNN MAE':>15} {'RMSSD MAE':>15} {'pNN50 MAE':>15}")
    print("-" * 150)

    best_by_task = {}
    for task in TASKS:
        best_hr = None
        best_row = None
        for config in args.configs:
            fold_metrics = []
            missing = []
            for fold in range(args.n_folds):
                path = root / task / config / "unet" / f"fold_{fold}" / "metrics.json"
                if not path.exists():
                    missing.append(fold)
                    continue
                with path.open() as f:
                    fold_metrics.append(json.load(f)["test"])
            if missing or not fold_metrics:
                print(f"{task:<48} {config:<12} missing folds {missing}")
                continue

            metric_values = {key: [row[key] for row in fold_metrics] for _label, key in METRICS}
            cells = [format_cell(metric_values[key]) for _label, key in METRICS]
            hr_mean, _hr_std = mean_std(metric_values["hr_bpm_mae"])
            print(f"{task:<48} {config:<12} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} {cells[3]:>15}")
            if best_hr is None or hr_mean < best_hr:
                best_hr = hr_mean
                best_row = (config, cells)
        if best_row:
            best_by_task[task] = best_row
        print("-" * 150)

    print("BEST CONFIG BY TASK, selected by lowest mean HR MAE")
    print("-" * 150)
    for task, (config, cells) in best_by_task.items():
        print(f"{task:<48} {config:<12} {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} {cells[3]:>15}")


if __name__ == "__main__":
    main()
