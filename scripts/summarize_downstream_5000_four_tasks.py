#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


TASKS = (
    "task1-12lead5000-to-our-wrist1ch",
    "task2-12lead5000-to-wecgdb-wrist2ch",
    "task3-v2v6-5000-to-our-wrist1ch",
    "task4-wrist2lead-5000-to-our-wrist1ch",
)
HEADS = ("cnn", "lstm", "unet", "mlp")


def main():
    parser = argparse.ArgumentParser(description="Summarize 5000-sample downstream four-task sweep.")
    parser.add_argument("--result_root", default="ml-famae/model_ckpt/downstream-5000-four-tasks")
    args = parser.parse_args()

    root = Path(args.result_root)
    print("=" * 116)
    print("DOWNSTREAM 5000 FOUR-TASK SUMMARY")
    print("=" * 116)
    print(f"{'Task':<42} {'Head':<6} {'HR MAE':>8} {'SDNN MAE':>10} {'RMSSD MAE':>11} {'pNN50 MAE':>11}")
    print("-" * 116)
    for task in TASKS:
        for head in HEADS:
            path = root / task / head / "metrics.json"
            if not path.exists():
                print(f"{task:<42} {head:<6} {'missing':>8}")
                continue
            with path.open() as f:
                metrics = json.load(f)["test"]
            print(
                f"{task:<42} {head:<6} "
                f"{metrics.get('hr_bpm_mae', float('nan')):>8.2f} "
                f"{metrics.get('sdnn_ms_mae', float('nan')):>10.2f} "
                f"{metrics.get('rmssd_ms_mae', float('nan')):>11.2f} "
                f"{metrics.get('pnn50_pct_mae', float('nan')):>11.2f}"
            )
        print("-" * 116)


if __name__ == "__main__":
    main()
