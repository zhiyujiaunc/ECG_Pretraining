import argparse
from pathlib import Path

import torch

from report_wrist_hrv import (
    activity_from_file,
    load_model,
    metrics,
    predict,
    split_metadata,
)


def summarize(rows):
    keys = ("hr_mae", "hr_rmse", "l1_lt_2", "l1_lt_5", "l1_lt_10pct", "rmssd", "sdnn")
    if any("pnn50" in row for row in rows):
        keys = (*keys, "pnn50")
    summary = {}
    for key in keys:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(unbiased=False))
    summary["n"] = int(round(sum(row["n"] for row in rows) / len(rows)))
    return summary


def print_summary(title, summary):
    has_pnn50 = any("pnn50_mean" in row for row in summary.values())
    print("=" * 110)
    print(title)
    print("Metrics are fold-wise mean+-std across 5 folds")
    print("HR: L1 regression metrics | HRV: within 5% of pooled GT range")
    print("=" * 110)
    pnn50_header = f"{'pNN50':>10}" if has_pnn50 else ""
    print(f"{'Dataset':<16}{'HR MAE+STD':>16}{'RMSE':>10}{'L1<2':>10}{'L1<5':>10}{'L1<10%r':>12}{'RMSSD':>10}{'SDNN':>10}{pnn50_header}{'N':>8}")
    print("-" * 110)
    for name in ("Gesture", "Static", "Free Form"):
        if name not in summary:
            continue
        row = summary[name]
        pnn50_value = f"{row['pnn50_mean']:>9.2f}%" if has_pnn50 else ""
        print(
            f"{name:<16}"
            f"{row['hr_mae_mean']:>8.2f}+{row['hr_mae_std']:<6.2f}"
            f"{row['hr_rmse_mean']:>10.2f}"
            f"{row['l1_lt_2_mean']:>9.2f}%"
            f"{row['l1_lt_5_mean']:>9.2f}%"
            f"{row['l1_lt_10pct_mean']:>11.2f}%"
            f"{row['rmssd_mean']:>9.2f}%"
            f"{row['sdnn_mean']:>9.2f}%"
            f"{pnn50_value}"
            f"{row['n']:>8d}"
        )
    print("-" * 110)


def evaluate_fold(fold_dir, checkpoint, device, batch_size):
    import json

    with (fold_dir / "metadata.json").open() as f:
        metadata = json.load(f)
    data = torch.load(fold_dir / "test.pt")
    rows = split_metadata(metadata, "test")
    if len(rows) != len(data["samples"]):
        raise ValueError(f"Metadata/sample mismatch for {fold_dir}: {len(rows)} vs {len(data['samples'])}")

    encoder, regressor, label_mean, label_std, head, _description = load_model(checkpoint, device)
    preds = predict(encoder, regressor, data["samples"], label_mean, label_std, head, device, batch_size)
    targets = data["labels"].float()

    group_preds = {}
    group_targets = {}
    for index, row in enumerate(rows):
        activity = activity_from_file(row["file"])
        group_preds.setdefault(activity, []).append(preds[index])
        group_targets.setdefault(activity, []).append(targets[index])

    group_metrics = {}
    for activity in group_preds:
        group_metrics[activity] = metrics(torch.stack(group_preds[activity]), torch.stack(group_targets[activity]))
    return group_metrics


def main():
    parser = argparse.ArgumentParser(description="Summarize 5-fold wrist HR/HRV regression metrics.")
    parser.add_argument("--fold_root", default="./data_our_wrist_hrv_3000_5fold")
    parser.add_argument("--model_root", default="./model_ckpt/our-wrist-hrv-3000-5fold")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    all_metrics = {}
    for fold_idx in range(args.n_folds):
        fold_dir = Path(args.fold_root) / f"fold_{fold_idx}"
        checkpoint = Path(args.model_root) / f"fold_{fold_idx}" / "best.pt"
        fold_metrics = evaluate_fold(fold_dir, checkpoint, device, args.batch_size)
        for activity, row in fold_metrics.items():
            all_metrics.setdefault(activity, []).append(row)

    summary = {activity: summarize(rows) for activity, rows in all_metrics.items()}
    print_summary("FINAL RESULTS - 5-FOLD WRIST ECG HR/HRV REGRESSION", summary)


if __name__ == "__main__":
    main()
