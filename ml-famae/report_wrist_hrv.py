import argparse
import json
import warnings
from pathlib import Path

import torch

from bioFAME.models.algorithms import get_algorithm_class
from bioFAME.models.hparams_registry import _hparams
from downstream_wrist_hr import HRCNNRegressor, HRRegressor, TARGET_NAMES, encode_for_head


warnings.filterwarnings("ignore", message="An output with one or more elements was resized.*")

ACTIVITY_LABELS = {
    "gesture": "Gesture",
    "static": "Static",
    "freeform": "Free Form",
}


def activity_from_file(path):
    parts = Path(path).parts
    for key, label in ACTIVITY_LABELS.items():
        if key in parts:
            return label
    return "Unknown"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})
    length = int(args.get("length", 3000))
    output_dim = int(args.get("output_dim", 3))
    head = args.get("head", "cnn")
    hidden_dim = int(args.get("hidden_dim", 64))
    dropout = float(args.get("dropout", 0.2))

    hparams = _hparams("bioFAME")
    hparams["encoder_only"] = True
    encoder = get_algorithm_class("bioFAME")(in_channel=1, length=length, n_classes=output_dim, hparams=hparams).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    if head == "cnn":
        regressor = HRCNNRegressor(hparams["dim"], hidden_dim=hidden_dim, dropout=dropout, output_dim=output_dim).to(device)
    else:
        regressor = HRRegressor(encoder.linear_clf_dim, hidden_dim=hidden_dim, dropout=dropout, output_dim=output_dim).to(device)
    regressor.load_state_dict(checkpoint["regressor"])
    regressor.eval()

    return encoder, regressor, checkpoint["label_mean"].to(device), checkpoint["label_std"].to(device), head


def predict(encoder, regressor, samples, label_mean, label_std, head, device, batch_size):
    preds = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size].float().to(device)
            features = encode_for_head(encoder, batch, head)
            pred_z = regressor(features)
            preds.append((pred_z * label_std + label_mean).cpu())
    return torch.cat(preds)


def split_metadata(metadata, split):
    return [row for row in metadata["windows"] if row["split"] == split]


def metrics(pred, target):
    err = pred - target
    hr_err = err[:, 0]
    hr_target = target[:, 0].clamp_min(1e-6)
    pooled_ranges = target.max(dim=0).values - target.min(dim=0).values
    sdnn_tol = 0.05 * pooled_ranges[1].clamp_min(1e-6)
    rmssd_tol = 0.05 * pooled_ranges[2].clamp_min(1e-6)
    return {
        "hr_mae": float(hr_err.abs().mean()),
        "hr_std": float(hr_err.abs().std(unbiased=False)),
        "hr_rmse": float(torch.sqrt((hr_err**2).mean())),
        "l1_lt_2": float((hr_err.abs() < 2.0).float().mean() * 100.0),
        "l1_lt_5": float((hr_err.abs() < 5.0).float().mean() * 100.0),
        "l1_lt_10pct": float((hr_err.abs() < 0.10 * hr_target).float().mean() * 100.0),
        "rmssd": float((err[:, 2].abs() < rmssd_tol).float().mean() * 100.0),
        "sdnn": float((err[:, 1].abs() < sdnn_tol).float().mean() * 100.0),
        "n": int(target.shape[0]),
    }


def print_table(title, group_metrics):
    print("=" * 110)
    print(title)
    print("HR: L1 regression metrics | HRV: within 5% of pooled GT range")
    print("=" * 110)
    print(f"{'Dataset':<16}{'HR MAE+STD':>16}{'RMSE':>10}{'L1<2':>10}{'L1<5':>10}{'L1<10%r':>12}{'RMSSD':>10}{'SDNN':>10}{'N':>8}")
    print("-" * 110)
    for name in ("Gesture", "Static", "Free Form"):
        if name not in group_metrics:
            continue
        row = group_metrics[name]
        print(
            f"{name:<16}"
            f"{row['hr_mae']:>8.2f}+{row['hr_std']:<6.2f}"
            f"{row['hr_rmse']:>10.2f}"
            f"{row['l1_lt_2']:>9.2f}%"
            f"{row['l1_lt_5']:>9.2f}%"
            f"{row['l1_lt_10pct']:>11.2f}%"
            f"{row['rmssd']:>9.2f}%"
            f"{row['sdnn']:>9.2f}%"
            f"{row['n']:>8d}"
        )
    print("-" * 110)


def main():
    parser = argparse.ArgumentParser(description="Report wrist ECG HR/HRV regression metrics by activity.")
    parser.add_argument("--data_dir", default="./data_our_wrist_hrv_3000")
    parser.add_argument("--checkpoint", default="./model_ckpt/our-wrist-hrv-3000-cnn-run2-from-12lead-50ep/best.pt")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    with (data_dir / "metadata.json").open() as f:
        metadata = json.load(f)

    encoder, regressor, label_mean, label_std, head = load_model(args.checkpoint, device)

    group_preds = {}
    group_targets = {}
    for split in args.splits:
        data = torch.load(data_dir / f"{split}.pt")
        samples = data["samples"]
        targets = data["labels"].float()
        rows = split_metadata(metadata, split)
        if len(rows) != len(samples):
            raise ValueError(f"Metadata/sample length mismatch for {split}: {len(rows)} vs {len(samples)}")

        preds = predict(encoder, regressor, samples, label_mean, label_std, head, device, args.batch_size)
        for index, row in enumerate(rows):
            activity = activity_from_file(row["file"])
            group_preds.setdefault(activity, []).append(preds[index])
            group_targets.setdefault(activity, []).append(targets[index])

    group_metrics = {}
    for activity in group_preds:
        pred = torch.stack(group_preds[activity])
        target = torch.stack(group_targets[activity])
        group_metrics[activity] = metrics(pred, target)

    split_text = ", ".join(args.splits)
    print_table(f"RESULTS - WRIST ECG HR/HRV REGRESSION ({split_text})", group_metrics)


if __name__ == "__main__":
    main()
