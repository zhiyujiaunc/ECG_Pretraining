#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Check 12-lead-as-1ch pretraining data and BioFAME token shape.")
    parser.add_argument("--data_dir", default="./data_wecg_12lead_as_1ch_5000")
    parser.add_argument("--data_name", default="wECGdb_12lead_as_1ch")
    parser.add_argument("--length", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(Path.cwd()))

    from bioFAME.datasets.dataloader import bioFAME_data
    from bioFAME.models.algorithms import get_algorithm_class
    from bioFAME.models.hparams_registry import _hparams

    data_dir = Path(args.data_dir)
    for split in ("train", "val", "test"):
        obj = torch.load(data_dir / f"{split}.pt", map_location="cpu")
        samples = obj["samples"]
        labels = obj["labels"]
        print(f"{split}: samples={tuple(samples.shape)} labels={tuple(labels.shape)}")
        if samples.ndim != 3 or samples.shape[1] != 1 or samples.shape[2] != args.length:
            raise SystemExit(f"{split} should be [N, 1, {args.length}], got {tuple(samples.shape)}")

    dataset = bioFAME_data(
        str(data_dir),
        filename="train.pt",
        channels=["ecg"],
        transforms=None,
        dataset_name=args.data_name,
    )
    batch, label = next(iter(DataLoader(dataset, batch_size=args.batch_size)))
    print(f"dataloader batch: data={tuple(batch.shape)} label={tuple(label.shape)}")

    hparams = _hparams("bioFAME")
    hparams["encoder_only"] = True
    model = get_algorithm_class("bioFAME")(in_channel=1, length=args.length, n_classes=1, hparams=hparams)
    model.eval()
    with torch.no_grad():
        patch = model.to_patch_embedding[0](batch)
        patch_emb = model.to_patch_embedding[1:](patch)
        tokens = model.encoder(patch_emb)
        tokens = tokens.reshape(batch.shape[0], model.in_channel * model.num_patches, -1)
    print(f"BioFAME token shape before downstream: {tuple(tokens.shape)}")
    expected = (batch.shape[0], args.length // hparams["patch_size"], hparams["dim"])
    if tuple(tokens.shape) != expected:
        raise SystemExit(f"Expected token shape {expected}, got {tuple(tokens.shape)}")
    print("OK: channel-independent pretraining data is [N*12, 1, L], encoder tokens are [B, 250, 64].")


if __name__ == "__main__":
    main()
