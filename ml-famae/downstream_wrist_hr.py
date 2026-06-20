import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bioFAME.datasets.dataloader import bioFAME_data
from bioFAME.models.algorithms import get_algorithm_class
from bioFAME.models.hparams_registry import _hparams


TARGET_NAMES = ("hr_bpm", "sdnn_ms", "rmssd_ms")


class HRRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class HRCNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=1):
        super().__init__()
        self.project = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            ResidualConvBlock(hidden_dim, dropout=dropout),
            ResidualConvBlock(hidden_dim, dropout=dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, tokens):
        x = tokens.transpose(1, 2)
        x = self.project(x)
        x = self.blocks(x)
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.mlp(pooled)


class HRLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens):
        seq, _ = self.lstm(tokens)
        pooled = torch.cat([seq.mean(dim=1), seq.amax(dim=1)], dim=-1)
        return self.mlp(pooled)


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HRUNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2, output_dim=1):
        super().__init__()
        self.stem = UNetBlock(input_dim, hidden_dim, dropout=dropout)
        self.down1 = UNetBlock(hidden_dim, hidden_dim * 2, dropout=dropout)
        self.down2 = UNetBlock(hidden_dim * 2, hidden_dim * 4, dropout=dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.up1_proj = nn.Conv1d(hidden_dim * 4, hidden_dim * 2, kernel_size=1)
        self.up1 = UNetBlock(hidden_dim * 4, hidden_dim * 2, dropout=dropout)
        self.up2_proj = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.up2 = UNetBlock(hidden_dim * 2, hidden_dim, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _match_length(x, target):
        if x.shape[-1] == target.shape[-1]:
            return x
        return nn.functional.interpolate(x, size=target.shape[-1], mode="linear", align_corners=False)

    def forward(self, tokens):
        x = tokens.transpose(1, 2)
        skip1 = self.stem(x)
        skip2 = self.down1(self.pool(skip1))
        bottleneck = self.down2(self.pool(skip2))

        x = nn.functional.interpolate(bottleneck, scale_factor=2, mode="linear", align_corners=False)
        x = self._match_length(self.up1_proj(x), skip2)
        x = self.up1(torch.cat([x, skip2], dim=1))

        x = nn.functional.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self._match_length(self.up2_proj(x), skip1)
        x = self.up2(torch.cat([x, skip1], dim=1))

        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1)], dim=-1)
        return self.mlp(pooled)


def build_regressor(head, token_dim, pooled_dim, hidden_dim, dropout, output_dim):
    if head == "cnn":
        return HRCNNRegressor(token_dim, hidden_dim=hidden_dim, dropout=dropout, output_dim=output_dim)
    if head == "lstm":
        return HRLSTMRegressor(token_dim, hidden_dim=hidden_dim, dropout=dropout, output_dim=output_dim)
    if head == "unet":
        return HRUNetRegressor(token_dim, hidden_dim=hidden_dim, dropout=dropout, output_dim=output_dim)
    if head == "mlp":
        return HRRegressor(pooled_dim, hidden_dim=hidden_dim, dropout=dropout, output_dim=output_dim)
    raise ValueError(f"Unsupported downstream head: {head}")


def extract_token_features(model, x):
    if model.hparams["revin"]:
        x = x.permute(0, 2, 1)
        x = model.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)

    patch = model.to_patch_embedding[0](x)
    patch_emb = model.to_patch_embedding[1:](patch)
    tokens = model.encoder(patch_emb)
    batch = x.shape[0]
    tokens = tokens.reshape(batch, model.in_channel * model.num_patches, -1)

    if model.cls_token_mode == "cls":
        cls_tokens = model.cls_token.repeat(batch, 1, 1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

    if model.hparams["modality_embedding_token"]:
        tokens = tokens + model.mod_embedding[:, : tokens.shape[1]]

    if model.hparams["include_Enc2"]:
        tokens = model.masked_encoder(tokens)

    if model.cls_token_mode == "cls":
        tokens = tokens[:, 1:]

    return tokens


def load_compatible_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Loaded compatible checkpoint tensors: {len(compatible)}")
    print(f"Skipped incompatible tensors: {len(skipped)}")
    if missing:
        print(f"Model tensors left randomly initialized: {len(missing)}")
    if unexpected:
        print(f"Unexpected tensors: {len(unexpected)}")
    return {"loaded": sorted(compatible.keys()), "skipped": sorted(skipped), "missing": sorted(missing)}


def make_loader(data_dir, data_name, split, batch_size, num_workers, shuffle):
    dataset = bioFAME_data(
        data_dir,
        filename=f"{split}.pt",
        channels=["wrist"],
        transforms=None,
        dataset_name=data_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=(num_workers != 0),
    )


def encode_for_head(encoder, data, head):
    if head in {"cnn", "lstm", "unet"}:
        return extract_token_features(encoder, data)
    return encoder(data, latent_mode=True)


def evaluate(encoder, regressor, loader, device, label_mean, label_std, head):
    encoder.eval()
    regressor.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.float().to(device)
            features = encode_for_head(encoder, data, head)
            pred_z = regressor(features)
            pred = pred_z * label_std + label_mean
            preds.append(pred.cpu())
            targets.append(label.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    preds = preds.reshape_as(targets)
    err = preds - targets
    metrics = {
        "mae": float(err.abs().mean()),
        "rmse": float(torch.sqrt((err**2).mean())),
        "pred_mean": float(preds.mean()),
        "target_mean": float(targets.mean()),
    }
    for index, name in enumerate(TARGET_NAMES[: targets.shape[1]]):
        target_err = err[:, index]
        metrics[f"{name}_mae"] = float(target_err.abs().mean())
        metrics[f"{name}_rmse"] = float(torch.sqrt((target_err**2).mean()))
        metrics[f"{name}_pred_mean"] = float(preds[:, index].mean())
        metrics[f"{name}_target_mean"] = float(targets[:, index].mean())
    return metrics


def train(args):
    device = torch.device(args.device)
    train_loader = make_loader(args.data_dir, args.data_name, "train", args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(args.data_dir, args.data_name, "val", args.batch_size, args.num_workers, shuffle=False)
    test_loader = make_loader(args.data_dir, args.data_name, "test", args.batch_size, args.num_workers, shuffle=False)

    model_hparams = _hparams("bioFAME")
    model_hparams["encoder_only"] = True
    model_class = get_algorithm_class("bioFAME")
    encoder = model_class(in_channel=1, length=args.length, n_classes=args.output_dim, hparams=model_hparams).to(device)
    load_report = load_compatible_weights(encoder, args.checkpoint, device)

    if args.chase_style:
        args.freeze_encoder = False
        if args.loss == "mse":
            args.loss = "smooth_l1"
        if args.optimizer == "adam":
            args.optimizer = "adamw"

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    elif args.unfreeze_last_n > 0:
        for param in encoder.parameters():
            param.requires_grad = False
        for module_name in ("encoder", "masked_encoder"):
            module = getattr(encoder, module_name, None)
            layers = getattr(module, "layers", None)
            if layers is None:
                continue
            for layer in layers[-args.unfreeze_last_n :]:
                for param in layer.parameters():
                    param.requires_grad = True

    regressor = build_regressor(
        args.head,
        token_dim=model_hparams["dim"],
        pooled_dim=encoder.linear_clf_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        output_dim=args.output_dim,
    ).to(device)
    trainable_encoder_params = [param for param in encoder.parameters() if param.requires_grad]
    head_params = list(regressor.parameters())
    param_groups = [{"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if trainable_encoder_params:
        param_groups.insert(
            0,
            {
                "params": trainable_encoder_params,
                "lr": args.encoder_lr,
                "weight_decay": args.weight_decay,
            },
        )

    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(param_groups)
    if args.loss == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    train_labels = train_loader.dataset.label.float()
    label_mean = train_labels.mean().to(device)
    label_std = train_labels.std().clamp_min(1e-6).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        encoder.train(not args.freeze_encoder)
        regressor.train()
        losses = []

        for data, label in train_loader:
            data = data.to(device)
            label = label.float().to(device)
            label_z = (label - label_mean) / label_std

            features = encode_for_head(encoder, data, args.head)
            pred_z = regressor(features)
            loss = criterion(pred_z, label_z)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(regressor.parameters()) + trainable_encoder_params,
                    max_norm=args.grad_clip,
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(encoder, regressor, val_loader, device, label_mean, label_std, args.head)
        train_loss = sum(losses) / max(len(losses), 1)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.4f} "
            + " ".join(
                f"val_{name}_mae={val_metrics[f'{name}_mae']:.2f}"
                for name in TARGET_NAMES[: args.output_dim]
            )
        )

        if val_metrics["hr_bpm_mae"] < best_val_mae:
            best_val_mae = val_metrics["hr_bpm_mae"]
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "regressor": regressor.state_dict(),
                    "label_mean": label_mean.detach().cpu(),
                    "label_std": label_std.detach().cpu(),
                    "args": vars(args),
                    "load_report": load_report,
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    regressor.load_state_dict(checkpoint["regressor"])
    test_metrics = evaluate(encoder, regressor, test_loader, device, label_mean, label_std, args.head)

    report = {"history": history, "best_val_mae": best_val_mae, "test": test_metrics, "args": vars(args)}
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(
        " ".join(
            f"test_{name}_mae={test_metrics[f'{name}_mae']:.2f}"
            for name in TARGET_NAMES[: args.output_dim]
        )
    )
    print(f"saved -> {output_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train a downstream HR/HRV regressor on our wrist ECG.")
    parser.add_argument("--data_dir", default="./data_our_wrist_hr")
    parser.add_argument("--data_name", default="our_wrist_hr")
    parser.add_argument("--checkpoint", default="./model_ckpt/wecg-12lead-pretrain-50ep_bioFAME/ckpt.pt")
    parser.add_argument("--output_dir", default="./model_ckpt/our-wrist-hr-cnn-from-12lead-50ep")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head", choices=["cnn", "lstm", "unet", "mlp"], default="cnn")
    parser.add_argument("--loss", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--unfreeze_last_n", type=int, default=0)
    parser.add_argument("--cosine_lr", action="store_true")
    parser.add_argument("--chase_style", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
