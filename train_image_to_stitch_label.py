from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset


class StitchLabelDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str):
        self.dataset_dir = dataset_dir
        manifest_path = dataset_dir / "manifest.csv"
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows found for split={split} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        image = Image.open(self.dataset_dir / row["input_png"]).convert("RGB")
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1))

        mask = Image.open(self.dataset_dir / row["mask_png"]).convert("L")
        mask_arr = (np.asarray(mask, dtype=np.float32) / 255.0)[None, :, :]
        density_arr = np.load(self.dataset_dir / row["density_npy"]).astype(np.float32)[None, :, :]
        direction_arr = np.load(self.dataset_dir / row["direction_npy"]).astype(np.float32)
        target = np.concatenate([mask_arr, density_arr, direction_arr], axis=0)
        return image_tensor, torch.from_numpy(target), row["pair_id"]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 4, base_channels: int = 24):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        d3 = self.up3(z)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        device = torch.device("cuda")
        _ = torch.ones(1, device=device) + 1
        torch.cuda.synchronize()
        return device
    except Exception as exc:
        print(f"warning: CUDA visible but unusable, falling back to CPU: {exc}")
        return torch.device("cpu")


def batch_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask_pos_weight: float,
    density_loss_weight: float,
    direction_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_mask = target[:, 0:1]
    target_density = target[:, 1:2]
    target_direction = target[:, 2:4]

    mask_logits = output[:, 0:1]
    density_pred = torch.sigmoid(output[:, 1:2])
    direction_pred = torch.tanh(output[:, 2:4])

    pos_weight = torch.tensor([mask_pos_weight], device=output.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask, pos_weight=pos_weight)
    density_loss = F.l1_loss(density_pred, target_density)
    direction_weight = target_mask.expand_as(target_direction)
    denom = direction_weight.sum().clamp_min(1.0)
    direction_loss = (((direction_pred - target_direction) ** 2) * direction_weight).sum() / denom
    total = mask_loss + density_loss_weight * density_loss + direction_loss_weight * direction_loss
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "density_loss": float(density_loss.detach().cpu()),
        "direction_loss": float(direction_loss.detach().cpu()),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_density_mae = 0.0
    total_direction_mae = 0.0
    count = 0
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss, _ = batch_loss(
                output,
                target,
                mask_pos_weight=args.mask_pos_weight,
                density_loss_weight=args.density_loss_weight,
                direction_loss_weight=args.direction_loss_weight,
            )
            mask_pred = torch.sigmoid(output[:, 0:1]) > args.mask_threshold
            mask_true = target[:, 0:1] > 0.5
            intersection = (mask_pred & mask_true).sum(dim=(1, 2, 3)).float()
            union = (mask_pred | mask_true).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            density_mae = (torch.sigmoid(output[:, 1:2]) - target[:, 1:2]).abs().mean(dim=(1, 2, 3))
            direction_weight = target[:, 0:1].expand_as(target[:, 2:4])
            direction_denom = direction_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
            direction_mae = (
                (torch.tanh(output[:, 2:4]) - target[:, 2:4]).abs() * direction_weight
            ).sum(dim=(1, 2, 3)) / direction_denom
            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_iou += float((intersection / union).sum().detach().cpu())
            total_density_mae += float(density_mae.sum().detach().cpu())
            total_direction_mae += float(direction_mae.sum().detach().cpu())
            count += batch_size
    return {
        "loss": total_loss / max(1, count),
        "mask_iou": total_iou / max(1, count),
        "density_mae": total_density_mae / max(1, count),
        "direction_mae_on_mask": total_direction_mae / max(1, count),
    }


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def gray_to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def save_prediction_previews(
    model: nn.Module,
    dataset: StitchLabelDataset,
    device: torch.device,
    output_dir: Path,
    max_items: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index in range(min(max_items, len(dataset))):
        image, target, pair_id = dataset[index]
        with torch.no_grad():
            output = model(image[None].to(device))[0].detach().cpu()
        input_image = tensor_to_image(image)
        true_mask = gray_to_image(target[0].numpy())
        pred_mask = gray_to_image(torch.sigmoid(output[0]).numpy())
        true_density = gray_to_image(target[1].numpy())
        pred_density = gray_to_image(torch.sigmoid(output[1]).numpy())
        width, height = input_image.size
        panel = Image.new("RGB", (width * 5, height + 22), (255, 255, 255))
        labels = ["input", "target mask", "pred mask", "target density", "pred density"]
        for i, item in enumerate([input_image, true_mask, pred_mask, true_density, pred_density]):
            panel.paste(item, (i * width, 22))
        draw = ImageDraw.Draw(panel)
        for i, label in enumerate(labels):
            draw.text((i * width + 6, 5), label, fill=(0, 0, 0))
        panel.save(output_dir / f"{pair_id}_prediction.png")


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = StitchLabelDataset(dataset_dir, split="train")
    val_dataset = StitchLabelDataset(dataset_dir, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = choose_device(args.cpu)
    model = TinyUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} start", flush=True)
        model.train()
        total_loss = 0.0
        total_parts = {"mask_loss": 0.0, "density_loss": 0.0, "direction_loss": 0.0}
        seen = 0
        for batch_index, (image, target, _) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss, parts = batch_loss(
                output,
                target,
                mask_pos_weight=args.mask_pos_weight,
                density_loss_weight=args.density_loss_weight,
                direction_loss_weight=args.direction_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                total_parts[key] += value * batch_size
            seen += batch_size
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "seen": seen,
                            "train_loss_so_far": total_loss / max(1, seen),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, args)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "train_mask_loss": total_parts["mask_loss"] / max(1, seen),
            "train_density_loss": total_parts["density_loss"] / max(1, seen),
            "train_direction_loss": total_parts["direction_loss"] / max(1, seen),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(json.dumps(record, ensure_ascii=False), flush=True)

    checkpoint_path = output_dir / "image_to_stitch_label_unet.pt"
    best_checkpoint_path = output_dir / "best_image_to_stitch_label_unet.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "TinyUNet",
            "base_channels": args.base_channels,
            "target_channels": ["mask", "density", "direction_x", "direction_y"],
            "dataset_dir": str(dataset_dir),
            "args": vars(args),
            "history": history,
        },
        checkpoint_path,
    )
    if best_state is not None:
        torch.save(
            {
                "model_state": best_state,
                "model": "TinyUNet",
                "base_channels": args.base_channels,
                "target_channels": ["mask", "density", "direction_x", "direction_y"],
                "dataset_dir": str(dataset_dir),
                "args": vars(args),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            best_checkpoint_path,
        )
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    save_prediction_previews(model, val_dataset, device, output_dir / "predictions", max_items=args.preview_count)
    summary = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "note": "This is a supervised image-to-stitch-label baseline. Quality depends on the paired supervision dataset used.",
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {checkpoint_path}")
    print(f"saved metrics: {metrics_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image-to-stitch-label U-Net baseline.")
    parser.add_argument("--dataset-dir", default="datasets/ml_stitch_supervision")
    parser.add_argument("--output-dir", default="models/image_to_stitch_label_unet")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-pos-weight", type=float, default=5.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.75)
    parser.add_argument("--direction-loss-weight", type=float, default=0.2)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
