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

from train_image_to_stitch_label import TinyUNet, choose_device


TARGET_CHANNELS = [
    "mask",
    "density",
    "axis_x",
    "axis_y",
    "direction_confidence",
    "boundary",
    "centerline",
]


class RichStitchDataset(Dataset):
    def __init__(self, dataset_dir: Path, split_field: str, split: str):
        self.dataset_dir = dataset_dir
        self.split_field = split_field
        manifest_path = dataset_dir / "manifest_rich.csv"
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row[split_field] == split]
        if not self.rows:
            raise ValueError(f"no rows found for {split_field}={split} in {manifest_path}")

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
        axis_arr = np.load(self.dataset_dir / row["direction_axis_npy"]).astype(np.float32)
        conf_arr = np.load(self.dataset_dir / row["direction_confidence_npy"]).astype(np.float32)[None, :, :]
        boundary = Image.open(self.dataset_dir / row["boundary_png"]).convert("L")
        boundary_arr = (np.asarray(boundary, dtype=np.float32) / 255.0)[None, :, :]
        centerline = Image.open(self.dataset_dir / row["centerline_png"]).convert("L")
        centerline_arr = (np.asarray(centerline, dtype=np.float32) / 255.0)[None, :, :]

        target = np.concatenate(
            [mask_arr, density_arr, axis_arr, conf_arr, boundary_arr, centerline_arr],
            axis=0,
        )
        return image_tensor, torch.from_numpy(target), row["pair_id"]


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    intersection = (pred_bool & target_bool).sum(dim=(1, 2, 3)).float()
    union = (pred_bool | target_bool).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
    return intersection / union


def binary_f1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bool = pred > 0.5
    target_bool = target > 0.5
    tp = (pred_bool & target_bool).sum(dim=(1, 2, 3)).float()
    fp = (pred_bool & ~target_bool).sum(dim=(1, 2, 3)).float()
    fn = (~pred_bool & target_bool).sum(dim=(1, 2, 3)).float()
    return (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)


def batch_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_mask = target[:, 0:1]
    target_density = target[:, 1:2]
    target_axis = target[:, 2:4]
    target_conf = target[:, 4:5]
    target_boundary = target[:, 5:6]
    target_centerline = target[:, 6:7]

    mask_logits = output[:, 0:1]
    density_pred = torch.sigmoid(output[:, 1:2])
    axis_pred = torch.tanh(output[:, 2:4])
    conf_pred = torch.sigmoid(output[:, 4:5])
    boundary_logits = output[:, 5:6]
    centerline_logits = output[:, 6:7]

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits,
        target_mask,
        pos_weight=torch.tensor([args.mask_pos_weight], device=output.device),
    )
    density_loss = F.l1_loss(density_pred, target_density)

    axis_weight = target_conf.expand_as(target_axis)
    axis_denom = axis_weight.sum().clamp_min(1.0)
    axis_loss = (((axis_pred - target_axis) ** 2) * axis_weight).sum() / axis_denom
    confidence_loss = F.l1_loss(conf_pred, target_conf)

    boundary_loss = F.binary_cross_entropy_with_logits(
        boundary_logits,
        target_boundary,
        pos_weight=torch.tensor([args.boundary_pos_weight], device=output.device),
    )
    centerline_loss = F.binary_cross_entropy_with_logits(
        centerline_logits,
        target_centerline,
        pos_weight=torch.tensor([args.centerline_pos_weight], device=output.device),
    )

    total = (
        mask_loss
        + args.density_loss_weight * density_loss
        + args.axis_loss_weight * axis_loss
        + args.confidence_loss_weight * confidence_loss
        + args.boundary_loss_weight * boundary_loss
        + args.centerline_loss_weight * centerline_loss
    )
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "density_loss": float(density_loss.detach().cpu()),
        "axis_loss": float(axis_loss.detach().cpu()),
        "confidence_loss": float(confidence_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "centerline_loss": float(centerline_loss.detach().cpu()),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mask_iou": 0.0,
        "density_mae": 0.0,
        "axis_mae_on_confident": 0.0,
        "confidence_mae": 0.0,
        "boundary_iou": 0.0,
        "boundary_f1": 0.0,
        "centerline_iou": 0.0,
        "centerline_f1": 0.0,
    }
    count = 0
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss, _ = batch_loss(output, target, args)

            mask_pred = torch.sigmoid(output[:, 0:1])
            density_pred = torch.sigmoid(output[:, 1:2])
            axis_pred = torch.tanh(output[:, 2:4])
            conf_pred = torch.sigmoid(output[:, 4:5])
            boundary_pred = torch.sigmoid(output[:, 5:6])
            centerline_pred = torch.sigmoid(output[:, 6:7])

            target_conf = target[:, 4:5]
            confident = (target_conf > args.axis_conf_threshold).expand_as(axis_pred)
            axis_denom = confident.sum(dim=(1, 2, 3)).clamp_min(1.0)
            axis_mae = ((axis_pred - target[:, 2:4]).abs() * confident).sum(dim=(1, 2, 3)) / axis_denom

            batch_size = image.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["mask_iou"] += float(binary_iou(mask_pred, target[:, 0:1]).sum().detach().cpu())
            totals["density_mae"] += float((density_pred - target[:, 1:2]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["axis_mae_on_confident"] += float(axis_mae.sum().detach().cpu())
            totals["confidence_mae"] += float((conf_pred - target_conf).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["boundary_iou"] += float(binary_iou(boundary_pred, target[:, 5:6]).sum().detach().cpu())
            totals["boundary_f1"] += float(binary_f1(boundary_pred, target[:, 5:6]).sum().detach().cpu())
            totals["centerline_iou"] += float(binary_iou(centerline_pred, target[:, 6:7]).sum().detach().cpu())
            totals["centerline_f1"] += float(binary_f1(centerline_pred, target[:, 6:7]).sum().detach().cpu())
            count += batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def gray(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def axis_preview(axis: np.ndarray, confidence: np.ndarray) -> Image.Image:
    rgb = np.zeros((axis.shape[1], axis.shape[2], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((axis[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((axis[1] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_prediction_previews(
    model: nn.Module,
    dataset: RichStitchDataset,
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

        input_img = tensor_to_image(image)
        pred_mask = gray(torch.sigmoid(output[0]).numpy())
        true_mask = gray(target[0].numpy())
        pred_density = gray(torch.sigmoid(output[1]).numpy())
        true_density = gray(target[1].numpy())
        pred_axis = axis_preview(torch.tanh(output[2:4]).numpy(), torch.sigmoid(output[4]).numpy())
        true_axis = axis_preview(target[2:4].numpy(), target[4].numpy())
        pred_boundary = gray(torch.sigmoid(output[5]).numpy())
        true_boundary = gray(target[5].numpy())
        pred_centerline = gray(torch.sigmoid(output[6]).numpy())
        true_centerline = gray(target[6].numpy())

        items = [
            ("input", input_img),
            ("true mask", true_mask),
            ("pred mask", pred_mask),
            ("true density", true_density),
            ("pred density", pred_density),
            ("true axis/conf", true_axis),
            ("pred axis/conf", pred_axis),
            ("true boundary", true_boundary),
            ("pred boundary", pred_boundary),
            ("true centerline", true_centerline),
            ("pred centerline", pred_centerline),
        ]
        width, height = input_img.size
        panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (label, item) in enumerate(items):
            x = i * width
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
            panel.paste(item, (x, 24))
        panel.save(output_dir / f"{pair_id}_rich_prediction.png")


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RichStitchDataset(dataset_dir, args.split_field, args.train_split)
    val_dataset = RichStitchDataset(dataset_dir, args.split_field, args.val_split)
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
    model = TinyUNet(out_channels=len(TARGET_CHANNELS), base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} start", flush=True)
        model.train()
        total_loss = 0.0
        part_totals = {
            "mask_loss": 0.0,
            "density_loss": 0.0,
            "axis_loss": 0.0,
            "confidence_loss": 0.0,
            "boundary_loss": 0.0,
            "centerline_loss": 0.0,
        }
        seen = 0
        for batch_index, (image, target, _) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss, parts = batch_loss(output, target, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                part_totals[key] += value * batch_size
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
            **{f"train_{key}": value / max(1, seen) for key, value in part_totals.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(json.dumps(record, ensure_ascii=False), flush=True)

    checkpoint_path = output_dir / "rich_stitch_label_unet.pt"
    best_checkpoint_path = output_dir / "best_rich_stitch_label_unet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "TinyUNet",
            "base_channels": args.base_channels,
            "target_channels": TARGET_CHANNELS,
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
                "target_channels": TARGET_CHANNELS,
                "dataset_dir": str(dataset_dir),
                "args": vars(args),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            },
            best_checkpoint_path,
        )
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})

    save_prediction_previews(model, val_dataset, device, output_dir / "predictions", args.preview_count)
    summary = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "split_field": args.split_field,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "target_channels": TARGET_CHANNELS,
        "history": history,
        "note": "Rich-label baseline trained on structural labels derived from DST files. Stitch type remains heuristic unless human-labeled.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {checkpoint_path}")
    print(f"saved metrics: {output_dir / 'metrics.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train rich structural stitch label U-Net.")
    parser.add_argument("--dataset-dir", default="datasets/dst_rendered_supervision_3322_rich_v2")
    parser.add_argument("--output-dir", default="models/rich_stitch_label_unet_v2")
    parser.add_argument("--split-field", default="split_stratified")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-pos-weight", type=float, default=5.0)
    parser.add_argument("--boundary-pos-weight", type=float, default=8.0)
    parser.add_argument("--centerline-pos-weight", type=float, default=12.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.75)
    parser.add_argument("--axis-loss-weight", type=float, default=0.75)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.35)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.45)
    parser.add_argument("--centerline-loss-weight", type=float, default=0.45)
    parser.add_argument("--axis-conf-threshold", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
