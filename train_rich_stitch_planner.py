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
from train_rich_stitch_label import (
    TARGET_CHANNELS as STRUCTURAL_CHANNELS,
    axis_preview,
    binary_f1,
    binary_iou,
    gray,
    tensor_to_image,
)


STITCH_TYPE_NAMES = ["no_stitch", "running", "satin", "fill"]
TARGET_CHANNELS = STRUCTURAL_CHANNELS + [f"stitch_type_{name}" for name in STITCH_TYPE_NAMES]
STITCH_TYPE_RGB = np.asarray(
    [
        (0, 0, 0),
        (80, 190, 255),
        (255, 190, 70),
        (180, 120, 230),
    ],
    dtype=np.int32,
)


class RichPlannerDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
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

        stitch_type = load_stitch_type(self.dataset_dir / row["stitch_type_heuristic_png"])
        structural = np.concatenate(
            [mask_arr, density_arr, axis_arr, conf_arr, boundary_arr, centerline_arr],
            axis=0,
        )
        return image_tensor, torch.from_numpy(structural), torch.from_numpy(stitch_type), row["pair_id"]


def load_stitch_type(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    distances = ((rgb[:, :, None, :] - STITCH_TYPE_RGB[None, None, :, :]) ** 2).sum(axis=3)
    return np.argmin(distances, axis=2).astype(np.int64)


def compute_stitch_type_weights(dataset: RichPlannerDataset, max_items: int = 0) -> torch.Tensor:
    counts = np.zeros(len(STITCH_TYPE_NAMES), dtype=np.float64)
    total = len(dataset) if max_items <= 0 else min(len(dataset), max_items)
    for index in range(total):
        row = dataset.rows[index]
        labels = load_stitch_type(dataset.dataset_dir / row["stitch_type_heuristic_png"])
        counts += np.bincount(labels.reshape(-1), minlength=len(STITCH_TYPE_NAMES))
    freq = counts / max(1.0, counts.sum())
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    weights = weights / weights.mean()
    weights[0] *= 0.45
    return torch.tensor(weights, dtype=torch.float32)


def batch_loss(
    output: torch.Tensor,
    structural: torch.Tensor,
    stitch_type: torch.Tensor,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_mask = structural[:, 0:1]
    target_density = structural[:, 1:2]
    target_axis = structural[:, 2:4]
    target_conf = structural[:, 4:5]
    target_boundary = structural[:, 5:6]
    target_centerline = structural[:, 6:7]

    mask_logits = output[:, 0:1]
    density_pred = torch.sigmoid(output[:, 1:2])
    axis_pred = torch.tanh(output[:, 2:4])
    conf_pred = torch.sigmoid(output[:, 4:5])
    boundary_logits = output[:, 5:6]
    centerline_logits = output[:, 6:7]
    stitch_logits = output[:, 7:11]

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
    stitch_type_loss = F.cross_entropy(stitch_logits, stitch_type, weight=class_weights.to(output.device))

    total = (
        mask_loss
        + args.density_loss_weight * density_loss
        + args.axis_loss_weight * axis_loss
        + args.confidence_loss_weight * confidence_loss
        + args.boundary_loss_weight * boundary_loss
        + args.centerline_loss_weight * centerline_loss
        + args.stitch_type_loss_weight * stitch_type_loss
    )
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "density_loss": float(density_loss.detach().cpu()),
        "axis_loss": float(axis_loss.detach().cpu()),
        "confidence_loss": float(confidence_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "centerline_loss": float(centerline_loss.detach().cpu()),
        "stitch_type_loss": float(stitch_type_loss.detach().cpu()),
    }


def class_iou(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> torch.Tensor:
    pred_bool = pred == class_id
    target_bool = target == class_id
    intersection = (pred_bool & target_bool).sum(dim=(1, 2)).float()
    union = (pred_bool | target_bool).sum(dim=(1, 2)).float().clamp_min(1.0)
    return intersection / union


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
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
        "stitch_type_acc_all": 0.0,
        "stitch_type_acc_on_mask": 0.0,
        "stitch_type_miou_nonzero": 0.0,
    }
    count = 0
    with torch.no_grad():
        for image, structural, stitch_type, _ in loader:
            image = image.to(device)
            structural = structural.to(device)
            stitch_type = stitch_type.to(device)
            output = model(image)
            loss, _ = batch_loss(output, structural, stitch_type, class_weights, args)

            mask_pred = torch.sigmoid(output[:, 0:1])
            density_pred = torch.sigmoid(output[:, 1:2])
            axis_pred = torch.tanh(output[:, 2:4])
            conf_pred = torch.sigmoid(output[:, 4:5])
            boundary_pred = torch.sigmoid(output[:, 5:6])
            centerline_pred = torch.sigmoid(output[:, 6:7])
            stitch_pred = output[:, 7:11].argmax(dim=1)

            target_conf = structural[:, 4:5]
            confident = (target_conf > args.axis_conf_threshold).expand_as(axis_pred)
            axis_denom = confident.sum(dim=(1, 2, 3)).clamp_min(1.0)
            axis_mae = ((axis_pred - structural[:, 2:4]).abs() * confident).sum(dim=(1, 2, 3)) / axis_denom

            batch_size = image.shape[0]
            mask_target_bool = stitch_type > 0
            on_mask_count = mask_target_bool.reshape(batch_size, -1).sum(dim=1).clamp_min(1)
            on_mask_acc = ((stitch_pred == stitch_type) & mask_target_bool).reshape(batch_size, -1).sum(dim=1).float() / on_mask_count
            nonzero_iou = torch.stack([class_iou(stitch_pred, stitch_type, class_id) for class_id in (1, 2, 3)], dim=0).mean(dim=0)

            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["mask_iou"] += float(binary_iou(mask_pred, structural[:, 0:1]).sum().detach().cpu())
            totals["density_mae"] += float((density_pred - structural[:, 1:2]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["axis_mae_on_confident"] += float(axis_mae.sum().detach().cpu())
            totals["confidence_mae"] += float((conf_pred - target_conf).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["boundary_iou"] += float(binary_iou(boundary_pred, structural[:, 5:6]).sum().detach().cpu())
            totals["boundary_f1"] += float(binary_f1(boundary_pred, structural[:, 5:6]).sum().detach().cpu())
            totals["centerline_iou"] += float(binary_iou(centerline_pred, structural[:, 6:7]).sum().detach().cpu())
            totals["centerline_f1"] += float(binary_f1(centerline_pred, structural[:, 6:7]).sum().detach().cpu())
            totals["stitch_type_acc_all"] += float((stitch_pred == stitch_type).float().mean(dim=(1, 2)).sum().detach().cpu())
            totals["stitch_type_acc_on_mask"] += float(on_mask_acc.sum().detach().cpu())
            totals["stitch_type_miou_nonzero"] += float(nonzero_iou.sum().detach().cpu())
            count += batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def stitch_type_preview(labels: np.ndarray) -> Image.Image:
    palette = STITCH_TYPE_RGB.astype(np.uint8)
    return Image.fromarray(palette[np.clip(labels, 0, len(palette) - 1)], mode="RGB")


def save_prediction_previews(
    model: nn.Module,
    dataset: RichPlannerDataset,
    device: torch.device,
    output_dir: Path,
    max_items: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index in range(min(max_items, len(dataset))):
        image, structural, stitch_type, pair_id = dataset[index]
        with torch.no_grad():
            output = model(image[None].to(device))[0].detach().cpu()
        input_img = tensor_to_image(image)
        pred_stitch = output[7:11].argmax(dim=0).numpy()
        items = [
            ("input", input_img),
            ("true mask", gray(structural[0].numpy())),
            ("pred mask", gray(torch.sigmoid(output[0]).numpy())),
            ("true axis/conf", axis_preview(structural[2:4].numpy(), structural[4].numpy())),
            ("pred axis/conf", axis_preview(torch.tanh(output[2:4]).numpy(), torch.sigmoid(output[4]).numpy())),
            ("true stitch type", stitch_type_preview(stitch_type.numpy())),
            ("pred stitch type", stitch_type_preview(pred_stitch)),
            ("true boundary", gray(structural[5].numpy())),
            ("pred boundary", gray(torch.sigmoid(output[5]).numpy())),
            ("true centerline", gray(structural[6].numpy())),
            ("pred centerline", gray(torch.sigmoid(output[6]).numpy())),
        ]
        width, height = input_img.size
        panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (label, item) in enumerate(items):
            x = i * width
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
            panel.paste(item, (x, 24))
        panel.save(output_dir / f"{pair_id}_planner_prediction.png")


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RichPlannerDataset(dataset_dir, args.split_field, args.train_split)
    val_dataset = RichPlannerDataset(dataset_dir, args.split_field, args.val_split)
    class_weights = compute_stitch_type_weights(train_dataset, args.class_weight_scan_items)
    print(
        json.dumps(
            {"stitch_type_names": STITCH_TYPE_NAMES, "stitch_type_class_weights": class_weights.tolist()},
            ensure_ascii=False,
        ),
        flush=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = choose_device(args.cpu)
    model = TinyUNet(out_channels=len(TARGET_CHANNELS), base_channels=args.base_channels).to(device)
    init_report: dict[str, object] = {"used": False}
    if args.init_structural_checkpoint:
        init_report = load_structural_initializer(model, Path(args.init_structural_checkpoint), device)
        print(json.dumps({"init_structural_checkpoint": init_report}, ensure_ascii=False), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_score = -float("inf")
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
            "stitch_type_loss": 0.0,
        }
        seen = 0
        for batch_index, (image, structural, stitch_type, _) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            structural = structural.to(device, non_blocking=True)
            stitch_type = stitch_type.to(device, non_blocking=True)
            output = model(image)
            loss, parts = batch_loss(output, structural, stitch_type, class_weights, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                part_totals[key] += value * batch_size
            seen += batch_size
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    json.dumps(
                        {"epoch": epoch, "batch": batch_index, "seen": seen, "train_loss_so_far": total_loss / max(1, seen)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, class_weights, args)
        score = (
            val_metrics["mask_iou"]
            + val_metrics["boundary_f1"]
            + val_metrics["centerline_f1"]
            + val_metrics["stitch_type_miou_nonzero"]
            - val_metrics["axis_mae_on_confident"]
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            **{f"train_{key}": value / max(1, seen) for key, value in part_totals.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_composite_score": float(score),
        }
        history.append(record)
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(json.dumps(record, ensure_ascii=False), flush=True)

    checkpoint_path = output_dir / "rich_stitch_planner_unet.pt"
    best_checkpoint_path = output_dir / "best_rich_stitch_planner_unet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "TinyUNet",
            "base_channels": args.base_channels,
            "target_channels": TARGET_CHANNELS,
            "stitch_type_names": STITCH_TYPE_NAMES,
            "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
            "stitch_type_class_weights": class_weights.tolist(),
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
                "stitch_type_names": STITCH_TYPE_NAMES,
                "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
                "stitch_type_class_weights": class_weights.tolist(),
                "dataset_dir": str(dataset_dir),
                "args": vars(args),
                "history": history,
                "best_epoch": best_epoch,
                "best_composite_score": best_score,
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
        "best_composite_score": best_score,
        "target_channels": TARGET_CHANNELS,
        "init_structural_checkpoint": init_report,
        "stitch_type_names": STITCH_TYPE_NAMES,
        "stitch_type_class_weights": class_weights.tolist(),
        "history": history,
        "note": "Planner model adds a stitch_type head so downstream DST export can choose running/satin/fill rules.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {checkpoint_path}")
    print(f"saved metrics: {output_dir / 'metrics.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train rich stitch planner U-Net with stitch type head.")
    parser.add_argument("--dataset-dir", default="datasets/dst_rendered_supervision_3322_rich_v2")
    parser.add_argument("--output-dir", default="models/rich_stitch_planner_unet_v2")
    parser.add_argument("--split-field", default="split_stratified")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--init-structural-checkpoint", default="")
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
    parser.add_argument("--stitch-type-loss-weight", type=float, default=0.7)
    parser.add_argument("--axis-conf-threshold", type=float, default=0.2)
    parser.add_argument("--class-weight-scan-items", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def load_structural_initializer(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    loaded = []
    skipped = []
    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(key)
            continue
        if target_state[key].shape == value.shape:
            target_state[key] = value
            loaded.append(key)
        elif key == "head.weight" and target_state[key].shape[1:] == value.shape[1:]:
            channels = min(target_state[key].shape[0], value.shape[0])
            target_state[key][:channels] = value[:channels]
            loaded.append(f"{key}[:{channels}]")
        elif key == "head.bias":
            channels = min(target_state[key].shape[0], value.shape[0])
            target_state[key][:channels] = value[:channels]
            loaded.append(f"{key}[:{channels}]")
        else:
            skipped.append(key)
    model.load_state_dict(target_state)
    model.to(device)
    return {
        "used": True,
        "path": str(checkpoint_path.resolve()),
        "source_target_channels": checkpoint.get("target_channels", []),
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "head_copied_channels": min(len(checkpoint.get("target_channels", [])), 7),
    }


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
