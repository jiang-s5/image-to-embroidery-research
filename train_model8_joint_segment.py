from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from train_model3_geometry_graph import TARGET_CHANNELS, GeometryGraphDataset, compute_stitch_type_weights, dice_loss_from_logits, save_prediction_previews
from train_model7_geometry_planner import (
    Model7GeometryPlannerCascade,
    batch_loss as model7_batch_loss,
    composite_score as model7_composite_score,
    evaluate as model7_evaluate,
    focal_bce_with_logits,
    pairwise_path_rank_accuracy,
    pairwise_path_rank_loss,
)
from train_image_to_stitch_label import choose_device
from train_rich_stitch_label import gray, tensor_to_image
from train_rich_stitch_planner import STITCH_TYPE_NAMES, STITCH_TYPE_RGB
from PIL import Image, ImageDraw


MODEL8_CHANNELS = TARGET_CHANNELS + ["segment_mask", "segment_boundary", "segment_order"]


class SegmentGeometryDataset(GeometryGraphDataset):
    def __getitem__(self, index: int):
        image, target, pair_id = super().__getitem__(index)
        row = self.rows[index]
        segment_map = np.load(self.data_path(row, "segment_id_map_npy")).astype(np.float32)
        segment_mask = (segment_map > 0).astype(np.float32)
        max_segment = max(1.0, float(segment_map.max()))
        segment_order = segment_map / max_segment
        padded = np.pad(segment_map, ((1, 1), (1, 1)), mode="edge")
        boundary = (
            ((padded[1:-1, 1:-1] != padded[:-2, 1:-1]) & (padded[1:-1, 1:-1] > 0))
            | ((padded[1:-1, 1:-1] != padded[2:, 1:-1]) & (padded[1:-1, 1:-1] > 0))
            | ((padded[1:-1, 1:-1] != padded[1:-1, :-2]) & (padded[1:-1, 1:-1] > 0))
            | ((padded[1:-1, 1:-1] != padded[1:-1, 2:]) & (padded[1:-1, 1:-1] > 0))
        ).astype(np.float32)
        boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)

        # Match the parent dataset's random flips by recomputing from image/target orientation is hard after the fact.
        # Segment labels are only used without augmentation in the stable model8 run.
        target["segment"] = torch.from_numpy(np.stack([segment_mask, boundary, segment_order], axis=0))
        return image, target, pair_id


class Model8JointSegmentPlanner(Model7GeometryPlannerCascade):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, geometry_channels: int = 13, detach_planner_geometry: bool = False):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            geometry_channels=geometry_channels,
            detach_planner_geometry=detach_planner_geometry,
        )
        b = base_channels
        self.segment_head = nn.Sequential(
            nn.Conv2d(b + geometry_channels, b, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        geometry = self.geometry_head(features)
        planner_input = torch.cat([features, geometry], dim=1)
        path_order = self.planner_head(planner_input)
        segment = self.segment_head(planner_input)
        return torch.cat([geometry, path_order, segment], dim=1)


def load_initializer(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
            loaded.append(key)
        elif key == "head.weight":
            if "geometry_head.weight" in target_state:
                channels = min(target_state["geometry_head.weight"].shape[0], value.shape[0])
                target_state["geometry_head.weight"][:channels] = value[:channels]
                loaded.append(f"head.weight -> geometry_head.weight[:{channels}]")
            if value.shape[0] > 13 and target_state["planner_head.2.weight"].shape == value[13:14].shape:
                target_state["planner_head.2.weight"] = value[13:14]
                loaded.append("head.weight[13] -> planner_head.2.weight")
        elif key == "head.bias":
            if "geometry_head.bias" in target_state:
                channels = min(target_state["geometry_head.bias"].shape[0], value.shape[0])
                target_state["geometry_head.bias"][:channels] = value[:channels]
                loaded.append(f"head.bias -> geometry_head.bias[:{channels}]")
            if value.shape[0] > 13 and target_state["planner_head.2.bias"].shape == value[13:14].shape:
                target_state["planner_head.2.bias"] = value[13:14]
                loaded.append("head.bias[13] -> planner_head.2.bias")
        else:
            skipped.append(key)
    model.load_state_dict(target_state)
    model.to(device)
    return {
        "used": True,
        "path": str(checkpoint_path.resolve()),
        "source_model": checkpoint.get("model", ""),
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "loaded_tail": loaded[-10:],
    }


def freeze_encoder(model: nn.Module) -> None:
    frozen_prefixes = ("enc1.", "enc2.", "enc3.", "bottleneck.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith(frozen_prefixes)


def segment_loss(output: torch.Tensor, target: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    segment = target["segment"]
    segment_mask = segment[:, 0:1]
    segment_boundary = segment[:, 1:2]
    segment_order = segment[:, 2:3]
    segment_mask_logits = output[:, 14:15]
    segment_boundary_logits = output[:, 15:16]
    segment_order_pred = torch.sigmoid(output[:, 16:17])
    mask_pos_weight = torch.tensor([args.segment_mask_pos_weight], device=output.device)
    boundary_pos_weight = torch.tensor([args.segment_boundary_pos_weight], device=output.device)
    mask_loss = F.binary_cross_entropy_with_logits(segment_mask_logits, segment_mask, pos_weight=mask_pos_weight)
    mask_loss = mask_loss + args.segment_dice_weight * dice_loss_from_logits(segment_mask_logits, segment_mask)
    boundary_loss = focal_bce_with_logits(segment_boundary_logits, segment_boundary, boundary_pos_weight, args.segment_boundary_focal_gamma)
    order_l1 = (torch.abs(segment_order_pred - segment_order) * segment_mask).sum() / segment_mask.sum().clamp_min(1.0)
    order_rank = pairwise_path_rank_loss(
        segment_order_pred,
        segment_order,
        segment_mask,
        args.rank_samples,
        args.rank_pairs,
        args.rank_target_margin,
        args.rank_temperature,
    )
    total = (
        args.segment_mask_loss_weight * mask_loss
        + args.segment_boundary_loss_weight * boundary_loss
        + args.segment_order_loss_weight * (order_l1 + args.segment_rank_loss_weight * order_rank)
    )
    return total, {
        "segment_mask_loss": float(mask_loss.detach().cpu()),
        "segment_boundary_loss": float(boundary_loss.detach().cpu()),
        "segment_order_l1_loss": float(order_l1.detach().cpu()),
        "segment_order_rank_loss": float(order_rank.detach().cpu()),
    }


def batch_loss(output: torch.Tensor, target: dict[str, torch.Tensor], class_weights: torch.Tensor, args: argparse.Namespace, epoch: int):
    base_loss, parts = model7_batch_loss(output[:, :14], target, class_weights, args, epoch)
    seg_loss, seg_parts = segment_loss(output, target, args)
    total = base_loss + seg_loss
    parts.update(seg_parts)
    parts["segment_total_loss"] = float(seg_loss.detach().cpu())
    return total, parts


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_weights: torch.Tensor, args: argparse.Namespace, epoch: int) -> dict[str, float]:
    base_metrics = model7_evaluate(model, loader, device, class_weights, args, epoch)
    model.eval()
    totals = {"segment_mask_iou": 0.0, "segment_boundary_iou": 0.0, "segment_order_mae": 0.0, "segment_order_pairwise_acc": 0.0}
    count = 0
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = {key: value.to(device) for key, value in target.items()}
            output = model(image)
            segment = target["segment"]
            seg_mask = segment[:, 0:1]
            seg_boundary = segment[:, 1:2]
            seg_order = segment[:, 2:3]
            pred_mask = torch.sigmoid(output[:, 14:15])
            pred_boundary = torch.sigmoid(output[:, 15:16])
            pred_order = torch.sigmoid(output[:, 16:17])
            b = image.shape[0]
            mask_bin = pred_mask > 0.5
            target_mask = seg_mask > 0.5
            boundary_bin = pred_boundary > 0.5
            target_boundary = seg_boundary > 0.5
            mask_iou = ((mask_bin & target_mask).sum(dim=(1, 2, 3)).float() / (mask_bin | target_mask).sum(dim=(1, 2, 3)).float().clamp_min(1.0))
            boundary_iou = ((boundary_bin & target_boundary).sum(dim=(1, 2, 3)).float() / (boundary_bin | target_boundary).sum(dim=(1, 2, 3)).float().clamp_min(1.0))
            order_mae = (torch.abs(pred_order - seg_order) * seg_mask).sum(dim=(1, 2, 3)) / seg_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            order_acc = pairwise_path_rank_accuracy(pred_order, seg_order, seg_mask, args.rank_eval_samples, args.rank_eval_pairs, args.rank_target_margin)
            totals["segment_mask_iou"] += float(mask_iou.sum().cpu())
            totals["segment_boundary_iou"] += float(boundary_iou.sum().cpu())
            totals["segment_order_mae"] += float(order_mae.sum().cpu())
            totals["segment_order_pairwise_acc"] += float(order_acc.sum().cpu())
            count += b
    base_metrics.update({key: value / max(1, count) for key, value in totals.items()})
    base_metrics["model8_composite_score"] = (
        model7_composite_score(base_metrics)
        + base_metrics["segment_mask_iou"]
        + 0.5 * base_metrics["segment_boundary_iou"]
        + base_metrics["segment_order_pairwise_acc"]
        - base_metrics["segment_order_mae"]
    )
    return base_metrics


def save_model8_previews(model: nn.Module, dataset: SegmentGeometryDataset, device: torch.device, output_dir: Path, max_items: int) -> None:
    save_prediction_previews(model, dataset, device, output_dir, max_items)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index in range(min(max_items, len(dataset))):
        image, target, pair_id = dataset[index]
        with torch.no_grad():
            output = model(image[None].to(device))[0].detach().cpu()
        input_img = tensor_to_image(image)
        items = [
            ("input", input_img),
            ("true seg mask", gray(target["segment"][0].numpy())),
            ("pred seg mask", gray(torch.sigmoid(output[14]).numpy())),
            ("true seg boundary", gray(target["segment"][1].numpy())),
            ("pred seg boundary", gray(torch.sigmoid(output[15]).numpy())),
            ("true seg order", gray(target["segment"][2].numpy())),
            ("pred seg order", gray(torch.sigmoid(output[16]).numpy())),
        ]
        width, height = input_img.size
        panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (label, item) in enumerate(items):
            x = i * width
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
            panel.paste(item, (x, 24))
        panel.save(output_dir / f"{pair_id}_model8_segment_prediction.png")


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SegmentGeometryDataset(dataset_dir, args.split_field, args.train_split, augment=False)
    val_dataset = SegmentGeometryDataset(dataset_dir, args.split_field, args.val_split, augment=False)
    class_weights = compute_stitch_type_weights(train_dataset, args.class_weight_scan_items)
    device = choose_device(args.cpu)
    model = Model8JointSegmentPlanner(base_channels=args.base_channels, detach_planner_geometry=False).to(device)
    init_report = {"used": False}
    if args.init_checkpoint:
        init_report = load_initializer(model, Path(args.init_checkpoint), device)
        print(json.dumps({"init_checkpoint": init_report}, ensure_ascii=False), flush=True)
    if args.freeze_encoder:
        freeze_encoder(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(json.dumps({"model8_channels": MODEL8_CHANNELS, "trainable_parameters": trainable, "frozen_parameters": frozen, "device": str(device)}, ensure_ascii=False), flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_plateau_patience, min_lr=args.min_lr)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        print(f"model8 epoch {epoch}/{args.epochs} start", flush=True)
        model.train()
        total_loss = 0.0
        part_totals: dict[str, float] = {}
        seen = 0
        for batch_index, (image, target, _) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            target = {key: value.to(device, non_blocking=True) for key, value in target.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(image)
                loss, parts = batch_loss(output, target, class_weights, args, epoch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            b = image.shape[0]
            total_loss += float(loss.detach().cpu()) * b
            for key, value in parts.items():
                part_totals[key] = part_totals.get(key, 0.0) + value * b
            seen += b
            if args.progress_every and batch_index % args.progress_every == 0:
                print(json.dumps({"epoch": epoch, "batch": batch_index, "seen": seen, "train_loss_so_far": total_loss / max(1, seen)}, ensure_ascii=False), flush=True)
        metrics = evaluate(model, val_loader, device, class_weights, args, epoch)
        score = metrics["model8_composite_score"]
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            **{f"train_{key}": value / max(1, seen) for key, value in part_totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        print(json.dumps(record, ensure_ascii=False), flush=True)

    checkpoint_path = output_dir / "model8_joint_segment.pt"
    best_checkpoint_path = output_dir / "best_model8_joint_segment.pt"
    payload = {
        "model_state": model.state_dict(),
        "model": "Model8JointSegmentPlanner",
        "base_channels": args.base_channels,
        "geometry_channels": 13,
        "target_channels": MODEL8_CHANNELS,
        "stitch_type_names": STITCH_TYPE_NAMES,
        "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
        "dataset_dir": str(dataset_dir),
        "args": vars(args),
        "history": history,
        "init_checkpoint": init_report,
        "model8_changes": {
            "partial_unfreeze": "encoder frozen, decoder + geometry_head + planner_head + segment_head trainable",
            "segment_supervision": ["segment_mask", "segment_boundary", "segment_order"],
            "inherits": "model7 geometry/planner cascade and model3-style geometry outputs",
        },
    }
    torch.save(payload, checkpoint_path)
    if best_state is not None:
        best_payload = dict(payload)
        best_payload["model_state"] = best_state
        best_payload["best_epoch"] = best_epoch
        best_payload["best_composite_score"] = best_score
        torch.save(best_payload, best_checkpoint_path)
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    save_model8_previews(model, val_dataset, device, output_dir / "predictions", args.preview_count)
    summary = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_composite_score": best_score,
        "target_channels": MODEL8_CHANNELS,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train model8 joint geometry/planner with segment-level supervision.")
    parser.add_argument("--dataset-dir", default="datasets/dataset2_collection_20260512_geometry_graph")
    parser.add_argument("--output-dir", default="models/model8_joint_segment_e20")
    parser.add_argument("--split-field", default="canonical_split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--init-checkpoint", default="models/model7_geometry_planner_frozen_e8/best_model7_geometry_planner_cascade.pt")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--mask-pos-weight", type=float, default=5.0)
    parser.add_argument("--boundary-pos-weight", type=float, default=8.0)
    parser.add_argument("--centerline-pos-weight", type=float, default=14.0)
    parser.add_argument("--endpoint-pos-weight", type=float, default=20.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.8)
    parser.add_argument("--axis-loss-weight", type=float, default=0.7)
    parser.add_argument("--axis-min-conf-weight", type=float, default=0.05)
    parser.add_argument("--axis-conf-threshold", type=float, default=0.12)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.3)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.6)
    parser.add_argument("--centerline-loss-weight", type=float, default=0.75)
    parser.add_argument("--stitch-type-loss-weight", type=float, default=0.75)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.7)
    parser.add_argument("--path-order-loss-weight", type=float, default=0.7)
    parser.add_argument("--path-order-start-weight", type=float, default=0.45)
    parser.add_argument("--path-order-warmup-epochs", type=int, default=4)
    parser.add_argument("--path-rank-loss-weight", type=float, default=0.12)
    parser.add_argument("--rank-samples", type=int, default=192)
    parser.add_argument("--rank-pairs", type=int, default=384)
    parser.add_argument("--rank-eval-samples", type=int, default=256)
    parser.add_argument("--rank-eval-pairs", type=int, default=512)
    parser.add_argument("--rank-target-margin", type=float, default=0.035)
    parser.add_argument("--rank-temperature", type=float, default=0.12)
    parser.add_argument("--mask-dice-weight", type=float, default=0.15)
    parser.add_argument("--boundary-dice-weight", type=float, default=0.25)
    parser.add_argument("--centerline-dice-weight", type=float, default=0.3)
    parser.add_argument("--endpoint-focal-gamma", type=float, default=1.5)
    parser.add_argument("--stitch-type-focal-gamma", type=float, default=1.2)
    parser.add_argument("--endpoint-eval-threshold", type=float, default=0.25)
    parser.add_argument("--segment-mask-pos-weight", type=float, default=4.0)
    parser.add_argument("--segment-boundary-pos-weight", type=float, default=10.0)
    parser.add_argument("--segment-mask-loss-weight", type=float, default=0.25)
    parser.add_argument("--segment-boundary-loss-weight", type=float, default=0.25)
    parser.add_argument("--segment-order-loss-weight", type=float, default=0.35)
    parser.add_argument("--segment-rank-loss-weight", type=float, default=0.12)
    parser.add_argument("--segment-dice-weight", type=float, default=0.15)
    parser.add_argument("--segment-boundary-focal-gamma", type=float, default=1.5)
    parser.add_argument("--class-weight-scan-items", type=int, default=0)
    parser.add_argument("--lr-plateau-patience", type=int, default=3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
