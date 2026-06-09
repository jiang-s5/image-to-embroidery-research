from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from train_image_to_stitch_label import choose_device
from train_model3_geometry_graph import compute_stitch_type_weights, dice_loss_from_logits
from train_model7_geometry_planner import focal_bce_with_logits
from train_model8_joint_segment import (
    MODEL8_CHANNELS,
    Model8JointSegmentPlanner,
    SegmentGeometryDataset,
    batch_loss as model8_batch_loss,
    evaluate as model8_evaluate,
    freeze_encoder,
    load_initializer,
)
from train_rich_stitch_label import binary_f1, gray, tensor_to_image
from train_rich_stitch_planner import STITCH_TYPE_NAMES, STITCH_TYPE_RGB


MODEL10_CONTINUITY_CHANNELS = ["stitch_trace", "near_connect", "jump_endpoint"]
MODEL10_CHANNELS = MODEL8_CHANNELS + MODEL10_CONTINUITY_CHANNELS


class VectorContinuityDataset(SegmentGeometryDataset):
    def __getitem__(self, index: int):
        image, target, pair_id = super().__getitem__(index)
        row = self.rows[index]
        continuity_path = self.dataset_dir / "labels" / f"{pair_id}_vector_continuity.npy"
        if not continuity_path.exists():
            raise FileNotFoundError(
                f"missing vector-continuity label: {continuity_path}. "
                "Run build_vector_continuity_labels.py first."
            )
        target["continuity"] = torch.from_numpy(np.load(continuity_path).astype(np.float32))
        return image, target, pair_id


class Model10VectorContinuityPlanner(Model8JointSegmentPlanner):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        geometry_channels: int = 13,
        detach_planner_geometry: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            geometry_channels=geometry_channels,
            detach_planner_geometry=detach_planner_geometry,
        )
        b = base_channels
        self.continuity_head = nn.Sequential(
            nn.Conv2d(b + geometry_channels, b, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, len(MODEL10_CONTINUITY_CHANNELS), kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        geometry = self.geometry_head(features)
        planner_input = torch.cat([features, geometry], dim=1)
        path_order = self.planner_head(planner_input)
        segment = self.segment_head(planner_input)
        continuity = self.continuity_head(planner_input)
        return torch.cat([geometry, path_order, segment, continuity], dim=1)


def continuity_loss(output: torch.Tensor, target: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    continuity = target["continuity"]
    logits = output[:, 17:20]
    pos_weight = torch.tensor(
        [args.stitch_trace_pos_weight, args.near_connect_pos_weight, args.jump_endpoint_pos_weight],
        device=output.device,
    )[:, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, continuity, pos_weight=pos_weight, reduction="none").mean(dim=(0, 2, 3))
    stitch_dice = dice_loss_from_logits(logits[:, 0:1], continuity[:, 0:1])
    near_dice = dice_loss_from_logits(logits[:, 1:2], continuity[:, 1:2])
    jump_focal = focal_bce_with_logits(
        logits[:, 2:3],
        continuity[:, 2:3],
        torch.tensor([args.jump_endpoint_pos_weight], device=output.device),
        args.jump_endpoint_focal_gamma,
    )
    total = (
        args.stitch_trace_loss_weight * (bce[0] + args.continuity_dice_weight * stitch_dice)
        + args.near_connect_loss_weight * (bce[1] + args.continuity_dice_weight * near_dice)
        + args.jump_endpoint_loss_weight * (bce[2] + args.jump_endpoint_focal_weight * jump_focal)
    )
    return total, {
        "stitch_trace_bce": float(bce[0].detach().cpu()),
        "near_connect_bce": float(bce[1].detach().cpu()),
        "jump_endpoint_bce": float(bce[2].detach().cpu()),
        "stitch_trace_dice": float(stitch_dice.detach().cpu()),
        "near_connect_dice": float(near_dice.detach().cpu()),
        "jump_endpoint_focal": float(jump_focal.detach().cpu()),
    }


def batch_loss(output: torch.Tensor, target: dict[str, torch.Tensor], class_weights: torch.Tensor, args: argparse.Namespace, epoch: int):
    base_loss, parts = model8_batch_loss(output[:, :17], target, class_weights, args, epoch)
    cont_loss, cont_parts = continuity_loss(output, target, args)
    total = base_loss + cont_loss
    parts.update(cont_parts)
    parts["continuity_total_loss"] = float(cont_loss.detach().cpu())
    return total, parts


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_weights: torch.Tensor, args: argparse.Namespace, epoch: int) -> dict[str, float]:
    metrics = model8_evaluate(model, loader, device, class_weights, args, epoch)
    totals = {
        "stitch_trace_f1": 0.0,
        "near_connect_f1": 0.0,
        "jump_endpoint_f1": 0.0,
        "continuity_mae": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = {key: value.to(device) for key, value in target.items()}
            output = model(image)
            pred = torch.sigmoid(output[:, 17:20])
            target_cont = target["continuity"]
            batch_size = image.shape[0]
            totals["stitch_trace_f1"] += float(binary_f1(pred[:, 0:1], target_cont[:, 0:1]).sum().cpu())
            totals["near_connect_f1"] += float(binary_f1(pred[:, 1:2], target_cont[:, 1:2]).sum().cpu())
            totals["jump_endpoint_f1"] += float(binary_f1(pred[:, 2:3], target_cont[:, 2:3]).sum().cpu())
            totals["continuity_mae"] += float((pred - target_cont).abs().mean(dim=(1, 2, 3)).sum().cpu())
            count += batch_size
    metrics.update({key: value / max(1, count) for key, value in totals.items()})
    metrics["model10_continuity_score"] = (
        metrics["model8_composite_score"]
        + 0.55 * metrics["stitch_trace_f1"]
        + 0.85 * metrics["near_connect_f1"]
        + 0.35 * metrics["jump_endpoint_f1"]
        - 0.50 * metrics["continuity_mae"]
    )
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    dataset_dir: Path,
    history: list[dict[str, float]],
    init_report: dict[str, object],
    best_epoch: int,
    best_score: float,
) -> None:
    payload = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": "Model10VectorContinuityPlanner",
        "base_channels": args.base_channels,
        "geometry_channels": 13,
        "target_channels": MODEL10_CHANNELS,
        "stitch_type_names": STITCH_TYPE_NAMES,
        "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
        "dataset_dir": str(dataset_dir),
        "args": vars(args),
        "history": history,
        "init_checkpoint": init_report,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "model10_changes": {
            "goal": "teach DST-derived vector continuity before graph planner export",
            "new_outputs": MODEL10_CONTINUITY_CHANNELS,
            "labels": {
                "stitch_trace": "true segment entry-to-exit vector trace",
                "near_connect": "same-color near gaps that should be continuous",
                "jump_endpoint": "long jump or color-change endpoints",
            },
        },
    }
    torch.save(payload, path)


def save_previews(model: nn.Module, dataset: VectorContinuityDataset, device: torch.device, output_dir: Path, max_items: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for index in range(min(max_items, len(dataset))):
        image, target, pair_id = dataset[index]
        with torch.no_grad():
            output = model(image[None].to(device))[0].detach().cpu()
        input_img = tensor_to_image(image)
        pred = torch.sigmoid(output[17:20]).numpy()
        true = target["continuity"].numpy()
        items = [
            ("input", input_img),
            ("true stitch trace", gray(true[0])),
            ("pred stitch trace", gray(pred[0])),
            ("true near connect", gray(true[1])),
            ("pred near connect", gray(pred[1])),
            ("true jump endpoint", gray(true[2])),
            ("pred jump endpoint", gray(pred[2])),
        ]
        width, height = input_img.size
        panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (label, item) in enumerate(items):
            x = i * width
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
            panel.paste(item, (x, 24))
        panel.save(output_dir / f"{pair_id}_model10_continuity_prediction.png")


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VectorContinuityDataset(dataset_dir, args.split_field, args.train_split, augment=False)
    val_dataset = VectorContinuityDataset(dataset_dir, args.split_field, args.val_split, augment=False)
    class_weights = compute_stitch_type_weights(train_dataset, args.class_weight_scan_items)
    device = choose_device(args.cpu)
    model = Model10VectorContinuityPlanner(base_channels=args.base_channels, detach_planner_geometry=False).to(device)
    init_report = {"used": False}
    if args.init_checkpoint:
        init_report = load_initializer(model, Path(args.init_checkpoint), device)
        print(json.dumps({"init_checkpoint": init_report}, ensure_ascii=False), flush=True)
    if args.freeze_encoder:
        freeze_encoder(model)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_plateau_patience, min_lr=args.min_lr)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0

    print(
        json.dumps(
            {
                "model": "model10_vector_continuity",
                "target_channels": MODEL10_CHANNELS,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "device": str(device),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        print(f"model10 vector continuity epoch {epoch}/{args.epochs} start", flush=True)
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
        score = metrics["model10_continuity_score"]
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
            save_checkpoint(output_dir / "best_model10_vector_continuity.pt", model, args, dataset_dir, copy.deepcopy(history), init_report, epoch, best_score)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    save_checkpoint(output_dir / "model10_vector_continuity_final.pt", model, args, dataset_dir, history, init_report, args.epochs, float(history[-1]["val_model10_continuity_score"]))
    best_checkpoint = output_dir / "best_model10_vector_continuity.pt"
    if best_checkpoint.exists():
        best_payload = torch.load(best_checkpoint, map_location="cpu")
        model.load_state_dict(best_payload["model_state"])
        model.to(device)
    save_previews(model, val_dataset, device, output_dir / "predictions", args.preview_count)
    summary = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "checkpoint": str(output_dir / "model10_vector_continuity_final.pt"),
        "best_checkpoint": str(best_checkpoint),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "target_channels": MODEL10_CHANNELS,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train model10 with DST-derived vector-continuity supervision.")
    parser.add_argument("--dataset-dir", default="datasets/dataset2_collection_20260512_geometry_graph")
    parser.add_argument("--output-dir", default="models/model10_vector_continuity")
    parser.add_argument("--split-field", default="canonical_split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--init-checkpoint", default="C:/Users/jiang/Desktop/科研/图片到dst/模型/model9_hard_experiments_20260513/model9A_joint_best_model9_hard.pt")
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
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
    parser.add_argument("--boundary-loss-weight", type=float, default=0.65)
    parser.add_argument("--centerline-loss-weight", type=float, default=0.85)
    parser.add_argument("--stitch-type-loss-weight", type=float, default=0.75)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.8)
    parser.add_argument("--path-order-loss-weight", type=float, default=0.55)
    parser.add_argument("--path-order-start-weight", type=float, default=0.35)
    parser.add_argument("--path-order-warmup-epochs", type=int, default=4)
    parser.add_argument("--path-rank-loss-weight", type=float, default=0.14)
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
    parser.add_argument("--segment-mask-loss-weight", type=float, default=0.22)
    parser.add_argument("--segment-boundary-loss-weight", type=float, default=0.25)
    parser.add_argument("--segment-order-loss-weight", type=float, default=0.28)
    parser.add_argument("--segment-rank-loss-weight", type=float, default=0.12)
    parser.add_argument("--segment-dice-weight", type=float, default=0.15)
    parser.add_argument("--segment-boundary-focal-gamma", type=float, default=1.5)
    parser.add_argument("--stitch-trace-pos-weight", type=float, default=8.0)
    parser.add_argument("--near-connect-pos-weight", type=float, default=28.0)
    parser.add_argument("--jump-endpoint-pos-weight", type=float, default=18.0)
    parser.add_argument("--stitch-trace-loss-weight", type=float, default=0.35)
    parser.add_argument("--near-connect-loss-weight", type=float, default=0.55)
    parser.add_argument("--jump-endpoint-loss-weight", type=float, default=0.25)
    parser.add_argument("--continuity-dice-weight", type=float, default=0.25)
    parser.add_argument("--jump-endpoint-focal-weight", type=float, default=0.35)
    parser.add_argument("--jump-endpoint-focal-gamma", type=float, default=1.5)
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
