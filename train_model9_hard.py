from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_image_to_stitch_label import choose_device
from train_model3_geometry_graph import compute_stitch_type_weights
from train_model8_joint_segment import (
    MODEL8_CHANNELS,
    Model8JointSegmentPlanner,
    SegmentGeometryDataset,
    batch_loss,
    evaluate,
    freeze_encoder,
    load_initializer,
    save_model8_previews,
)
from train_rich_stitch_planner import STITCH_TYPE_NAMES, STITCH_TYPE_RGB


HARD_THRESHOLDS = {
    "mask_iou": ("max", 0.9580),
    "boundary_f1": ("max", 0.7100),
    "centerline_f1": ("max", 0.5000),
    "endpoint_mae": ("min", 0.0800),
    "path_order_mae_on_mask": ("min", 0.2050),
    "path_order_pairwise_acc": ("max", 0.6800),
    "segment_order_mae": ("min", 0.2150),
    "segment_order_pairwise_acc": ("max", 0.6600),
}


def hard_score(metrics: dict[str, float]) -> float:
    """A stricter score than model8 composite: it rewards planner gains but penalizes geometry collapse."""
    return (
        1.20 * metrics["model8_composite_score"]
        + 0.80 * metrics["centerline_f1"]
        + 0.55 * metrics["boundary_f1"]
        + 0.70 * metrics["path_order_pairwise_acc"]
        + 0.70 * metrics["segment_order_pairwise_acc"]
        - 0.80 * metrics["endpoint_mae"]
        - 0.55 * metrics["path_order_mae_on_mask"]
        - 0.55 * metrics["segment_order_mae"]
    )


def metric_is_better(name: str, value: float, best: float | None) -> bool:
    mode, _ = HARD_THRESHOLDS.get(name, ("max", 0.0))
    if best is None:
        return True
    if mode == "min":
        return value < best
    return value > best


def freeze_geometry_backbone(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("planner_head.") or name.startswith("segment_head.")


def hard_status(metrics: dict[str, float]) -> dict[str, object]:
    checks = {}
    passed = 0
    for name, (mode, threshold) in HARD_THRESHOLDS.items():
        value = float(metrics[name])
        ok = value <= threshold if mode == "min" else value >= threshold
        checks[name] = {"value": value, "threshold": threshold, "mode": mode, "pass": ok}
        passed += int(ok)
    return {"passed": passed, "total": len(HARD_THRESHOLDS), "checks": checks, "hard_pass": passed == len(HARD_THRESHOLDS)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    dataset_dir: Path,
    history: list[dict[str, float]],
    init_report: dict[str, object],
    epoch: int,
    score: float,
    tag: str,
) -> None:
    payload = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": "Model9HardJointSegmentPlanner",
        "base_channels": args.base_channels,
        "geometry_channels": 13,
        "target_channels": MODEL8_CHANNELS,
        "stitch_type_names": STITCH_TYPE_NAMES,
        "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
        "dataset_dir": str(dataset_dir),
        "args": vars(args),
        "history": history,
        "init_checkpoint": init_report,
        "best_epoch": epoch,
        "best_score": score,
        "best_tag": tag,
        "hard_thresholds": HARD_THRESHOLDS,
        "model9_changes": {
            "goal": "hard-standard continuation from model8 with stricter geometry/planner gates",
            "training": "geometry backbone frozen" if args.freeze_geometry_backbone else "encoder frozen, decoder + geometry_head + planner_head + segment_head trainable",
            "loss_focus": [
                "higher centerline/boundary preservation",
                "higher endpoint weight",
                "stronger path and segment ranking",
                "longer low-learning-rate continuation",
            ],
        },
    }
    torch.save(payload, path)


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
    if args.freeze_geometry_backbone:
        freeze_geometry_backbone(model)
    elif args.freeze_encoder:
        freeze_encoder(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        json.dumps(
            {
                "model": "model9_hard",
                "channels": MODEL8_CHANNELS,
                "trainable_parameters": trainable,
                "frozen_parameters": frozen,
                "device": str(device),
                "hard_thresholds": HARD_THRESHOLDS,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_plateau_patience, min_lr=args.min_lr
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, float]] = []
    best_hard_score = -float("inf")
    best_hard_epoch = 0
    best_metric_values: dict[str, float | None] = {name: None for name in HARD_THRESHOLDS}
    best_metric_paths: dict[str, str] = {}

    for epoch in range(1, args.epochs + 1):
        print(f"model9 hard epoch {epoch}/{args.epochs} start", flush=True)
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
                print(
                    json.dumps(
                        {"epoch": epoch, "batch": batch_index, "seen": seen, "train_loss_so_far": total_loss / max(1, seen)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        metrics = evaluate(model, val_loader, device, class_weights, args, epoch)
        score = hard_score(metrics)
        scheduler.step(score)
        status = hard_status(metrics)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            **{f"train_{key}": value / max(1, seen) for key, value in part_totals.items()},
            **{f"val_{key}": value for key, value in metrics.items()},
            "hard_score": float(score),
            "hard_passed": int(status["passed"]),
            "hard_total": int(status["total"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        if score > best_hard_score:
            best_hard_score = float(score)
            best_hard_epoch = epoch
            save_checkpoint(
                output_dir / "best_model9_hard.pt",
                model,
                args,
                dataset_dir,
                copy.deepcopy(history),
                init_report,
                epoch,
                best_hard_score,
                "hard_score",
            )

        for metric_name in HARD_THRESHOLDS:
            value = float(metrics[metric_name])
            if metric_is_better(metric_name, value, best_metric_values[metric_name]):
                best_metric_values[metric_name] = value
                metric_path = output_dir / f"best_{metric_name}.pt"
                save_checkpoint(
                    metric_path,
                    model,
                    args,
                    dataset_dir,
                    copy.deepcopy(history),
                    init_report,
                    epoch,
                    value,
                    metric_name,
                )
                best_metric_paths[metric_name] = str(metric_path)

        print(json.dumps({**record, "hard_status": status}, ensure_ascii=False), flush=True)

    checkpoint_path = output_dir / "model9_hard_final.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        args,
        dataset_dir,
        history,
        init_report,
        args.epochs,
        float(history[-1]["hard_score"]) if history else 0.0,
        "final",
    )
    save_model8_previews(model, val_dataset, device, output_dir / "predictions", args.preview_count)

    best_record = max(history, key=lambda item: item["hard_score"]) if history else {}
    summary = {
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(output_dir / "best_model9_hard.pt"),
        "best_hard_epoch": best_hard_epoch,
        "best_hard_score": best_hard_score,
        "best_metric_paths": best_metric_paths,
        "best_metric_values": best_metric_values,
        "hard_thresholds": HARD_THRESHOLDS,
        "best_record": best_record,
        "best_hard_status": hard_status({key[4:]: value for key, value in best_record.items() if key.startswith("val_")}) if best_record else {},
        "target_channels": MODEL8_CHANNELS,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hard-standard continuation training for geometry/planner embroidery model.")
    parser.add_argument("--dataset-dir", default="datasets/dataset2_collection_20260512_geometry_graph")
    parser.add_argument("--output-dir", default="models/model9_hard_e40_from_model8")
    parser.add_argument("--split-field", default="canonical_split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--init-checkpoint", default="models/model8_joint_segment_e20_from_e2/best_model8_joint_segment.pt")
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--min-lr", type=float, default=8e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--freeze-geometry-backbone", action="store_true")
    parser.add_argument("--mask-pos-weight", type=float, default=5.0)
    parser.add_argument("--boundary-pos-weight", type=float, default=9.0)
    parser.add_argument("--centerline-pos-weight", type=float, default=16.0)
    parser.add_argument("--endpoint-pos-weight", type=float, default=24.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.75)
    parser.add_argument("--axis-loss-weight", type=float, default=0.8)
    parser.add_argument("--axis-min-conf-weight", type=float, default=0.05)
    parser.add_argument("--axis-conf-threshold", type=float, default=0.12)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.3)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.75)
    parser.add_argument("--centerline-loss-weight", type=float, default=0.95)
    parser.add_argument("--stitch-type-loss-weight", type=float, default=0.75)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.9)
    parser.add_argument("--path-order-loss-weight", type=float, default=0.9)
    parser.add_argument("--path-order-start-weight", type=float, default=0.55)
    parser.add_argument("--path-order-warmup-epochs", type=int, default=6)
    parser.add_argument("--path-rank-loss-weight", type=float, default=0.2)
    parser.add_argument("--rank-samples", type=int, default=256)
    parser.add_argument("--rank-pairs", type=int, default=512)
    parser.add_argument("--rank-eval-samples", type=int, default=320)
    parser.add_argument("--rank-eval-pairs", type=int, default=768)
    parser.add_argument("--rank-target-margin", type=float, default=0.035)
    parser.add_argument("--rank-temperature", type=float, default=0.1)
    parser.add_argument("--mask-dice-weight", type=float, default=0.15)
    parser.add_argument("--boundary-dice-weight", type=float, default=0.3)
    parser.add_argument("--centerline-dice-weight", type=float, default=0.35)
    parser.add_argument("--endpoint-focal-gamma", type=float, default=1.6)
    parser.add_argument("--stitch-type-focal-gamma", type=float, default=1.2)
    parser.add_argument("--endpoint-eval-threshold", type=float, default=0.25)
    parser.add_argument("--segment-mask-pos-weight", type=float, default=4.0)
    parser.add_argument("--segment-boundary-pos-weight", type=float, default=11.0)
    parser.add_argument("--segment-mask-loss-weight", type=float, default=0.25)
    parser.add_argument("--segment-boundary-loss-weight", type=float, default=0.3)
    parser.add_argument("--segment-order-loss-weight", type=float, default=0.5)
    parser.add_argument("--segment-rank-loss-weight", type=float, default=0.2)
    parser.add_argument("--segment-dice-weight", type=float, default=0.15)
    parser.add_argument("--segment-boundary-focal-gamma", type=float, default=1.6)
    parser.add_argument("--class-weight-scan-items", type=int, default=0)
    parser.add_argument("--lr-plateau-patience", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=30)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
