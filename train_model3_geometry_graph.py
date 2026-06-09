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
from train_rich_stitch_label import axis_preview, binary_f1, binary_iou, gray, tensor_to_image
from train_rich_stitch_planner import STITCH_TYPE_NAMES, STITCH_TYPE_RGB, load_stitch_type, stitch_type_preview


TARGET_CHANNELS = [
    "mask",
    "density",
    "axis_x",
    "axis_y",
    "axis_confidence",
    "boundary",
    "centerline",
    "stitch_type_no_stitch",
    "stitch_type_running",
    "stitch_type_satin",
    "stitch_type_fill",
    "entry_endpoint_heatmap",
    "exit_endpoint_heatmap",
    "path_order",
]


class GeometryGraphDataset(Dataset):
    def __init__(self, dataset_dir: Path, split_field: str, split: str, augment: bool = False):
        self.dataset_dir = dataset_dir
        self.augment = augment
        summary_path = dataset_dir / "summary_dataset2.json"
        self.source_dir = dataset_dir
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_dir = summary.get("source_dir")
            if source_dir:
                self.source_dir = Path(source_dir)
        manifest_path = dataset_dir / "manifest_dataset2.csv"
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row[split_field] == split]
        if not self.rows:
            raise ValueError(f"no rows found for {split_field}={split} in {manifest_path}")

    def data_path(self, row: dict[str, str], key: str) -> Path:
        path = self.dataset_dir / row[key]
        if path.exists():
            return path
        return self.source_dir / row[key]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        row = self.rows[index]
        image = Image.open(self.data_path(row, "input_png")).convert("RGB")
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1))

        mask_arr = (np.asarray(Image.open(self.data_path(row, "mask_png")).convert("L"), dtype=np.float32) / 255.0)[None, :, :]
        density_arr = np.load(self.data_path(row, "density_npy")).astype(np.float32)[None, :, :]
        axis_arr = np.load(self.data_path(row, "direction_axis_npy")).astype(np.float32)
        conf_arr = np.load(self.data_path(row, "direction_confidence_npy")).astype(np.float32)[None, :, :]
        boundary_arr = (np.asarray(Image.open(self.data_path(row, "boundary_png")).convert("L"), dtype=np.float32) / 255.0)[None, :, :]
        centerline_arr = (np.asarray(Image.open(self.data_path(row, "centerline_png")).convert("L"), dtype=np.float32) / 255.0)[None, :, :]
        axis_valid_arr = (np.asarray(Image.open(self.data_path(row, "axis_valid_mask_png")).convert("L"), dtype=np.float32) / 255.0)[None, :, :]
        endpoint_arr = np.load(self.data_path(row, "endpoint_heatmap_npy")).astype(np.float32)
        path_order_arr = np.load(self.data_path(row, "path_order_npy")).astype(np.float32)[None, :, :]
        stitch_type = load_stitch_type(self.data_path(row, "stitch_type_heuristic_png"))

        if self.augment:
            if random.random() < 0.5:
                image_arr = image_arr[:, ::-1, :].copy()
                mask_arr = mask_arr[:, :, ::-1].copy()
                density_arr = density_arr[:, :, ::-1].copy()
                axis_arr = axis_arr[:, :, ::-1].copy()
                axis_arr[0] *= -1.0
                conf_arr = conf_arr[:, :, ::-1].copy()
                boundary_arr = boundary_arr[:, :, ::-1].copy()
                centerline_arr = centerline_arr[:, :, ::-1].copy()
                axis_valid_arr = axis_valid_arr[:, :, ::-1].copy()
                endpoint_arr = endpoint_arr[:, :, ::-1].copy()
                path_order_arr = path_order_arr[:, :, ::-1].copy()
                stitch_type = stitch_type[:, ::-1].copy()
            if random.random() < 0.25:
                image_arr = image_arr[::-1, :, :].copy()
                mask_arr = mask_arr[:, ::-1, :].copy()
                density_arr = density_arr[:, ::-1, :].copy()
                axis_arr = axis_arr[:, ::-1, :].copy()
                axis_arr[1] *= -1.0
                conf_arr = conf_arr[:, ::-1, :].copy()
                boundary_arr = boundary_arr[:, ::-1, :].copy()
                centerline_arr = centerline_arr[:, ::-1, :].copy()
                axis_valid_arr = axis_valid_arr[:, ::-1, :].copy()
                endpoint_arr = endpoint_arr[:, ::-1, :].copy()
                path_order_arr = path_order_arr[:, ::-1, :].copy()
                stitch_type = stitch_type[::-1, :].copy()

        structural = np.concatenate(
            [mask_arr, density_arr, axis_arr, conf_arr, boundary_arr, centerline_arr],
            axis=0,
        )
        target = {
            "structural": torch.from_numpy(structural),
            "stitch_type": torch.from_numpy(stitch_type),
            "axis_valid": torch.from_numpy(axis_valid_arr),
            "endpoint": torch.from_numpy(endpoint_arr),
            "path_order": torch.from_numpy(path_order_arr),
        }
        return image_tensor, target, row["pair_id"]


def compute_stitch_type_weights(dataset: GeometryGraphDataset, max_items: int = 0) -> torch.Tensor:
    counts = np.zeros(len(STITCH_TYPE_NAMES), dtype=np.float64)
    total = len(dataset) if max_items <= 0 else min(len(dataset), max_items)
    for index in range(total):
        row = dataset.rows[index]
        labels = load_stitch_type(dataset.data_path(row, "stitch_type_heuristic_png"))
        counts += np.bincount(labels.reshape(-1), minlength=len(STITCH_TYPE_NAMES))
    freq = counts / max(1.0, counts.sum())
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    weights = weights / weights.mean()
    weights[0] *= 0.45
    return torch.tensor(weights, dtype=torch.float32)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    if gamma <= 0:
        return bce.mean()
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob).clamp(1e-4, 1.0 - 1e-4)
    return ((1.0 - pt) ** gamma * bce).mean()


def batch_loss(
    output: torch.Tensor,
    target: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    structural = target["structural"]
    stitch_type = target["stitch_type"]
    axis_valid = target["axis_valid"]
    endpoint = target["endpoint"]
    path_order = target["path_order"]

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
    endpoint_logits = output[:, 11:13]
    path_order_pred = torch.sigmoid(output[:, 13:14]) if output.shape[1] > 13 else None

    mask_pos_weight = torch.tensor([args.mask_pos_weight], device=output.device)
    boundary_pos_weight = torch.tensor([args.boundary_pos_weight], device=output.device)
    centerline_pos_weight = torch.tensor([args.centerline_pos_weight], device=output.device)
    endpoint_pos_weight = torch.tensor([args.endpoint_pos_weight], device=output.device)

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits,
        target_mask,
        pos_weight=mask_pos_weight,
    )
    if args.mask_dice_weight > 0:
        mask_loss = mask_loss + args.mask_dice_weight * dice_loss_from_logits(mask_logits, target_mask)
    density_loss = F.l1_loss(density_pred, target_density)
    axis_weight = (axis_valid * target_conf.clamp_min(args.axis_min_conf_weight)).expand_as(target_axis)
    axis_loss = (((axis_pred - target_axis) ** 2) * axis_weight).sum() / axis_weight.sum().clamp_min(1.0)
    confidence_loss = F.l1_loss(conf_pred, target_conf)
    boundary_loss = F.binary_cross_entropy_with_logits(
        boundary_logits,
        target_boundary,
        pos_weight=boundary_pos_weight,
    )
    if args.boundary_dice_weight > 0:
        boundary_loss = boundary_loss + args.boundary_dice_weight * dice_loss_from_logits(boundary_logits, target_boundary)
    centerline_loss = F.binary_cross_entropy_with_logits(
        centerline_logits,
        target_centerline,
        pos_weight=centerline_pos_weight,
    )
    if args.centerline_dice_weight > 0:
        centerline_loss = centerline_loss + args.centerline_dice_weight * dice_loss_from_logits(centerline_logits, target_centerline)
    stitch_type_loss = F.cross_entropy(stitch_logits, stitch_type, weight=class_weights.to(output.device))
    endpoint_loss = focal_bce_with_logits(
        endpoint_logits,
        endpoint,
        pos_weight=endpoint_pos_weight,
        gamma=args.endpoint_focal_gamma,
    )
    if path_order_pred is not None:
        order_weight = target_mask.clamp_min(0.0)
        path_order_loss = (torch.abs(path_order_pred - path_order) * order_weight).sum() / order_weight.sum().clamp_min(1.0)
    else:
        path_order_loss = torch.zeros((), device=output.device)

    total = (
        mask_loss
        + args.density_loss_weight * density_loss
        + args.axis_loss_weight * axis_loss
        + args.confidence_loss_weight * confidence_loss
        + args.boundary_loss_weight * boundary_loss
        + args.centerline_loss_weight * centerline_loss
        + args.stitch_type_loss_weight * stitch_type_loss
        + args.endpoint_loss_weight * endpoint_loss
        + args.path_order_loss_weight * path_order_loss
    )
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "density_loss": float(density_loss.detach().cpu()),
        "axis_valid_loss": float(axis_loss.detach().cpu()),
        "confidence_loss": float(confidence_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "centerline_loss": float(centerline_loss.detach().cpu()),
        "stitch_type_loss": float(stitch_type_loss.detach().cpu()),
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "path_order_loss": float(path_order_loss.detach().cpu()),
    }


def class_iou(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> torch.Tensor:
    pred_bool = pred == class_id
    target_bool = target == class_id
    intersection = (pred_bool & target_bool).sum(dim=(1, 2)).float()
    union = (pred_bool | target_bool).sum(dim=(1, 2)).float().clamp_min(1.0)
    return intersection / union


def endpoint_peak_recall(endpoint_pred: torch.Tensor, endpoint_target: torch.Tensor, threshold: float) -> torch.Tensor:
    pred = endpoint_pred > threshold
    target = endpoint_target > threshold
    hits = (pred & target).reshape(endpoint_pred.shape[0], -1).sum(dim=1).float()
    total = target.reshape(endpoint_pred.shape[0], -1).sum(dim=1).float().clamp_min(1.0)
    return hits / total


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
        "axis_mae_on_valid": 0.0,
        "confidence_mae": 0.0,
        "boundary_f1": 0.0,
        "centerline_f1": 0.0,
        "stitch_type_acc_on_mask": 0.0,
        "stitch_type_miou_nonzero": 0.0,
        "endpoint_mae": 0.0,
        "endpoint_peak_recall": 0.0,
        "path_order_mae_on_mask": 0.0,
    }
    count = 0
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = {key: value.to(device) for key, value in target.items()}
            output = model(image)
            loss, _ = batch_loss(output, target, class_weights, args)

            structural = target["structural"]
            stitch_type = target["stitch_type"]
            axis_valid = target["axis_valid"]
            endpoint = target["endpoint"]
            path_order = target["path_order"]

            mask_pred = torch.sigmoid(output[:, 0:1])
            density_pred = torch.sigmoid(output[:, 1:2])
            axis_pred = torch.tanh(output[:, 2:4])
            conf_pred = torch.sigmoid(output[:, 4:5])
            boundary_pred = torch.sigmoid(output[:, 5:6])
            centerline_pred = torch.sigmoid(output[:, 6:7])
            stitch_pred = output[:, 7:11].argmax(dim=1)
            endpoint_pred = torch.sigmoid(output[:, 11:13])
            path_order_pred = torch.sigmoid(output[:, 13:14]) if output.shape[1] > 13 else None

            valid = axis_valid.expand_as(axis_pred) > 0.5
            axis_denom = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
            axis_mae = ((axis_pred - structural[:, 2:4]).abs() * valid).sum(dim=(1, 2, 3)) / axis_denom

            batch_size = image.shape[0]
            mask_target_bool = stitch_type > 0
            on_mask_count = mask_target_bool.reshape(batch_size, -1).sum(dim=1).clamp_min(1)
            on_mask_acc = ((stitch_pred == stitch_type) & mask_target_bool).reshape(batch_size, -1).sum(dim=1).float() / on_mask_count
            nonzero_iou = torch.stack([class_iou(stitch_pred, stitch_type, class_id) for class_id in (1, 2, 3)], dim=0).mean(dim=0)
            endpoint_recall = endpoint_peak_recall(endpoint_pred, endpoint, args.endpoint_eval_threshold)
            if path_order_pred is not None:
                order_mask = structural[:, 0:1].clamp_min(0.0)
                order_mae = (torch.abs(path_order_pred - path_order) * order_mask).sum(dim=(1, 2, 3)) / order_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            else:
                order_mae = torch.zeros(batch_size, device=device)

            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["mask_iou"] += float(binary_iou(mask_pred, structural[:, 0:1]).sum().detach().cpu())
            totals["density_mae"] += float((density_pred - structural[:, 1:2]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["axis_mae_on_valid"] += float(axis_mae.sum().detach().cpu())
            totals["confidence_mae"] += float((conf_pred - structural[:, 4:5]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["boundary_f1"] += float(binary_f1(boundary_pred, structural[:, 5:6]).sum().detach().cpu())
            totals["centerline_f1"] += float(binary_f1(centerline_pred, structural[:, 6:7]).sum().detach().cpu())
            totals["stitch_type_acc_on_mask"] += float(on_mask_acc.sum().detach().cpu())
            totals["stitch_type_miou_nonzero"] += float(nonzero_iou.sum().detach().cpu())
            totals["endpoint_mae"] += float((endpoint_pred - endpoint).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["endpoint_peak_recall"] += float(endpoint_recall.sum().detach().cpu())
            totals["path_order_mae_on_mask"] += float(order_mae.sum().detach().cpu())
            count += batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def heatmap_preview(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (arr * 255).astype(np.uint8)
    rgb[..., 1] = (np.sqrt(arr) * 190).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_prediction_previews(
    model: nn.Module,
    dataset: GeometryGraphDataset,
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
        pred_stitch = output[7:11].argmax(dim=0).numpy()
        endpoint_pred = torch.sigmoid(output[11:13]).numpy()
        path_order_pred = torch.sigmoid(output[13]).numpy() if output.shape[0] > 13 else np.zeros_like(target["structural"][0].numpy())
        items = [
            ("input", input_img),
            ("true mask", gray(target["structural"][0].numpy())),
            ("pred mask", gray(torch.sigmoid(output[0]).numpy())),
            ("true axis/valid", axis_preview(target["structural"][2:4].numpy(), target["axis_valid"][0].numpy())),
            ("pred axis/conf", axis_preview(torch.tanh(output[2:4]).numpy(), torch.sigmoid(output[4]).numpy())),
            ("true stitch", stitch_type_preview(target["stitch_type"].numpy())),
            ("pred stitch", stitch_type_preview(pred_stitch)),
            ("true boundary", gray(target["structural"][5].numpy())),
            ("pred boundary", gray(torch.sigmoid(output[5]).numpy())),
            ("true endpoints", heatmap_preview(np.maximum(target["endpoint"][0].numpy(), target["endpoint"][1].numpy()))),
            ("pred endpoints", heatmap_preview(np.maximum(endpoint_pred[0], endpoint_pred[1]))),
            ("true order", gray(target["path_order"][0].numpy())),
            ("pred order", gray(path_order_pred)),
        ]
        width, height = input_img.size
        panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for i, (label, item) in enumerate(items):
            x = i * width
            draw.text((x + 6, 6), label, fill=(0, 0, 0))
            panel.paste(item, (x, 24))
        panel.save(output_dir / f"{pair_id}_model3_prediction.png")


def load_partial_initializer(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
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
        "loaded_tail": loaded[-8:],
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = GeometryGraphDataset(dataset_dir, args.split_field, args.train_split, augment=args.augment)
    val_dataset = GeometryGraphDataset(dataset_dir, args.split_field, args.val_split, augment=False)
    class_weights = compute_stitch_type_weights(train_dataset, args.class_weight_scan_items)
    print(json.dumps({"target_channels": TARGET_CHANNELS, "class_weights": class_weights.tolist()}, ensure_ascii=False), flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = choose_device(args.cpu)
    model = TinyUNet(out_channels=len(TARGET_CHANNELS), base_channels=args.base_channels).to(device)
    init_report: dict[str, object] = {"used": False}
    if args.init_checkpoint:
        init_report = load_partial_initializer(model, Path(args.init_checkpoint), device)
        print(json.dumps({"init_checkpoint": init_report}, ensure_ascii=False), flush=True)
    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
            min_lr=args.min_lr,
        )
    else:
        scheduler = None
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} start", flush=True)
        model.train()
        total_loss = 0.0
        part_totals = {
            "mask_loss": 0.0,
            "density_loss": 0.0,
            "axis_valid_loss": 0.0,
            "confidence_loss": 0.0,
            "boundary_loss": 0.0,
            "centerline_loss": 0.0,
            "stitch_type_loss": 0.0,
            "endpoint_loss": 0.0,
            "path_order_loss": 0.0,
        }
        seen = 0
        for batch_index, (image, target, _) in enumerate(train_loader, start=1):
            image = image.to(device, non_blocking=True)
            target = {key: value.to(device, non_blocking=True) for key, value in target.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(image)
                loss, parts = batch_loss(output, target, class_weights, args)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

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
            + val_metrics["endpoint_peak_recall"]
            - val_metrics["axis_mae_on_valid"]
            - val_metrics["endpoint_mae"]
            - val_metrics["path_order_mae_on_mask"]
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
            epochs_without_improvement = 0
        elif score <= best_score + args.early_stop_min_delta:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(float(score))
            else:
                scheduler.step()
        record["lr"] = float(optimizer.param_groups[0]["lr"])
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
            break

    checkpoint_path = output_dir / "model3_geometry_graph_unet.pt"
    best_checkpoint_path = output_dir / "best_model3_geometry_graph_unet.pt"
    payload = {
        "model_state": model.state_dict(),
        "model": "TinyUNet",
        "base_channels": args.base_channels,
        "target_channels": TARGET_CHANNELS,
        "stitch_type_names": STITCH_TYPE_NAMES,
        "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
        "dataset_dir": str(dataset_dir),
        "args": vars(args),
        "history": history,
        "init_checkpoint": init_report,
        "training_optimizations": {
            "augment": args.augment,
            "amp": amp_enabled,
            "scheduler": args.scheduler,
            "mask_dice_weight": args.mask_dice_weight,
            "boundary_dice_weight": args.boundary_dice_weight,
            "centerline_dice_weight": args.centerline_dice_weight,
            "endpoint_focal_gamma": args.endpoint_focal_gamma,
        },
    }
    torch.save(payload, checkpoint_path)
    if best_state is not None:
        payload_best = dict(payload)
        payload_best["model_state"] = best_state
        payload_best["best_epoch"] = best_epoch
        payload_best["best_composite_score"] = best_score
        torch.save(payload_best, best_checkpoint_path)
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
        "init_checkpoint": init_report,
        "history": history,
        "note": "Model3 uses dataset2 canonical split, axis_valid_mask, and entry/exit endpoint heatmap supervision.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model: {checkpoint_path}")
    print(f"saved best model: {best_checkpoint_path}")
    print(f"saved metrics: {output_dir / 'metrics.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train model3 geometry/graph U-Net from dataset2.")
    parser.add_argument("--dataset-dir", default="datasets/dataset2_geometry_graph")
    parser.add_argument("--output-dir", default="models/model3_geometry_graph_unet")
    parser.add_argument("--split-field", default="canonical_split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-pos-weight", type=float, default=5.0)
    parser.add_argument("--boundary-pos-weight", type=float, default=8.0)
    parser.add_argument("--centerline-pos-weight", type=float, default=12.0)
    parser.add_argument("--endpoint-pos-weight", type=float, default=18.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.7)
    parser.add_argument("--axis-loss-weight", type=float, default=0.55)
    parser.add_argument("--axis-min-conf-weight", type=float, default=0.05)
    parser.add_argument("--confidence-loss-weight", type=float, default=0.3)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--centerline-loss-weight", type=float, default=0.5)
    parser.add_argument("--stitch-type-loss-weight", type=float, default=0.7)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.55)
    parser.add_argument("--path-order-loss-weight", type=float, default=0.45)
    parser.add_argument("--mask-dice-weight", type=float, default=0.15)
    parser.add_argument("--boundary-dice-weight", type=float, default=0.25)
    parser.add_argument("--centerline-dice-weight", type=float, default=0.25)
    parser.add_argument("--endpoint-focal-gamma", type=float, default=1.5)
    parser.add_argument("--endpoint-eval-threshold", type=float, default=0.25)
    parser.add_argument("--class-weight-scan-items", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-plateau-patience", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
