from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from train_image_to_stitch_label import ConvBlock, choose_device
from train_model3_geometry_graph import (
    TARGET_CHANNELS,
    GeometryGraphDataset,
    compute_stitch_type_weights,
    dice_loss_from_logits,
    endpoint_peak_recall,
    save_prediction_previews,
)
from train_rich_stitch_label import binary_f1, binary_iou
from train_rich_stitch_planner import STITCH_TYPE_NAMES, STITCH_TYPE_RGB


class Model7GeometryPlannerCascade(nn.Module):
    """Shared geometry encoder with separated geometry and planner heads."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        geometry_channels: int = 13,
        detach_planner_geometry: bool = True,
    ):
        super().__init__()
        b = base_channels
        self.geometry_channels = geometry_channels
        self.detach_planner_geometry = detach_planner_geometry
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
        self.geometry_head = nn.Conv2d(b, geometry_channels, kernel_size=1)
        self.planner_head = nn.Sequential(
            nn.Conv2d(b + geometry_channels, b, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, 1, kernel_size=1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        d3 = self.up3(z)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        return self.dec1(torch.cat([d1, e1], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        geometry = self.geometry_head(features)
        if self.detach_planner_geometry:
            planner_input = torch.cat([features.detach(), geometry.detach()], dim=1)
        else:
            planner_input = torch.cat([features, geometry], dim=1)
        path_order = self.planner_head(planner_input)
        return torch.cat([geometry, path_order], dim=1)


def multiclass_focal_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    if gamma <= 0:
        return ce.mean()
    prob = torch.softmax(logits, dim=1)
    pt = prob.gather(1, target[:, None]).squeeze(1).clamp(1e-4, 1.0 - 1e-4)
    return (((1.0 - pt) ** gamma) * ce).mean()


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


def undirected_axis_loss(
    axis_pred: torch.Tensor,
    target_axis: torch.Tensor,
    axis_valid: torch.Tensor,
    target_conf: torch.Tensor,
    min_conf_weight: float,
    conf_threshold: float,
) -> torch.Tensor:
    pred_norm = F.normalize(axis_pred, dim=1, eps=1e-6)
    target_norm = F.normalize(target_axis, dim=1, eps=1e-6)
    dot = (pred_norm * target_norm).sum(dim=1, keepdim=True).abs()
    conf = target_conf.clamp_min(min_conf_weight)
    valid = (axis_valid > 0.5).float() * (target_conf >= conf_threshold).float()
    weight = valid * conf
    return ((1.0 - dot) * weight).sum() / weight.sum().clamp_min(1.0)


def pairwise_path_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    samples: int,
    pairs: int,
    target_margin: float,
    temperature: float,
) -> torch.Tensor:
    losses = []
    flat_pred = pred[:, 0].reshape(pred.shape[0], -1)
    flat_target = target[:, 0].reshape(target.shape[0], -1)
    flat_mask = mask[:, 0].reshape(mask.shape[0], -1)
    for batch_index in range(pred.shape[0]):
        idx = torch.nonzero(flat_mask[batch_index] > 0.5, as_tuple=False).flatten()
        if idx.numel() < 3:
            continue
        if idx.numel() > samples:
            perm = torch.randperm(idx.numel(), device=pred.device)[:samples]
            idx = idx[perm]
        point_count = idx.numel()
        left = torch.randint(0, point_count, (pairs,), device=pred.device)
        right = torch.randint(0, point_count, (pairs,), device=pred.device)
        target_diff = flat_target[batch_index, idx[left]] - flat_target[batch_index, idx[right]]
        valid = target_diff.abs() > target_margin
        if not bool(valid.any()):
            continue
        sign = target_diff[valid].sign()
        pred_diff = flat_pred[batch_index, idx[left[valid]]] - flat_pred[batch_index, idx[right[valid]]]
        losses.append(F.softplus(-sign * pred_diff / max(temperature, 1e-4)).mean())
    if not losses:
        return torch.zeros((), device=pred.device)
    return torch.stack(losses).mean()


def pairwise_path_rank_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    samples: int,
    pairs: int,
    target_margin: float,
) -> torch.Tensor:
    scores = []
    flat_pred = pred[:, 0].reshape(pred.shape[0], -1)
    flat_target = target[:, 0].reshape(target.shape[0], -1)
    flat_mask = mask[:, 0].reshape(mask.shape[0], -1)
    for batch_index in range(pred.shape[0]):
        idx = torch.nonzero(flat_mask[batch_index] > 0.5, as_tuple=False).flatten()
        if idx.numel() < 3:
            scores.append(torch.zeros((), device=pred.device))
            continue
        if idx.numel() > samples:
            idx = idx[torch.randperm(idx.numel(), device=pred.device)[:samples]]
        point_count = idx.numel()
        left = torch.randint(0, point_count, (pairs,), device=pred.device)
        right = torch.randint(0, point_count, (pairs,), device=pred.device)
        target_diff = flat_target[batch_index, idx[left]] - flat_target[batch_index, idx[right]]
        valid = target_diff.abs() > target_margin
        if not bool(valid.any()):
            scores.append(torch.zeros((), device=pred.device))
            continue
        pred_diff = flat_pred[batch_index, idx[left[valid]]] - flat_pred[batch_index, idx[right[valid]]]
        scores.append((pred_diff.sign() == target_diff[valid].sign()).float().mean())
    return torch.stack(scores)


def endpoint_top1_error_norm(endpoint_pred: torch.Tensor, endpoint_target: torch.Tensor) -> torch.Tensor:
    b, c, h, w = endpoint_pred.shape
    pred_flat = endpoint_pred.reshape(b, c, -1).argmax(dim=2)
    target_flat = endpoint_target.reshape(b, c, -1).argmax(dim=2)
    px = (pred_flat % w).float()
    py = (pred_flat // w).float()
    tx = (target_flat % w).float()
    ty = (target_flat // w).float()
    distance = torch.sqrt((px - tx) ** 2 + (py - ty) ** 2)
    return distance.mean(dim=1) / float((h * h + w * w) ** 0.5)


def class_iou(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> torch.Tensor:
    pred_bool = pred == class_id
    target_bool = target == class_id
    intersection = (pred_bool & target_bool).sum(dim=(1, 2)).float()
    union = (pred_bool | target_bool).sum(dim=(1, 2)).float().clamp_min(1.0)
    return intersection / union


def scheduled_path_order_weight(args: argparse.Namespace, epoch: int) -> float:
    if args.path_order_warmup_epochs <= 1:
        return args.path_order_loss_weight
    progress = min(1.0, max(0.0, (epoch - 1) / float(args.path_order_warmup_epochs - 1)))
    return args.path_order_start_weight + progress * (args.path_order_loss_weight - args.path_order_start_weight)


def batch_loss(
    output: torch.Tensor,
    target: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
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
    path_order_pred = torch.sigmoid(output[:, 13:14])

    mask_pos_weight = torch.tensor([args.mask_pos_weight], device=output.device)
    boundary_pos_weight = torch.tensor([args.boundary_pos_weight], device=output.device)
    centerline_pos_weight = torch.tensor([args.centerline_pos_weight], device=output.device)
    endpoint_pos_weight = torch.tensor([args.endpoint_pos_weight], device=output.device)

    mask_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask, pos_weight=mask_pos_weight)
    if args.mask_dice_weight > 0:
        mask_loss = mask_loss + args.mask_dice_weight * dice_loss_from_logits(mask_logits, target_mask)
    density_loss = F.l1_loss(density_pred, target_density)
    axis_loss = undirected_axis_loss(
        axis_pred,
        target_axis,
        axis_valid,
        target_conf,
        args.axis_min_conf_weight,
        args.axis_conf_threshold,
    )
    confidence_loss = F.l1_loss(conf_pred, target_conf)
    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, target_boundary, pos_weight=boundary_pos_weight)
    if args.boundary_dice_weight > 0:
        boundary_loss = boundary_loss + args.boundary_dice_weight * dice_loss_from_logits(boundary_logits, target_boundary)
    centerline_loss = F.binary_cross_entropy_with_logits(centerline_logits, target_centerline, pos_weight=centerline_pos_weight)
    if args.centerline_dice_weight > 0:
        centerline_loss = centerline_loss + args.centerline_dice_weight * dice_loss_from_logits(centerline_logits, target_centerline)
    stitch_type_loss = multiclass_focal_ce(
        stitch_logits,
        stitch_type,
        class_weights.to(output.device),
        args.stitch_type_focal_gamma,
    )
    endpoint_loss = focal_bce_with_logits(endpoint_logits, endpoint, endpoint_pos_weight, args.endpoint_focal_gamma)

    order_weight = target_mask.clamp_min(0.0)
    path_order_l1 = (torch.abs(path_order_pred - path_order) * order_weight).sum() / order_weight.sum().clamp_min(1.0)
    path_order_rank = pairwise_path_rank_loss(
        path_order_pred,
        path_order,
        target_mask,
        args.rank_samples,
        args.rank_pairs,
        args.rank_target_margin,
        args.rank_temperature,
    )
    path_order_loss = path_order_l1 + args.path_rank_loss_weight * path_order_rank
    path_order_weight = scheduled_path_order_weight(args, epoch)

    total = (
        mask_loss
        + args.density_loss_weight * density_loss
        + args.axis_loss_weight * axis_loss
        + args.confidence_loss_weight * confidence_loss
        + args.boundary_loss_weight * boundary_loss
        + args.centerline_loss_weight * centerline_loss
        + args.stitch_type_loss_weight * stitch_type_loss
        + args.endpoint_loss_weight * endpoint_loss
        + path_order_weight * path_order_loss
    )
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "density_loss": float(density_loss.detach().cpu()),
        "axis_undirected_loss": float(axis_loss.detach().cpu()),
        "confidence_loss": float(confidence_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "centerline_loss": float(centerline_loss.detach().cpu()),
        "stitch_type_focal_loss": float(stitch_type_loss.detach().cpu()),
        "endpoint_loss": float(endpoint_loss.detach().cpu()),
        "path_order_l1_loss": float(path_order_l1.detach().cpu()),
        "path_order_rank_loss": float(path_order_rank.detach().cpu()),
        "path_order_loss": float(path_order_loss.detach().cpu()),
        "path_order_weight": float(path_order_weight),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mask_iou": 0.0,
        "density_mae": 0.0,
        "axis_mae_on_valid": 0.0,
        "axis_undirected_error": 0.0,
        "confidence_mae": 0.0,
        "boundary_f1": 0.0,
        "centerline_f1": 0.0,
        "stitch_type_acc_on_mask": 0.0,
        "stitch_type_miou_nonzero": 0.0,
        "endpoint_mae": 0.0,
        "endpoint_top1_error_norm": 0.0,
        "endpoint_peak_recall": 0.0,
        "path_order_mae_on_mask": 0.0,
        "path_order_pairwise_acc": 0.0,
    }
    count = 0
    with torch.no_grad():
        for image, target, _ in loader:
            image = image.to(device)
            target = {key: value.to(device) for key, value in target.items()}
            output = model(image)
            loss, _ = batch_loss(output, target, class_weights, args, epoch)

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
            path_order_pred = torch.sigmoid(output[:, 13:14])

            valid = axis_valid.expand_as(axis_pred) > 0.5
            axis_denom = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
            axis_mae = ((axis_pred - structural[:, 2:4]).abs() * valid).sum(dim=(1, 2, 3)) / axis_denom
            pred_norm = F.normalize(axis_pred, dim=1, eps=1e-6)
            target_norm = F.normalize(structural[:, 2:4], dim=1, eps=1e-6)
            axis_err = (1.0 - (pred_norm * target_norm).sum(dim=1, keepdim=True).abs())
            axis_err = (axis_err * axis_valid).sum(dim=(1, 2, 3)) / axis_valid.sum(dim=(1, 2, 3)).clamp_min(1.0)

            batch_size = image.shape[0]
            mask_target_bool = stitch_type > 0
            on_mask_count = mask_target_bool.reshape(batch_size, -1).sum(dim=1).clamp_min(1)
            on_mask_acc = ((stitch_pred == stitch_type) & mask_target_bool).reshape(batch_size, -1).sum(dim=1).float() / on_mask_count
            nonzero_iou = torch.stack([class_iou(stitch_pred, stitch_type, class_id) for class_id in (1, 2, 3)], dim=0).mean(dim=0)
            endpoint_recall = endpoint_peak_recall(endpoint_pred, endpoint, args.endpoint_eval_threshold)
            endpoint_top1 = endpoint_top1_error_norm(endpoint_pred, endpoint)
            order_mask = structural[:, 0:1].clamp_min(0.0)
            order_mae = (torch.abs(path_order_pred - path_order) * order_mask).sum(dim=(1, 2, 3)) / order_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            order_pairwise_acc = pairwise_path_rank_accuracy(
                path_order_pred,
                path_order,
                order_mask,
                args.rank_eval_samples,
                args.rank_eval_pairs,
                args.rank_target_margin,
            )

            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["mask_iou"] += float(binary_iou(mask_pred, structural[:, 0:1]).sum().detach().cpu())
            totals["density_mae"] += float((density_pred - structural[:, 1:2]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["axis_mae_on_valid"] += float(axis_mae.sum().detach().cpu())
            totals["axis_undirected_error"] += float(axis_err.sum().detach().cpu())
            totals["confidence_mae"] += float((conf_pred - structural[:, 4:5]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["boundary_f1"] += float(binary_f1(boundary_pred, structural[:, 5:6]).sum().detach().cpu())
            totals["centerline_f1"] += float(binary_f1(centerline_pred, structural[:, 6:7]).sum().detach().cpu())
            totals["stitch_type_acc_on_mask"] += float(on_mask_acc.sum().detach().cpu())
            totals["stitch_type_miou_nonzero"] += float(nonzero_iou.sum().detach().cpu())
            totals["endpoint_mae"] += float((endpoint_pred - endpoint).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
            totals["endpoint_top1_error_norm"] += float(endpoint_top1.sum().detach().cpu())
            totals["endpoint_peak_recall"] += float(endpoint_recall.sum().detach().cpu())
            totals["path_order_mae_on_mask"] += float(order_mae.sum().detach().cpu())
            totals["path_order_pairwise_acc"] += float(order_pairwise_acc.sum().detach().cpu())
            count += batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def load_model7_initializer(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, object]:
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
        "source_target_channels": checkpoint.get("target_channels", []),
        "loaded_count": len(loaded),
        "skipped_count": len(skipped),
        "loaded_tail": loaded[-10:],
    }


def freeze_geometry_modules(model: Model7GeometryPlannerCascade) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("planner_head.")


def keep_frozen_geometry_eval(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if name and not name.startswith("planner_head"):
            module.eval()


def composite_score(metrics: dict[str, float]) -> float:
    return float(
        metrics["mask_iou"]
        + metrics["boundary_f1"]
        + metrics["centerline_f1"]
        + metrics["stitch_type_miou_nonzero"]
        + metrics["path_order_pairwise_acc"]
        - metrics["axis_undirected_error"]
        - metrics["endpoint_top1_error_norm"]
        - metrics["path_order_mae_on_mask"]
    )


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
    print(json.dumps({"model7_target_channels": TARGET_CHANNELS, "class_weights": class_weights.tolist()}, ensure_ascii=False), flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = choose_device(args.cpu)
    model = Model7GeometryPlannerCascade(
        base_channels=args.base_channels,
        detach_planner_geometry=args.detach_planner_geometry,
    ).to(device)
    init_report: dict[str, object] = {"used": False}
    if args.init_checkpoint:
        init_report = load_model7_initializer(model, Path(args.init_checkpoint), device)
        print(json.dumps({"init_checkpoint": init_report}, ensure_ascii=False), flush=True)
    if args.freeze_geometry:
        freeze_geometry_modules(model)
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        print(json.dumps({"freeze_geometry": True, "trainable_parameters": trainable, "frozen_parameters": frozen}, ensure_ascii=False), flush=True)

    if args.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
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
        print(f"model7 epoch {epoch}/{args.epochs} start", flush=True)
        model.train()
        if args.freeze_geometry:
            keep_frozen_geometry_eval(model)
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

            batch_size = image.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            for key, value in parts.items():
                part_totals[key] = part_totals.get(key, 0.0) + value * batch_size
            seen += batch_size
            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "seen": seen,
                            "train_loss_so_far": total_loss / max(1, seen),
                            "path_order_weight": scheduled_path_order_weight(args, epoch),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, class_weights, args, epoch)
        score = composite_score(val_metrics)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            **{f"train_{key}": value / max(1, seen) for key, value in part_totals.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_composite_score": score,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        elif score <= best_score + args.early_stop_min_delta:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
            break

    checkpoint_path = output_dir / "model7_geometry_planner_cascade.pt"
    best_checkpoint_path = output_dir / "best_model7_geometry_planner_cascade.pt"
    payload = {
        "model_state": model.state_dict(),
        "model": "Model7GeometryPlannerCascade",
        "base_channels": args.base_channels,
        "geometry_channels": 13,
        "detach_planner_geometry": args.detach_planner_geometry,
        "target_channels": TARGET_CHANNELS,
        "stitch_type_names": STITCH_TYPE_NAMES,
        "stitch_type_rgb": STITCH_TYPE_RGB.astype(int).tolist(),
        "dataset_dir": str(dataset_dir),
        "args": vars(args),
        "history": history,
        "init_checkpoint": init_report,
        "model7_changes": {
            "architecture": "geometry head + detached planner head cascade",
            "freeze_geometry": args.freeze_geometry,
            "axis_loss": "confidence-gated undirected angular loss",
            "stitch_type_loss": "weighted focal cross entropy",
            "path_order_loss": "masked L1 + pairwise ranking loss",
            "path_order_schedule": "warmup from path_order_start_weight to path_order_loss_weight",
            "endpoint_metrics": "heatmap MAE plus top-1 localization error",
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
        "checkpoint": str(checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_composite_score": best_score,
        "target_channels": TARGET_CHANNELS,
        "init_checkpoint": init_report,
        "history": history,
        "note": "Model7 separates geometry prediction from path-order planning and adds rank/undirected-axis losses.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved model7: {checkpoint_path}")
    print(f"saved best model7: {best_checkpoint_path}")
    print(f"saved metrics: {output_dir / 'metrics.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train model7 geometry-planner cascade from collection geometry graph data.")
    parser.add_argument("--dataset-dir", default="datasets/dataset2_collection_20260512_geometry_graph")
    parser.add_argument("--output-dir", default="models/model7_geometry_planner_cascade_e5")
    parser.add_argument("--split-field", default="canonical_split")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--init-checkpoint", default="models/model6_collection_path_order_e5/best_model3_geometry_graph_unet.pt")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument("--path-order-loss-weight", type=float, default=0.45)
    parser.add_argument("--path-order-start-weight", type=float, default=0.18)
    parser.add_argument("--path-order-warmup-epochs", type=int, default=3)
    parser.add_argument("--path-rank-loss-weight", type=float, default=0.08)
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
    parser.add_argument("--class-weight-scan-items", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--detach-planner-geometry", dest="detach_planner_geometry", action="store_true", default=True)
    parser.add_argument("--no-detach-planner-geometry", dest="detach_planner_geometry", action="store_false")
    parser.add_argument("--freeze-geometry", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="plateau")
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
