from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pyembroidery import COLOR_CHANGE, COMMAND_MASK, END, JUMP, STITCH, TRIM, read_dst, read_pes


COMMAND_NAMES = {
    STITCH: "stitch",
    JUMP: "jump",
    TRIM: "trim",
    COLOR_CHANGE: "color_change",
    END: "end",
}


def read_pattern(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pes":
        return read_pes(str(path))
    if suffix == ".dst":
        return read_dst(str(path))
    raise ValueError(f"Unsupported embroidery file: {path}")


def command_name(command: int) -> str:
    return COMMAND_NAMES.get(command & COMMAND_MASK, f"unsupported_{command & COMMAND_MASK}")


def load_mask(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.float32) / 255.0) >= 0.5


def mm_to_mask_xy(point: tuple[float, float], mask_shape: tuple[int, int], target_width_mm: float) -> tuple[float, float]:
    height, width = mask_shape
    scale_mm = target_width_mm / max(1, width)
    return (point[0] / scale_mm + width / 2.0, point[1] / scale_mm + height / 2.0)


def mask_value(mask: np.ndarray, point_mm: tuple[float, float], target_width_mm: float) -> bool:
    x, y = mm_to_mask_xy(point_mm, mask.shape, target_width_mm)
    ix = int(round(x))
    iy = int(round(y))
    if ix < 0 or iy < 0 or iy >= mask.shape[0] or ix >= mask.shape[1]:
        return False
    return bool(mask[iy, ix])


def off_mask_fraction(
    mask: np.ndarray | None,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    distance_mm: float,
    target_width_mm: float,
    sample_step_mm: float,
) -> float:
    if mask is None or distance_mm <= 1e-6:
        return 0.0
    steps = max(1, int(math.ceil(distance_mm / max(0.1, sample_step_mm))))
    off_mask = 0
    total = steps + 1
    for index in range(total):
        t = index / max(1, steps)
        point = (
            start_mm[0] + (end_mm[0] - start_mm[0]) * t,
            start_mm[1] + (end_mm[1] - start_mm[1]) * t,
        )
        if not mask_value(mask, point, target_width_mm):
            off_mask += 1
    return off_mask / total


def analyze_file(
    path: Path,
    max_stitch_mm: float,
    high_risk_jump_mm: float,
    mask_path: Path | None = None,
    target_width_mm: float = 90.0,
    visible_connector_min_mm: float = 1.5,
    visible_connector_off_mask_ratio: float = 0.65,
    mask_sample_step_mm: float = 1.0,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "path": str(path),
        "round_trip_parse_success": False,
        "error": "",
    }
    try:
        pattern = read_pattern(path)
    except Exception as exc:
        metrics["error"] = str(exc)
        return metrics
    mask = load_mask(mask_path)

    command_counts: dict[str, int] = {}
    last = None
    stitch_count = 0
    jump_count = 0
    trim_count = 0
    color_change_count = 0
    unsupported_count = 0
    illegal_long_stitches = 0
    high_risk_jumps = 0
    total_stitch_mm = 0.0
    total_jump_mm = 0.0
    max_stitch_observed = 0.0
    max_jump_observed = 0.0
    lock_tack_like_count = 0
    short_stitch_run = 0
    off_mask_stitch_length_mm = 0.0
    off_mask_stitch_count = 0
    visible_connector_count = 0
    visible_connector_length_mm = 0.0

    for x_raw, y_raw, command_raw in pattern.stitches:
        command = command_raw & COMMAND_MASK
        name = command_name(command)
        command_counts[name] = command_counts.get(name, 0) + 1
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        distance = math.dist(last, (x, y)) if last is not None else 0.0

        if command == STITCH:
            stitch_count += 1
            total_stitch_mm += distance
            max_stitch_observed = max(max_stitch_observed, distance)
            if last is not None and mask is not None:
                off_fraction = off_mask_fraction(
                    mask,
                    last,
                    (x, y),
                    distance,
                    target_width_mm,
                    mask_sample_step_mm,
                )
                off_length = distance * off_fraction
                off_mask_stitch_length_mm += off_length
                if off_fraction > 0.0:
                    off_mask_stitch_count += 1
                if distance >= visible_connector_min_mm and off_fraction >= visible_connector_off_mask_ratio:
                    visible_connector_count += 1
                    visible_connector_length_mm += distance
            if distance > max_stitch_mm:
                illegal_long_stitches += 1
            if 0.05 <= distance <= 1.2:
                short_stitch_run += 1
            else:
                if short_stitch_run >= 3:
                    lock_tack_like_count += 1
                short_stitch_run = 0
        elif command == JUMP:
            jump_count += 1
            total_jump_mm += distance
            max_jump_observed = max(max_jump_observed, distance)
            if distance >= high_risk_jump_mm:
                high_risk_jumps += 1
            if short_stitch_run >= 3:
                lock_tack_like_count += 1
            short_stitch_run = 0
        elif command == TRIM:
            trim_count += 1
            if short_stitch_run >= 3:
                lock_tack_like_count += 1
            short_stitch_run = 0
        elif command == COLOR_CHANGE:
            color_change_count += 1
            short_stitch_run = 0
        elif command != END:
            unsupported_count += 1
            short_stitch_run = 0
        last = (x, y)

    if short_stitch_run >= 3:
        lock_tack_like_count += 1

    metrics.update(
        {
            "round_trip_parse_success": True,
            "commands": len(pattern.stitches),
            "command_counts": command_counts,
            "stitch_count": stitch_count,
            "jump_count": jump_count,
            "trim_count": trim_count,
            "color_change_count": color_change_count,
            "unsupported_command_count": unsupported_count,
            "illegal_long_stitch_count": illegal_long_stitches,
            "high_risk_jump_count": high_risk_jumps,
            "max_stitch_mm": round(max_stitch_observed, 4),
            "max_jump_mm": round(max_jump_observed, 4),
            "stitch_path_mm": round(total_stitch_mm, 4),
            "jump_path_mm": round(total_jump_mm, 4),
            "lock_tack_like_count": lock_tack_like_count,
            "off_mask_stitch_length_mm": round(off_mask_stitch_length_mm, 4),
            "off_mask_stitch_count": off_mask_stitch_count,
            "visible_connector_count": visible_connector_count,
            "visible_connector_length_mm": round(visible_connector_length_mm, 4),
            "mask_path": str(mask_path) if mask_path else "",
            "target_width_mm": target_width_mm,
            "max_stitch_threshold_mm": max_stitch_mm,
            "high_risk_jump_threshold_mm": high_risk_jump_mm,
            "visible_connector_min_mm": visible_connector_min_mm,
            "visible_connector_off_mask_ratio": visible_connector_off_mask_ratio,
        }
    )
    return metrics


def compare_metrics(pred: dict[str, object], gt: dict[str, object]) -> dict[str, object]:
    keys = [
        "stitch_count",
        "jump_count",
        "trim_count",
        "color_change_count",
        "illegal_long_stitch_count",
        "high_risk_jump_count",
        "max_jump_mm",
        "stitch_path_mm",
        "jump_path_mm",
        "off_mask_stitch_length_mm",
        "visible_connector_count",
        "visible_connector_length_mm",
    ]
    comparison: dict[str, object] = {}
    for key in keys:
        p = pred.get(key)
        g = gt.get(key)
        if isinstance(p, (int, float)) and isinstance(g, (int, float)):
            comparison[f"{key}_delta"] = round(float(p) - float(g), 4)
            comparison[f"{key}_ratio"] = round(float(p) / max(1e-6, float(g)), 4)
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DST/PES executability and path-risk metrics.")
    parser.add_argument("--pred", required=True, help="Predicted DST/PES file.")
    parser.add_argument("--gt", default="", help="Optional ground-truth DST/PES file.")
    parser.add_argument("--report", default="", help="Optional JSON report path.")
    parser.add_argument("--mask", default="", help="Optional foreground/export mask used for off-mask stitch metrics.")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--max-stitch-mm", type=float, default=4.0)
    parser.add_argument("--high-risk-jump-mm", type=float, default=8.0)
    parser.add_argument("--visible-connector-min-mm", type=float, default=1.5)
    parser.add_argument("--visible-connector-off-mask-ratio", type=float, default=0.65)
    parser.add_argument("--mask-sample-step-mm", type=float, default=1.0)
    args = parser.parse_args()

    mask_path = Path(args.mask) if args.mask else None
    pred = analyze_file(
        Path(args.pred),
        args.max_stitch_mm,
        args.high_risk_jump_mm,
        mask_path=mask_path,
        target_width_mm=args.target_width_mm,
        visible_connector_min_mm=args.visible_connector_min_mm,
        visible_connector_off_mask_ratio=args.visible_connector_off_mask_ratio,
        mask_sample_step_mm=args.mask_sample_step_mm,
    )
    report: dict[str, object] = {"pred": pred}
    if args.gt:
        gt = analyze_file(
            Path(args.gt),
            args.max_stitch_mm,
            args.high_risk_jump_mm,
            mask_path=mask_path,
            target_width_mm=args.target_width_mm,
            visible_connector_min_mm=args.visible_connector_min_mm,
            visible_connector_off_mask_ratio=args.visible_connector_off_mask_ratio,
            mask_sample_step_mm=args.mask_sample_step_mm,
        )
        report["gt"] = gt
        if pred.get("round_trip_parse_success") and gt.get("round_trip_parse_success"):
            report["comparison"] = compare_metrics(pred, gt)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
