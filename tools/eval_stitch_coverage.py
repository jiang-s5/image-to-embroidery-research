from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pyembroidery import COMMAND_MASK, STITCH, read_dst


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def mm_to_px(point_mm: tuple[float, float], shape: tuple[int, int], target_width_mm: float) -> tuple[int, int]:
    height, width = shape
    scale = target_width_mm / max(1, width)
    x = int(round(point_mm[0] / scale + width / 2.0))
    y = int(round(point_mm[1] / scale + height / 2.0))
    return x, y


def render_stitch_mask(
    dst_path: Path,
    shape: tuple[int, int],
    target_width_mm: float,
    line_radius_px: int,
) -> np.ndarray:
    pattern = read_dst(str(dst_path))
    canvas = np.zeros(shape, dtype=np.uint8)
    last: tuple[float, float] | None = None
    thickness = max(1, line_radius_px * 2 + 1)
    for x_raw, y_raw, command_raw in pattern.stitches:
        point = (float(x_raw) / 10.0, float(y_raw) / 10.0)
        command = command_raw & COMMAND_MASK
        if command == STITCH and last is not None:
            start_px = mm_to_px(last, shape, target_width_mm)
            end_px = mm_to_px(point, shape, target_width_mm)
            cv2.line(canvas, start_px, end_px, 255, thickness=thickness, lineType=cv2.LINE_AA)
        last = point
    return (canvas > 0).astype(np.uint8)


def coverage_metrics(
    dst_path: Path,
    target_mask_path: Path,
    target_width_mm: float = 90.0,
    line_radius_px: int = 2,
) -> dict[str, Any]:
    target = load_mask(target_mask_path)
    stitch = render_stitch_mask(dst_path, target.shape, target_width_mm, line_radius_px)
    target_pixels = int(target.sum())
    stitch_pixels = int(stitch.sum())
    overlap = int((target & stitch).sum())
    off_target = int((stitch & (1 - target)).sum())
    coverage = overlap / max(1, target_pixels)
    precision = overlap / max(1, stitch_pixels)
    return {
        "dst_path": str(dst_path),
        "target_mask_path": str(target_mask_path),
        "target_pixels": target_pixels,
        "stitch_pixels": stitch_pixels,
        "overlap_pixels": overlap,
        "off_target_pixels": off_target,
        "coverage_ratio": round(coverage, 6),
        "stitch_precision_ratio": round(precision, 6),
        "line_radius_px": line_radius_px,
        "target_width_mm": target_width_mm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rendered stitch coverage against a target mask.")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--line-radius-px", type=int, default=2)
    args = parser.parse_args()

    report = coverage_metrics(
        Path(args.dst),
        Path(args.mask),
        target_width_mm=args.target_width_mm,
        line_radius_px=args.line_radius_px,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
