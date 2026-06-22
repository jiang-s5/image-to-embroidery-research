from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pyembroidery import END, JUMP, STITCH, EmbPattern, EmbThread, write_dst


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def add_thread(pattern: EmbPattern, color: tuple[int, int, int]) -> None:
    thread = EmbThread()
    thread.set_color(*color)
    pattern.add_thread(thread)


def pixel_to_mm_xy(x: float, y: float, shape: tuple[int, int], target_width_mm: float) -> tuple[float, float]:
    height, width = shape
    scale = target_width_mm / max(1, width)
    return ((x - width / 2.0) * scale, (y - height / 2.0) * scale)


def add_abs(pattern: EmbPattern, command: int, point_mm: tuple[float, float]) -> None:
    pattern.add_stitch_absolute(command, int(round(point_mm[0] * 10)), int(round(point_mm[1] * 10)))


def add_segment(pattern: EmbPattern, start_mm: tuple[float, float], end_mm: tuple[float, float], max_stitch_mm: float) -> int:
    distance = math.dist(start_mm, end_mm)
    if distance <= 1e-6:
        return 0
    steps = max(1, int(math.ceil(distance / max(0.1, max_stitch_mm))))
    for index in range(1, steps + 1):
        t = index / steps
        point = (
            start_mm[0] + (end_mm[0] - start_mm[0]) * t,
            start_mm[1] + (end_mm[1] - start_mm[1]) * t,
        )
        add_abs(pattern, STITCH, point)
    return steps


def segment_inside_fraction(
    mask: np.ndarray,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    target_width_mm: float,
    sample_step_mm: float = 0.5,
) -> float:
    distance = math.dist(start_mm, end_mm)
    if distance <= 1e-6:
        return 1.0
    height, width = mask.shape
    scale = target_width_mm / max(1, width)
    steps = max(1, int(math.ceil(distance / max(0.1, sample_step_mm))))
    inside = 0
    total = steps + 1
    for index in range(total):
        t = index / steps
        x_mm = start_mm[0] + (end_mm[0] - start_mm[0]) * t
        y_mm = start_mm[1] + (end_mm[1] - start_mm[1]) * t
        x = int(round(x_mm / scale + width / 2.0))
        y = int(round(y_mm / scale + height / 2.0))
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            inside += 1
    return inside / total


def row_runs(component_mask: np.ndarray, y: int, min_run_px: int) -> list[tuple[int, int]]:
    xs = np.where(component_mask[y] > 0)[0]
    if xs.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for raw_x in xs[1:]:
        x = int(raw_x)
        if x == prev + 1:
            prev = x
            continue
        if prev - start + 1 >= min_run_px:
            runs.append((start, prev))
        start = prev = x
    if prev - start + 1 >= min_run_px:
        runs.append((start, prev))
    return runs


def component_fill_rows(
    component_mask: np.ndarray,
    row_spacing_px: int,
    min_run_px: int,
    reverse_first: bool = False,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ys, xs = np.where(component_mask > 0)
    if ys.size == 0:
        return []
    y_min = int(ys.min())
    y_max = int(ys.max())
    rows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    reverse = reverse_first
    for y in range(y_min, y_max + 1, max(1, row_spacing_px)):
        runs = row_runs(component_mask, y, min_run_px)
        if reverse:
            runs = list(reversed(runs))
        for x0, x1 in runs:
            if reverse:
                rows.append(((x1, y), (x0, y)))
            else:
                rows.append(((x0, y), (x1, y)))
            reverse = not reverse
    return rows


def component_order(labels: np.ndarray, stats: np.ndarray, min_component_pixels: int) -> list[int]:
    labels_out = []
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_pixels:
            labels_out.append(label)
    return sorted(labels_out, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)


def generate_mask_fill_dst(
    mask_path: Path,
    output_dst: Path,
    connector_mask_path: Path | None = None,
    target_width_mm: float = 90.0,
    row_spacing_mm: float = 1.2,
    max_stitch_mm: float = 3.2,
    max_connect_mm: float = 6.0,
    min_connect_inside_fraction: float = 0.95,
    min_component_pixels: int = 64,
    min_run_mm: float = 1.0,
    thread_rgb: tuple[int, int, int] = (30, 120, 200),
) -> dict[str, Any]:
    mask = load_mask(mask_path)
    connector_mask = load_mask(connector_mask_path) if connector_mask_path is not None and connector_mask_path.exists() else mask
    height, width = mask.shape
    scale = target_width_mm / max(1, width)
    row_spacing_px = max(1, int(round(row_spacing_mm / max(scale, 1e-6))))
    min_run_px = max(1, int(round(min_run_mm / max(scale, 1e-6))))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    pattern = EmbPattern()
    add_thread(pattern, thread_rgb)

    current_mm: tuple[float, float] | None = None
    jumps = 0
    safe_connects = 0
    rejected_connects = 0
    stitch_segments = 0
    fill_rows = 0
    components_used = 0

    for component_index, label in enumerate(component_order(labels, stats, min_component_pixels)):
        component_mask = (labels == label).astype(np.uint8)
        rows = component_fill_rows(component_mask, row_spacing_px, min_run_px, reverse_first=component_index % 2 == 1)
        if not rows:
            continue
        components_used += 1
        for start_px, end_px in rows:
            start_mm = pixel_to_mm_xy(start_px[0], start_px[1], mask.shape, target_width_mm)
            end_mm = pixel_to_mm_xy(end_px[0], end_px[1], mask.shape, target_width_mm)
            if current_mm is None:
                add_abs(pattern, JUMP, start_mm)
                jumps += 1
            else:
                distance = math.dist(current_mm, start_mm)
                inside = segment_inside_fraction(connector_mask, current_mm, start_mm, target_width_mm)
                if distance <= max_connect_mm and inside >= min_connect_inside_fraction:
                    stitch_segments += add_segment(pattern, current_mm, start_mm, max_stitch_mm)
                    safe_connects += 1
                else:
                    add_abs(pattern, JUMP, start_mm)
                    jumps += 1
                    rejected_connects += 1
            stitch_segments += add_segment(pattern, start_mm, end_mm, max_stitch_mm)
            fill_rows += 1
            current_mm = end_mm

    if current_mm is None:
        current_mm = (0.0, 0.0)
        add_abs(pattern, JUMP, current_mm)
        jumps += 1
    add_abs(pattern, END, current_mm)
    output_dst.parent.mkdir(parents=True, exist_ok=True)
    write_dst(pattern, str(output_dst))
    return {
        "mask_path": str(mask_path),
        "connector_mask_path": str(connector_mask_path) if connector_mask_path else "",
        "output_dst": str(output_dst),
        "components_total": int(count - 1),
        "components_used": components_used,
        "fill_rows": fill_rows,
        "jump_commands_added": jumps,
        "safe_connects": safe_connects,
        "rejected_connects": rejected_connects,
        "stitch_segments_added": stitch_segments,
        "target_width_mm": target_width_mm,
        "row_spacing_mm": row_spacing_mm,
        "row_spacing_px": row_spacing_px,
        "max_stitch_mm": max_stitch_mm,
        "max_connect_mm": max_connect_mm,
        "min_connect_inside_fraction": min_connect_inside_fraction,
        "min_component_pixels": min_component_pixels,
        "min_run_mm": min_run_mm,
        "min_run_px": min_run_px,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a simple segment-level fill DST from a binary mask.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--connector-mask", default="")
    parser.add_argument("--output-dst", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--row-spacing-mm", type=float, default=1.2)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--max-connect-mm", type=float, default=6.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.95)
    parser.add_argument("--min-component-pixels", type=int, default=64)
    parser.add_argument("--min-run-mm", type=float, default=1.0)
    args = parser.parse_args()

    report = generate_mask_fill_dst(
        Path(args.mask),
        Path(args.output_dst),
        connector_mask_path=Path(args.connector_mask) if args.connector_mask else None,
        target_width_mm=args.target_width_mm,
        row_spacing_mm=args.row_spacing_mm,
        max_stitch_mm=args.max_stitch_mm,
        max_connect_mm=args.max_connect_mm,
        min_connect_inside_fraction=args.min_connect_inside_fraction,
        min_component_pixels=args.min_component_pixels,
        min_run_mm=args.min_run_mm,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
