from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pyembroidery import END, JUMP, STITCH, EmbPattern, EmbThread, write_dst

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_executability import analyze_file
from eval_stitch_coverage import coverage_metrics
from score_unified import score_metrics


Pixel = tuple[int, int]

NEIGHBORS_8 = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_binary(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return (cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0).astype(np.uint8)


def pixel_to_mm(pixel: Pixel, shape: tuple[int, int], target_width_mm: float) -> tuple[float, float]:
    y, x = pixel
    height, width = shape
    scale = target_width_mm / max(1, width)
    return ((x - width / 2.0) * scale, (y - height / 2.0) * scale)


def add_thread(pattern: EmbPattern, color: tuple[int, int, int]) -> None:
    thread = EmbThread()
    thread.set_color(*color)
    pattern.add_thread(thread)


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


def component_pixels(skeleton: np.ndarray, min_pixels: int) -> list[list[Pixel]]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    components: list[list[Pixel]] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_pixels:
            continue
        ys, xs = np.where(labels == label)
        components.append(list(zip(ys.tolist(), xs.tolist())))
    return components


def build_graph(pixels: list[Pixel]) -> dict[Pixel, list[Pixel]]:
    pixel_set = set(pixels)
    graph: dict[Pixel, list[Pixel]] = {}
    for y, x in pixels:
        neighbors: list[Pixel] = []
        for dx, dy in NEIGHBORS_8:
            candidate = (y + dy, x + dx)
            if candidate in pixel_set:
                neighbors.append(candidate)
        graph[(y, x)] = sorted(neighbors)
    return graph


def choose_start(
    graph: dict[Pixel, list[Pixel]],
    current_mm: tuple[float, float] | None,
    shape: tuple[int, int],
    target_width_mm: float,
) -> Pixel:
    endpoints = [pixel for pixel, neighbors in graph.items() if len(neighbors) <= 1]
    candidates = endpoints or list(graph.keys())
    if current_mm is None:
        return min(candidates)
    return min(candidates, key=lambda pixel: math.dist(pixel_to_mm(pixel, shape, target_width_mm), current_mm))


def dfs_edge_walk(graph: dict[Pixel, list[Pixel]], start: Pixel) -> list[Pixel]:
    path: list[Pixel] = [start]
    seen_edges: set[tuple[Pixel, Pixel]] = set()

    def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    def visit(node: Pixel) -> None:
        neighbors = sorted(graph[node], key=lambda pixel: (len(graph[pixel]) != 1, pixel))
        for neighbor in neighbors:
            key = edge_key(node, neighbor)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            path.append(neighbor)
            visit(neighbor)
            path.append(node)

    visit(start)
    return path


def generate_line_stroke_dst(
    skeleton_path: Path,
    mask_path: Path,
    output_dst: Path,
    target_width_mm: float,
    max_stitch_mm: float,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    short_gap_mm: float,
    short_gap_inside_fraction: float,
    connector_dilation_px: int,
    min_component_pixels: int,
    thread_rgb: tuple[int, int, int] = (20, 80, 150),
) -> dict[str, Any]:
    skeleton = load_binary(skeleton_path)
    mask = load_binary(mask_path)
    connector_mask = dilate(mask, connector_dilation_px)
    components = [(pixels, build_graph(pixels)) for pixels in component_pixels(skeleton, min_component_pixels)]

    pattern = EmbPattern()
    add_thread(pattern, thread_rgb)
    current_mm: tuple[float, float] | None = None
    stitch_segments = 0
    jumps = 0
    safe_connects = 0
    short_gap_connects = 0
    rejected_connects = 0
    remaining = [(pixels, graph) for pixels, graph in components if graph]

    while remaining:
        if current_mm is None:
            next_index = max(range(len(remaining)), key=lambda index: len(remaining[index][0]))
        else:
            next_index = min(
                range(len(remaining)),
                key=lambda index: math.dist(
                    current_mm,
                    pixel_to_mm(
                        choose_start(remaining[index][1], current_mm, skeleton.shape, target_width_mm),
                        skeleton.shape,
                        target_width_mm,
                    ),
                ),
            )
        _pixels, graph = remaining.pop(next_index)
        start = choose_start(graph, current_mm, skeleton.shape, target_width_mm)
        walk = dfs_edge_walk(graph, start)
        if not walk:
            continue
        first_mm = pixel_to_mm(walk[0], skeleton.shape, target_width_mm)
        if current_mm is None:
            add_abs(pattern, JUMP, first_mm)
            jumps += 1
        else:
            distance = math.dist(current_mm, first_mm)
            inside = segment_inside_fraction(connector_mask, current_mm, first_mm, target_width_mm)
            short_inside = segment_inside_fraction(mask, current_mm, first_mm, target_width_mm)
            can_connect = distance <= max_connect_mm and inside >= min_connect_inside_fraction
            can_short_gap = distance <= short_gap_mm and inside >= short_gap_inside_fraction and short_inside > 0.0
            if can_connect or can_short_gap:
                stitch_segments += add_segment(pattern, current_mm, first_mm, max_stitch_mm)
                safe_connects += 1
                short_gap_connects += 1 if can_short_gap and not can_connect else 0
            else:
                add_abs(pattern, JUMP, first_mm)
                jumps += 1
                rejected_connects += 1
        current_mm = first_mm
        for pixel in walk[1:]:
            point_mm = pixel_to_mm(pixel, skeleton.shape, target_width_mm)
            stitch_segments += add_segment(pattern, current_mm, point_mm, max_stitch_mm)
            current_mm = point_mm

    if current_mm is None:
        current_mm = (0.0, 0.0)
    add_abs(pattern, END, current_mm)
    output_dst.parent.mkdir(parents=True, exist_ok=True)
    write_dst(pattern, str(output_dst))
    return {
        "mode": "line_domain_stroke_trace",
        "skeleton_path": str(skeleton_path),
        "mask_path": str(mask_path),
        "output_dst": str(output_dst),
        "components": len(components),
        "jump_commands_added": jumps,
        "safe_component_connects": safe_connects,
        "short_gap_connects": short_gap_connects,
        "rejected_component_connects": rejected_connects,
        "stitch_segments_added": stitch_segments,
        "target_width_mm": target_width_mm,
        "max_stitch_mm": max_stitch_mm,
        "max_connect_mm": max_connect_mm,
        "min_connect_inside_fraction": min_connect_inside_fraction,
        "short_gap_mm": short_gap_mm,
        "short_gap_inside_fraction": short_gap_inside_fraction,
        "connector_dilation_px": connector_dilation_px,
        "min_component_pixels": min_component_pixels,
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "unified_loss",
        "jump_count",
        "trim_count",
        "off_mask_stitch_length_mm",
        "visible_connector_count",
        "coverage_ratio",
        "stitch_precision_ratio",
    ]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "hard_fail": sum(1 for row in rows if row.get("quality_level") == "hard_fail"),
    }
    for key in keys:
        summary[f"mean_{key}"] = round(mean(safe_float(row.get(key)) for row in rows), 6) if rows else 0.0
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_name", ""))].append(row)
    summary["by_source"] = {
        source: {
            "samples": len(items),
            "mean_unified_loss": round(mean(safe_float(row.get("unified_loss")) for row in items), 6),
            "mean_jump_count": round(mean(safe_float(row.get("jump_count")) for row in items), 6),
            "mean_coverage_ratio": round(mean(safe_float(row.get("coverage_ratio")) for row in items), 6),
            "mean_stitch_precision_ratio": round(mean(safe_float(row.get("stitch_precision_ratio")) for row in items), 6),
        }
        for source, items in sorted(by_source.items())
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a line-domain stroke-tracing candidate for M2.56.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-selection", required=True)
    parser.add_argument("--fallback-output-dir", default="")
    parser.add_argument("--line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--max-connect-mm", type=float, default=18.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.70)
    parser.add_argument("--short-gap-mm", type=float, default=3.0)
    parser.add_argument("--short-gap-inside-fraction", type=float, default=0.35)
    parser.add_argument("--connector-dilation-px", type=int, default=3)
    parser.add_argument("--min-component-pixels", type=int, default=4)
    parser.add_argument("--line-radius-px", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(dataset_dir / "manifest.csv")
    fallback_rows = {row["sample_id"]: row for row in read_csv(Path(args.fallback_selection))}
    line_sources = {item.strip() for item in args.line_sources.split(",") if item.strip()}

    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for sample in manifest:
        sample_id = sample["sample_id"]
        sample_out = output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        dst_path = sample_out / "prediction.dst"
        report: dict[str, Any]
        if sample.get("source_name", "") in line_sources:
            report = generate_line_stroke_dst(
                dataset_dir / sample["skeleton_path"],
                dataset_dir / sample["mask_path"],
                dst_path,
                target_width_mm=args.target_width_mm,
                max_stitch_mm=args.max_stitch_mm,
                max_connect_mm=args.max_connect_mm,
                min_connect_inside_fraction=args.min_connect_inside_fraction,
                short_gap_mm=args.short_gap_mm,
                short_gap_inside_fraction=args.short_gap_inside_fraction,
                connector_dilation_px=args.connector_dilation_px,
                min_component_pixels=args.min_component_pixels,
            )
            pred = analyze_file(
                dst_path,
                max_stitch_mm=4.0,
                high_risk_jump_mm=8.0,
                mask_path=dataset_dir / sample["mask_path"],
                target_width_mm=args.target_width_mm,
            )
            coverage = coverage_metrics(
                dst_path,
                dataset_dir / sample["mask_path"],
                target_width_mm=args.target_width_mm,
                line_radius_px=args.line_radius_px,
            )
            score = score_metrics(pred)
            (sample_out / "eval_executability.json").write_text(json.dumps({"pred": pred}, ensure_ascii=False, indent=2), encoding="utf-8")
            (sample_out / "coverage_report.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
            quality = score["quality_level"]
            metric_row = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "mode": "line_domain_stroke_trace",
                "quality_level": quality,
                "unified_loss": score["unified_loss"],
                "exec_score": score["exec_score"],
                "visual_risk": score["visual_risk"],
                "jump_count": pred.get("jump_count", ""),
                "trim_count": pred.get("trim_count", ""),
                "jump_path_mm": pred.get("jump_path_mm", ""),
                "off_mask_stitch_length_mm": pred.get("off_mask_stitch_length_mm", ""),
                "visible_connector_count": pred.get("visible_connector_count", ""),
                "visible_connector_length_mm": pred.get("visible_connector_length_mm", ""),
            }
            coverage_row = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "coverage_ratio": coverage["coverage_ratio"],
                "stitch_precision_ratio": coverage["stitch_precision_ratio"],
                "target_pixels": coverage["target_pixels"],
                "stitch_pixels": coverage["stitch_pixels"],
                "overlap_pixels": coverage["overlap_pixels"],
                "off_target_pixels": coverage["off_target_pixels"],
            }
        else:
            fallback = fallback_rows[sample_id]
            metric_row = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "mode": "fallback_selection",
                "quality_level": fallback.get("quality_level", ""),
                "unified_loss": fallback.get("unified_loss", ""),
                "exec_score": "",
                "visual_risk": "",
                "jump_count": fallback.get("jump_count", ""),
                "trim_count": fallback.get("trim_count", ""),
                "jump_path_mm": "",
                "off_mask_stitch_length_mm": fallback.get("off_mask_stitch_length_mm", ""),
                "visible_connector_count": fallback.get("visible_connector_count", ""),
                "visible_connector_length_mm": "",
            }
            coverage_row = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "coverage_ratio": fallback.get("coverage_ratio", ""),
                "stitch_precision_ratio": fallback.get("stitch_precision_ratio", ""),
            }
            report = {"mode": "fallback_selection", "fallback_candidate": fallback.get("chosen_candidate", "")}
            if args.fallback_output_dir:
                source_dir = Path(args.fallback_output_dir) / sample_id
                for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
                    source = source_dir / filename
                    if source.exists():
                        shutil.copy2(source, sample_out / filename)
        (sample_out / "generator_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_rows.append(metric_row)
        coverage_rows.append(coverage_row)

    write_csv(metric_rows, output_dir / "source_aware_hybrid_rows.csv")
    write_csv(coverage_rows, output_dir / "coverage_rows.csv")
    summary = summarize_metric_rows(
        [
            {
                **metric_row,
                "coverage_ratio": coverage_row.get("coverage_ratio", ""),
                "stitch_precision_ratio": coverage_row.get("stitch_precision_ratio", ""),
            }
            for metric_row, coverage_row in zip(metric_rows, coverage_rows)
        ]
    )
    summary["line_sources"] = sorted(line_sources)
    summary["candidate_args"] = {
        "max_connect_mm": args.max_connect_mm,
        "min_connect_inside_fraction": args.min_connect_inside_fraction,
        "short_gap_mm": args.short_gap_mm,
        "short_gap_inside_fraction": args.short_gap_inside_fraction,
        "connector_dilation_px": args.connector_dilation_px,
        "min_component_pixels": args.min_component_pixels,
    }
    (output_dir / "line_domain_stroke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
