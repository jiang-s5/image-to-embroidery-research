from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from pyembroidery import END, JUMP, EmbPattern, EmbThread, write_dst

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_executability import analyze_file
from eval_stitch_coverage import coverage_metrics
from run_line_domain_grouped_topology_candidate import (
    choose_grouped_component,
    safe_float,
)
from run_line_domain_stroke_candidate import (
    Pixel,
    add_abs,
    add_segment,
    build_graph,
    component_pixels,
    dilate,
    load_binary,
    pixel_to_mm,
    segment_inside_fraction,
)
from run_line_domain_topology_candidate import (
    closed_edge_walk,
    farthest_endpoint_by_graph_distance,
    open_edge_walk,
)
from score_unified import score_metrics


ROUTE_NEIGHBORS: list[tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2.0)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2.0)),
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


def add_thread(pattern: EmbPattern, color: tuple[int, int, int]) -> None:
    thread = EmbThread()
    thread.set_color(*color)
    pattern.add_thread(thread)


def in_bounds(mask, pixel: Pixel) -> bool:
    y, x = pixel
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]


def nearest_passable(mask, pixel: Pixel, radius: int) -> Pixel | None:
    if in_bounds(mask, pixel) and mask[pixel] > 0:
        return pixel
    y0, x0 = pixel
    best: tuple[float, Pixel] | None = None
    for radius_i in range(1, radius + 1):
        for y in range(y0 - radius_i, y0 + radius_i + 1):
            for x in range(x0 - radius_i, x0 + radius_i + 1):
                candidate = (y, x)
                if not in_bounds(mask, candidate) or mask[candidate] <= 0:
                    continue
                distance = math.hypot(y - y0, x - x0)
                if best is None or distance < best[0]:
                    best = (distance, candidate)
        if best is not None:
            return best[1]
    return None


def heuristic(a: Pixel, b: Pixel) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar_mask_path(mask, start: Pixel, end: Pixel, max_cost_px: float) -> list[Pixel]:
    start = nearest_passable(mask, start, radius=3) or start
    end = nearest_passable(mask, end, radius=3) or end
    if not (in_bounds(mask, start) and in_bounds(mask, end)):
        return []
    if mask[start] <= 0 or mask[end] <= 0:
        return []

    open_heap: list[tuple[float, int, Pixel]] = []
    heapq.heappush(open_heap, (heuristic(start, end), 0, start))
    parent: dict[Pixel, Pixel | None] = {start: None}
    g_score: dict[Pixel, float] = {start: 0.0}
    counter = 0
    while open_heap:
        _f_score, _counter, node = heapq.heappop(open_heap)
        current_g = g_score[node]
        if current_g > max_cost_px:
            continue
        if node == end:
            path: list[Pixel] = []
            cursor: Pixel | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            return list(reversed(path))
        for dy, dx, step_cost in ROUTE_NEIGHBORS:
            neighbor = (node[0] + dy, node[1] + dx)
            if not in_bounds(mask, neighbor) or mask[neighbor] <= 0:
                continue
            tentative = current_g + step_cost
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            if tentative + heuristic(neighbor, end) > max_cost_px:
                continue
            parent[neighbor] = node
            g_score[neighbor] = tentative
            counter += 1
            heapq.heappush(open_heap, (tentative + heuristic(neighbor, end), counter, neighbor))
    return []


def pixel_path_length_mm(path: list[Pixel], shape: tuple[int, int], target_width_mm: float) -> float:
    if len(path) <= 1:
        return 0.0
    scale = target_width_mm / max(1, shape[1])
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(a[0] - b[0], a[1] - b[1]) * scale
    return total


def add_pixel_path(
    pattern: EmbPattern,
    path: list[Pixel],
    shape: tuple[int, int],
    target_width_mm: float,
    max_stitch_mm: float,
) -> int:
    if len(path) <= 1:
        return 0
    stitch_segments = 0
    current_mm = pixel_to_mm(path[0], shape, target_width_mm)
    for pixel in path[1:]:
        point_mm = pixel_to_mm(pixel, shape, target_width_mm)
        stitch_segments += add_segment(pattern, current_mm, point_mm, max_stitch_mm)
        current_mm = point_mm
    return stitch_segments


def generate_maskpath_merger_dst(
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
    unsafe_connection_penalty: float,
    inside_penalty: float,
    max_start_candidates: int,
    max_route_mm: float,
    max_route_factor: float,
    route_mask_dilation_px: int,
    thread_rgb: tuple[int, int, int] = (20, 80, 150),
) -> dict[str, Any]:
    skeleton = load_binary(skeleton_path)
    mask = load_binary(mask_path)
    connector_mask = dilate(mask, connector_dilation_px)
    route_mask = dilate(mask, route_mask_dilation_px)
    components = [(pixels, build_graph(pixels)) for pixels in component_pixels(skeleton, min_component_pixels)]

    pattern = EmbPattern()
    add_thread(pattern, thread_rgb)
    current_mm: tuple[float, float] | None = None
    current_pixel: Pixel | None = None
    stitch_segments = 0
    jumps = 0
    safe_connects = 0
    routed_connects = 0
    short_gap_connects = 0
    rejected_connects = 0
    route_failures = 0
    route_length_mm_total = 0.0
    open_components = 0
    remaining = [(pixels, graph) for pixels, graph in components if graph]

    while remaining:
        next_index, start, _features = choose_grouped_component(
            remaining,
            current_mm,
            skeleton.shape,
            target_width_mm,
            mask,
            connector_mask,
            max_connect_mm,
            min_connect_inside_fraction,
            short_gap_mm,
            short_gap_inside_fraction,
            unsafe_connection_penalty,
            inside_penalty,
            max_start_candidates,
        )
        _pixels, graph = remaining.pop(next_index)
        end = farthest_endpoint_by_graph_distance(graph, start)
        walk = open_edge_walk(graph, start, end) if end != start else closed_edge_walk(graph, start)
        open_components += 1 if end != start else 0
        if not walk:
            continue

        first_mm = pixel_to_mm(walk[0], skeleton.shape, target_width_mm)
        if current_mm is None or current_pixel is None:
            add_abs(pattern, JUMP, first_mm)
            jumps += 1
        else:
            distance = math.dist(current_mm, first_mm)
            inside = segment_inside_fraction(connector_mask, current_mm, first_mm, target_width_mm)
            strict_inside = segment_inside_fraction(mask, current_mm, first_mm, target_width_mm)
            can_connect = distance <= max_connect_mm and inside >= min_connect_inside_fraction
            can_short_gap = distance <= short_gap_mm and inside >= short_gap_inside_fraction and strict_inside > 0.0
            if can_connect or can_short_gap:
                stitch_segments += add_segment(pattern, current_mm, first_mm, max_stitch_mm)
                safe_connects += 1
                short_gap_connects += 1 if can_short_gap and not can_connect else 0
            else:
                scale = target_width_mm / max(1, skeleton.shape[1])
                straight_px = max(1.0, distance / scale)
                max_cost_px = min(max_route_mm / scale, straight_px * max_route_factor)
                route = astar_mask_path(route_mask, current_pixel, walk[0], max_cost_px=max_cost_px)
                route_length_mm = pixel_path_length_mm(route, skeleton.shape, target_width_mm)
                if route and route_length_mm <= max_route_mm + 1e-9 and route_length_mm <= distance * max_route_factor + 1e-9:
                    stitch_segments += add_pixel_path(pattern, route, skeleton.shape, target_width_mm, max_stitch_mm)
                    routed_connects += 1
                    route_length_mm_total += route_length_mm
                else:
                    add_abs(pattern, JUMP, first_mm)
                    jumps += 1
                    rejected_connects += 1
                    route_failures += 1
        current_mm = first_mm
        current_pixel = walk[0]
        for pixel in walk[1:]:
            point_mm = pixel_to_mm(pixel, skeleton.shape, target_width_mm)
            stitch_segments += add_segment(pattern, current_mm, point_mm, max_stitch_mm)
            current_mm = point_mm
            current_pixel = pixel

    if current_mm is None:
        current_mm = (0.0, 0.0)
    add_abs(pattern, END, current_mm)
    output_dst.parent.mkdir(parents=True, exist_ok=True)
    write_dst(pattern, str(output_dst))
    return {
        "mode": "line_domain_maskpath_merger",
        "skeleton_path": str(skeleton_path),
        "mask_path": str(mask_path),
        "output_dst": str(output_dst),
        "components": len(components),
        "open_components": open_components,
        "jump_commands_added": jumps,
        "safe_component_connects": safe_connects,
        "routed_connects": routed_connects,
        "short_gap_connects": short_gap_connects,
        "rejected_component_connects": rejected_connects,
        "route_failures": route_failures,
        "route_length_mm_total": round(route_length_mm_total, 4),
        "stitch_segments_added": stitch_segments,
        "target_width_mm": target_width_mm,
        "max_stitch_mm": max_stitch_mm,
        "max_connect_mm": max_connect_mm,
        "min_connect_inside_fraction": min_connect_inside_fraction,
        "short_gap_mm": short_gap_mm,
        "short_gap_inside_fraction": short_gap_inside_fraction,
        "connector_dilation_px": connector_dilation_px,
        "min_component_pixels": min_component_pixels,
        "unsafe_connection_penalty": unsafe_connection_penalty,
        "inside_penalty": inside_penalty,
        "max_start_candidates": max_start_candidates,
        "max_route_mm": max_route_mm,
        "max_route_factor": max_route_factor,
        "route_mask_dilation_px": route_mask_dilation_px,
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
    parser = argparse.ArgumentParser(description="Generate an M2.60 line-domain mask-path segment merger candidate.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-selection", required=True)
    parser.add_argument("--fallback-output-dir", default="")
    parser.add_argument("--line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--max-connect-mm", type=float, default=20.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.95)
    parser.add_argument("--short-gap-mm", type=float, default=4.0)
    parser.add_argument("--short-gap-inside-fraction", type=float, default=0.80)
    parser.add_argument("--connector-dilation-px", type=int, default=0)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--unsafe-connection-penalty", type=float, default=100.0)
    parser.add_argument("--inside-penalty", type=float, default=20.0)
    parser.add_argument("--max-start-candidates", type=int, default=48)
    parser.add_argument("--max-route-mm", type=float, default=28.0)
    parser.add_argument("--max-route-factor", type=float, default=2.4)
    parser.add_argument("--route-mask-dilation-px", type=int, default=0)
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
            report = generate_maskpath_merger_dst(
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
                unsafe_connection_penalty=args.unsafe_connection_penalty,
                inside_penalty=args.inside_penalty,
                max_start_candidates=args.max_start_candidates,
                max_route_mm=args.max_route_mm,
                max_route_factor=args.max_route_factor,
                route_mask_dilation_px=args.route_mask_dilation_px,
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
            metric_row = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "mode": "line_domain_maskpath_merger",
                "quality_level": score["quality_level"],
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
    merged = [
        {**metric_row, "coverage_ratio": coverage_row.get("coverage_ratio", ""), "stitch_precision_ratio": coverage_row.get("stitch_precision_ratio", "")}
        for metric_row, coverage_row in zip(metric_rows, coverage_rows)
    ]
    summary = summarize_metric_rows(merged)
    summary["line_sources"] = sorted(line_sources)
    summary["candidate_args"] = {
        "max_connect_mm": args.max_connect_mm,
        "min_connect_inside_fraction": args.min_connect_inside_fraction,
        "short_gap_mm": args.short_gap_mm,
        "short_gap_inside_fraction": args.short_gap_inside_fraction,
        "connector_dilation_px": args.connector_dilation_px,
        "min_component_pixels": args.min_component_pixels,
        "unsafe_connection_penalty": args.unsafe_connection_penalty,
        "inside_penalty": args.inside_penalty,
        "max_start_candidates": args.max_start_candidates,
        "max_route_mm": args.max_route_mm,
        "max_route_factor": args.max_route_factor,
        "route_mask_dilation_px": args.route_mask_dilation_px,
    }
    (output_dir / "line_domain_maskpath_merger_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

