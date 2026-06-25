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

from pyembroidery import END, JUMP, EmbPattern, EmbThread, write_dst

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_executability import analyze_file
from eval_stitch_coverage import coverage_metrics
from run_line_domain_grouped_topology_candidate import candidate_starts, safe_float
from run_line_domain_maskpath_merger_candidate import (
    add_pixel_path,
    astar_mask_path,
    pixel_path_length_mm,
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


def connection_plan(
    current_mm: tuple[float, float] | None,
    current_pixel: Pixel | None,
    target_pixel: Pixel,
    shape: tuple[int, int],
    target_width_mm: float,
    mask,
    connector_mask,
    route_mask,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    short_gap_mm: float,
    short_gap_inside_fraction: float,
    max_route_mm: float,
    max_route_factor: float,
) -> dict[str, Any]:
    target_mm = pixel_to_mm(target_pixel, shape, target_width_mm)
    if current_mm is None or current_pixel is None:
        return {
            "action": "start_jump",
            "distance_mm": 0.0,
            "inside": 1.0,
            "strict_inside": 1.0,
            "route": [],
            "route_length_mm": 0.0,
            "target_mm": target_mm,
        }
    distance = math.dist(current_mm, target_mm)
    inside = segment_inside_fraction(connector_mask, current_mm, target_mm, target_width_mm)
    strict_inside = segment_inside_fraction(mask, current_mm, target_mm, target_width_mm)
    can_connect = distance <= max_connect_mm and inside >= min_connect_inside_fraction
    can_short_gap = distance <= short_gap_mm and inside >= short_gap_inside_fraction and strict_inside > 0.0
    if can_connect or can_short_gap:
        return {
            "action": "safe_connect",
            "distance_mm": distance,
            "inside": inside,
            "strict_inside": strict_inside,
            "route": [],
            "route_length_mm": 0.0,
            "target_mm": target_mm,
        }

    scale = target_width_mm / max(1, shape[1])
    straight_px = max(1.0, distance / scale)
    max_cost_px = min(max_route_mm / scale, straight_px * max_route_factor)
    route = astar_mask_path(route_mask, current_pixel, target_pixel, max_cost_px=max_cost_px)
    route_length_mm = pixel_path_length_mm(route, shape, target_width_mm)
    route_ok = (
        bool(route)
        and route_length_mm <= max_route_mm + 1e-9
        and route_length_mm <= distance * max_route_factor + 1e-9
    )
    return {
        "action": "mask_route" if route_ok else "jump",
        "distance_mm": distance,
        "inside": inside,
        "strict_inside": strict_inside,
        "route": route if route_ok else [],
        "route_length_mm": route_length_mm if route_ok else 0.0,
        "target_mm": target_mm,
    }


def action_cost(plan: dict[str, Any], args: argparse.Namespace) -> float:
    action = plan["action"]
    if action == "start_jump":
        return 0.0
    distance = safe_float(plan.get("distance_mm"))
    inside = safe_float(plan.get("inside"))
    if action == "safe_connect":
        return distance
    if action == "mask_route":
        return args.route_cost_weight * safe_float(plan.get("route_length_mm"))
    return args.jump_penalty + distance + args.inside_penalty * max(0.0, 1.0 - inside)


def component_options(
    pixels: list[Pixel],
    graph: dict[Pixel, list[Pixel]],
    shape: tuple[int, int],
    target_width_mm: float,
    max_start_candidates: int,
    walk_mode: str,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[tuple[Pixel, Pixel]] = set()
    starts = euler_candidate_starts(graph, max_start_candidates) if walk_mode == "euler" else candidate_starts(graph, max_start_candidates)
    for start in starts:
        if walk_mode == "euler":
            walk = euler_or_fallback_walk(graph, start)
        else:
            end = farthest_endpoint_by_graph_distance(graph, start)
            walk = open_edge_walk(graph, start, end) if end != start else closed_edge_walk(graph, start)
        if not walk:
            continue
        key = (walk[0], walk[-1])
        if key in seen:
            continue
        seen.add(key)
        options.append(
            {
                "start": walk[0],
                "end": walk[-1],
                "start_mm": pixel_to_mm(walk[0], shape, target_width_mm),
                "end_mm": pixel_to_mm(walk[-1], shape, target_width_mm),
                "walk": walk,
                "pixels": len(pixels),
            }
        )
    return options


def edge_count(graph: dict[Pixel, list[Pixel]]) -> int:
    return sum(len(neighbors) for neighbors in graph.values()) // 2


def odd_degree_nodes(graph: dict[Pixel, list[Pixel]]) -> list[Pixel]:
    return sorted(pixel for pixel, neighbors in graph.items() if len(neighbors) % 2 == 1)


def euler_candidate_starts(graph: dict[Pixel, list[Pixel]], max_candidates: int) -> list[Pixel]:
    odds = odd_degree_nodes(graph)
    if len(odds) == 2:
        return odds
    if len(odds) == 0:
        nodes = sorted(graph)
        if len(nodes) <= max_candidates:
            return nodes
        step = max(1, len(nodes) // max_candidates)
        return nodes[::step][:max_candidates]
    return candidate_starts(graph, max_candidates)


def euler_edge_walk(graph: dict[Pixel, list[Pixel]], start: Pixel) -> list[Pixel]:
    odds = odd_degree_nodes(graph)
    if len(odds) not in (0, 2):
        return []
    if len(odds) == 2 and start not in odds:
        return []
    adjacency = {node: list(neighbors) for node, neighbors in graph.items()}
    stack = [start]
    path: list[Pixel] = []
    while stack:
        node = stack[-1]
        while adjacency[node] and node not in adjacency[adjacency[node][-1]]:
            adjacency[node].pop()
        if adjacency[node]:
            neighbor = adjacency[node].pop()
            adjacency[neighbor].remove(node)
            stack.append(neighbor)
        else:
            path.append(stack.pop())
    walk = list(reversed(path))
    if len(walk) - 1 != edge_count(graph):
        return []
    return walk


def euler_or_fallback_walk(graph: dict[Pixel, list[Pixel]], start: Pixel) -> list[Pixel]:
    walk = euler_edge_walk(graph, start)
    if walk:
        return walk
    end = farthest_endpoint_by_graph_distance(graph, start)
    return open_edge_walk(graph, start, end) if end != start else closed_edge_walk(graph, start)


def best_future_cost(
    current_mm: tuple[float, float],
    current_pixel: Pixel,
    remaining_options: list[list[dict[str, Any]]],
    shape: tuple[int, int],
    target_width_mm: float,
    mask,
    connector_mask,
    route_mask,
    args: argparse.Namespace,
) -> float:
    best = float("inf")
    for options in remaining_options:
        for option in options:
            target_mm = option["start_mm"]
            distance = math.dist(current_mm, target_mm)
            inside = segment_inside_fraction(connector_mask, current_mm, target_mm, target_width_mm)
            strict_inside = segment_inside_fraction(mask, current_mm, target_mm, target_width_mm)
            safe = distance <= args.max_connect_mm and inside >= args.min_connect_inside_fraction
            short_safe = distance <= args.short_gap_mm and inside >= args.short_gap_inside_fraction and strict_inside > 0.0
            if safe or short_safe:
                cost = distance
            else:
                cost = args.jump_penalty + distance + args.inside_penalty * max(0.0, 1.0 - inside)
            best = min(best, cost)
    return 0.0 if best == float("inf") else best


def choose_segment_graph_option(
    remaining: list[dict[str, Any]],
    current_mm: tuple[float, float] | None,
    current_pixel: Pixel | None,
    shape: tuple[int, int],
    target_width_mm: float,
    mask,
    connector_mask,
    route_mask,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any], dict[str, Any], float]:
    best: tuple[float, int, dict[str, Any], dict[str, Any]] | None = None
    for index, component in enumerate(remaining):
        other_options = [item["options"] for other_index, item in enumerate(remaining) if other_index != index]
        for option in component["options"]:
            plan = connection_plan(
                current_mm,
                current_pixel,
                option["start"],
                shape,
                target_width_mm,
                mask,
                connector_mask,
                route_mask,
                args.max_connect_mm,
                args.min_connect_inside_fraction,
                args.short_gap_mm,
                args.short_gap_inside_fraction,
                args.max_route_mm,
                args.max_route_factor,
            )
            incoming = action_cost(plan, args)
            future = best_future_cost(
                option["end_mm"],
                option["end"],
                other_options,
                shape,
                target_width_mm,
                mask,
                connector_mask,
                route_mask,
                args,
            )
            size_bonus = min(args.component_size_bonus, option["pixels"] / max(1.0, args.component_size_scale))
            score = incoming + args.lookahead_weight * future - size_bonus
            if current_mm is None:
                # First component should still favor the largest stroke, but
                # choose its orientation by outgoing connectivity.
                score = -min(2.0, option["pixels"] / 250.0) + args.lookahead_weight * future
            if best is None or score < best[0]:
                best = (score, index, option, plan)
    if best is None:
        raise ValueError("No segment graph option available.")
    score, index, option, plan = best
    return index, option, plan, score


def add_connection(
    pattern: EmbPattern,
    current_mm: tuple[float, float] | None,
    first_mm: tuple[float, float],
    plan: dict[str, Any],
    shape: tuple[int, int],
    target_width_mm: float,
    max_stitch_mm: float,
) -> dict[str, Any]:
    if plan["action"] == "start_jump" or current_mm is None:
        add_abs(pattern, JUMP, first_mm)
        return {"jump": 1, "safe": 0, "routed": 0, "segments": 0, "route_length_mm": 0.0}
    if plan["action"] == "safe_connect":
        return {
            "jump": 0,
            "safe": 1,
            "routed": 0,
            "segments": add_segment(pattern, current_mm, first_mm, max_stitch_mm),
            "route_length_mm": 0.0,
        }
    if plan["action"] == "mask_route":
        route = plan.get("route") or []
        return {
            "jump": 0,
            "safe": 0,
            "routed": 1,
            "segments": add_pixel_path(pattern, route, shape, target_width_mm, max_stitch_mm),
            "route_length_mm": safe_float(plan.get("route_length_mm")),
        }
    add_abs(pattern, JUMP, first_mm)
    return {"jump": 1, "safe": 0, "routed": 0, "segments": 0, "route_length_mm": 0.0}


def generate_segment_graph_merge_dst(
    skeleton_path: Path,
    mask_path: Path,
    output_dst: Path,
    args: argparse.Namespace,
    thread_rgb: tuple[int, int, int] = (20, 80, 150),
) -> dict[str, Any]:
    skeleton = load_binary(skeleton_path)
    mask = load_binary(mask_path)
    connector_mask = dilate(mask, args.connector_dilation_px)
    route_mask = dilate(mask, args.route_mask_dilation_px)
    components = [(pixels, build_graph(pixels)) for pixels in component_pixels(skeleton, args.min_component_pixels)]
    remaining = [
        {
            "pixels": pixels,
            "graph": graph,
            "options": component_options(
                pixels,
                graph,
                skeleton.shape,
                args.target_width_mm,
                args.max_start_candidates,
                args.walk_mode,
            ),
        }
        for pixels, graph in components
        if graph
    ]
    remaining = [item for item in remaining if item["options"]]

    pattern = EmbPattern()
    add_thread(pattern, thread_rgb)
    current_mm: tuple[float, float] | None = None
    current_pixel: Pixel | None = None
    stitch_segments = 0
    jumps = 0
    safe_connects = 0
    routed_connects = 0
    rejected_connects = 0
    route_failures = 0
    route_length_mm_total = 0.0
    choice_trace: list[dict[str, Any]] = []

    while remaining:
        index, option, plan, score = choose_segment_graph_option(
            remaining,
            current_mm,
            current_pixel,
            skeleton.shape,
            args.target_width_mm,
            mask,
            connector_mask,
            route_mask,
            args,
        )
        remaining.pop(index)
        walk = option["walk"]
        first_mm = option["start_mm"]
        stats = add_connection(pattern, current_mm, first_mm, plan, skeleton.shape, args.target_width_mm, args.max_stitch_mm)
        jumps += stats["jump"]
        safe_connects += stats["safe"]
        routed_connects += stats["routed"]
        rejected_connects += 1 if plan["action"] == "jump" else 0
        route_failures += 1 if plan["action"] == "jump" and current_mm is not None else 0
        stitch_segments += stats["segments"]
        route_length_mm_total += stats["route_length_mm"]
        current_mm = first_mm
        current_pixel = walk[0]
        for pixel in walk[1:]:
            point_mm = pixel_to_mm(pixel, skeleton.shape, args.target_width_mm)
            stitch_segments += add_segment(pattern, current_mm, point_mm, args.max_stitch_mm)
            current_mm = point_mm
            current_pixel = pixel
        choice_trace.append(
            {
                "action": plan["action"],
                "score": round(score, 6),
                "start": list(option["start"]),
                "end": list(option["end"]),
                "distance_mm": round(safe_float(plan.get("distance_mm")), 4),
                "inside": round(safe_float(plan.get("inside")), 4),
                "route_length_mm": round(safe_float(plan.get("route_length_mm")), 4),
                "pixels": option["pixels"],
            }
        )

    if current_mm is None:
        current_mm = (0.0, 0.0)
    add_abs(pattern, END, current_mm)
    output_dst.parent.mkdir(parents=True, exist_ok=True)
    write_dst(pattern, str(output_dst))
    return {
        "mode": "line_domain_segment_graph_merge",
        "skeleton_path": str(skeleton_path),
        "mask_path": str(mask_path),
        "output_dst": str(output_dst),
        "components": len(components),
        "used_components": len(choice_trace),
        "jump_commands_added": jumps,
        "safe_component_connects": safe_connects,
        "routed_connects": routed_connects,
        "rejected_component_connects": rejected_connects,
        "route_failures": route_failures,
        "route_length_mm_total": round(route_length_mm_total, 4),
        "stitch_segments_added": stitch_segments,
        "target_width_mm": args.target_width_mm,
        "max_stitch_mm": args.max_stitch_mm,
        "max_connect_mm": args.max_connect_mm,
        "min_connect_inside_fraction": args.min_connect_inside_fraction,
        "short_gap_mm": args.short_gap_mm,
        "short_gap_inside_fraction": args.short_gap_inside_fraction,
        "connector_dilation_px": args.connector_dilation_px,
        "route_mask_dilation_px": args.route_mask_dilation_px,
        "min_component_pixels": args.min_component_pixels,
        "max_route_mm": args.max_route_mm,
        "max_route_factor": args.max_route_factor,
        "lookahead_weight": args.lookahead_weight,
        "jump_penalty": args.jump_penalty,
        "route_cost_weight": args.route_cost_weight,
        "inside_penalty": args.inside_penalty,
        "walk_mode": args.walk_mode,
        "choice_trace": choice_trace,
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
    parser = argparse.ArgumentParser(description="Generate an M2.61 line-domain segment-graph merge candidate.")
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
    parser.add_argument("--route-mask-dilation-px", type=int, default=0)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--max-start-candidates", type=int, default=48)
    parser.add_argument("--max-route-mm", type=float, default=32.0)
    parser.add_argument("--max-route-factor", type=float, default=2.6)
    parser.add_argument("--lookahead-weight", type=float, default=0.35)
    parser.add_argument("--jump-penalty", type=float, default=100.0)
    parser.add_argument("--inside-penalty", type=float, default=20.0)
    parser.add_argument("--route-cost-weight", type=float, default=0.65)
    parser.add_argument("--component-size-bonus", type=float, default=1.0)
    parser.add_argument("--component-size-scale", type=float, default=250.0)
    parser.add_argument("--walk-mode", choices=["open", "euler"], default="open")
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
            report = generate_segment_graph_merge_dst(
                dataset_dir / sample["skeleton_path"],
                dataset_dir / sample["mask_path"],
                dst_path,
                args,
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
                "mode": "line_domain_segment_graph_merge",
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
        "route_mask_dilation_px": args.route_mask_dilation_px,
        "min_component_pixels": args.min_component_pixels,
        "max_start_candidates": args.max_start_candidates,
        "max_route_mm": args.max_route_mm,
        "max_route_factor": args.max_route_factor,
        "lookahead_weight": args.lookahead_weight,
        "jump_penalty": args.jump_penalty,
        "inside_penalty": args.inside_penalty,
        "route_cost_weight": args.route_cost_weight,
        "component_size_bonus": args.component_size_bonus,
        "component_size_scale": args.component_size_scale,
        "walk_mode": args.walk_mode,
    }
    (output_dir / "line_domain_segment_graph_merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
