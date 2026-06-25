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

from pyembroidery import END, JUMP, STITCH, EmbPattern, EmbThread, write_dst

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_executability import analyze_file
from eval_stitch_coverage import coverage_metrics
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
    endpoints,
    farthest_endpoint_by_graph_distance,
    open_edge_walk,
    closed_edge_walk,
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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def add_thread(pattern: EmbPattern, color: tuple[int, int, int]) -> None:
    thread = EmbThread()
    thread.set_color(*color)
    pattern.add_thread(thread)


def candidate_starts(graph: dict[Pixel, list[Pixel]], max_candidates: int) -> list[Pixel]:
    candidates = endpoints(graph) or list(graph.keys())
    if len(candidates) <= max_candidates:
        return sorted(candidates)
    # Keep endpoints distributed across the component without depending on randomness.
    step = max(1, len(candidates) // max_candidates)
    return sorted(candidates)[::step][:max_candidates]


def connection_features(
    current_mm: tuple[float, float],
    start: Pixel,
    shape: tuple[int, int],
    target_width_mm: float,
    mask,
    connector_mask,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    short_gap_mm: float,
    short_gap_inside_fraction: float,
) -> dict[str, Any]:
    start_mm = pixel_to_mm(start, shape, target_width_mm)
    distance = math.dist(current_mm, start_mm)
    inside = segment_inside_fraction(connector_mask, current_mm, start_mm, target_width_mm)
    strict_inside = segment_inside_fraction(mask, current_mm, start_mm, target_width_mm)
    safe = distance <= max_connect_mm and inside >= min_connect_inside_fraction
    short_safe = distance <= short_gap_mm and inside >= short_gap_inside_fraction and strict_inside > 0.0
    return {
        "start_mm": start_mm,
        "distance": distance,
        "inside": inside,
        "strict_inside": strict_inside,
        "safe": safe,
        "short_safe": short_safe,
        "connectable": safe or short_safe,
    }


def choose_grouped_component(
    remaining: list[tuple[list[Pixel], dict[Pixel, list[Pixel]]]],
    current_mm: tuple[float, float] | None,
    shape: tuple[int, int],
    target_width_mm: float,
    mask,
    connector_mask,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    short_gap_mm: float,
    short_gap_inside_fraction: float,
    unsafe_connection_penalty: float,
    inside_penalty: float,
    max_start_candidates: int,
) -> tuple[int, Pixel, dict[str, Any]]:
    if current_mm is None:
        index = max(range(len(remaining)), key=lambda item: len(remaining[item][0]))
        graph = remaining[index][1]
        start = min(candidate_starts(graph, max_start_candidates))
        return index, start, {"mode": "first_largest_component", "connectable": False}

    best: tuple[float, int, Pixel, dict[str, Any]] | None = None
    for index, (_pixels, graph) in enumerate(remaining):
        for start in candidate_starts(graph, max_start_candidates):
            features = connection_features(
                current_mm,
                start,
                shape,
                target_width_mm,
                mask,
                connector_mask,
                max_connect_mm,
                min_connect_inside_fraction,
                short_gap_mm,
                short_gap_inside_fraction,
            )
            # Prefer mask-safe links first, then short distance. This is the
            # stroke grouping step: a farther but safe connector can beat a
            # nearer connector that would immediately become a jump.
            score = (
                (0.0 if features["connectable"] else unsafe_connection_penalty)
                + features["distance"]
                + inside_penalty * max(0.0, 1.0 - features["inside"])
                - min(1.0, len(_pixels) / 250.0)
            )
            if best is None or score < best[0]:
                best = (score, index, start, features)
    if best is None:
        index = 0
        graph = remaining[index][1]
        start = min(candidate_starts(graph, max_start_candidates))
        return index, start, {"mode": "fallback", "connectable": False}
    score, index, start, features = best
    features["mode"] = "safe_grouped_choice" if features["connectable"] else "jump_grouped_choice"
    features["choice_score"] = score
    return index, start, features


def generate_grouped_topology_dst(
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
    open_components = 0
    grouped_safe_choices = 0
    grouped_jump_choices = 0
    remaining = [(pixels, graph) for pixels, graph in components if graph]

    while remaining:
        next_index, start, features = choose_grouped_component(
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
        if current_mm is None:
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
                grouped_safe_choices += 1
            else:
                add_abs(pattern, JUMP, first_mm)
                jumps += 1
                rejected_connects += 1
                grouped_jump_choices += 1
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
        "mode": "line_domain_grouped_topology",
        "skeleton_path": str(skeleton_path),
        "mask_path": str(mask_path),
        "output_dst": str(output_dst),
        "components": len(components),
        "open_components": open_components,
        "jump_commands_added": jumps,
        "safe_component_connects": safe_connects,
        "short_gap_connects": short_gap_connects,
        "rejected_component_connects": rejected_connects,
        "grouped_safe_choices": grouped_safe_choices,
        "grouped_jump_choices": grouped_jump_choices,
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
    parser = argparse.ArgumentParser(description="Generate an M2.59 grouped topology line-domain candidate.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-selection", required=True)
    parser.add_argument("--fallback-output-dir", default="")
    parser.add_argument("--line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--max-connect-mm", type=float, default=16.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.85)
    parser.add_argument("--short-gap-mm", type=float, default=4.0)
    parser.add_argument("--short-gap-inside-fraction", type=float, default=0.60)
    parser.add_argument("--connector-dilation-px", type=int, default=0)
    parser.add_argument("--min-component-pixels", type=int, default=12)
    parser.add_argument("--unsafe-connection-penalty", type=float, default=80.0)
    parser.add_argument("--inside-penalty", type=float, default=20.0)
    parser.add_argument("--max-start-candidates", type=int, default=48)
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
            report = generate_grouped_topology_dst(
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
                "mode": "line_domain_grouped_topology",
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
    }
    (output_dir / "line_domain_grouped_topology_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

