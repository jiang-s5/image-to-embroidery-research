from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from pyembroidery import END, JUMP, STITCH, EmbPattern, EmbThread, write_dst

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_executability import analyze_file
from eval_stitch_coverage import coverage_metrics
from run_line_domain_stroke_candidate import add_abs, add_segment, load_binary, segment_inside_fraction
from score_unified import score_metrics


Point = tuple[float, float]
Stroke = list[Point]


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


def pixel_to_mm_xy(x: float, y: float, width: int, height: int, target_width_mm: float) -> Point:
    scale = target_width_mm / max(1, width)
    return ((x - width / 2.0) * scale, (y - height / 2.0) * scale)


def load_quickdraw_strokes(raw_path: Path, width: int, height: int, target_width_mm: float, min_points: int) -> list[Stroke]:
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    strokes: list[Stroke] = []
    for raw_stroke in payload.get("drawing", []):
        if not isinstance(raw_stroke, list) or len(raw_stroke) < 2:
            continue
        xs, ys = raw_stroke[0], raw_stroke[1]
        if len(xs) != len(ys) or len(xs) < min_points:
            continue
        stroke = [pixel_to_mm_xy(float(x), float(y), width, height, target_width_mm) for x, y in zip(xs, ys)]
        strokes.append(stroke)
    return strokes


def stroke_length(stroke: Stroke) -> float:
    if len(stroke) <= 1:
        return 0.0
    return sum(math.dist(a, b) for a, b in zip(stroke, stroke[1:]))


def choose_next_stroke(
    remaining: list[Stroke],
    current: Point | None,
    order: str,
    allow_reverse: bool,
) -> tuple[int, Stroke, bool]:
    if order == "original" or current is None:
        stroke = remaining[0]
        if allow_reverse and current is not None and math.dist(current, stroke[-1]) < math.dist(current, stroke[0]):
            return 0, list(reversed(stroke)), True
        return 0, stroke, False

    best: tuple[float, int, Stroke, bool] | None = None
    for index, stroke in enumerate(remaining):
        candidates = [(math.dist(current, stroke[0]), stroke, False)]
        if allow_reverse:
            candidates.append((math.dist(current, stroke[-1]), list(reversed(stroke)), True))
        for distance, oriented, reversed_flag in candidates:
            score = distance - 0.01 * stroke_length(stroke)
            if best is None or score < best[0]:
                best = (score, index, oriented, reversed_flag)
    if best is None:
        return 0, remaining[0], False
    _score, index, oriented, reversed_flag = best
    return index, oriented, reversed_flag


def mm_to_pixel(point: Point, width: int, height: int, target_width_mm: float) -> tuple[int, int]:
    scale = target_width_mm / max(1, width)
    x = int(round(point[0] / scale + width / 2.0))
    y = int(round(point[1] / scale + height / 2.0))
    return y, x


def can_safe_connect(mask, start: Point, end: Point, target_width_mm: float, max_connect_mm: float, min_inside: float) -> bool:
    distance = math.dist(start, end)
    if distance > max_connect_mm:
        return False
    return segment_inside_fraction(mask, start, end, target_width_mm) >= min_inside


def generate_vector_stroke_dst(
    raw_path: Path,
    mask_path: Path,
    output_dst: Path,
    width: int,
    height: int,
    target_width_mm: float,
    max_stitch_mm: float,
    order: str,
    allow_reverse: bool,
    connect_strokes: bool,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    min_points_per_stroke: int,
    thread_rgb: tuple[int, int, int] = (20, 80, 150),
) -> dict[str, Any]:
    strokes = load_quickdraw_strokes(raw_path, width, height, target_width_mm, min_points_per_stroke)
    mask = load_binary(mask_path)
    pattern = EmbPattern()
    add_thread(pattern, thread_rgb)
    remaining = list(strokes)
    current: Point | None = None
    jumps = 0
    safe_connects = 0
    reversed_strokes = 0
    stitch_segments = 0
    stroke_index = 0

    while remaining:
        index, stroke, reversed_flag = choose_next_stroke(remaining, current, order, allow_reverse)
        remaining.pop(index)
        if reversed_flag:
            reversed_strokes += 1
        start = stroke[0]
        if current is None:
            add_abs(pattern, JUMP, start)
            jumps += 1
        elif connect_strokes and can_safe_connect(mask, current, start, target_width_mm, max_connect_mm, min_connect_inside_fraction):
            stitch_segments += add_segment(pattern, current, start, max_stitch_mm)
            safe_connects += 1
        else:
            add_abs(pattern, JUMP, start)
            jumps += 1
        current = start
        for point in stroke[1:]:
            stitch_segments += add_segment(pattern, current, point, max_stitch_mm)
            current = point
        stroke_index += 1

    pattern.add_command(END)
    output_dst.parent.mkdir(parents=True, exist_ok=True)
    write_dst(pattern, str(output_dst))
    return {
        "mode": "quickdraw_vector_stroke",
        "raw_path": str(raw_path),
        "stroke_count": len(strokes),
        "strokes_written": stroke_index,
        "jumps_inserted": jumps,
        "safe_connects": safe_connects,
        "reversed_strokes": reversed_strokes,
        "stitch_segments": stitch_segments,
        "order": order,
        "allow_reverse": allow_reverse,
        "connect_strokes": connect_strokes,
        "max_connect_mm": max_connect_mm,
        "min_connect_inside_fraction": min_connect_inside_fraction,
        "target_width_mm": target_width_mm,
        "max_stitch_mm": max_stitch_mm,
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row.get("source_name", "")), []).append(row)

    def avg(items: list[dict[str, Any]], key: str) -> float:
        return round(mean(safe_float(item.get(key)) for item in items), 6)

    return {
        "samples": len(rows),
        "hard_fail": sum(1 for row in rows if str(row.get("quality_level", "")).lower() == "hard_fail"),
        "mean_unified_loss": avg(rows, "unified_loss"),
        "mean_jump_count": avg(rows, "jump_count"),
        "mean_trim_count": avg(rows, "trim_count"),
        "mean_off_mask_stitch_length_mm": avg(rows, "off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg(rows, "visible_connector_count"),
        "mean_coverage_ratio": avg(rows, "coverage_ratio"),
        "mean_stitch_precision_ratio": avg(rows, "stitch_precision_ratio"),
        "mean_adaptive_coverage_ratio": avg(rows, "adaptive_coverage_ratio"),
        "mean_adaptive_stitch_precision_ratio": avg(rows, "adaptive_stitch_precision_ratio"),
        "by_source": {
            source: {
                "samples": len(items),
                "hard_fail": sum(1 for row in items if str(row.get("quality_level", "")).lower() == "hard_fail"),
                "mean_unified_loss": avg(items, "unified_loss"),
                "mean_jump_count": avg(items, "jump_count"),
                "mean_trim_count": avg(items, "trim_count"),
                "mean_off_mask_stitch_length_mm": avg(items, "off_mask_stitch_length_mm"),
                "mean_visible_connector_count": avg(items, "visible_connector_count"),
                "mean_coverage_ratio": avg(items, "coverage_ratio"),
                "mean_stitch_precision_ratio": avg(items, "stitch_precision_ratio"),
                "mean_adaptive_coverage_ratio": avg(items, "adaptive_coverage_ratio"),
                "mean_adaptive_stitch_precision_ratio": avg(items, "adaptive_stitch_precision_ratio"),
            }
            for source, items in sorted(by_source.items())
        },
    }


def copy_fallback_outputs(source_dir: Path, sample_out: Path) -> None:
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json", "coverage_report.json"):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate QuickDraw DST candidates directly from original vector strokes.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fallback-selection", required=True)
    parser.add_argument("--fallback-output-dir", default="")
    parser.add_argument("--line-sources", default="QuickDraw")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--stroke-order", choices=["original", "nearest"], default="original")
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--connect-strokes", action="store_true")
    parser.add_argument("--max-connect-mm", type=float, default=8.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.95)
    parser.add_argument("--min-points-per-stroke", type=int, default=2)
    parser.add_argument("--line-radius-px", type=int, default=2)
    parser.add_argument("--adaptive-precision-radius-px", type=int, default=2)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(dataset_dir / "manifest.csv")
    fallback_rows = {row["sample_id"]: row for row in read_csv(Path(args.fallback_selection))}
    line_sources = {item.strip() for item in args.line_sources.split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    for sample in manifest:
        sample_id = sample["sample_id"]
        sample_out = output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        dst_path = sample_out / "prediction.dst"
        raw_path = dataset_dir / sample.get("raw_path", "")
        mask_path = dataset_dir / sample["mask_path"]
        if sample.get("source_name", "") in line_sources and raw_path.exists():
            report = generate_vector_stroke_dst(
                raw_path,
                mask_path,
                dst_path,
                width=int(sample.get("width_px", 256) or 256),
                height=int(sample.get("height_px", 256) or 256),
                target_width_mm=args.target_width_mm,
                max_stitch_mm=args.max_stitch_mm,
                order=args.stroke_order,
                allow_reverse=args.allow_reverse,
                connect_strokes=args.connect_strokes,
                max_connect_mm=args.max_connect_mm,
                min_connect_inside_fraction=args.min_connect_inside_fraction,
                min_points_per_stroke=args.min_points_per_stroke,
            )
            pred = analyze_file(
                dst_path,
                max_stitch_mm=4.0,
                high_risk_jump_mm=8.0,
                mask_path=mask_path,
                target_width_mm=args.target_width_mm,
            )
            coverage = coverage_metrics(
                dst_path,
                mask_path,
                target_width_mm=args.target_width_mm,
                line_radius_px=args.line_radius_px,
                adaptive_precision_radius_px=args.adaptive_precision_radius_px,
            )
            score = score_metrics(pred)
            (sample_out / "eval_executability.json").write_text(json.dumps({"pred": pred}, ensure_ascii=False, indent=2), encoding="utf-8")
            (sample_out / "coverage_report.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
            (sample_out / "generator_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "mode": "quickdraw_vector_stroke",
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
                "coverage_ratio": coverage["coverage_ratio"],
                "stitch_precision_ratio": coverage["stitch_precision_ratio"],
                "adaptive_coverage_ratio": coverage["adaptive_coverage_ratio"],
                "adaptive_stitch_precision_ratio": coverage["adaptive_stitch_precision_ratio"],
                "overfill_outside_adaptive_ratio": coverage["overfill_outside_adaptive_ratio"],
                "stroke_count": report["stroke_count"],
                "safe_connects": report["safe_connects"],
                "reversed_strokes": report["reversed_strokes"],
            }
        else:
            fallback = fallback_rows[sample_id]
            if args.fallback_output_dir:
                copy_fallback_outputs(Path(args.fallback_output_dir) / sample_id, sample_out)
            row = {
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
                "coverage_ratio": fallback.get("coverage_ratio", ""),
                "stitch_precision_ratio": fallback.get("stitch_precision_ratio", ""),
                "adaptive_coverage_ratio": fallback.get("adaptive_coverage_ratio", ""),
                "adaptive_stitch_precision_ratio": fallback.get("adaptive_stitch_precision_ratio", ""),
                "overfill_outside_adaptive_ratio": fallback.get("overfill_outside_adaptive_ratio", ""),
            }
            (sample_out / "generator_report.json").write_text(
                json.dumps({"mode": "fallback_selection", "fallback_candidate": fallback.get("chosen_candidate", "")}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        rows.append(row)

    write_csv(rows, output_dir / "source_aware_hybrid_rows.csv")
    summary = summarize_metric_rows(rows)
    summary["line_sources"] = sorted(line_sources)
    summary["candidate_args"] = vars(args)
    (output_dir / "quickdraw_vector_stroke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
