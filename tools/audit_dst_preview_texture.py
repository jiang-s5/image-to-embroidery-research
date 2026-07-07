from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image, ImageDraw, ImageFilter
from pyembroidery import COMMAND_MASK, JUMP, STITCH, TRIM, read_dst


TEXTURE_KEYS = (
    "fill_rows",
    "outline_points",
    "satin_columns",
    "satin_rail_segments",
    "dt_satin_segments",
    "style_running_points",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pattern_segments(dst_path: Path) -> tuple[list[dict[str, float]], dict[str, int]]:
    pattern = read_dst(str(dst_path))
    segments: list[dict[str, float]] = []
    command_counts: dict[str, int] = defaultdict(int)
    last: tuple[float, float] | None = None
    for x_raw, y_raw, command_raw in pattern.stitches:
        command = command_raw & COMMAND_MASK
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        if command == STITCH:
            command_counts["stitch"] += 1
            if last is not None:
                dx = x - last[0]
                dy = y - last[1]
                length = math.hypot(dx, dy)
                if length > 1e-6:
                    angle = math.atan2(dy, dx) % math.pi
                    segments.append(
                        {
                            "x0": last[0],
                            "y0": last[1],
                            "x1": x,
                            "y1": y,
                            "length_mm": length,
                            "angle": angle,
                        }
                    )
        elif command == JUMP:
            command_counts["jump"] += 1
        elif command == TRIM:
            command_counts["trim"] += 1
        last = (x, y)
    return segments, dict(command_counts)


def angle_entropy(angles: list[float], bins: int = 18) -> tuple[float, float]:
    if not angles:
        return 0.0, 0.0
    counts = [0] * bins
    for angle in angles:
        index = min(bins - 1, int((angle % math.pi) / math.pi * bins))
        counts[index] += 1
    total = sum(counts)
    probs = [count / total for count in counts if count]
    entropy = -sum(prob * math.log(prob) for prob in probs) / math.log(bins)
    dominant_mass = max(counts) / total
    return round(entropy, 8), round(dominant_mass, 8)


def generator_texture_score(report: dict[str, Any]) -> float:
    texture = {key: safe_float(report.get(key)) for key in TEXTURE_KEYS}
    return round(
        0.002 * texture["fill_rows"]
        + 0.001 * texture["outline_points"]
        + 0.006 * texture["satin_columns"]
        + 0.010 * texture["satin_rail_segments"]
        + 0.010 * texture["dt_satin_segments"]
        + 0.004 * texture["style_running_points"],
        8,
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_preview_texture(metrics: dict[str, Any]) -> dict[str, float]:
    coverage = safe_float(metrics.get("coverage_ratio"))
    precision = safe_float(metrics.get("stitch_precision_ratio"))
    adaptive_precision = safe_float(metrics.get("adaptive_stitch_precision_ratio"), precision)
    entropy = safe_float(metrics.get("angle_entropy"))
    dominant = safe_float(metrics.get("dominant_angle_mass"))
    mean_len = safe_float(metrics.get("mean_stitch_length_mm"))
    cv_len = safe_float(metrics.get("stitch_length_cv"))
    long_ratio = safe_float(metrics.get("long_stitch_ratio"))
    density = safe_float(metrics.get("stitch_density_per_100mm2"))
    texture_score = safe_float(metrics.get("generator_texture_score"))
    visible = safe_float(metrics.get("visible_connector_count"))
    off_mask = safe_float(metrics.get("off_mask_stitch_length_mm"))
    illegal = safe_float(metrics.get("illegal_long_stitch_count"))

    density_score = clamp01(density / 5.0)
    rhythm_score = clamp01(1.0 - abs(mean_len - 2.2) / 2.8) * clamp01(1.0 - 0.75 * cv_len)
    structure_balance = clamp01(1.0 - abs(entropy - 0.42) / 0.42)
    directional_score = clamp01(0.55 * dominant + 0.45 * structure_balance)
    family_score = clamp01(texture_score / 0.75)
    safety_score = clamp01(1.0 - 0.20 * visible - 0.02 * off_mask - 0.10 * illegal - 0.35 * long_ratio)
    coverage_score = clamp01(0.55 * coverage + 0.45 * max(precision, adaptive_precision))

    professional_preview_score = clamp01(
        0.25 * coverage_score
        + 0.18 * density_score
        + 0.18 * rhythm_score
        + 0.17 * directional_score
        + 0.14 * family_score
        + 0.08 * safety_score
    )
    return {
        "coverage_score": round(coverage_score, 8),
        "density_score": round(density_score, 8),
        "rhythm_score": round(rhythm_score, 8),
        "directional_score": round(directional_score, 8),
        "family_texture_score_norm": round(family_score, 8),
        "safety_score": round(safety_score, 8),
        "professional_preview_score": round(professional_preview_score, 8),
    }


def audit_sample(sample_dir: Path, selected_row: dict[str, Any] | None = None) -> dict[str, Any]:
    selected_row = selected_row or {}
    dst_path = sample_dir / "prediction.dst"
    if not dst_path.exists():
        return {"round_trip_parse_success": False, "error": "missing_prediction_dst"}
    try:
        segments, command_counts = pattern_segments(dst_path)
    except Exception as exc:
        return {"round_trip_parse_success": False, "error": str(exc)}

    lengths = [seg["length_mm"] for seg in segments]
    angles = [seg["angle"] for seg in segments]
    xs = [coord for seg in segments for coord in (seg["x0"], seg["x1"])]
    ys = [coord for seg in segments for coord in (seg["y0"], seg["y1"])]
    bbox_w = max(xs) - min(xs) if xs else 0.0
    bbox_h = max(ys) - min(ys) if ys else 0.0
    bbox_area = max(1e-6, bbox_w * bbox_h)
    entropy, dominant = angle_entropy(angles)
    mean_length = mean(lengths) if lengths else 0.0
    median_length = median(lengths) if lengths else 0.0
    length_std = math.sqrt(mean((value - mean_length) ** 2 for value in lengths)) if lengths else 0.0

    eval_report = read_json(sample_dir / "eval_executability.json").get("pred", {})
    coverage = read_json(sample_dir / "coverage.json")
    generator = read_json(sample_dir / "generator_report.json")
    row: dict[str, Any] = {
        "round_trip_parse_success": True,
        "dst_path": str(dst_path),
        "stitch_segment_count": len(segments),
        "stitch_command_count": command_counts.get("stitch", 0),
        "jump_count": command_counts.get("jump", safe_int(eval_report.get("jump_count"))),
        "trim_count": command_counts.get("trim", safe_int(eval_report.get("trim_count"))),
        "bbox_width_mm": round(bbox_w, 8),
        "bbox_height_mm": round(bbox_h, 8),
        "bbox_area_mm2": round(bbox_area, 8),
        "stitch_path_mm": round(sum(lengths), 8),
        "mean_stitch_length_mm": round(mean_length, 8),
        "median_stitch_length_mm": round(median_length, 8),
        "stitch_length_cv": round(length_std / max(1e-6, mean_length), 8),
        "short_stitch_ratio": round(sum(1 for value in lengths if value < 0.8) / max(1, len(lengths)), 8),
        "long_stitch_ratio": round(sum(1 for value in lengths if value > 4.0) / max(1, len(lengths)), 8),
        "angle_entropy": entropy,
        "dominant_angle_mass": dominant,
        "stitch_density_per_100mm2": round(100.0 * len(segments) / bbox_area, 8),
        "path_density_per_100mm2": round(100.0 * sum(lengths) / bbox_area, 8),
        "generator_texture_score": generator_texture_score(generator),
        "fill_rows": safe_float(generator.get("fill_rows")),
        "outline_points": safe_float(generator.get("outline_points")),
        "satin_columns": safe_float(generator.get("satin_columns")),
        "satin_rail_segments": safe_float(generator.get("satin_rail_segments")),
        "dt_satin_segments": safe_float(generator.get("dt_satin_segments")),
        "style_running_points": safe_float(generator.get("style_running_points")),
        "coverage_ratio": safe_float(coverage.get("coverage_ratio"), safe_float(selected_row.get("coverage_ratio"))),
        "stitch_precision_ratio": safe_float(
            coverage.get("stitch_precision_ratio"), safe_float(selected_row.get("stitch_precision_ratio"))
        ),
        "adaptive_coverage_ratio": safe_float(
            coverage.get("adaptive_coverage_ratio"), safe_float(selected_row.get("adaptive_coverage_ratio"))
        ),
        "adaptive_stitch_precision_ratio": safe_float(
            coverage.get("adaptive_stitch_precision_ratio"),
            safe_float(selected_row.get("adaptive_stitch_precision_ratio"), safe_float(selected_row.get("stitch_precision_ratio"))),
        ),
        "visible_connector_count": safe_float(
            eval_report.get("visible_connector_count"), safe_float(selected_row.get("visible_connector_count"))
        ),
        "off_mask_stitch_length_mm": safe_float(
            eval_report.get("off_mask_stitch_length_mm"), safe_float(selected_row.get("off_mask_stitch_length_mm"))
        ),
        "illegal_long_stitch_count": safe_float(
            eval_report.get("illegal_long_stitch_count"), safe_float(selected_row.get("illegal_long_stitch_count"))
        ),
        "high_risk_jump_count": safe_float(
            eval_report.get("high_risk_jump_count"), safe_float(selected_row.get("high_risk_jump_count"))
        ),
    }
    row.update(score_preview_texture(row))
    return row


def line_color(angle: float) -> tuple[int, int, int]:
    phase = angle / math.pi
    return (
        int(50 + 95 * phase),
        int(95 + 80 * (1.0 - abs(phase - 0.5) * 2.0)),
        int(130 + 70 * (1.0 - phase)),
    )


def render_preview(sample_dir: Path, output_path: Path, size: int = 768, supersample: int = 2) -> None:
    dst_path = sample_dir / "prediction.dst"
    segments, _ = pattern_segments(dst_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        Image.new("RGB", (size, size), (244, 240, 224)).save(output_path)
        return
    xs = [coord for seg in segments for coord in (seg["x0"], seg["x1"])]
    ys = [coord for seg in segments for coord in (seg["y0"], seg["y1"])]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    canvas_size = size * supersample
    margin = int(canvas_size * 0.08)
    scale = (canvas_size - 2 * margin) / span

    def project(x: float, y: float) -> tuple[float, float]:
        px = margin + (x - min_x) * scale
        py = margin + (y - min_y) * scale
        return px, py

    fabric = Image.new("RGB", (canvas_size, canvas_size), (244, 240, 224))
    grid = ImageDraw.Draw(fabric)
    grid_step = max(8, canvas_size // 96)
    for x in range(0, canvas_size, grid_step):
        grid.line([(x, 0), (x, canvas_size)], fill=(232, 228, 212), width=1)
    for y in range(0, canvas_size, grid_step):
        grid.line([(0, y), (canvas_size, y)], fill=(232, 228, 212), width=1)

    shadow = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    thread = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    thread_draw = ImageDraw.Draw(thread)
    width = max(2, int(canvas_size / 390))
    for seg in segments:
        p0 = project(seg["x0"], seg["y0"])
        p1 = project(seg["x1"], seg["y1"])
        shadow_draw.line([p0, p1], fill=(64, 52, 38, 55), width=width + 2)
        color = line_color(seg["angle"])
        thread_draw.line([p0, p1], fill=(*color, 230), width=width)
        highlight = tuple(min(255, int(c * 1.28)) for c in color)
        thread_draw.line([p0, p1], fill=(*highlight, 80), width=max(1, width // 2))

    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1, width // 2)))
    composed = Image.alpha_composite(fabric.convert("RGBA"), shadow)
    composed = Image.alpha_composite(composed, thread)
    composed = composed.resize((size, size), Image.Resampling.LANCZOS)
    composed.convert("RGB").save(output_path)


def load_manifest(dataset_dir: Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in read_csv(dataset_dir / "manifest.csv")}


def parse_profile(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Profile must use name=path format.")
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_profile_rows(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Profile rows must use name=path format.")
    name, path = value.split("=", 1)
    return name, Path(path)


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out = []
    for value, items in sorted(groups.items()):
        out.append(
            {
                key: value,
                "samples": len(items),
                "mean_professional_preview_score": avg(items, "professional_preview_score"),
                "mean_generator_texture_score": avg(items, "generator_texture_score"),
                "mean_angle_entropy": avg(items, "angle_entropy"),
                "mean_dominant_angle_mass": avg(items, "dominant_angle_mass"),
                "mean_stitch_density_per_100mm2": avg(items, "stitch_density_per_100mm2"),
                "mean_rhythm_score": avg(items, "rhythm_score"),
                "mean_coverage_ratio": avg(items, "coverage_ratio"),
                "mean_stitch_precision_ratio": avg(items, "stitch_precision_ratio"),
                "mean_visible_connector_count": avg(items, "visible_connector_count"),
            }
        )
    return out


def build_delta_rows(rows: list[dict[str, Any]], base_profile: str, compare_profile: str) -> list[dict[str, Any]]:
    by_key = {(row["profile"], row["sample_id"]): row for row in rows}
    sample_ids = sorted({row["sample_id"] for row in rows})
    delta_keys = (
        "professional_preview_score",
        "generator_texture_score",
        "coverage_ratio",
        "stitch_precision_ratio",
        "angle_entropy",
        "dominant_angle_mass",
        "stitch_density_per_100mm2",
        "rhythm_score",
        "visible_connector_count",
        "off_mask_stitch_length_mm",
    )
    delta_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        base = by_key.get((base_profile, sample_id))
        comp = by_key.get((compare_profile, sample_id))
        if not base or not comp:
            continue
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_name": comp.get("source_name", ""),
            "category": comp.get("category", ""),
            "base_profile": base_profile,
            "compare_profile": compare_profile,
        }
        for key in delta_keys:
            row[f"base_{key}"] = base.get(key, "")
            row[f"compare_{key}"] = comp.get(key, "")
            row[f"delta_{key}"] = round(safe_float(comp.get(key)) - safe_float(base.get(key)), 8)
        delta_rows.append(row)
    return delta_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DST preview texture and thread-structure quality.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/public_benchmark_v1_ext33"))
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help="Profile in name=path format. Can be passed multiple times.",
    )
    parser.add_argument(
        "--profile-rows",
        action="append",
        default=[],
        help="Selected-row CSV in name=path format. Used to backfill coverage/precision metrics.",
    )
    parser.add_argument("--base-profile", default="m2_81")
    parser.add_argument("--compare-profile", default="m2_82")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit"),
    )
    parser.add_argument("--preview-size", type=int, default=768)
    parser.add_argument("--render-previews", action="store_true")
    args = parser.parse_args()

    profile_items = args.profile or [
        "m2_81=results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile",
        "m2_82=results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy",
    ]
    profile_row_items = args.profile_rows or [
        "m2_81=results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv",
        "m2_82=results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_selected_rows.csv",
    ]
    profiles = [parse_profile(item) for item in profile_items]
    profile_rows = {
        name: {row["sample_id"]: row for row in read_csv(path)}
        for name, path in [parse_profile_rows(item) for item in profile_row_items]
    }
    manifest = load_manifest(args.dataset_dir)

    rows: list[dict[str, Any]] = []
    for profile_name, root in profiles:
        for sample_id, manifest_row in sorted(manifest.items()):
            sample_dir = root / sample_id
            row = audit_sample(sample_dir, profile_rows.get(profile_name, {}).get(sample_id, {}))
            row.update(
                {
                    "profile": profile_name,
                    "sample_id": sample_id,
                    "source_name": manifest_row.get("source_name", ""),
                    "category": manifest_row.get("category", ""),
                    "sample_dir": str(sample_dir),
                }
            )
            rows.append(row)
            if args.render_previews and row.get("round_trip_parse_success"):
                render_preview(
                    sample_dir,
                    args.output_dir / "previews" / profile_name / f"{sample_id}_stitch_preview.png",
                    size=args.preview_size,
                )

    delta_rows = build_delta_rows(rows, args.base_profile, args.compare_profile)
    summary = {
        "model_id": "m2_83_dst_preview_texture_audit",
        "profiles": {name: str(path) for name, path in profiles},
        "rows": len(rows),
        "delta_rows": len(delta_rows),
        "profile_summary": group_summary(rows, "profile"),
        "source_summary": group_summary(rows, "source_name"),
        "delta_summary": {
            "base_profile": args.base_profile,
            "compare_profile": args.compare_profile,
            "samples": len(delta_rows),
            "mean_delta_professional_preview_score": avg(delta_rows, "delta_professional_preview_score"),
            "mean_delta_generator_texture_score": avg(delta_rows, "delta_generator_texture_score"),
            "mean_delta_coverage_ratio": avg(delta_rows, "delta_coverage_ratio"),
            "mean_delta_stitch_precision_ratio": avg(delta_rows, "delta_stitch_precision_ratio"),
            "mean_delta_visible_connector_count": avg(delta_rows, "delta_visible_connector_count"),
            "mean_delta_off_mask_stitch_length_mm": avg(delta_rows, "delta_off_mask_stitch_length_mm"),
            "improved_preview_samples": sum(1 for row in delta_rows if safe_float(row.get("delta_professional_preview_score")) > 0.0),
            "worse_preview_samples": sum(1 for row in delta_rows if safe_float(row.get("delta_professional_preview_score")) < 0.0),
        },
        "interpretation": "This audit reads DST stitches directly and adds a preview-texture proxy score. It is a diagnostic evaluator, not a replacement for human stitch-preview review or professional digitizer ground truth.",
    }

    write_csv(rows, args.output_dir / "dst_preview_texture_rows.csv")
    write_csv(delta_rows, args.output_dir / "dst_preview_texture_delta_rows.csv")
    write_json(summary, args.output_dir / "dst_preview_texture_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
