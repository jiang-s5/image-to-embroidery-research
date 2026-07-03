from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from auto_planner_branch import compute_branch_features, select_planner_branch


METRIC_KEYS = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
)

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


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def generator_texture(report: dict[str, Any]) -> dict[str, float]:
    texture = {key: safe_float(report.get(key)) for key in TEXTURE_KEYS}
    texture["texture_score"] = round(
        0.002 * texture["fill_rows"]
        + 0.001 * texture["outline_points"]
        + 0.006 * texture["satin_columns"]
        + 0.010 * texture["satin_rail_segments"]
        + 0.010 * texture["dt_satin_segments"]
        + 0.004 * texture["style_running_points"],
        8,
    )
    return texture


def load_texture(root: Path, sample_id: str) -> dict[str, float]:
    return generator_texture(read_json(root / sample_id / "generator_report.json"))


def candidate_kind(name: str) -> str:
    lowered = name.lower()
    if lowered == "base_keep":
        return "base_keep"
    if "dtsatin" in lowered or "dt_satin" in lowered:
        return "dt_satin"
    if "satinrail" in lowered or "satin_rail" in lowered:
        return "satin_rail"
    if "satin" in lowered:
        return "satin"
    if "outline" in lowered:
        return "outline"
    if "style" in lowered:
        return "style_aware"
    if "skeleton" in lowered or "line" in lowered:
        return "running"
    return "fill"


def candidate_is_satin_like(texture: dict[str, float]) -> bool:
    return safe_float(texture.get("satin_columns")) + safe_float(texture.get("satin_rail_segments")) + safe_float(
        texture.get("dt_satin_segments")
    ) > 0


def metric_delta(candidate: dict[str, Any], base: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(base.get(key)), 8)


def texture_delta(candidate_texture: dict[str, float], base_texture: dict[str, float], key: str) -> float:
    return round(safe_float(candidate_texture.get(key)) - safe_float(base_texture.get(key)), 8)


def parse_candidate_specs(items: list[list[str]]) -> list[tuple[str, Path, dict[str, dict[str, str]], dict[str, dict[str, str]]]]:
    specs = []
    for name, root_text in items:
        root = Path(root_text)
        rows = {row["sample_id"]: row for row in read_csv(root / "source_aware_hybrid_rows.csv")}
        coverage = {row["sample_id"]: row for row in read_csv(root / "coverage_rows.csv")}
        specs.append((name, root, rows, coverage))
    names = [name for name, *_ in specs]
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique.")
    return specs


def sample_geometry(dataset_dir: Path, manifest_row: dict[str, str]) -> dict[str, Any]:
    branch = select_planner_branch(
        compute_branch_features(dataset_dir / manifest_row["mask_path"], dataset_dir / manifest_row["skeleton_path"])
    )
    features = branch["features"]
    return {
        "planner_branch": branch["branch"],
        "branch_confidence": branch["confidence"],
        "branch_line_score": branch["line_score"],
        "mask_area_ratio": features["mask_area_ratio"],
        "skeleton_area_ratio": features["skeleton_area_ratio"],
        "skeleton_to_mask_ratio": features["skeleton_to_mask_ratio"],
        "skeleton_components": features["skeleton_components"],
        "large_skeleton_components": features["large_skeleton_components"],
        "largest_skeleton_component": features["largest_skeleton_component"],
    }


def pass_reason(
    row: dict[str, Any],
    base: dict[str, Any],
    texture: dict[str, float],
    base_texture: dict[str, float],
    source_name: str,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if str(row.get("quality_level", "")).lower() == "hard_fail":
        return False, "candidate_hard_fail"
    if source_name in set(args.exclude_sources):
        return False, "excluded_source"
    if metric_delta(row, base, "off_mask_stitch_length_mm") > args.max_off_mask_increase:
        return False, "off_mask_increase"
    if metric_delta(row, base, "visible_connector_count") > args.max_visible_increase:
        return False, "visible_increase"
    if metric_delta(row, base, "jump_count") > args.max_jump_increase:
        return False, "jump_increase"
    if metric_delta(row, base, "trim_count") > args.max_trim_increase:
        return False, "trim_increase"
    if metric_delta(row, base, "unified_loss") > args.max_loss_increase:
        return False, "loss_increase"

    coverage_delta = metric_delta(row, base, "coverage_ratio")
    precision_delta = metric_delta(row, base, "stitch_precision_ratio")
    score_delta = texture_delta(texture, base_texture, "texture_score")
    satin_like = (
        safe_float(texture.get("satin_columns"))
        + safe_float(texture.get("satin_rail_segments"))
        + safe_float(texture.get("dt_satin_segments"))
    )
    if coverage_delta < args.min_coverage_gain:
        return False, "coverage_gain_too_small"
    if precision_delta < -args.max_precision_drop:
        return False, "precision_drop_too_large"
    if score_delta < args.min_texture_score_gain:
        return False, "texture_gain_too_small"
    if satin_like < args.min_satin_like_segments:
        return False, "not_enough_satin_like_segments"
    return True, "professional_texture_allowed"


def professional_utility(row: dict[str, Any], base: dict[str, Any], texture: dict[str, float], base_texture: dict[str, float], args: argparse.Namespace) -> float:
    coverage_delta = metric_delta(row, base, "coverage_ratio")
    precision_delta = metric_delta(row, base, "stitch_precision_ratio")
    return round(
        args.texture_weight * texture_delta(texture, base_texture, "texture_score")
        + args.coverage_weight * coverage_delta
        + args.precision_weight * precision_delta
        - args.loss_weight * max(0.0, metric_delta(row, base, "unified_loss"))
        - args.jump_weight * max(0.0, metric_delta(row, base, "jump_count"))
        - args.trim_weight * max(0.0, metric_delta(row, base, "trim_count"))
        - args.off_mask_weight * max(0.0, metric_delta(row, base, "off_mask_stitch_length_mm"))
        - args.visible_weight * max(0.0, metric_delta(row, base, "visible_connector_count")),
        8,
    )


def build_candidate_row(
    sample: dict[str, str],
    geometry: dict[str, Any],
    candidate_name: str,
    candidate_root: str,
    row: dict[str, Any],
    base: dict[str, Any],
    texture: dict[str, float],
    base_texture: dict[str, float],
    allowed: bool,
    reason: str,
    utility: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "source_name": sample.get("source_name", ""),
        "category": sample.get("category", ""),
        "phase": sample.get("phase", ""),
        "target_split": sample.get("target_split", ""),
        "candidate": candidate_name,
        "candidate_root": candidate_root,
        "candidate_kind": candidate_kind(candidate_name),
        "is_base_keep": 1 if candidate_name == "base_keep" else 0,
        "is_satin_like": 1 if candidate_is_satin_like(texture) else 0,
        "allowed_by_professional_gate": 1 if allowed else 0,
        "gate_reason": reason,
        "professional_utility": utility,
    }
    output.update(geometry)
    for key in METRIC_KEYS:
        output[key] = row.get(key, "")
        output[f"base_{key}"] = base.get(key, "")
        output[f"delta_{key}"] = metric_delta(row, base, key)
    for key in TEXTURE_KEYS + ("texture_score",):
        output[key] = texture.get(key, 0.0)
        output[f"base_{key}"] = base_texture.get(key, 0.0)
        output[f"delta_{key}"] = texture_delta(texture, base_texture, key)
    return output


def summarize(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        output.append(
            {
                group_key: key,
                "rows": len(items),
                "teacher_choices": sum(1 for item in items if safe_float(item.get("is_teacher_choice")) > 0.5),
                "allowed_candidates": sum(1 for item in items if safe_float(item.get("allowed_by_professional_gate")) > 0.5),
                "mean_professional_utility": avg(items, "professional_utility"),
                "mean_delta_texture_score": avg(items, "delta_texture_score"),
                "mean_delta_unified_loss": avg(items, "delta_unified_loss"),
                "mean_delta_jump_count": avg(items, "delta_jump_count"),
                "mean_delta_stitch_precision_ratio": avg(items, "delta_stitch_precision_ratio"),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stitch-type planner teacher dataset from texture candidates.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "OUTPUT_DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="m2_75_stitch_type_teacher_dataset")
    parser.add_argument("--exclude-sources", nargs="*", default=["QuickDraw", "TextRender"])
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-jump-increase", type=float, default=8.0)
    parser.add_argument("--max-trim-increase", type=float, default=3.0)
    parser.add_argument("--max-loss-increase", type=float, default=0.03)
    parser.add_argument("--min-coverage-gain", type=float, default=-0.015)
    parser.add_argument("--max-precision-drop", type=float, default=0.04)
    parser.add_argument("--min-texture-score-gain", type=float, default=0.35)
    parser.add_argument("--min-satin-like-segments", type=float, default=35.0)
    parser.add_argument("--texture-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=2.0)
    parser.add_argument("--precision-weight", type=float, default=3.0)
    parser.add_argument("--loss-weight", type=float, default=6.0)
    parser.add_argument("--jump-weight", type=float, default=0.15)
    parser.add_argument("--trim-weight", type=float, default=0.10)
    parser.add_argument("--off-mask-weight", type=float, default=20.0)
    parser.add_argument("--visible-weight", type=float, default=4.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["sample_id"]: row for row in read_csv(dataset_dir / "manifest.csv")}
    base_root = Path(args.base_output_dir)
    base_rows = {row["sample_id"]: row for row in read_csv(Path(args.base_selection))}
    candidates = parse_candidate_specs(args.candidate)

    candidate_rows: list[dict[str, Any]] = []
    teacher_choices: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()

    for sample_id, base in base_rows.items():
        sample = manifest[sample_id]
        geometry = sample_geometry(dataset_dir, sample)
        base_texture = load_texture(base_root, sample_id)
        base_row = build_candidate_row(
            sample,
            geometry,
            "base_keep",
            str(base_root),
            base,
            base,
            base_texture,
            base_texture,
            True,
            "base_keep",
            0.0,
        )
        per_sample = [base_row]

        for name, root, rows, coverage_rows in candidates:
            candidate = dict(rows.get(sample_id, {}))
            candidate_coverage = coverage_rows.get(sample_id, {})
            if not candidate or not candidate_coverage:
                failure_counts[f"{name}:missing_candidate"] += 1
                continue
            for key in ("coverage_ratio", "stitch_precision_ratio"):
                candidate[key] = candidate_coverage.get(key, candidate.get(key, ""))
            texture = load_texture(root, sample_id)
            allowed, reason = pass_reason(candidate, base, texture, base_texture, sample.get("source_name", ""), args)
            failure_counts[f"{name}:{reason}"] += 1
            utility = professional_utility(candidate, base, texture, base_texture, args)
            per_sample.append(
                build_candidate_row(
                    sample,
                    geometry,
                    name,
                    str(root),
                    candidate,
                    base,
                    texture,
                    base_texture,
                    allowed,
                    reason,
                    utility,
                )
            )

        allowed_candidates = [row for row in per_sample if safe_float(row.get("allowed_by_professional_gate")) > 0.5 and row["candidate"] != "base_keep"]
        if allowed_candidates:
            teacher = max(allowed_candidates, key=lambda row: safe_float(row.get("professional_utility")))
        else:
            teacher = base_row
        for row in per_sample:
            row["is_teacher_choice"] = 1 if row["candidate"] == teacher["candidate"] else 0
            row["teacher_candidate"] = teacher["candidate"]
            row["teacher_kind"] = teacher["candidate_kind"]
            row["teacher_is_texture_switch"] = 1 if teacher["candidate"] != "base_keep" else 0
            candidate_rows.append(row)
        teacher_choices.append(dict(teacher))

    summary = {
        "model_id": args.model_id,
        "samples": len(base_rows),
        "candidate_rows": len(candidate_rows),
        "teacher_texture_switches": sum(1 for row in teacher_choices if row["candidate"] != "base_keep"),
        "teacher_keep_base": sum(1 for row in teacher_choices if row["candidate"] == "base_keep"),
        "teacher_kinds": dict(Counter(str(row["candidate_kind"]) for row in teacher_choices)),
        "allowed_candidate_rows": sum(1 for row in candidate_rows if safe_float(row.get("allowed_by_professional_gate")) > 0.5),
        "gate_failure_counts": dict(sorted(failure_counts.items())),
        "teacher_mean_unified_loss": avg(teacher_choices, "unified_loss"),
        "teacher_mean_jump_count": avg(teacher_choices, "jump_count"),
        "teacher_mean_trim_count": avg(teacher_choices, "trim_count"),
        "teacher_mean_off_mask_stitch_length_mm": avg(teacher_choices, "off_mask_stitch_length_mm"),
        "teacher_mean_visible_connector_count": avg(teacher_choices, "visible_connector_count"),
        "teacher_mean_coverage_ratio": avg(teacher_choices, "coverage_ratio"),
        "teacher_mean_stitch_precision_ratio": avg(teacher_choices, "stitch_precision_ratio"),
        "teacher_mean_texture_score": avg(teacher_choices, "texture_score"),
        "args": vars(args),
    }
    write_csv(candidate_rows, output_dir / "stitch_type_candidate_rows.csv")
    write_csv(teacher_choices, output_dir / "stitch_type_teacher_choices.csv")
    write_csv(summarize(candidate_rows, "candidate_kind"), output_dir / "stitch_type_by_candidate_kind.csv")
    write_csv(summarize(candidate_rows, "source_name"), output_dir / "stitch_type_by_source.csv")
    write_csv(summarize(candidate_rows, "gate_reason"), output_dir / "stitch_type_by_gate_reason.csv")
    write_json(summary, output_dir / "stitch_type_teacher_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
