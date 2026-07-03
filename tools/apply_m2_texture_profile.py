from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


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


def summarize(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            group_key: key,
            "samples": len(items),
            "texture_switches": sum(1 for item in items if str(item.get("texture_profile_switched", "")).lower() == "true"),
            "hard_fail": sum(1 for item in items if str(item.get("quality_level", "")).lower() == "hard_fail"),
        }
        for metric in METRIC_KEYS + TEXTURE_KEYS + ("texture_score",):
            summary[f"mean_{metric}"] = avg(items, metric)
        output.append(summary)
    return output


def metric_delta(candidate: dict[str, Any], base: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(base.get(key)), 8)


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


def copy_outputs(source_root: Path, sample_id: str, output_root: Path) -> None:
    sample_out = output_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json", "coverage.json", "coverage_report.json"):
        source = source_root / sample_id / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def candidate_passes(
    candidate: dict[str, Any],
    candidate_coverage: dict[str, Any],
    base: dict[str, Any],
    base_texture: dict[str, float],
    candidate_texture: dict[str, float],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if str(candidate.get("quality_level", "")).lower() == "hard_fail":
        return False, "candidate_hard_fail"
    if str(base.get("source_name", "")) in set(args.exclude_sources):
        return False, "excluded_source"
    if metric_delta(candidate, base, "off_mask_stitch_length_mm") > args.max_off_mask_increase:
        return False, "off_mask_increase"
    if metric_delta(candidate, base, "visible_connector_count") > args.max_visible_increase:
        return False, "visible_increase"
    if metric_delta(candidate, base, "jump_count") > args.max_jump_increase:
        return False, "jump_increase"
    if metric_delta(candidate, base, "trim_count") > args.max_trim_increase:
        return False, "trim_increase"
    if metric_delta(candidate, base, "unified_loss") > args.max_loss_increase:
        return False, "loss_increase"

    coverage_delta = safe_float(candidate_coverage.get("coverage_ratio")) - safe_float(base.get("coverage_ratio"))
    precision_delta = safe_float(candidate_coverage.get("stitch_precision_ratio")) - safe_float(base.get("stitch_precision_ratio"))
    texture_delta = safe_float(candidate_texture.get("texture_score")) - safe_float(base_texture.get("texture_score"))
    satin_like = safe_float(candidate_texture.get("satin_rail_segments")) + safe_float(candidate_texture.get("dt_satin_segments"))

    if coverage_delta < args.min_coverage_gain:
        return False, "coverage_gain_too_small"
    if precision_delta < -args.max_precision_drop:
        return False, "precision_drop_too_large"
    if texture_delta < args.min_texture_score_gain:
        return False, "texture_gain_too_small"
    if satin_like < args.min_satin_like_segments:
        return False, "not_enough_satin_like_segments"
    return True, "texture_layer_safe"


def candidate_rank(
    candidate: dict[str, Any],
    candidate_coverage: dict[str, Any],
    base: dict[str, Any],
    base_texture: dict[str, float],
    candidate_texture: dict[str, float],
) -> float:
    texture_delta = safe_float(candidate_texture.get("texture_score")) - safe_float(base_texture.get("texture_score"))
    coverage_delta = safe_float(candidate_coverage.get("coverage_ratio")) - safe_float(base.get("coverage_ratio"))
    precision_delta = safe_float(candidate_coverage.get("stitch_precision_ratio")) - safe_float(base.get("stitch_precision_ratio"))
    return round(
        texture_delta
        + 2.0 * max(0.0, coverage_delta)
        + 3.0 * precision_delta
        - 6.0 * max(0.0, metric_delta(candidate, base, "unified_loss"))
        - 0.15 * max(0.0, metric_delta(candidate, base, "jump_count"))
        - 0.10 * max(0.0, metric_delta(candidate, base, "trim_count")),
        8,
    )


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely promote stitch-type texture candidates on top of M2.72.")
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "OUTPUT_DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-sources", nargs="*", default=["QuickDraw", "TextRender"])
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-jump-increase", type=float, default=0.0)
    parser.add_argument("--max-trim-increase", type=float, default=0.0)
    parser.add_argument("--max-loss-increase", type=float, default=0.002)
    parser.add_argument("--min-coverage-gain", type=float, default=0.0)
    parser.add_argument("--max-precision-drop", type=float, default=0.01)
    parser.add_argument("--min-texture-score-gain", type=float, default=0.5)
    parser.add_argument("--min-satin-like-segments", type=float, default=60.0)
    args = parser.parse_args()

    base_root = Path(args.base_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv(Path(args.base_selection))
    candidates = parse_candidate_specs(args.candidate)

    selected: list[dict[str, Any]] = []
    switched: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for base in base_rows:
        sample_id = base["sample_id"]
        base_texture = load_texture(base_root, sample_id)
        best: tuple[float, str, Path, dict[str, str], dict[str, str], dict[str, float], str] | None = None
        decision = {
            "sample_id": sample_id,
            "source_name": base.get("source_name", ""),
            "category": base.get("category", ""),
            "base_candidate": base.get("chosen_candidate", ""),
            "decision": "base_kept",
            "reason": "no_texture_candidate_passed",
            **{f"base_{key}": value for key, value in base_texture.items()},
        }
        rejected_reasons: list[str] = []

        for name, root, rows, coverage_rows in candidates:
            candidate = rows.get(sample_id)
            candidate_coverage = coverage_rows.get(sample_id)
            if not candidate or not candidate_coverage:
                rejected_reasons.append(f"{name}:missing_candidate")
                continue
            candidate_texture = load_texture(root, sample_id)
            allowed, reason = candidate_passes(candidate, candidate_coverage, base, base_texture, candidate_texture, args)
            if not allowed:
                rejected_reasons.append(f"{name}:{reason}")
                continue
            rank = candidate_rank(candidate, candidate_coverage, base, base_texture, candidate_texture)
            if best is None or rank > best[0]:
                best = (rank, name, root, candidate, candidate_coverage, candidate_texture, reason)

        if best is None:
            row = dict(base)
            row["texture_profile_switched"] = False
            row["texture_profile_reason"] = "base_kept"
            for key, value in base_texture.items():
                row[key] = value
            decision["rejected_candidates"] = "|".join(rejected_reasons)
            selected.append(row)
            decisions.append(decision)
            copy_outputs(base_root, sample_id, output_dir)
            continue

        rank, name, root, candidate, candidate_coverage, candidate_texture, reason = best
        row = dict(base)
        row["texture_profile_switched"] = True
        row["active_profile"] = "stitch_type_texture"
        row["texture_profile_reason"] = reason
        row["texture_candidate"] = name
        row["base_candidate"] = base.get("chosen_candidate", "")
        row["chosen_candidate"] = str(root.name)
        row["texture_rank_score"] = rank
        for key in ("unified_loss", "jump_count", "trim_count", "off_mask_stitch_length_mm", "visible_connector_count"):
            row[key] = candidate.get(key, "")
            row[f"delta_vs_base_{key}"] = metric_delta(candidate, base, key)
        for key in ("coverage_ratio", "stitch_precision_ratio"):
            row[key] = candidate_coverage.get(key, "")
            row[f"delta_vs_base_{key}"] = round(safe_float(candidate_coverage.get(key)) - safe_float(base.get(key)), 8)
        for key, value in candidate_texture.items():
            row[key] = value
            row[f"base_{key}"] = base_texture.get(key, 0.0)
            row[f"delta_vs_base_{key}"] = round(value - base_texture.get(key, 0.0), 8)
        selected.append(row)
        switched.append(row)
        decisions.append(
            {
                **decision,
                "decision": "switched",
                "reason": reason,
                "selected_candidate": name,
                "texture_rank_score": rank,
                "loss_delta": row.get("delta_vs_base_unified_loss", ""),
                "jump_delta": row.get("delta_vs_base_jump_count", ""),
                "trim_delta": row.get("delta_vs_base_trim_count", ""),
                "coverage_delta": row.get("delta_vs_base_coverage_ratio", ""),
                "precision_delta": row.get("delta_vs_base_stitch_precision_ratio", ""),
                "texture_score_delta": row.get("delta_vs_base_texture_score", ""),
            }
        )
        copy_outputs(root, sample_id, output_dir)

    summary = {
        "model_id": "m2_73_stitch_type_texture_profile",
        "base_profile_system": "m2_72_line_skeleton_fidelity_profile",
        "samples": len(selected),
        "texture_switches": len(switched),
        "hard_fail": sum(1 for row in selected if str(row.get("quality_level", "")).lower() == "hard_fail"),
        **{f"mean_{key}": avg(selected, key) for key in METRIC_KEYS + TEXTURE_KEYS + ("texture_score",)},
        "by_profile": summarize(selected, "active_profile"),
        "by_source": summarize(selected, "source_name"),
        "by_category": summarize(selected, "category"),
        "switched_samples": [
            {
                "sample_id": row.get("sample_id", ""),
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "texture_candidate": row.get("texture_candidate", ""),
                "loss_delta": row.get("delta_vs_base_unified_loss", ""),
                "jump_delta": row.get("delta_vs_base_jump_count", ""),
                "trim_delta": row.get("delta_vs_base_trim_count", ""),
                "coverage_delta": row.get("delta_vs_base_coverage_ratio", ""),
                "precision_delta": row.get("delta_vs_base_stitch_precision_ratio", ""),
                "texture_score_delta": row.get("delta_vs_base_texture_score", ""),
                "satin_rail_segments": row.get("satin_rail_segments", ""),
                "dt_satin_segments": row.get("dt_satin_segments", ""),
            }
            for row in switched
        ],
        "args": vars(args),
    }
    write_csv(selected, output_dir / "texture_selected_rows.csv")
    write_csv(switched, output_dir / "texture_switched_rows.csv")
    write_csv(decisions, output_dir / "texture_decision_rows.csv")
    write_csv(summarize(selected, "active_profile"), output_dir / "texture_by_profile.csv")
    write_csv(summarize(selected, "source_name"), output_dir / "texture_by_source.csv")
    write_csv(summarize(selected, "category"), output_dir / "texture_by_category.csv")
    write_json(summary, output_dir / "texture_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
