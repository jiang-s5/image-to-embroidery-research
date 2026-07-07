"""Select DST candidates with an explicit professional-texture objective.

M2.86 is intentionally conservative: it requires positive preview gain while
strongly limiting loss, jump, trim, off-mask, visible-connector, coverage, and
precision regressions. M2.90 explores the next research target: allowing a
bounded execution tradeoff when the DST-derived preview-texture gain is large.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMMAND_METRICS = [
    "unified_loss",
    "exec_score",
    "visual_risk",
    "jump_count",
    "trim_count",
    "jump_path_mm",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "visible_connector_length_mm",
    "coverage_ratio",
    "stitch_precision_ratio",
]

PREVIEW_METRICS = [
    "professional_preview_score",
    "generator_texture_score",
    "angle_entropy",
    "dominant_angle_mass",
    "stitch_density_per_100mm2",
    "rhythm_score",
    "coverage_score",
    "density_score",
    "directional_score",
    "family_texture_score_norm",
    "safety_score",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(safe_float(row.get(key)) for row in rows) / len(rows), 8)


def hard_fail(row: dict[str, Any]) -> bool:
    return str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower() == "hard_fail"


def gate_candidate(row: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if str(row.get("round_trip_parse_success", "1")).lower() in {"0", "false"}:
        return False, "round_trip_parse_failed"
    if hard_fail(row):
        return False, "hard_fail_quality_level"
    if safe_float(row.get("delta_professional_preview_score")) < args.min_preview_gain:
        return False, "preview_gain_too_small"
    if safe_float(row.get("delta_generator_texture_score")) < args.min_texture_gain:
        return False, "texture_regression_too_large"
    if safe_float(row.get("visible_connector_count")) > safe_float(baseline.get("visible_connector_count")) + args.max_visible_increase:
        return False, "visible_connector_absolute"
    if safe_float(row.get("delta_visible_connector_count")) > args.max_visible_increase:
        return False, "visible_connector_increase"
    if safe_float(row.get("off_mask_stitch_length_mm")) > args.max_abs_off_mask:
        return False, "off_mask_absolute"
    if safe_float(row.get("delta_off_mask_stitch_length_mm")) > args.max_off_mask_increase:
        return False, "off_mask_increase"
    if safe_float(row.get("delta_unified_loss")) > args.max_loss_increase:
        return False, "loss_increase"
    if safe_float(row.get("delta_jump_count")) > args.max_jump_increase:
        return False, "jump_increase"
    if safe_float(row.get("delta_trim_count")) > args.max_trim_increase:
        return False, "trim_increase"
    if safe_float(row.get("delta_coverage_ratio")) < -args.max_coverage_drop:
        return False, "coverage_drop"
    if safe_float(row.get("delta_stitch_precision_ratio")) < -args.max_precision_drop:
        return False, "precision_drop"
    if safe_float(row.get("stitch_precision_ratio")) < args.min_abs_precision:
        return False, "precision_absolute"
    if safe_float(row.get("coverage_ratio")) < args.min_abs_coverage:
        return False, "coverage_absolute"
    return True, "professional_texture_candidate_pass"


def professional_texture_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    preview_gain = safe_float(row.get("delta_professional_preview_score"))
    texture_gain = safe_float(row.get("delta_generator_texture_score"))
    family_texture_gain = safe_float(row.get("delta_family_texture_score_norm"))
    directional_gain = safe_float(row.get("delta_directional_score"))
    rhythm_gain = safe_float(row.get("delta_rhythm_score"))
    coverage_gain = safe_float(row.get("delta_coverage_ratio"))
    precision_drop = max(0.0, -safe_float(row.get("delta_stitch_precision_ratio")))
    loss_increase = max(0.0, safe_float(row.get("delta_unified_loss")))
    jump_increase = max(0.0, safe_float(row.get("delta_jump_count")))
    trim_increase = max(0.0, safe_float(row.get("delta_trim_count")))
    family_bonus = 0.0
    if str(row.get("candidate_family")) in {"satin_like", "dt_satin", "outline_border"}:
        family_bonus += args.structured_family_bonus
    if str(row.get("candidate_family")) in {"edgewalk_fill", "nearestrow_fill"}:
        family_bonus += args.fill_family_bonus
    score = (
        args.preview_weight * preview_gain
        + args.texture_weight * texture_gain
        + args.family_texture_weight * family_texture_gain
        + args.direction_weight * directional_gain
        + args.rhythm_weight * rhythm_gain
        + args.coverage_weight * coverage_gain
        + family_bonus
        - args.loss_penalty * loss_increase
        - args.jump_penalty * jump_increase
        - args.trim_penalty * trim_increase
        - args.precision_drop_penalty * precision_drop
    )
    return round(score, 8)


def summarize(rows: list[dict[str, Any]], selected_key: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(rows),
        "professional_texture_switches": sum(1 for row in rows if str(row.get(selected_key)) == "professional_texture_candidate"),
        "hard_fail": sum(1 for row in rows if hard_fail(row)),
        "selected_candidate_counts": dict(Counter(str(row.get("candidate", "")) for row in rows)),
        "selected_family_counts": dict(Counter(str(row.get("deployed_family", row.get("candidate_family", ""))) for row in rows)),
    }
    for key in COMMAND_METRICS + PREVIEW_METRICS:
        summary[f"mean_{key}"] = avg(rows, key)
    return summary


def group_summary(rows: list[dict[str, Any]], key: str, selected_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out = []
    for value, items in sorted(groups.items()):
        item = {key: value}
        item.update(summarize(items, selected_key))
        out.append(item)
    return out


def metric_deltas(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    keys = [
        "samples",
        "hard_fail",
        "mean_unified_loss",
        "mean_jump_count",
        "mean_trim_count",
        "mean_off_mask_stitch_length_mm",
        "mean_visible_connector_count",
        "mean_coverage_ratio",
        "mean_stitch_precision_ratio",
        "mean_professional_preview_score",
        "mean_generator_texture_score",
        "mean_family_texture_score_norm",
    ]
    return {
        f"delta_{key}": round(safe_float(current.get(key)) - safe_float(reference.get(key)), 8)
        for key in keys
    }


def build_policy(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates = read_csv(args.candidate_rows)
    baselines = {row["sample_id"]: row for row in read_csv(args.baseline_rows) if row.get("sample_id")}

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_rows: list[dict[str, Any]] = []
    for row in candidates:
        sample_id = str(row.get("sample_id", ""))
        baseline = baselines.get(sample_id)
        if not baseline:
            continue
        passed, reason = gate_candidate(row, baseline, args)
        out = dict(row)
        out["professional_texture_gate_pass"] = int(passed)
        out["professional_texture_gate_reason"] = reason
        out["professional_texture_objective_score"] = professional_texture_score(out, args) if passed else -999.0
        decision_rows.append(out)
        if passed:
            by_sample[sample_id].append(out)

    selected_rows: list[dict[str, Any]] = []
    for sample_id, baseline in sorted(baselines.items()):
        passing = by_sample.get(sample_id, [])
        if passing:
            selected = max(passing, key=lambda row: safe_float(row.get("professional_texture_objective_score"), -999.0))
            if safe_float(selected.get("professional_texture_objective_score"), -999.0) < args.min_objective_score:
                selected = dict(baseline)
                selected["professional_texture_selected_from"] = "baseline"
                selected["professional_texture_objective_score"] = 0.0
            else:
                selected = dict(selected)
                selected["professional_texture_selected_from"] = "professional_texture_candidate"
        else:
            selected = dict(baseline)
            selected["professional_texture_selected_from"] = "baseline"
            selected["professional_texture_objective_score"] = 0.0
        selected["m2_89_candidate"] = baseline.get("candidate", "")
        selected["m2_89_candidate_root"] = baseline.get("candidate_root", "")
        selected["differs_from_m2_89"] = int(
            str(selected.get("candidate", "")) != str(baseline.get("candidate", ""))
            or str(selected.get("candidate_root", "")) != str(baseline.get("candidate_root", ""))
        )
        selected_rows.append(selected)

    selected_summary = summarize(selected_rows, "professional_texture_selected_from")
    baseline_summary = summarize(list(baselines.values()), "learned_preview_selected_from")
    summary = {
        "model_id": args.model_id,
        "candidate_rows": str(args.candidate_rows),
        "baseline_rows": str(args.baseline_rows),
        "policy": {
            "min_preview_gain": args.min_preview_gain,
            "min_texture_gain": args.min_texture_gain,
            "max_visible_increase": args.max_visible_increase,
            "max_abs_off_mask": args.max_abs_off_mask,
            "max_off_mask_increase": args.max_off_mask_increase,
            "max_loss_increase": args.max_loss_increase,
            "max_jump_increase": args.max_jump_increase,
            "max_trim_increase": args.max_trim_increase,
            "max_coverage_drop": args.max_coverage_drop,
            "max_precision_drop": args.max_precision_drop,
            "min_abs_precision": args.min_abs_precision,
            "min_abs_coverage": args.min_abs_coverage,
            "score_weights": {
                "preview_weight": args.preview_weight,
                "texture_weight": args.texture_weight,
                "family_texture_weight": args.family_texture_weight,
                "direction_weight": args.direction_weight,
                "rhythm_weight": args.rhythm_weight,
                "coverage_weight": args.coverage_weight,
                "loss_penalty": args.loss_penalty,
                "jump_penalty": args.jump_penalty,
                "trim_penalty": args.trim_penalty,
                "precision_drop_penalty": args.precision_drop_penalty,
                "structured_family_bonus": args.structured_family_bonus,
                "fill_family_bonus": args.fill_family_bonus,
            },
        },
        "candidate_audit": {
            "candidate_rows": len(decision_rows),
            "gate_pass_rows": sum(1 for row in decision_rows if int(row.get("professional_texture_gate_pass", 0)) == 1),
            "gate_reasons": dict(Counter(str(row.get("professional_texture_gate_reason", "")) for row in decision_rows)),
            "gate_pass_by_family": dict(
                Counter(
                    str(row.get("candidate_family", ""))
                    for row in decision_rows
                    if int(row.get("professional_texture_gate_pass", 0)) == 1
                )
            ),
        },
        "baseline_summary": baseline_summary,
        "selected_summary": selected_summary,
        "comparison_to_m2_89": metric_deltas(selected_summary, baseline_summary),
        "by_source": group_summary(selected_rows, "source_name", "professional_texture_selected_from"),
        "by_family": group_summary(selected_rows, "candidate_family", "professional_texture_selected_from"),
        "switch_sample_ids": [
            row["sample_id"]
            for row in selected_rows
            if str(row.get("professional_texture_selected_from")) == "professional_texture_candidate"
        ],
        "interpretation": (
            "M2.90 is an explicit professional-texture objective. It allows bounded execution tradeoffs "
            "only when DST-derived preview texture improves, while keeping hard_fail and visible connectors gated out."
        ),
    }
    return decision_rows, selected_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select candidates with an explicit professional DST-preview texture objective.")
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv"),
    )
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_89_hard_failure_deployment_strict/preview_learned_selected_rows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_90_professional_texture_objective"),
    )
    parser.add_argument("--model-id", default="m2_90_professional_texture_objective")
    parser.add_argument("--min-preview-gain", type=float, default=0.001)
    parser.add_argument("--min-texture-gain", type=float, default=-0.05)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-abs-off-mask", type=float, default=0.5)
    parser.add_argument("--max-off-mask-increase", type=float, default=0.25)
    parser.add_argument("--max-loss-increase", type=float, default=0.12)
    parser.add_argument("--max-jump-increase", type=float, default=20.0)
    parser.add_argument("--max-trim-increase", type=float, default=8.0)
    parser.add_argument("--max-coverage-drop", type=float, default=0.03)
    parser.add_argument("--max-precision-drop", type=float, default=0.12)
    parser.add_argument("--min-abs-precision", type=float, default=0.70)
    parser.add_argument("--min-abs-coverage", type=float, default=0.90)
    parser.add_argument("--min-objective-score", type=float, default=0.0)
    parser.add_argument("--preview-weight", type=float, default=10.0)
    parser.add_argument("--texture-weight", type=float, default=1.0)
    parser.add_argument("--family-texture-weight", type=float, default=0.8)
    parser.add_argument("--direction-weight", type=float, default=0.3)
    parser.add_argument("--rhythm-weight", type=float, default=0.2)
    parser.add_argument("--coverage-weight", type=float, default=0.2)
    parser.add_argument("--loss-penalty", type=float, default=0.7)
    parser.add_argument("--jump-penalty", type=float, default=0.015)
    parser.add_argument("--trim-penalty", type=float, default=0.015)
    parser.add_argument("--precision-drop-penalty", type=float, default=0.5)
    parser.add_argument("--structured-family-bonus", type=float, default=0.005)
    parser.add_argument("--fill-family-bonus", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision_rows, selected_rows, summary = build_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(decision_rows, args.output_dir / "professional_texture_decision_rows.csv")
    write_csv(selected_rows, args.output_dir / "professional_texture_selected_rows.csv")
    write_csv(
        [row for row in selected_rows if str(row.get("professional_texture_selected_from")) == "professional_texture_candidate"],
        args.output_dir / "professional_texture_switched_rows.csv",
    )
    write_json(summary, args.output_dir / "professional_texture_summary.json")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
