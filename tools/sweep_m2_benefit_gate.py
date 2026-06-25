from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from apply_m2_candidate_selector import (
    choose_with_benefit_gate,
    semantic_rerank,
    summarize,
    write_csv,
)
from train_m2_candidate_selector import (
    TASK_PROFILE_PRESETS,
    apply_task_profile,
    build_candidate_rows,
    groups_by_sample,
    predict,
    safe_float,
    selectable_candidates,
)


def parse_csv_floats(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return [float(item) for item in values]


def parse_csv_strings(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def gate_objective(summary: dict[str, Any], min_mean_coverage: float, max_hard_fail: int) -> float:
    hard_fail = int(summary.get("hard_fail", 0))
    coverage = safe_float(summary.get("mean_coverage_ratio"))
    loss = safe_float(summary.get("mean_unified_loss"))
    jump = safe_float(summary.get("mean_jump_count"))
    trim = safe_float(summary.get("mean_trim_count"))
    off_mask = safe_float(summary.get("mean_off_mask_stitch_length_mm"))
    visible = safe_float(summary.get("mean_visible_connector_count"))
    deficit = max(0.0, min_mean_coverage - coverage)
    hard_penalty = 10.0 * max(0, hard_fail - max_hard_fail)
    return (
        loss
        + 0.003 * jump
        + 0.002 * trim
        + 0.010 * off_mask
        + 0.002 * visible
        + 0.500 * deficit
        + hard_penalty
    )


def select_rows_for_config(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    profile: str,
    strength: float,
    coverage_floor: float,
    penalty_weight: float,
    min_coverage: float,
    min_jump_gain: float,
    min_precision: float,
    max_loss_slack: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}
    selected: list[dict[str, Any]] = []
    reverse = str(model.get("select_direction", "min")) == "max"
    for sample_id, group in sorted(groups_by_sample(rows).items()):
        selectable = selectable_candidates(
            group,
            args.exclude_hard_fail,
            False,
            args.flat_min_coverage,
            args.line_min_coverage,
            0.0,
            "branch",
            coverage_floor_line_sources,
            False,
            0.05,
            15.0,
            3.0,
            0.75,
            0.80,
            False,
            0.05,
            20.0,
            3.0,
            0.70,
            0.80,
        )
        predictions = predict(model, selectable)
        ranked = sorted(zip(selectable, predictions), key=lambda item: item[1], reverse=reverse)
        semantic_ranked = semantic_rerank(
            ranked,
            profile,
            strength,
            coverage_floor,
            penalty_weight,
        )
        chosen, predicted_score, semantic_score, calibrated_score, gate_details = choose_with_benefit_gate(
            semantic_ranked,
            coverage_floor,
            min_coverage,
            min_jump_gain,
            min_precision,
            max_loss_slack,
        )
        selected.append(
            {
                "sample_id": sample_id,
                "source_name": chosen.get("source_name", ""),
                "category": chosen.get("category", ""),
                "task_profile": profile,
                "chosen_candidate": chosen["candidate"],
                "predicted_score": round(float(predicted_score), 8),
                "semantic_score": round(float(semantic_score), 8),
                "calibrated_score": round(float(calibrated_score), 8),
                "coverage_floor": coverage_floor,
                "coverage_gate_mode": gate_details.get("coverage_gate_mode", ""),
                "coverage_gate_baseline_candidate": gate_details.get("coverage_gate_baseline_candidate", ""),
                "coverage_gate_baseline_coverage": gate_details.get("coverage_gate_baseline_coverage", ""),
                "coverage_gate_baseline_jump": gate_details.get("coverage_gate_baseline_jump", ""),
                "coverage_gate_jump_gain": gate_details.get("coverage_gate_jump_gain", ""),
                "coverage_gate_loss_slack": gate_details.get("coverage_gate_loss_slack", ""),
                "oracle_score": chosen["oracle_score"],
                "quality_level": chosen["oracle_quality_level"],
                "unified_loss": chosen["unified_loss"],
                "jump_count": chosen["jump_count"],
                "trim_count": chosen["trim_count"],
                "off_mask_stitch_length_mm": chosen["off_mask_stitch_length_mm"],
                "visible_connector_count": chosen["visible_connector_count"],
                "coverage_ratio": chosen["coverage_ratio"],
                "stitch_precision_ratio": chosen["stitch_precision_ratio"],
            }
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep M2.50 benefit-gate thresholds without copying DST outputs.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--profiles", default="balanced,low_jump")
    parser.add_argument("--strengths", default="0.90")
    parser.add_argument("--coverage-floors", default="0.85,0.90")
    parser.add_argument("--penalty-weights", default="0.20,0.35,0.50")
    parser.add_argument("--min-coverages", default="0.60,0.65,0.70")
    parser.add_argument("--min-jump-gains", default="2,3,4")
    parser.add_argument("--min-precisions", default="0.78,0.80,0.85")
    parser.add_argument("--max-loss-slacks", default="0.00,0.02,0.04")
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument("--precision-weight", type=float, default=0.15)
    parser.add_argument("--hard-fail-penalty", type=float, default=0.04)
    parser.add_argument("--coverage-floor-line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--exclude-hard-fail", action="store_true")
    parser.add_argument("--min-mean-coverage", type=float, default=0.90)
    parser.add_argument("--max-hard-fail", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = read_json(Path(args.model))
    candidates = [(name, Path(path)) for name, path in args.candidate]
    base_rows = build_candidate_rows(
        Path(args.dataset_dir),
        candidates,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_weight,
        args.precision_weight,
        args.hard_fail_penalty,
    )
    model_task_profile_presets = model.get("task_profile_presets") or {}
    available_presets = model_task_profile_presets if isinstance(model_task_profile_presets, dict) and model_task_profile_presets else TASK_PROFILE_PRESETS
    profile_models = model.get("profile_models") if isinstance(model.get("profile_models"), dict) else {}

    profiles = parse_csv_strings(args.profiles)
    strengths = parse_csv_floats(args.strengths)
    coverage_floors = parse_csv_floats(args.coverage_floors)
    penalty_weights = parse_csv_floats(args.penalty_weights)
    min_coverages = parse_csv_floats(args.min_coverages)
    min_jump_gains = parse_csv_floats(args.min_jump_gains)
    min_precisions = parse_csv_floats(args.min_precisions)
    max_loss_slacks = parse_csv_floats(args.max_loss_slacks)

    summary_rows: list[dict[str, Any]] = []
    best_by_profile: dict[str, dict[str, Any]] = {}
    best_selected_by_profile: dict[str, list[dict[str, Any]]] = {}
    for profile in profiles:
        if profile not in available_presets:
            raise ValueError(f"Unknown profile: {profile}")
        profiled_rows = apply_task_profile(base_rows, profile, available_presets)
        active_model = profile_models.get(profile, model)
        for strength, coverage_floor, penalty_weight, min_coverage, min_jump_gain, min_precision, max_loss_slack in itertools.product(
            strengths,
            coverage_floors,
            penalty_weights,
            min_coverages,
            min_jump_gains,
            min_precisions,
            max_loss_slacks,
        ):
            selected = select_rows_for_config(
                profiled_rows,
                active_model,
                profile,
                strength,
                coverage_floor,
                penalty_weight,
                min_coverage,
                min_jump_gain,
                min_precision,
                max_loss_slack,
                args,
            )
            summary = summarize(selected)
            modes = {
                mode: sum(1 for row in selected if row.get("coverage_gate_mode") == mode)
                for mode in sorted({str(row.get("coverage_gate_mode", "")) for row in selected if row.get("coverage_gate_mode")})
            }
            row = {
                "profile": profile,
                "strength": strength,
                "coverage_floor": coverage_floor,
                "penalty_weight": penalty_weight,
                "min_coverage": min_coverage,
                "min_jump_gain": min_jump_gain,
                "min_precision": min_precision,
                "max_loss_slack": max_loss_slack,
                "objective": round(gate_objective(summary, args.min_mean_coverage, args.max_hard_fail), 8),
                "samples": summary["samples"],
                "hard_fail": summary["hard_fail"],
                "mean_unified_loss": summary["mean_unified_loss"],
                "mean_jump_count": summary["mean_jump_count"],
                "mean_trim_count": summary["mean_trim_count"],
                "mean_off_mask_stitch_length_mm": summary["mean_off_mask_stitch_length_mm"],
                "mean_visible_connector_count": summary["mean_visible_connector_count"],
                "mean_coverage_ratio": summary["mean_coverage_ratio"],
                "mean_stitch_precision_ratio": summary["mean_stitch_precision_ratio"],
                "coverage_gate_modes": json.dumps(modes, ensure_ascii=False, sort_keys=True),
                "chosen_counts": json.dumps(summary["chosen_counts"], ensure_ascii=False, sort_keys=True),
            }
            summary_rows.append(row)
            current_best = best_by_profile.get(profile)
            if current_best is None or row["objective"] < current_best["objective"]:
                best_by_profile[profile] = row
                best_selected_by_profile[profile] = selected

    summary_rows = sorted(summary_rows, key=lambda row: (row["profile"], row["objective"]))
    write_csv(summary_rows, output_dir / "benefit_gate_sweep_rows.csv")
    best_rows = [best_by_profile[profile] for profile in profiles]
    write_csv(best_rows, output_dir / "benefit_gate_best_rows.csv")
    for profile, selected in best_selected_by_profile.items():
        write_csv(selected, output_dir / f"best_selected_{profile}.csv")
    write_json(
        {
            "dataset_dir": args.dataset_dir,
            "model": args.model,
            "profiles": profiles,
            "grid_size": len(summary_rows),
            "min_mean_coverage": args.min_mean_coverage,
            "max_hard_fail": args.max_hard_fail,
            "best_by_profile": best_rows,
        },
        output_dir / "benefit_gate_sweep_summary.json",
    )
    print(json.dumps({"grid_size": len(summary_rows), "best_by_profile": best_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

