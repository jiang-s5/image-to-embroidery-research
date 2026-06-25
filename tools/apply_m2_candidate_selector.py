from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from train_m2_candidate_selector import (
    TASK_PROFILE_PRESETS,
    apply_task_profile,
    build_candidate_rows,
    groups_by_sample,
    predict,
    safe_float,
    selectable_candidates,
)

DEFAULT_PROFILE_COVERAGE_FLOORS: dict[str, float] = {
    "low_loss": 0.90,
    "balanced": 0.90,
    "precision": 0.88,
    "coverage": 0.97,
    "low_jump": 0.90,
    "strict_precision": 0.85,
    "strict_coverage": 0.99,
    "strict_low_jump": 0.90,
}


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return round(mean(safe_float(row.get(key)) for row in rows), 6)

    chosen_counts: dict[str, int] = {}
    for row in rows:
        chosen_counts[str(row["chosen_candidate"])] = chosen_counts.get(str(row["chosen_candidate"]), 0) + 1
    return {
        "samples": len(rows),
        "hard_fail": sum(1 for row in rows if row.get("quality_level") == "hard_fail"),
        "mean_predicted_score": avg("predicted_score"),
        "mean_oracle_score": avg("oracle_score"),
        "mean_unified_loss": avg("unified_loss"),
        "mean_jump_count": avg("jump_count"),
        "mean_trim_count": avg("trim_count"),
        "mean_off_mask_stitch_length_mm": avg("off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg("visible_connector_count"),
        "mean_coverage_ratio": avg("coverage_ratio"),
        "mean_stitch_precision_ratio": avg("stitch_precision_ratio"),
        "chosen_counts": chosen_counts,
    }


def normalized_metric(row: dict[str, Any], key: str, ranges: dict[str, tuple[float, float]]) -> float:
    value = safe_float(row.get(key))
    low, high = ranges.get(key, (value, value))
    span = high - low
    if abs(span) < 1e-9:
        return 0.0
    return max(0.0, min(1.0, (value - low) / span))


def inverse_normalized_metric(row: dict[str, Any], key: str, ranges: dict[str, tuple[float, float]]) -> float:
    return 1.0 - normalized_metric(row, key, ranges)


def profile_semantic_score(row: dict[str, Any], profile: str, ranges: dict[str, tuple[float, float]]) -> float:
    loss = normalized_metric(row, "unified_loss", ranges)
    jump = normalized_metric(row, "jump_count", ranges)
    trim = normalized_metric(row, "trim_count", ranges)
    off_mask = normalized_metric(row, "off_mask_stitch_length_mm", ranges)
    visible = normalized_metric(row, "visible_connector_count", ranges)
    low_coverage = inverse_normalized_metric(row, "coverage_ratio", ranges)
    low_precision = inverse_normalized_metric(row, "stitch_precision_ratio", ranges)

    if profile in {"precision", "strict_precision"}:
        precision_weight = 0.62 if profile == "strict_precision" else 0.52
        return (
            precision_weight * low_precision
            + 0.14 * loss
            + 0.10 * low_coverage
            + 0.08 * jump
            + 0.03 * trim
            + 0.03 * off_mask
        )
    if profile in {"coverage", "strict_coverage"}:
        coverage_weight = 0.62 if profile == "strict_coverage" else 0.52
        return (
            coverage_weight * low_coverage
            + 0.16 * loss
            + 0.10 * low_precision
            + 0.07 * jump
            + 0.03 * trim
            + 0.02 * off_mask
        )
    if profile in {"low_jump", "strict_low_jump"}:
        jump_weight = 0.62 if profile == "strict_low_jump" else 0.52
        return (
            jump_weight * jump
            + 0.12 * trim
            + 0.14 * loss
            + 0.08 * low_coverage
            + 0.02 * low_precision
            + 0.02 * off_mask
        )
    if profile == "low_loss":
        return 0.60 * loss + 0.12 * jump + 0.06 * trim + 0.12 * low_coverage + 0.04 * low_precision + 0.06 * off_mask
    return (
        0.34 * loss
        + 0.12 * jump
        + 0.06 * trim
        + 0.20 * low_coverage
        + 0.14 * low_precision
        + 0.08 * off_mask
        + 0.06 * visible
    )


def semantic_rerank(
    ranked: list[tuple[dict[str, Any], float]],
    profile: str,
    strength: float,
    coverage_penalty_floor: float = -1.0,
    coverage_penalty_weight: float = 0.0,
) -> list[tuple[dict[str, Any], float, float, float]]:
    if not ranked:
        return []
    metric_keys = [
        "unified_loss",
        "jump_count",
        "trim_count",
        "off_mask_stitch_length_mm",
        "visible_connector_count",
        "coverage_ratio",
        "stitch_precision_ratio",
    ]
    rows = [row for row, _score in ranked]
    ranges = {
        key: (
            min(safe_float(row.get(key)) for row in rows),
            max(safe_float(row.get(key)) for row in rows),
        )
        for key in metric_keys
    }
    denom = max(1, len(ranked) - 1)
    reranked: list[tuple[dict[str, Any], float, float, float]] = []
    for index, (row, predicted_score) in enumerate(ranked):
        learned_rank_score = index / denom
        semantic_score = profile_semantic_score(row, profile, ranges)
        if coverage_penalty_floor >= 0.0 and coverage_penalty_weight > 0.0:
            coverage = safe_float(row.get("coverage_ratio"))
            coverage_deficit = max(0.0, coverage_penalty_floor - coverage)
            semantic_score += coverage_penalty_weight * coverage_deficit
        calibrated_score = (1.0 - strength) * learned_rank_score + strength * semantic_score
        reranked.append((row, predicted_score, semantic_score, calibrated_score))
    return sorted(reranked, key=lambda item: item[3])


def parse_profile_thresholds(text: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected PROFILE=VALUE threshold entry, got: {item!r}")
        name, value_text = item.split("=", 1)
        thresholds[name.strip()] = safe_float(value_text.strip())
    return thresholds


def profile_coverage_floor(
    profile: str,
    global_floor: float,
    profile_floors: dict[str, float],
    use_defaults: bool,
) -> float:
    if profile in profile_floors:
        return profile_floors[profile]
    if global_floor >= 0.0:
        return global_floor
    if use_defaults:
        return DEFAULT_PROFILE_COVERAGE_FLOORS.get(profile, 0.90)
    return -1.0


def choose_with_coverage_constraint(
    semantic_ranked: list[tuple[dict[str, Any], float, float, float]],
    min_coverage: float,
    fallback_margin: float,
) -> tuple[dict[str, Any], float, float, float, bool, float]:
    if not semantic_ranked:
        raise ValueError("Cannot choose from an empty ranked list.")
    if min_coverage < 0.0:
        chosen, predicted_score, semantic_score, calibrated_score = semantic_ranked[0]
        return chosen, predicted_score, semantic_score, calibrated_score, True, safe_float(chosen.get("coverage_ratio"))

    valid = [
        item for item in semantic_ranked
        if safe_float(item[0].get("coverage_ratio")) + 1e-9 >= min_coverage
    ]
    if valid:
        chosen, predicted_score, semantic_score, calibrated_score = valid[0]
        return chosen, predicted_score, semantic_score, calibrated_score, True, safe_float(chosen.get("coverage_ratio"))

    relaxed_floor = max(0.0, min_coverage - max(0.0, fallback_margin))
    relaxed = [
        item for item in semantic_ranked
        if safe_float(item[0].get("coverage_ratio")) + 1e-9 >= relaxed_floor
    ]
    if relaxed:
        chosen, predicted_score, semantic_score, calibrated_score = relaxed[0]
        return chosen, predicted_score, semantic_score, calibrated_score, False, safe_float(chosen.get("coverage_ratio"))

    chosen, predicted_score, semantic_score, calibrated_score = semantic_ranked[0]
    return chosen, predicted_score, semantic_score, calibrated_score, False, safe_float(chosen.get("coverage_ratio"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a trained M2 learned candidate selector.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument("--precision-weight", type=float, default=0.15)
    parser.add_argument("--hard-fail-penalty", type=float, default=0.04)
    parser.add_argument("--exclude-hard-fail", action="store_true")
    parser.add_argument("--enforce-coverage-floor", action="store_true")
    parser.add_argument("--coverage-floor-tolerance", type=float, default=0.0)
    parser.add_argument("--coverage-floor-mode", choices=["branch", "source"], default="branch")
    parser.add_argument("--coverage-floor-line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--style-aware-hard-gate", action="store_true")
    parser.add_argument("--style-aware-max-off-mask-mm", type=float, default=0.05)
    parser.add_argument("--style-aware-max-jump-count", type=float, default=15.0)
    parser.add_argument("--style-aware-max-trim-count", type=float, default=3.0)
    parser.add_argument("--style-aware-min-precision", type=float, default=0.75)
    parser.add_argument("--style-aware-min-coverage", type=float, default=0.80)
    parser.add_argument("--mask-fill-hard-gate", action="store_true")
    parser.add_argument("--mask-fill-max-off-mask-mm", type=float, default=0.05)
    parser.add_argument("--mask-fill-max-jump-count", type=float, default=20.0)
    parser.add_argument("--mask-fill-max-trim-count", type=float, default=3.0)
    parser.add_argument("--mask-fill-min-precision", type=float, default=0.70)
    parser.add_argument("--mask-fill-min-coverage", type=float, default=0.80)
    parser.add_argument("--task-profile", default="", help="Task-conditioned profile to apply when the model was trained with --task-profiles.")
    parser.add_argument("--profile-semantic-rerank", action="store_true", help="Apply profile-specific metric calibration after learned ranking.")
    parser.add_argument("--profile-rerank-strength", type=float, default=0.65, help="Blend strength for --profile-semantic-rerank. 0 keeps learned rank; 1 uses semantic metric score.")
    parser.add_argument("--profile-coverage-constrained-rerank", action="store_true", help="After semantic reranking, choose the best candidate that satisfies a profile-aware coverage floor.")
    parser.add_argument("--profile-coverage-soft-penalty", action="store_true", help="Add a profile-aware soft coverage deficit penalty to semantic reranking.")
    parser.add_argument("--profile-min-coverage", type=float, default=-1.0, help="Global coverage floor for --profile-coverage-constrained-rerank. Negative uses defaults or per-profile overrides.")
    parser.add_argument("--profile-min-coverage-by-name", default="", help="Comma-separated profile coverage floors such as balanced=0.93,low_jump=0.90,coverage=0.98.")
    parser.add_argument("--profile-coverage-fallback-margin", type=float, default=0.03, help="If no candidate satisfies the coverage floor, allow this much floor relaxation before falling back to the unconstrained top candidate.")
    parser.add_argument("--profile-coverage-penalty-weight", type=float, default=0.35, help="Soft coverage deficit penalty weight used with --profile-coverage-soft-penalty.")
    parser.add_argument("--disable-default-profile-coverage-floors", action="store_true", help="Disable built-in profile coverage floors when --profile-min-coverage is negative.")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [(name, Path(path)) for name, path in args.candidate]
    candidate_dirs = {name: path for name, path in candidates}
    rows = build_candidate_rows(
        Path(args.dataset_dir),
        candidates,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_weight,
        args.precision_weight,
        args.hard_fail_penalty,
    )
    active_task_profile = ""
    model_task_profiles = model.get("task_profiles") or []
    model_task_profile_presets = model.get("task_profile_presets") or {}
    if model_task_profiles:
        active_task_profile = args.task_profile or str(model.get("default_task_profile") or model_task_profiles[0])
        available_presets = model_task_profile_presets if isinstance(model_task_profile_presets, dict) and model_task_profile_presets else TASK_PROFILE_PRESETS
        if active_task_profile not in available_presets:
            raise ValueError(f"Unknown task profile: {active_task_profile}")
        if active_task_profile not in model_task_profiles:
            raise ValueError(
                f"Task profile {active_task_profile!r} was not present during training. "
                f"Available profiles: {', '.join(str(item) for item in model_task_profiles)}"
            )
        rows = apply_task_profile(rows, active_task_profile, available_presets)
        profile_models = model.get("profile_models")
        if isinstance(profile_models, dict) and active_task_profile in profile_models:
            model = profile_models[active_task_profile]
    elif args.task_profile:
        raise ValueError("--task-profile can only be used with a task-conditioned selector model.")
    profile_min_coverage_by_name = parse_profile_thresholds(args.profile_min_coverage_by_name)
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}
    selected: list[dict[str, Any]] = []
    for sample_id, group in sorted(groups_by_sample(rows).items()):
        selectable = selectable_candidates(
            group,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
            args.style_aware_hard_gate,
            args.style_aware_max_off_mask_mm,
            args.style_aware_max_jump_count,
            args.style_aware_max_trim_count,
            args.style_aware_min_precision,
            args.style_aware_min_coverage,
            args.mask_fill_hard_gate,
            args.mask_fill_max_off_mask_mm,
            args.mask_fill_max_jump_count,
            args.mask_fill_max_trim_count,
            args.mask_fill_min_precision,
            args.mask_fill_min_coverage,
        )
        predictions = predict(model, selectable)
        reverse = str(model.get("select_direction", "min")) == "max"
        ranked = sorted(zip(selectable, predictions), key=lambda item: item[1], reverse=reverse)
        semantic_score = 0.0
        calibrated_score = 0.0
        coverage_constraint_enabled = bool(args.profile_coverage_constrained_rerank)
        coverage_constraint_min = -1.0
        coverage_constraint_met = True
        if args.profile_semantic_rerank:
            profile_name = active_task_profile or "balanced"
            coverage_constraint_min = profile_coverage_floor(
                profile_name,
                args.profile_min_coverage,
                profile_min_coverage_by_name,
                not args.disable_default_profile_coverage_floors,
            )
            semantic_ranked = semantic_rerank(
                ranked,
                profile_name,
                max(0.0, min(1.0, args.profile_rerank_strength)),
                coverage_constraint_min if args.profile_coverage_soft_penalty else -1.0,
                max(0.0, args.profile_coverage_penalty_weight),
            )
            if coverage_constraint_enabled:
                (
                    chosen,
                    predicted_score,
                    semantic_score,
                    calibrated_score,
                    coverage_constraint_met,
                    _coverage_value,
                ) = choose_with_coverage_constraint(
                    semantic_ranked,
                    coverage_constraint_min,
                    args.profile_coverage_fallback_margin,
                )
            else:
                chosen, predicted_score, semantic_score, calibrated_score = semantic_ranked[0]
        else:
            chosen, predicted_score = ranked[0]
        sample_out = output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        source_dir = candidate_dirs[str(chosen["candidate"])] / sample_id
        for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
            source = source_dir / filename
            if source.exists():
                shutil.copy2(source, sample_out / filename)
        selected.append(
            {
                "sample_id": sample_id,
                "source_name": chosen.get("source_name", ""),
                "category": chosen.get("category", ""),
                "task_profile": active_task_profile,
                "chosen_candidate": chosen["candidate"],
                "predicted_score": round(float(predicted_score), 8),
                "semantic_score": round(float(semantic_score), 8),
                "calibrated_score": round(float(calibrated_score), 8),
                "coverage_constraint_enabled": coverage_constraint_enabled,
                "coverage_constraint_min": coverage_constraint_min,
                "coverage_constraint_met": coverage_constraint_met,
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

    write_csv(selected, output_dir / "learned_selected_rows.csv")
    summary = summarize(selected)
    summary["task_profile"] = active_task_profile
    summary["profile_semantic_rerank"] = bool(args.profile_semantic_rerank)
    summary["profile_rerank_strength"] = args.profile_rerank_strength
    summary["profile_coverage_constrained_rerank"] = bool(args.profile_coverage_constrained_rerank)
    summary["profile_coverage_soft_penalty"] = bool(args.profile_coverage_soft_penalty)
    summary["profile_min_coverage"] = args.profile_min_coverage
    summary["profile_min_coverage_by_name"] = profile_min_coverage_by_name
    summary["profile_coverage_fallback_margin"] = args.profile_coverage_fallback_margin
    summary["profile_coverage_penalty_weight"] = args.profile_coverage_penalty_weight
    summary["coverage_constraint_failed_samples"] = sum(
        1 for row in selected
        if row.get("coverage_constraint_enabled") and not row.get("coverage_constraint_met")
    )
    (output_dir / "learned_selected_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
