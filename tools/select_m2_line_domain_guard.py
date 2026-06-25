from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def profile_coverage_floor(profile: str, global_floor: float, profile_floors: dict[str, float]) -> float:
    if profile in profile_floors:
        return profile_floors[profile]
    if global_floor >= 0.0:
        return global_floor
    return 0.85


def is_line_domain(row: dict[str, Any], line_sources: set[str]) -> bool:
    return str(row.get("source_name", "")) in line_sources


def line_guard_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    coverage = safe_float(row.get("coverage_ratio"))
    precision = safe_float(row.get("stitch_precision_ratio"))
    coverage_deficit = max(0.0, args.line_target_coverage - coverage)
    precision_deficit = max(0.0, args.line_target_precision - precision)
    candidate = str(row.get("candidate", "")).lower()
    skeleton_bonus = args.line_skeleton_bonus if "skeleton" in candidate else 0.0
    return (
        args.line_loss_weight * safe_float(row.get("unified_loss"))
        + args.line_jump_weight * safe_float(row.get("jump_count"))
        + args.line_trim_weight * safe_float(row.get("trim_count"))
        + args.line_off_mask_weight * safe_float(row.get("off_mask_stitch_length_mm"))
        + args.line_visible_weight * safe_float(row.get("visible_connector_count"))
        + args.line_coverage_deficit_weight * coverage_deficit
        + args.line_precision_deficit_weight * precision_deficit
        - skeleton_bonus
    )


def passes_line_guard(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        safe_float(row.get("coverage_ratio")) + 1e-9 >= args.line_absolute_min_coverage
        and safe_float(row.get("stitch_precision_ratio")) + 1e-9 >= args.line_absolute_min_precision
        and safe_float(row.get("jump_count")) <= args.line_max_jump_count + 1e-9
        and safe_float(row.get("off_mask_stitch_length_mm")) <= args.line_max_off_mask_mm + 1e-9
        and safe_float(row.get("visible_connector_count")) <= args.line_max_visible_count + 1e-9
        and safe_float(row.get("trim_count")) <= args.line_max_trim_count + 1e-9
    )


def choose_line_guard(
    semantic_ranked: list[tuple[dict[str, Any], float, float, float]],
    baseline: tuple[dict[str, Any], float, float, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, float, float, dict[str, Any]]:
    baseline_row, baseline_predicted, baseline_semantic, baseline_calibrated = baseline
    valid = [item for item in semantic_ranked if passes_line_guard(item[0], args)]
    guard_pool = valid if valid else semantic_ranked
    scored = sorted(
        (
            (line_guard_score(row, args), row, predicted_score, semantic_score, calibrated_score)
            for row, predicted_score, semantic_score, calibrated_score in guard_pool
        ),
        key=lambda item: item[0],
    )
    guard_score, row, predicted_score, semantic_score, calibrated_score = scored[0]
    base_loss = safe_float(baseline_row.get("unified_loss"))
    loss_slack = safe_float(row.get("unified_loss")) - base_loss
    if loss_slack > args.line_max_loss_slack + 1e-9:
        return baseline_row, baseline_predicted, baseline_semantic, baseline_calibrated, {
            "line_guard_mode": "baseline_loss_guard",
            "line_guard_score": line_guard_score(baseline_row, args),
            "line_guard_pool_size": len(guard_pool),
            "line_guard_valid_pool_size": len(valid),
            "line_guard_rejected_candidate": row.get("candidate", ""),
            "line_guard_loss_slack": loss_slack,
        }
    return row, predicted_score, semantic_score, calibrated_score, {
        "line_guard_mode": "line_domain_guard",
        "line_guard_score": guard_score,
        "line_guard_pool_size": len(guard_pool),
        "line_guard_valid_pool_size": len(valid),
        "line_guard_rejected_candidate": "",
        "line_guard_loss_slack": loss_slack,
    }


def choose_line_relative_rescue(
    semantic_ranked: list[tuple[dict[str, Any], float, float, float]],
    baseline: tuple[dict[str, Any], float, float, float],
    current: tuple[dict[str, Any], float, float, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, float, float, dict[str, Any]]:
    baseline_row, _baseline_predicted, _baseline_semantic, _baseline_calibrated = baseline
    current_row, current_predicted, current_semantic, current_calibrated = current
    baseline_loss = safe_float(baseline_row.get("unified_loss"))
    baseline_jump = safe_float(baseline_row.get("jump_count"))
    baseline_coverage = safe_float(baseline_row.get("coverage_ratio"))
    baseline_precision = safe_float(baseline_row.get("stitch_precision_ratio"))
    baseline_visible = safe_float(baseline_row.get("visible_connector_count"))
    baseline_off_mask = safe_float(baseline_row.get("off_mask_stitch_length_mm"))

    rescue_pool: list[tuple[float, dict[str, Any], float, float, float]] = []
    for row, predicted_score, semantic_score, calibrated_score in semantic_ranked:
        loss = safe_float(row.get("unified_loss"))
        jump = safe_float(row.get("jump_count"))
        coverage = safe_float(row.get("coverage_ratio"))
        precision = safe_float(row.get("stitch_precision_ratio"))
        visible = safe_float(row.get("visible_connector_count"))
        off_mask = safe_float(row.get("off_mask_stitch_length_mm"))
        if coverage + 1e-9 < args.line_rescue_min_coverage:
            continue
        if jump > baseline_jump - args.line_rescue_min_jump_gain + 1e-9:
            continue
        if loss > baseline_loss + args.line_rescue_max_loss_slack + 1e-9:
            continue
        if coverage + args.line_rescue_max_coverage_drop + 1e-9 < baseline_coverage:
            continue
        if precision + args.line_rescue_max_precision_drop + 1e-9 < baseline_precision:
            continue
        if visible > baseline_visible + args.line_rescue_max_visible_increase + 1e-9:
            continue
        if off_mask > baseline_off_mask + args.line_rescue_max_off_mask_increase + 1e-9:
            continue
        rescue_score = (
            loss
            + args.line_rescue_jump_weight * jump
            + args.line_rescue_coverage_deficit_weight * max(0.0, args.line_rescue_target_coverage - coverage)
            + args.line_rescue_precision_deficit_weight * max(0.0, args.line_rescue_target_precision - precision)
        )
        rescue_pool.append((rescue_score, row, predicted_score, semantic_score, calibrated_score))

    if not rescue_pool:
        return current_row, current_predicted, current_semantic, current_calibrated, {
            "line_rescue_mode": "no_rescue_candidate",
            "line_rescue_score": "",
            "line_rescue_pool_size": 0,
            "line_rescue_baseline_candidate": baseline_row.get("candidate", ""),
            "line_rescue_baseline_jump": baseline_jump,
            "line_rescue_baseline_loss": baseline_loss,
            "line_rescue_baseline_coverage": baseline_coverage,
        }
    rescue_score, row, predicted_score, semantic_score, calibrated_score = sorted(rescue_pool, key=lambda item: item[0])[0]
    return row, predicted_score, semantic_score, calibrated_score, {
        "line_rescue_mode": "relative_rescue",
        "line_rescue_score": rescue_score,
        "line_rescue_pool_size": len(rescue_pool),
        "line_rescue_baseline_candidate": baseline_row.get("candidate", ""),
        "line_rescue_baseline_jump": baseline_jump,
        "line_rescue_baseline_loss": baseline_loss,
        "line_rescue_baseline_coverage": baseline_coverage,
        "line_rescue_jump_gain": baseline_jump - safe_float(row.get("jump_count")),
        "line_rescue_loss_delta": safe_float(row.get("unified_loss")) - baseline_loss,
        "line_rescue_coverage_delta": safe_float(row.get("coverage_ratio")) - baseline_coverage,
    }


def choose_line_precision_rescue(
    semantic_ranked: list[tuple[dict[str, Any], float, float, float]],
    current: tuple[dict[str, Any], float, float, float],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, float, float, dict[str, Any]]:
    current_row, current_predicted, current_semantic, current_calibrated = current
    current_loss = safe_float(current_row.get("unified_loss"))
    current_jump = safe_float(current_row.get("jump_count"))
    current_trim = safe_float(current_row.get("trim_count"))
    current_coverage = safe_float(current_row.get("coverage_ratio"))
    current_precision = safe_float(current_row.get("stitch_precision_ratio"))
    current_visible = safe_float(current_row.get("visible_connector_count"))
    current_off_mask = safe_float(current_row.get("off_mask_stitch_length_mm"))

    if current_precision + 1e-9 >= args.line_precision_rescue_trigger_precision:
        return current_row, current_predicted, current_semantic, current_calibrated, {
            "line_precision_rescue_mode": "precision_not_low",
            "line_precision_rescue_score": "",
            "line_precision_rescue_pool_size": 0,
            "line_precision_rescue_baseline_candidate": current_row.get("candidate", ""),
            "line_precision_rescue_precision_gain": "",
            "line_precision_rescue_loss_delta": "",
            "line_precision_rescue_jump_delta": "",
            "line_precision_rescue_coverage_delta": "",
        }

    rescue_pool: list[tuple[float, dict[str, Any], float, float, float]] = []
    for row, predicted_score, semantic_score, calibrated_score in semantic_ranked:
        loss = safe_float(row.get("unified_loss"))
        jump = safe_float(row.get("jump_count"))
        trim = safe_float(row.get("trim_count"))
        coverage = safe_float(row.get("coverage_ratio"))
        precision = safe_float(row.get("stitch_precision_ratio"))
        visible = safe_float(row.get("visible_connector_count"))
        off_mask = safe_float(row.get("off_mask_stitch_length_mm"))
        precision_gain = precision - current_precision
        if precision + 1e-9 < args.line_precision_rescue_min_precision:
            continue
        if precision_gain + 1e-9 < args.line_precision_rescue_min_gain:
            continue
        if coverage + 1e-9 < args.line_precision_rescue_min_coverage:
            continue
        if coverage + args.line_precision_rescue_max_coverage_drop + 1e-9 < current_coverage:
            continue
        if loss > current_loss + args.line_precision_rescue_max_loss_slack + 1e-9:
            continue
        if jump > current_jump + args.line_precision_rescue_max_jump_increase + 1e-9:
            continue
        if trim > current_trim + args.line_precision_rescue_max_trim_increase + 1e-9:
            continue
        if visible > current_visible + args.line_precision_rescue_max_visible_increase + 1e-9:
            continue
        if off_mask > current_off_mask + args.line_precision_rescue_max_off_mask_increase + 1e-9:
            continue
        rescue_score = (
            loss
            + args.line_precision_rescue_jump_weight * jump
            + args.line_precision_rescue_trim_weight * trim
            + args.line_precision_rescue_coverage_deficit_weight * max(0.0, args.line_precision_rescue_target_coverage - coverage)
            - args.line_precision_rescue_precision_gain_weight * precision_gain
        )
        rescue_pool.append((rescue_score, row, predicted_score, semantic_score, calibrated_score))

    if not rescue_pool:
        return current_row, current_predicted, current_semantic, current_calibrated, {
            "line_precision_rescue_mode": "no_precision_rescue_candidate",
            "line_precision_rescue_score": "",
            "line_precision_rescue_pool_size": 0,
            "line_precision_rescue_baseline_candidate": current_row.get("candidate", ""),
            "line_precision_rescue_precision_gain": "",
            "line_precision_rescue_loss_delta": "",
            "line_precision_rescue_jump_delta": "",
            "line_precision_rescue_coverage_delta": "",
        }

    rescue_score, row, predicted_score, semantic_score, calibrated_score = sorted(rescue_pool, key=lambda item: item[0])[0]
    return row, predicted_score, semantic_score, calibrated_score, {
        "line_precision_rescue_mode": "precision_rescue",
        "line_precision_rescue_score": rescue_score,
        "line_precision_rescue_pool_size": len(rescue_pool),
        "line_precision_rescue_baseline_candidate": current_row.get("candidate", ""),
        "line_precision_rescue_precision_gain": safe_float(row.get("stitch_precision_ratio")) - current_precision,
        "line_precision_rescue_loss_delta": safe_float(row.get("unified_loss")) - current_loss,
        "line_precision_rescue_jump_delta": safe_float(row.get("jump_count")) - current_jump,
        "line_precision_rescue_coverage_delta": safe_float(row.get("coverage_ratio")) - current_coverage,
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 6)


def source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_name", ""))].append(row)
    output: list[dict[str, Any]] = []
    for source_name, items in sorted(grouped.items()):
        output.append(
            {
                "source_name": source_name,
                "samples": len(items),
                "changed_samples": sum(1 for row in items if str(row.get("changed_by_line_guard", "")).lower() == "true"),
                "hard_fail": sum(1 for row in items if row.get("quality_level") == "hard_fail"),
                "mean_unified_loss": mean_metric(items, "unified_loss"),
                "mean_jump_count": mean_metric(items, "jump_count"),
                "mean_trim_count": mean_metric(items, "trim_count"),
                "mean_off_mask_stitch_length_mm": mean_metric(items, "off_mask_stitch_length_mm"),
                "mean_visible_connector_count": mean_metric(items, "visible_connector_count"),
                "mean_coverage_ratio": mean_metric(items, "coverage_ratio"),
                "mean_stitch_precision_ratio": mean_metric(items, "stitch_precision_ratio"),
            }
        )
    return output


def compare_to_baseline(selected: list[dict[str, Any]], baseline_path: Path) -> list[dict[str, Any]]:
    baseline_rows = {row["sample_id"]: row for row in read_csv(baseline_path)}
    comparison: list[dict[str, Any]] = []
    for row in selected:
        base = baseline_rows.get(str(row["sample_id"]))
        if not base:
            continue
        comparison.append(
            {
                "sample_id": row["sample_id"],
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "baseline_candidate": base.get("chosen_candidate", ""),
                "selected_candidate": row.get("chosen_candidate", ""),
                "changed": str(base.get("chosen_candidate", "")) != str(row.get("chosen_candidate", "")),
                "delta_unified_loss": round(safe_float(row.get("unified_loss")) - safe_float(base.get("unified_loss")), 8),
                "delta_jump_count": round(safe_float(row.get("jump_count")) - safe_float(base.get("jump_count")), 8),
                "delta_trim_count": round(safe_float(row.get("trim_count")) - safe_float(base.get("trim_count")), 8),
                "delta_off_mask_stitch_length_mm": round(
                    safe_float(row.get("off_mask_stitch_length_mm")) - safe_float(base.get("off_mask_stitch_length_mm")), 8
                ),
                "delta_visible_connector_count": round(
                    safe_float(row.get("visible_connector_count")) - safe_float(base.get("visible_connector_count")), 8
                ),
                "delta_coverage_ratio": round(safe_float(row.get("coverage_ratio")) - safe_float(base.get("coverage_ratio")), 8),
                "delta_stitch_precision_ratio": round(
                    safe_float(row.get("stitch_precision_ratio")) - safe_float(base.get("stitch_precision_ratio")), 8
                ),
            }
        )
    return comparison


def anchor_baseline_choice(
    semantic_ranked: list[tuple[dict[str, Any], float, float, float]],
    anchor_row: dict[str, Any] | None,
    fallback: tuple[dict[str, Any], float, float, float],
) -> tuple[dict[str, Any], float, float, float, dict[str, Any]]:
    if not anchor_row:
        return (*fallback, {"anchor_mode": "no_anchor_row", "anchor_candidate": ""})
    anchor_candidate = str(anchor_row.get("chosen_candidate", "")).strip()
    if not anchor_candidate:
        return (*fallback, {"anchor_mode": "missing_anchor_candidate", "anchor_candidate": ""})
    for row, predicted_score, semantic_score, calibrated_score in semantic_ranked:
        if str(row.get("candidate", "")) == anchor_candidate:
            return row, predicted_score, semantic_score, calibrated_score, {
                "anchor_mode": "anchored_baseline",
                "anchor_candidate": anchor_candidate,
            }
    return (*fallback, {"anchor_mode": "anchor_candidate_not_found", "anchor_candidate": anchor_candidate})


def copy_outputs(candidate_dirs: dict[str, Path], chosen: dict[str, Any], sample_out: Path) -> None:
    sample_out.mkdir(parents=True, exist_ok=True)
    source_dir = candidate_dirs[str(chosen["candidate"])] / str(chosen["sample_id"])
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply M2.55 line-domain guard on top of the robust M2 benefit gate.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--baseline-selection", default="")
    parser.add_argument("--anchor-baseline-selection", default="")
    parser.add_argument("--profile", default="low_jump")
    parser.add_argument("--exclude-hard-fail", action="store_true")
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument("--precision-weight", type=float, default=0.15)
    parser.add_argument("--hard-fail-penalty", type=float, default=0.04)
    parser.add_argument("--profile-rerank-strength", type=float, default=0.90)
    parser.add_argument("--profile-coverage-penalty-weight", type=float, default=0.20)
    parser.add_argument("--profile-min-coverage", type=float, default=-1.0)
    parser.add_argument("--profile-min-coverage-by-name", default="balanced=0.85,low_jump=0.85,strict_low_jump=0.85")
    parser.add_argument("--profile-benefit-min-coverage", type=float, default=0.60)
    parser.add_argument("--profile-benefit-min-jump-gain", type=float, default=2.0)
    parser.add_argument("--profile-benefit-min-precision", type=float, default=0.78)
    parser.add_argument("--profile-benefit-max-loss-slack", type=float, default=0.0)
    parser.add_argument("--coverage-floor-line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--line-domain-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--line-target-coverage", type=float, default=0.88)
    parser.add_argument("--line-target-precision", type=float, default=0.75)
    parser.add_argument("--line-absolute-min-coverage", type=float, default=0.60)
    parser.add_argument("--line-absolute-min-precision", type=float, default=0.45)
    parser.add_argument("--line-max-jump-count", type=float, default=10.0)
    parser.add_argument("--line-max-off-mask-mm", type=float, default=1.0)
    parser.add_argument("--line-max-visible-count", type=float, default=0.0)
    parser.add_argument("--line-max-trim-count", type=float, default=6.0)
    parser.add_argument("--line-max-loss-slack", type=float, default=0.03)
    parser.add_argument("--line-loss-weight", type=float, default=1.0)
    parser.add_argument("--line-jump-weight", type=float, default=0.004)
    parser.add_argument("--line-trim-weight", type=float, default=0.002)
    parser.add_argument("--line-off-mask-weight", type=float, default=0.010)
    parser.add_argument("--line-visible-weight", type=float, default=0.005)
    parser.add_argument("--line-coverage-deficit-weight", type=float, default=0.55)
    parser.add_argument("--line-precision-deficit-weight", type=float, default=0.22)
    parser.add_argument("--line-skeleton-bonus", type=float, default=0.0)
    parser.add_argument("--line-relative-rescue", action="store_true")
    parser.add_argument("--line-rescue-min-coverage", type=float, default=0.70)
    parser.add_argument("--line-rescue-min-jump-gain", type=float, default=1.0)
    parser.add_argument("--line-rescue-max-loss-slack", type=float, default=0.0)
    parser.add_argument("--line-rescue-max-coverage-drop", type=float, default=0.05)
    parser.add_argument("--line-rescue-max-precision-drop", type=float, default=0.02)
    parser.add_argument("--line-rescue-max-visible-increase", type=float, default=0.0)
    parser.add_argument("--line-rescue-max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--line-rescue-jump-weight", type=float, default=0.004)
    parser.add_argument("--line-rescue-target-coverage", type=float, default=0.80)
    parser.add_argument("--line-rescue-target-precision", type=float, default=0.70)
    parser.add_argument("--line-rescue-coverage-deficit-weight", type=float, default=0.20)
    parser.add_argument("--line-rescue-precision-deficit-weight", type=float, default=0.10)
    parser.add_argument("--line-precision-rescue", action="store_true")
    parser.add_argument("--line-precision-rescue-trigger-precision", type=float, default=0.60)
    parser.add_argument("--line-precision-rescue-min-precision", type=float, default=0.65)
    parser.add_argument("--line-precision-rescue-min-gain", type=float, default=0.10)
    parser.add_argument("--line-precision-rescue-min-coverage", type=float, default=0.85)
    parser.add_argument("--line-precision-rescue-max-coverage-drop", type=float, default=0.08)
    parser.add_argument("--line-precision-rescue-max-loss-slack", type=float, default=0.06)
    parser.add_argument("--line-precision-rescue-max-jump-increase", type=float, default=6.0)
    parser.add_argument("--line-precision-rescue-max-trim-increase", type=float, default=2.0)
    parser.add_argument("--line-precision-rescue-max-visible-increase", type=float, default=0.0)
    parser.add_argument("--line-precision-rescue-max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--line-precision-rescue-jump-weight", type=float, default=0.004)
    parser.add_argument("--line-precision-rescue-trim-weight", type=float, default=0.002)
    parser.add_argument("--line-precision-rescue-target-coverage", type=float, default=0.90)
    parser.add_argument("--line-precision-rescue-coverage-deficit-weight", type=float, default=0.20)
    parser.add_argument("--line-precision-rescue-precision-gain-weight", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = read_json(Path(args.model))
    candidates = [(name, Path(path)) for name, path in args.candidate]
    candidate_dirs = {name: path for name, path in candidates}
    anchor_rows = {row["sample_id"]: row for row in read_csv(Path(args.anchor_baseline_selection))} if args.anchor_baseline_selection else {}
    rows = build_candidate_rows(
        Path(args.dataset_dir),
        candidates,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_weight,
        args.precision_weight,
        args.hard_fail_penalty,
    )

    profile_models = model.get("profile_models") if isinstance(model.get("profile_models"), dict) else {}
    presets = model.get("task_profile_presets") if isinstance(model.get("task_profile_presets"), dict) else TASK_PROFILE_PRESETS
    if args.profile not in presets:
        raise ValueError(f"Unknown profile: {args.profile}")
    rows = apply_task_profile(rows, args.profile, presets)
    active_model = profile_models.get(args.profile, model)
    reverse = str(active_model.get("select_direction", "min")) == "max"
    coverage_floor_by_profile = parse_profile_thresholds(args.profile_min_coverage_by_name)
    coverage_floor = profile_coverage_floor(args.profile, args.profile_min_coverage, coverage_floor_by_profile)
    line_sources = {item.strip() for item in args.line_domain_sources.split(",") if item.strip()}
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}

    selected: list[dict[str, Any]] = []
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
        predictions = predict(active_model, selectable)
        ranked = sorted(zip(selectable, predictions), key=lambda item: item[1], reverse=reverse)
        semantic_ranked = semantic_rerank(
            ranked,
            args.profile,
            max(0.0, min(1.0, args.profile_rerank_strength)),
            coverage_floor,
            max(0.0, args.profile_coverage_penalty_weight),
        )
        base_chosen, base_predicted, base_semantic, base_calibrated, gate_details = choose_with_benefit_gate(
            semantic_ranked,
            coverage_floor,
            max(0.0, args.profile_benefit_min_coverage),
            max(0.0, args.profile_benefit_min_jump_gain),
            max(0.0, args.profile_benefit_min_precision),
            args.profile_benefit_max_loss_slack,
        )
        base_chosen, base_predicted, base_semantic, base_calibrated, anchor_details = anchor_baseline_choice(
            semantic_ranked,
            anchor_rows.get(sample_id),
            (base_chosen, base_predicted, base_semantic, base_calibrated),
        )
        chosen = base_chosen
        predicted_score = base_predicted
        semantic_score = base_semantic
        calibrated_score = base_calibrated
        line_details: dict[str, Any] = {
            "line_guard_mode": "non_line_domain",
            "line_guard_score": "",
            "line_guard_pool_size": "",
            "line_guard_valid_pool_size": "",
            "line_guard_rejected_candidate": "",
            "line_guard_loss_slack": "",
            "line_rescue_mode": "",
            "line_rescue_score": "",
            "line_rescue_pool_size": "",
            "line_rescue_baseline_candidate": "",
            "line_rescue_baseline_jump": "",
            "line_rescue_baseline_loss": "",
            "line_rescue_baseline_coverage": "",
            "line_rescue_jump_gain": "",
            "line_rescue_loss_delta": "",
            "line_rescue_coverage_delta": "",
            "line_precision_rescue_mode": "",
            "line_precision_rescue_score": "",
            "line_precision_rescue_pool_size": "",
            "line_precision_rescue_baseline_candidate": "",
            "line_precision_rescue_precision_gain": "",
            "line_precision_rescue_loss_delta": "",
            "line_precision_rescue_jump_delta": "",
            "line_precision_rescue_coverage_delta": "",
        }
        if is_line_domain(base_chosen, line_sources):
            chosen, predicted_score, semantic_score, calibrated_score, line_details = choose_line_guard(
                semantic_ranked,
                (base_chosen, base_predicted, base_semantic, base_calibrated),
                args,
            )
            if args.line_relative_rescue:
                rescue_chosen, rescue_predicted, rescue_semantic, rescue_calibrated, rescue_details = choose_line_relative_rescue(
                    semantic_ranked,
                    (base_chosen, base_predicted, base_semantic, base_calibrated),
                    (chosen, predicted_score, semantic_score, calibrated_score),
                    args,
                )
                chosen = rescue_chosen
                predicted_score = rescue_predicted
                semantic_score = rescue_semantic
                calibrated_score = rescue_calibrated
                line_details.update(rescue_details)
            if args.line_precision_rescue:
                precision_chosen, precision_predicted, precision_semantic, precision_calibrated, precision_details = choose_line_precision_rescue(
                    semantic_ranked,
                    (chosen, predicted_score, semantic_score, calibrated_score),
                    args,
                )
                chosen = precision_chosen
                predicted_score = precision_predicted
                semantic_score = precision_semantic
                calibrated_score = precision_calibrated
                line_details.update(precision_details)

        sample_out = output_dir / str(sample_id)
        copy_outputs(candidate_dirs, chosen, sample_out)
        selected.append(
            {
                "sample_id": sample_id,
                "source_name": chosen.get("source_name", ""),
                "category": chosen.get("category", ""),
                "task_profile": args.profile,
                "chosen_candidate": chosen["candidate"],
                "base_candidate": base_chosen["candidate"],
                "changed_by_line_guard": str(chosen["candidate"]) != str(base_chosen["candidate"]),
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
                "anchor_mode": anchor_details.get("anchor_mode", ""),
                "anchor_candidate": anchor_details.get("anchor_candidate", ""),
                **line_details,
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

    write_csv(selected, output_dir / "line_guard_selected_rows.csv")
    changed_rows = [row for row in selected if str(row.get("changed_by_line_guard", "")).lower() == "true"]
    write_csv(changed_rows, output_dir / "line_guard_changed_rows.csv")
    write_csv(source_summary(selected), output_dir / "line_guard_source_summary.csv")

    summary = summarize(selected)
    summary.update(
        {
            "dataset_dir": args.dataset_dir,
            "profile": args.profile,
            "line_domain_sources": sorted(line_sources),
            "changed_samples": len(changed_rows),
            "line_domain_samples": sum(1 for row in selected if str(row.get("source_name", "")) in line_sources),
            "line_guard_args": {
                "anchor_baseline_selection": args.anchor_baseline_selection,
                "line_target_coverage": args.line_target_coverage,
                "line_target_precision": args.line_target_precision,
                "line_absolute_min_coverage": args.line_absolute_min_coverage,
                "line_absolute_min_precision": args.line_absolute_min_precision,
                "line_max_jump_count": args.line_max_jump_count,
                "line_max_off_mask_mm": args.line_max_off_mask_mm,
                "line_max_visible_count": args.line_max_visible_count,
                "line_max_trim_count": args.line_max_trim_count,
                "line_max_loss_slack": args.line_max_loss_slack,
                "line_loss_weight": args.line_loss_weight,
                "line_jump_weight": args.line_jump_weight,
                "line_trim_weight": args.line_trim_weight,
                "line_off_mask_weight": args.line_off_mask_weight,
                "line_visible_weight": args.line_visible_weight,
                "line_coverage_deficit_weight": args.line_coverage_deficit_weight,
                "line_precision_deficit_weight": args.line_precision_deficit_weight,
                "line_skeleton_bonus": args.line_skeleton_bonus,
                "line_relative_rescue": bool(args.line_relative_rescue),
                "line_rescue_min_coverage": args.line_rescue_min_coverage,
                "line_rescue_min_jump_gain": args.line_rescue_min_jump_gain,
                "line_rescue_max_loss_slack": args.line_rescue_max_loss_slack,
                "line_rescue_max_coverage_drop": args.line_rescue_max_coverage_drop,
                "line_rescue_max_precision_drop": args.line_rescue_max_precision_drop,
                "line_rescue_max_visible_increase": args.line_rescue_max_visible_increase,
                "line_rescue_max_off_mask_increase": args.line_rescue_max_off_mask_increase,
                "line_rescue_jump_weight": args.line_rescue_jump_weight,
                "line_rescue_target_coverage": args.line_rescue_target_coverage,
                "line_rescue_target_precision": args.line_rescue_target_precision,
                "line_rescue_coverage_deficit_weight": args.line_rescue_coverage_deficit_weight,
                "line_rescue_precision_deficit_weight": args.line_rescue_precision_deficit_weight,
                "line_precision_rescue": bool(args.line_precision_rescue),
                "line_precision_rescue_trigger_precision": args.line_precision_rescue_trigger_precision,
                "line_precision_rescue_min_precision": args.line_precision_rescue_min_precision,
                "line_precision_rescue_min_gain": args.line_precision_rescue_min_gain,
                "line_precision_rescue_min_coverage": args.line_precision_rescue_min_coverage,
                "line_precision_rescue_max_coverage_drop": args.line_precision_rescue_max_coverage_drop,
                "line_precision_rescue_max_loss_slack": args.line_precision_rescue_max_loss_slack,
                "line_precision_rescue_max_jump_increase": args.line_precision_rescue_max_jump_increase,
                "line_precision_rescue_max_trim_increase": args.line_precision_rescue_max_trim_increase,
                "line_precision_rescue_max_visible_increase": args.line_precision_rescue_max_visible_increase,
                "line_precision_rescue_max_off_mask_increase": args.line_precision_rescue_max_off_mask_increase,
                "line_precision_rescue_jump_weight": args.line_precision_rescue_jump_weight,
                "line_precision_rescue_trim_weight": args.line_precision_rescue_trim_weight,
                "line_precision_rescue_target_coverage": args.line_precision_rescue_target_coverage,
                "line_precision_rescue_coverage_deficit_weight": args.line_precision_rescue_coverage_deficit_weight,
                "line_precision_rescue_precision_gain_weight": args.line_precision_rescue_precision_gain_weight,
            },
        }
    )
    if args.baseline_selection:
        comparison = compare_to_baseline(selected, Path(args.baseline_selection))
        write_csv(comparison, output_dir / "line_guard_comparison_to_baseline.csv")
        summary["baseline_selection"] = args.baseline_selection
        summary["baseline_changed_samples"] = sum(1 for row in comparison if row["changed"])
        summary["mean_delta_unified_loss"] = mean_metric(comparison, "delta_unified_loss")
        summary["mean_delta_jump_count"] = mean_metric(comparison, "delta_jump_count")
        summary["mean_delta_trim_count"] = mean_metric(comparison, "delta_trim_count")
        summary["mean_delta_off_mask_stitch_length_mm"] = mean_metric(comparison, "delta_off_mask_stitch_length_mm")
        summary["mean_delta_visible_connector_count"] = mean_metric(comparison, "delta_visible_connector_count")
        summary["mean_delta_coverage_ratio"] = mean_metric(comparison, "delta_coverage_ratio")
        summary["mean_delta_stitch_precision_ratio"] = mean_metric(comparison, "delta_stitch_precision_ratio")
    write_json(summary, output_dir / "line_guard_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
