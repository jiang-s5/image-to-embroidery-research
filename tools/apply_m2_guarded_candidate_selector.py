from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
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
    "adaptive_coverage_ratio",
    "adaptive_stitch_precision_ratio",
    "overfill_outside_adaptive_ratio",
)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def summarize(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    summaries: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        summaries.append(
            {
                group_key: key,
                "samples": len(items),
                "promoted_samples": sum(1 for row in items if str(row.get("m2_71_promoted", "")).lower() == "true"),
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
                "mean_overfill_outside_adaptive_ratio": avg(items, "overfill_outside_adaptive_ratio"),
            }
        )
    return summaries


def copy_outputs(source_root: Path, sample_id: str, output_root: Path) -> None:
    sample_out = output_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
        source = source_root / sample_id / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def metric_delta(candidate: dict[str, Any], base: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(base.get(key)), 8)


def guarded_score(candidate: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> float:
    loss_reduction = -metric_delta(candidate, base, "unified_loss")
    jump_reduction = -metric_delta(candidate, base, "jump_count")
    trim_reduction = -metric_delta(candidate, base, "trim_count")
    coverage_gain = metric_delta(candidate, base, "coverage_ratio")
    precision_gain = metric_delta(candidate, base, "stitch_precision_ratio")
    adaptive_coverage_gain = metric_delta(candidate, base, "adaptive_coverage_ratio")
    adaptive_precision_gain = metric_delta(candidate, base, "adaptive_stitch_precision_ratio")
    return round(
        args.loss_gain_weight * loss_reduction
        + args.jump_gain_weight * jump_reduction
        + args.trim_gain_weight * trim_reduction
        + args.coverage_gain_weight * coverage_gain
        + args.precision_gain_weight * precision_gain
        + args.adaptive_coverage_gain_weight * adaptive_coverage_gain
        + args.adaptive_precision_gain_weight * adaptive_precision_gain,
        8,
    )


def improvement_reason(candidate: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> str:
    reasons: list[str] = []
    if -metric_delta(candidate, base, "unified_loss") >= args.min_loss_reduction:
        reasons.append("loss_reduced")
    if -metric_delta(candidate, base, "jump_count") >= args.min_jump_reduction:
        reasons.append("jump_reduced")
    if -metric_delta(candidate, base, "trim_count") >= args.min_trim_reduction:
        reasons.append("trim_reduced")
    if metric_delta(candidate, base, "coverage_ratio") >= args.min_coverage_gain:
        reasons.append("coverage_improved")
    if metric_delta(candidate, base, "stitch_precision_ratio") >= args.min_precision_gain:
        reasons.append("strict_precision_improved")
    if metric_delta(candidate, base, "adaptive_coverage_ratio") >= args.min_adaptive_coverage_gain:
        reasons.append("adaptive_coverage_improved")
    if metric_delta(candidate, base, "adaptive_stitch_precision_ratio") >= args.min_adaptive_precision_gain:
        reasons.append("adaptive_precision_improved")
    return "|".join(reasons)


def gate_candidate(candidate: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if str(candidate.get("quality_level", "")).lower() == "hard_fail":
        return False, "candidate_hard_fail"
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
    if metric_delta(candidate, base, "coverage_ratio") < -args.max_coverage_drop:
        return False, "coverage_drop"
    if metric_delta(candidate, base, "stitch_precision_ratio") < -args.max_precision_drop:
        return False, "strict_precision_drop"
    if metric_delta(candidate, base, "adaptive_coverage_ratio") < -args.max_adaptive_coverage_drop:
        return False, "adaptive_coverage_drop"
    if metric_delta(candidate, base, "adaptive_stitch_precision_ratio") < -args.max_adaptive_precision_drop:
        return False, "adaptive_precision_drop"
    if safe_float(candidate.get("coverage_ratio")) < args.min_absolute_coverage:
        return False, "absolute_coverage_too_low"
    if safe_float(candidate.get("adaptive_coverage_ratio")) < args.min_absolute_adaptive_coverage:
        return False, "absolute_adaptive_coverage_too_low"
    if safe_float(candidate.get("adaptive_stitch_precision_ratio")) < args.min_absolute_adaptive_precision:
        return False, "absolute_adaptive_precision_too_low"
    reason = improvement_reason(candidate, base, args)
    if not reason:
        return False, "no_material_improvement"
    score = guarded_score(candidate, base, args)
    if score < args.min_guarded_score:
        return False, "guarded_score_too_low"
    return True, reason


def load_candidate_scans(paths: list[str]) -> dict[str, list[dict[str, str]]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for item in paths:
        path = Path(item)
        for row in read_csv(path):
            sample_id = row.get("sample_id", "")
            candidate = row.get("candidate", "")
            key = (sample_id, candidate)
            if not sample_id or not candidate or key in seen:
                continue
            seen.add(key)
            row["candidate_scan_path"] = str(path)
            by_sample[sample_id].append(row)
    return by_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a no-regression guarded candidate selector on top of M2.70.")
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--candidate-scan", action="append", required=True)
    parser.add_argument("--candidate-results-root", default="results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="m2_71_guarded_candidate_selector")
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-jump-increase", type=float, default=0.0)
    parser.add_argument("--max-trim-increase", type=float, default=0.0)
    parser.add_argument("--max-loss-increase", type=float, default=0.0)
    parser.add_argument("--max-coverage-drop", type=float, default=0.005)
    parser.add_argument("--max-precision-drop", type=float, default=0.005)
    parser.add_argument("--max-adaptive-coverage-drop", type=float, default=0.01)
    parser.add_argument("--max-adaptive-precision-drop", type=float, default=0.01)
    parser.add_argument("--min-absolute-coverage", type=float, default=0.90)
    parser.add_argument("--min-absolute-adaptive-coverage", type=float, default=0.85)
    parser.add_argument("--min-absolute-adaptive-precision", type=float, default=0.94)
    parser.add_argument("--min-loss-reduction", type=float, default=0.0005)
    parser.add_argument("--min-jump-reduction", type=float, default=1.0)
    parser.add_argument("--min-trim-reduction", type=float, default=1.0)
    parser.add_argument("--min-coverage-gain", type=float, default=0.01)
    parser.add_argument("--min-precision-gain", type=float, default=0.01)
    parser.add_argument("--min-adaptive-coverage-gain", type=float, default=0.01)
    parser.add_argument("--min-adaptive-precision-gain", type=float, default=0.005)
    parser.add_argument("--min-guarded-score", type=float, default=0.0)
    parser.add_argument("--loss-gain-weight", type=float, default=1.0)
    parser.add_argument("--jump-gain-weight", type=float, default=0.006)
    parser.add_argument("--trim-gain-weight", type=float, default=0.003)
    parser.add_argument("--coverage-gain-weight", type=float, default=0.02)
    parser.add_argument("--precision-gain-weight", type=float, default=0.02)
    parser.add_argument("--adaptive-coverage-gain-weight", type=float, default=0.02)
    parser.add_argument("--adaptive-precision-gain-weight", type=float, default=0.02)
    parser.add_argument("--near-miss-count", type=int, default=80)
    args = parser.parse_args()

    base_root = Path(args.base_output_dir)
    output_dir = Path(args.output_dir)
    results_root = Path(args.candidate_results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_csv(Path(args.base_selection))
    candidates_by_sample = load_candidate_scans(args.candidate_scan)
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []

    for base in base_rows:
        sample_id = base["sample_id"]
        best: tuple[float, dict[str, str], str] | None = None
        base_candidate = base.get("chosen_candidate", "")
        sample_rejections = Counter()
        for candidate in candidates_by_sample.get(sample_id, []):
            if candidate.get("candidate") == base_candidate:
                continue
            allowed, reason = gate_candidate(candidate, base, args)
            score = guarded_score(candidate, base, args)
            delta_row: dict[str, Any] = {
                "sample_id": sample_id,
                "source_name": base.get("source_name", ""),
                "category": base.get("category", ""),
                "base_candidate": base_candidate,
                "candidate": candidate.get("candidate", ""),
                "allowed": allowed,
                "reason": reason,
                "guarded_score": score,
            }
            for key in METRIC_KEYS:
                delta_row[f"delta_{key}"] = metric_delta(candidate, base, key)
                delta_row[f"base_{key}"] = base.get(key, "")
                delta_row[f"candidate_{key}"] = candidate.get(key, "")
            if allowed:
                if best is None or (score, -safe_float(candidate.get("unified_loss"))) > (
                    best[0],
                    -safe_float(best[1].get("unified_loss")),
                ):
                    best = (score, candidate, reason)
            else:
                sample_rejections[reason] += 1
                rejections.append(delta_row)
                near_misses.append(delta_row)

        if best is None:
            kept = dict(base)
            kept["m2_71_promoted"] = False
            kept["m2_71_guard_status"] = "base_kept"
            kept["m2_71_rejection_reasons"] = "|".join(f"{key}:{value}" for key, value in sorted(sample_rejections.items()))
            copy_outputs(base_root, sample_id, output_dir)
            selected.append(kept)
            decisions.append(
                {
                    "sample_id": sample_id,
                    "source_name": base.get("source_name", ""),
                    "category": base.get("category", ""),
                    "decision": "base_kept",
                    "base_candidate": base_candidate,
                    "selected_candidate": base_candidate,
                    "selected_profile": base.get("active_profile", ""),
                    "promoted": False,
                    "rejection_reasons": kept["m2_71_rejection_reasons"],
                }
            )
            continue

        score, candidate, reason = best
        updated: dict[str, Any] = dict(base)
        updated["m2_71_promoted"] = True
        updated["m2_71_guard_status"] = "promoted"
        updated["m2_71_promotion_reason"] = reason
        updated["m2_71_guarded_score"] = score
        updated["base_candidate"] = base_candidate
        updated["chosen_candidate"] = candidate.get("candidate", "")
        updated["active_profile"] = "m2_71_guarded_candidate"
        for key in METRIC_KEYS:
            updated[key] = candidate.get(key, "")
            updated[f"delta_vs_base_{key}"] = metric_delta(candidate, base, key)
        copy_outputs(results_root / candidate["candidate"], sample_id, output_dir)
        selected.append(updated)
        promotions.append(updated)
        decisions.append(
            {
                "sample_id": sample_id,
                "source_name": base.get("source_name", ""),
                "category": base.get("category", ""),
                "decision": "promoted",
                "base_candidate": base_candidate,
                "selected_candidate": candidate.get("candidate", ""),
                "selected_profile": "m2_71_guarded_candidate",
                "promoted": True,
                "promotion_reason": reason,
                "guarded_score": score,
            }
        )

    near_misses = sorted(
        near_misses,
        key=lambda row: (
            -safe_float(row.get("guarded_score")),
            safe_float(row.get("delta_unified_loss")),
            safe_float(row.get("delta_jump_count")),
        ),
    )[: args.near_miss_count]
    rejection_counts = Counter(row["reason"] for row in rejections)
    summary = {
        "model_id": args.model_id,
        "base_profile_system": "m2_70_line_art_jump_profile",
        "samples": len(selected),
        "promoted_samples": len(promotions),
        "hard_fail": sum(1 for row in selected if str(row.get("quality_level", "")).lower() == "hard_fail"),
        "mean_unified_loss": avg(selected, "unified_loss"),
        "mean_jump_count": avg(selected, "jump_count"),
        "mean_trim_count": avg(selected, "trim_count"),
        "mean_off_mask_stitch_length_mm": avg(selected, "off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg(selected, "visible_connector_count"),
        "mean_coverage_ratio": avg(selected, "coverage_ratio"),
        "mean_stitch_precision_ratio": avg(selected, "stitch_precision_ratio"),
        "mean_adaptive_coverage_ratio": avg(selected, "adaptive_coverage_ratio"),
        "mean_adaptive_stitch_precision_ratio": avg(selected, "adaptive_stitch_precision_ratio"),
        "mean_overfill_outside_adaptive_ratio": avg(selected, "overfill_outside_adaptive_ratio"),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "by_profile": summarize(selected, "active_profile"),
        "by_source": summarize(selected, "source_name"),
        "by_category": summarize(selected, "category"),
        "promotions": [
            {
                "sample_id": row.get("sample_id", ""),
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "base_candidate": row.get("base_candidate", ""),
                "chosen_candidate": row.get("chosen_candidate", ""),
                "reason": row.get("m2_71_promotion_reason", ""),
                "guarded_score": row.get("m2_71_guarded_score", ""),
                "loss_delta": row.get("delta_vs_base_unified_loss", ""),
                "jump_delta": row.get("delta_vs_base_jump_count", ""),
                "coverage_delta": row.get("delta_vs_base_coverage_ratio", ""),
                "precision_delta": row.get("delta_vs_base_stitch_precision_ratio", ""),
            }
            for row in promotions
        ],
        "args": vars(args),
    }
    write_csv(selected, output_dir / "guarded_selected_rows.csv")
    write_csv(decisions, output_dir / "guarded_decision_rows.csv")
    write_csv(promotions, output_dir / "guarded_promoted_rows.csv")
    write_csv(rejections, output_dir / "guarded_rejected_rows.csv")
    write_csv(near_misses, output_dir / "guarded_near_misses.csv")
    write_csv(summarize(selected, "active_profile"), output_dir / "guarded_by_profile.csv")
    write_csv(summarize(selected, "source_name"), output_dir / "guarded_by_source.csv")
    write_csv(summarize(selected, "category"), output_dir / "guarded_by_category.csv")
    write_json(summary, output_dir / "guarded_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
