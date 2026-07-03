from __future__ import annotations

import argparse
import csv
import json
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
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        output.append(
            {
                group_key: key,
                "samples": len(items),
                "profile_switches": sum(1 for row in items if str(row.get("profile_switched", "")).lower() == "true"),
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
    return output


def copy_outputs(source_root: Path, sample_id: str, output_root: Path) -> None:
    sample_out = output_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
        source = source_root / sample_id / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def metric_delta(candidate: dict[str, Any], base: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(base.get(key)), 8)


def relief_score(candidate: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> float:
    jump_reduction = -metric_delta(candidate, base, "jump_count")
    loss_reduction = -metric_delta(candidate, base, "unified_loss")
    precision_drop = max(0.0, -metric_delta(candidate, base, "adaptive_stitch_precision_ratio"))
    strict_drop = max(0.0, -metric_delta(candidate, base, "stitch_precision_ratio"))
    coverage_deficit = max(0.0, args.target_coverage - safe_float(candidate.get("coverage_ratio")))
    adaptive_coverage_deficit = max(0.0, args.target_adaptive_coverage - safe_float(candidate.get("adaptive_coverage_ratio")))
    return round(
        safe_float(candidate.get("unified_loss"))
        + args.jump_weight * safe_float(candidate.get("jump_count"))
        + args.trim_weight * safe_float(candidate.get("trim_count"))
        + args.coverage_deficit_weight * coverage_deficit
        + args.adaptive_coverage_deficit_weight * adaptive_coverage_deficit
        + args.precision_drop_weight * precision_drop
        + args.strict_drop_weight * strict_drop
        - args.jump_reduction_weight * jump_reduction
        - args.loss_reduction_weight * loss_reduction,
        8,
    )


def passes_gate(candidate: dict[str, Any], base: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if str(candidate.get("quality_level", "")).lower() == "hard_fail":
        return False, "candidate_hard_fail"
    if safe_float(base.get("jump_count")) < args.trigger_min_jump:
        return False, "base_jump_not_high"
    jump_reduction = -metric_delta(candidate, base, "jump_count")
    if jump_reduction < args.min_jump_reduction:
        return False, "jump_reduction_too_small"
    if safe_float(candidate.get("jump_count")) > args.max_candidate_jump:
        return False, "candidate_jump_too_high"
    if safe_float(candidate.get("adaptive_stitch_precision_ratio")) < args.min_adaptive_precision:
        return False, "adaptive_precision_too_low"
    if safe_float(candidate.get("adaptive_coverage_ratio")) < args.min_adaptive_coverage:
        return False, "adaptive_coverage_too_low"
    if safe_float(candidate.get("coverage_ratio")) < args.min_coverage:
        return False, "coverage_too_low"
    if safe_float(candidate.get("stitch_precision_ratio")) < args.min_strict_precision:
        return False, "strict_precision_too_low"
    if -metric_delta(candidate, base, "adaptive_stitch_precision_ratio") > args.max_adaptive_precision_drop:
        return False, "adaptive_precision_drop_too_large"
    if -metric_delta(candidate, base, "stitch_precision_ratio") > args.max_strict_precision_drop:
        return False, "strict_precision_drop_too_large"
    if metric_delta(candidate, base, "unified_loss") > args.max_loss_increase:
        return False, "loss_increase_too_large"
    if metric_delta(candidate, base, "trim_count") > args.max_trim_increase:
        return False, "trim_increase_too_large"
    if metric_delta(candidate, base, "visible_connector_count") > args.max_visible_increase:
        return False, "visible_increase"
    if metric_delta(candidate, base, "off_mask_stitch_length_mm") > args.max_off_mask_increase:
        return False, "off_mask_increase"
    return True, "line_art_jump_relief"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a jump-aware QuickDraw line-art profile on top of M2.69.")
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--candidate-scan", required=True)
    parser.add_argument("--candidate-results-root", default="results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-source-name", default="QuickDraw")
    parser.add_argument("--trigger-min-jump", type=float, default=12.0)
    parser.add_argument("--min-jump-reduction", type=float, default=4.0)
    parser.add_argument("--max-candidate-jump", type=float, default=10.0)
    parser.add_argument("--min-adaptive-precision", type=float, default=0.94)
    parser.add_argument("--min-adaptive-coverage", type=float, default=0.85)
    parser.add_argument("--min-coverage", type=float, default=0.94)
    parser.add_argument("--min-strict-precision", type=float, default=0.70)
    parser.add_argument("--max-adaptive-precision-drop", type=float, default=0.04)
    parser.add_argument("--max-strict-precision-drop", type=float, default=0.06)
    parser.add_argument("--max-loss-increase", type=float, default=0.0)
    parser.add_argument("--max-trim-increase", type=float, default=0.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--target-coverage", type=float, default=0.96)
    parser.add_argument("--target-adaptive-coverage", type=float, default=0.86)
    parser.add_argument("--jump-weight", type=float, default=0.004)
    parser.add_argument("--trim-weight", type=float, default=0.002)
    parser.add_argument("--coverage-deficit-weight", type=float, default=0.08)
    parser.add_argument("--adaptive-coverage-deficit-weight", type=float, default=0.10)
    parser.add_argument("--precision-drop-weight", type=float, default=0.14)
    parser.add_argument("--strict-drop-weight", type=float, default=0.06)
    parser.add_argument("--jump-reduction-weight", type=float, default=0.006)
    parser.add_argument("--loss-reduction-weight", type=float, default=0.10)
    args = parser.parse_args()

    base_root = Path(args.base_output_dir)
    results_root = Path(args.candidate_results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv(Path(args.base_selection))
    scan_rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(Path(args.candidate_scan)):
        scan_rows_by_sample[row["sample_id"]].append(row)

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    switched: list[dict[str, Any]] = []

    for base_row in base_rows:
        sample_id = base_row["sample_id"]
        current = dict(base_row)
        current["base_profile"] = base_row.get("active_profile", "safe_low_cost")
        current["base_candidate"] = base_row.get("chosen_candidate", "")
        current["profile_switched"] = False
        decision = {
            "sample_id": sample_id,
            "source_name": base_row.get("source_name", ""),
            "category": base_row.get("category", ""),
            "base_profile": base_row.get("active_profile", "safe_low_cost"),
            "base_candidate": base_row.get("chosen_candidate", ""),
            "selected_profile": base_row.get("active_profile", "safe_low_cost"),
            "selected_candidate": base_row.get("chosen_candidate", ""),
            "decision": "base_kept",
            "reason": "non_line_source",
        }

        if base_row.get("source_name") != args.line_source_name:
            copy_outputs(base_root, sample_id, output_dir)
            selected.append(current)
            decisions.append(decision)
            continue

        candidates: list[tuple[float, dict[str, str], str]] = []
        rejected: list[str] = []
        for candidate in scan_rows_by_sample.get(sample_id, []):
            allowed, reason = passes_gate(candidate, base_row, args)
            if not allowed:
                rejected.append(f"{candidate.get('candidate','')}:{reason}")
                continue
            candidates.append((relief_score(candidate, base_row, args), candidate, reason))

        if not candidates:
            decision.update({"reason": ";".join(rejected[:12]) or "no_candidate"})
            copy_outputs(base_root, sample_id, output_dir)
            selected.append(current)
            decisions.append(decision)
            continue

        score, chosen, reason = sorted(candidates, key=lambda item: (item[0], safe_float(item[1].get("jump_count"))))[0]
        copy_outputs(results_root / chosen["candidate"], sample_id, output_dir)
        updated: dict[str, Any] = dict(base_row)
        updated["base_profile"] = base_row.get("active_profile", "safe_low_cost")
        updated["base_candidate"] = base_row.get("chosen_candidate", "")
        updated["active_profile"] = "line_art_jump_relief"
        updated["profile_reason"] = reason
        updated["profile_switched"] = True
        updated["chosen_candidate"] = chosen["candidate"]
        updated["line_jump_profile_score"] = score
        for key in METRIC_KEYS:
            updated[key] = chosen.get(key, "")
            updated[f"delta_vs_base_{key}"] = metric_delta(chosen, base_row, key)
        selected.append(updated)
        switched.append(updated)
        decision.update(
            {
                "selected_profile": "line_art_jump_relief",
                "selected_candidate": chosen["candidate"],
                "decision": "switched",
                "reason": reason,
                "selector_score": score,
                "jump_reduction": -metric_delta(chosen, base_row, "jump_count"),
                "loss_delta": metric_delta(chosen, base_row, "unified_loss"),
                "adaptive_precision_delta": metric_delta(chosen, base_row, "adaptive_stitch_precision_ratio"),
                "strict_precision_delta": metric_delta(chosen, base_row, "stitch_precision_ratio"),
                "coverage_delta": metric_delta(chosen, base_row, "coverage_ratio"),
                "adaptive_coverage_delta": metric_delta(chosen, base_row, "adaptive_coverage_ratio"),
                "trim_delta": metric_delta(chosen, base_row, "trim_count"),
                "visible_delta": metric_delta(chosen, base_row, "visible_connector_count"),
                "off_mask_delta": metric_delta(chosen, base_row, "off_mask_stitch_length_mm"),
            }
        )
        decisions.append(decision)

    by_profile = summarize(selected, "active_profile")
    by_source = summarize(selected, "source_name")
    by_category = summarize(selected, "category")
    summary = {
        "model_id": "m2_70_line_art_jump_profile",
        "base_profile_system": "m2_69_text_wku_precision_profile",
        "samples": len(selected),
        "profile_switches": len(switched),
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
        "by_profile": by_profile,
        "by_source": by_source,
        "by_category": by_category,
        "switched_samples": [
            {
                "sample_id": row.get("sample_id", ""),
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "base_profile": row.get("base_profile", ""),
                "active_profile": row.get("active_profile", ""),
                "base_candidate": row.get("base_candidate", ""),
                "chosen_candidate": row.get("chosen_candidate", ""),
                "jump_reduction": -safe_float(row.get("delta_vs_base_jump_count")),
                "loss_delta": row.get("delta_vs_base_unified_loss", ""),
                "adaptive_precision_delta": row.get("delta_vs_base_adaptive_stitch_precision_ratio", ""),
                "strict_precision_delta": row.get("delta_vs_base_stitch_precision_ratio", ""),
                "coverage_delta": row.get("delta_vs_base_coverage_ratio", ""),
                "adaptive_coverage_delta": row.get("delta_vs_base_adaptive_coverage_ratio", ""),
                "trim_delta": row.get("delta_vs_base_trim_count", ""),
            }
            for row in switched
        ],
        "args": vars(args),
    }
    write_csv(selected, output_dir / "line_jump_profile_selected_rows.csv")
    write_csv(decisions, output_dir / "line_jump_profile_decision_rows.csv")
    write_csv(switched, output_dir / "line_jump_profile_switched_rows.csv")
    write_csv(by_profile, output_dir / "line_jump_profile_by_profile.csv")
    write_csv(by_source, output_dir / "line_jump_profile_by_source.csv")
    write_csv(by_category, output_dir / "line_jump_profile_by_category.csv")
    write_json(summary, output_dir / "line_jump_profile_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
