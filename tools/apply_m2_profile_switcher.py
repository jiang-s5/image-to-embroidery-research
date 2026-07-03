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
        src = source_root / sample_id / filename
        if src.exists():
            shutil.copy2(src, sample_out / filename)


def profile_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(baseline.get(key)), 8)


def text_profile_is_worth_it(candidate: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    precision_gain = profile_delta(candidate, baseline, "adaptive_stitch_precision_ratio")
    overfill_reduction = -profile_delta(candidate, baseline, "overfill_outside_adaptive_ratio")
    if precision_gain < args.min_text_precision_gain:
        return False, "precision_gain_too_small"
    if overfill_reduction < args.min_text_overfill_reduction:
        return False, "overfill_reduction_too_small"
    if profile_delta(candidate, baseline, "visible_connector_count") > args.max_visible_increase:
        return False, "visible_connector_increase"
    if profile_delta(candidate, baseline, "off_mask_stitch_length_mm") > args.max_off_mask_increase:
        return False, "off_mask_increase"
    if profile_delta(candidate, baseline, "unified_loss") > args.max_loss_increase:
        return False, "loss_increase_too_large"
    if profile_delta(candidate, baseline, "jump_count") > args.max_jump_increase:
        return False, "jump_increase_too_large"
    if profile_delta(candidate, baseline, "trim_count") > args.max_trim_increase:
        return False, "trim_increase_too_large"
    if safe_float(candidate.get("adaptive_stitch_precision_ratio")) < args.min_text_adaptive_precision:
        return False, "adaptive_precision_too_low"
    if safe_float(candidate.get("adaptive_coverage_ratio")) < args.min_text_adaptive_coverage:
        return False, "adaptive_coverage_too_low"
    return True, "text_legibility_gain"


def select_profile(
    safe_row: dict[str, str],
    text_row: dict[str, str],
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], str]:
    source_name = safe_row.get("source_name", "")
    category = safe_row.get("category", "")
    if source_name != args.text_source_name:
        return "safe_low_cost", dict(safe_row), "non_text_source"
    allowed, reason = text_profile_is_worth_it(text_row, safe_row, args)
    if allowed:
        return "text_legibility", dict(text_row), reason
    return "safe_low_cost", dict(safe_row), f"text_safe_fallback:{reason}:{category}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select professional-style M2 generation profiles per sample.")
    parser.add_argument("--safe-selection", required=True)
    parser.add_argument("--safe-output-dir", required=True)
    parser.add_argument("--text-selection", required=True)
    parser.add_argument("--text-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-source-name", default="Rendered text")
    parser.add_argument("--min-text-precision-gain", type=float, default=0.015)
    parser.add_argument("--min-text-overfill-reduction", type=float, default=0.015)
    parser.add_argument("--min-text-adaptive-precision", type=float, default=0.95)
    parser.add_argument("--min-text-adaptive-coverage", type=float, default=0.88)
    parser.add_argument("--max-loss-increase", type=float, default=0.12)
    parser.add_argument("--max-jump-increase", type=float, default=12.0)
    parser.add_argument("--max-trim-increase", type=float, default=4.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-off-mask-increase", type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    safe_root = Path(args.safe_output_dir)
    text_root = Path(args.text_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_rows = {row["sample_id"]: row for row in read_csv(Path(args.safe_selection))}
    text_rows = {row["sample_id"]: row for row in read_csv(Path(args.text_selection))}
    missing = sorted(set(safe_rows) ^ set(text_rows))
    if missing:
        raise ValueError(f"Selection files do not cover the same samples: {missing[:10]}")

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for sample_id in sorted(safe_rows):
        safe_row = safe_rows[sample_id]
        text_row = text_rows[sample_id]
        profile, chosen, reason = select_profile(safe_row, text_row, args)
        source_root = text_root if profile == "text_legibility" else safe_root
        copy_outputs(source_root, sample_id, output_dir)

        selected_row: dict[str, Any] = dict(chosen)
        selected_row["active_profile"] = profile
        selected_row["profile_reason"] = reason
        selected_row["profile_switched"] = profile != "safe_low_cost"
        for key in METRIC_KEYS:
            selected_row[f"delta_vs_safe_{key}"] = profile_delta(chosen, safe_row, key)
        selected.append(selected_row)

        decisions.append(
            {
                "sample_id": sample_id,
                "source_name": safe_row.get("source_name", ""),
                "category": safe_row.get("category", ""),
                "active_profile": profile,
                "profile_reason": reason,
                "profile_switched": profile != "safe_low_cost",
                "safe_candidate": safe_row.get("chosen_candidate", ""),
                "chosen_candidate": chosen.get("chosen_candidate", ""),
                "adaptive_precision_gain": profile_delta(chosen, safe_row, "adaptive_stitch_precision_ratio"),
                "overfill_reduction": -profile_delta(chosen, safe_row, "overfill_outside_adaptive_ratio"),
                "loss_delta": profile_delta(chosen, safe_row, "unified_loss"),
                "jump_delta": profile_delta(chosen, safe_row, "jump_count"),
                "trim_delta": profile_delta(chosen, safe_row, "trim_count"),
                "visible_delta": profile_delta(chosen, safe_row, "visible_connector_count"),
                "off_mask_delta": profile_delta(chosen, safe_row, "off_mask_stitch_length_mm"),
            }
        )

    by_profile = summarize(selected, "active_profile")
    by_source = summarize(selected, "source_name")
    by_category = summarize(selected, "category")
    switched = [row for row in decisions if row["profile_switched"]]
    summary = {
        "model_id": "m2_66_profile_switcher",
        "samples": len(selected),
        "profiles": {
            "safe_low_cost": "m2_64_text_precision_rescue",
            "text_legibility": "m2_65_text_legibility_rescue",
        },
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
        "switched_samples": switched,
        "args": vars(args),
    }

    write_csv(selected, output_dir / "profile_selected_rows.csv")
    write_csv(decisions, output_dir / "profile_decision_rows.csv")
    write_csv(switched, output_dir / "profile_switched_rows.csv")
    write_csv(by_profile, output_dir / "profile_by_profile.csv")
    write_csv(by_source, output_dir / "profile_by_source.csv")
    write_csv(by_category, output_dir / "profile_by_category.csv")
    write_json(summary, output_dir / "profile_switch_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
