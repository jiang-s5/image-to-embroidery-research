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

SKELETON_KEYS = (
    "skeleton_coverage_ratio",
    "skeleton_precision_ratio",
    "adaptive_skeleton_coverage_ratio",
    "adaptive_skeleton_precision_ratio",
    "skeleton_overfill_outside_adaptive_ratio",
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
        row: dict[str, Any] = {
            group_key: key,
            "samples": len(items),
            "profile_switches": sum(1 for item in items if str(item.get("profile_switched", "")).lower() == "true"),
            "hard_fail": sum(1 for item in items if str(item.get("quality_level", "")).lower() == "hard_fail"),
        }
        for metric in METRIC_KEYS + SKELETON_KEYS:
            row[f"mean_{metric}"] = avg(items, metric)
        summaries.append(row)
    return summaries


def copy_outputs(source_root: Path, sample_id: str, output_root: Path) -> None:
    sample_out = output_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json", "coverage_report.json"):
        source = source_root / sample_id / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def metric_delta(candidate: dict[str, Any], base: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(base.get(key)), 8)


def skeleton_delta(
    candidate_skeleton: dict[str, Any],
    base_skeleton: dict[str, Any],
    key: str,
) -> float:
    return round(safe_float(candidate_skeleton.get(key)) - safe_float(base_skeleton.get(key)), 8)


def passes_gate(
    candidate: dict[str, Any],
    base: dict[str, Any],
    candidate_skeleton: dict[str, Any],
    base_skeleton: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str]:
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
    if safe_float(candidate.get("adaptive_stitch_precision_ratio")) < args.min_adaptive_precision:
        return False, "adaptive_precision_too_low"
    if safe_float(candidate_skeleton.get("adaptive_skeleton_coverage_ratio")) < args.min_adaptive_skeleton_coverage:
        return False, "adaptive_skeleton_coverage_too_low"
    if skeleton_delta(candidate_skeleton, base_skeleton, "skeleton_coverage_ratio") < args.min_skeleton_coverage_gain:
        return False, "skeleton_coverage_gain_too_small"
    if skeleton_delta(candidate_skeleton, base_skeleton, "adaptive_skeleton_coverage_ratio") < args.min_adaptive_skeleton_coverage_gain:
        return False, "adaptive_skeleton_coverage_gain_too_small"
    return True, "line_skeleton_fidelity"


def load_skeleton_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        rows[(row["selection"], row["sample_id"])] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote QuickDraw vector-stroke outputs when skeleton fidelity improves safely.")
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--candidate-selection", required=True)
    parser.add_argument("--candidate-output-dir", required=True)
    parser.add_argument("--skeleton-audit-rows", required=True)
    parser.add_argument("--base-skeleton-selection-name", default="m2_71")
    parser.add_argument("--candidate-skeleton-selection-name", default="vector_nearest_reverse_connect8")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--line-source-name", default="QuickDraw")
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-jump-increase", type=float, default=0.0)
    parser.add_argument("--max-trim-increase", type=float, default=0.0)
    parser.add_argument("--max-loss-increase", type=float, default=0.0)
    parser.add_argument("--min-adaptive-precision", type=float, default=0.95)
    parser.add_argument("--min-adaptive-skeleton-coverage", type=float, default=0.95)
    parser.add_argument("--min-skeleton-coverage-gain", type=float, default=0.005)
    parser.add_argument("--min-adaptive-skeleton-coverage-gain", type=float, default=0.005)
    args = parser.parse_args()

    base_root = Path(args.base_output_dir)
    candidate_root = Path(args.candidate_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_csv(Path(args.base_selection))
    candidate_rows = {row["sample_id"]: row for row in read_csv(Path(args.candidate_selection))}
    skeleton_rows = load_skeleton_rows(Path(args.skeleton_audit_rows))

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    switched: list[dict[str, Any]] = []

    for base in base_rows:
        sample_id = base["sample_id"]
        current = dict(base)
        current["profile_switched"] = False
        current["base_candidate"] = base.get("chosen_candidate", "")
        decision = {
            "sample_id": sample_id,
            "source_name": base.get("source_name", ""),
            "category": base.get("category", ""),
            "base_candidate": base.get("chosen_candidate", ""),
            "decision": "base_kept",
            "reason": "non_line_source",
        }

        candidate = candidate_rows.get(sample_id)
        base_skeleton = skeleton_rows.get((args.base_skeleton_selection_name, sample_id), {})
        candidate_skeleton = skeleton_rows.get((args.candidate_skeleton_selection_name, sample_id), {})
        if base.get("source_name") != args.line_source_name or not candidate or not base_skeleton or not candidate_skeleton:
            copy_outputs(base_root, sample_id, output_dir)
            for key in SKELETON_KEYS:
                current[key] = base_skeleton.get(key, "")
            selected.append(current)
            decisions.append(decision)
            continue

        allowed, reason = passes_gate(candidate, base, candidate_skeleton, base_skeleton, args)
        if not allowed:
            copy_outputs(base_root, sample_id, output_dir)
            for key in SKELETON_KEYS:
                current[key] = base_skeleton.get(key, "")
                current[f"candidate_{key}"] = candidate_skeleton.get(key, "")
                current[f"delta_vs_candidate_{key}"] = skeleton_delta(candidate_skeleton, base_skeleton, key)
            decision.update({"reason": reason, "candidate": candidate.get("mode", "quickdraw_vector_stroke")})
            selected.append(current)
            decisions.append(decision)
            continue

        updated: dict[str, Any] = dict(base)
        updated["profile_switched"] = True
        updated["active_profile"] = "line_skeleton_fidelity"
        updated["profile_reason"] = reason
        updated["base_candidate"] = base.get("chosen_candidate", "")
        updated["chosen_candidate"] = str(candidate_root.name)
        for key in METRIC_KEYS:
            updated[key] = candidate.get(key, "")
            updated[f"delta_vs_base_{key}"] = metric_delta(candidate, base, key)
        for key in SKELETON_KEYS:
            updated[key] = candidate_skeleton.get(key, "")
            updated[f"base_{key}"] = base_skeleton.get(key, "")
            updated[f"delta_vs_base_{key}"] = skeleton_delta(candidate_skeleton, base_skeleton, key)
        copy_outputs(candidate_root, sample_id, output_dir)
        switched.append(updated)
        selected.append(updated)
        decision.update(
            {
                "decision": "switched",
                "reason": reason,
                "selected_candidate": str(candidate_root.name),
                "loss_delta": metric_delta(candidate, base, "unified_loss"),
                "jump_delta": metric_delta(candidate, base, "jump_count"),
                "trim_delta": metric_delta(candidate, base, "trim_count"),
                "coverage_delta": metric_delta(candidate, base, "coverage_ratio"),
                "strict_precision_delta": metric_delta(candidate, base, "stitch_precision_ratio"),
                "adaptive_precision_delta": metric_delta(candidate, base, "adaptive_stitch_precision_ratio"),
                "skeleton_coverage_delta": skeleton_delta(candidate_skeleton, base_skeleton, "skeleton_coverage_ratio"),
                "adaptive_skeleton_coverage_delta": skeleton_delta(candidate_skeleton, base_skeleton, "adaptive_skeleton_coverage_ratio"),
            }
        )
        decisions.append(decision)

    summary = {
        "model_id": "m2_72_line_skeleton_fidelity_profile",
        "base_profile_system": "m2_71_guarded_candidate_selector",
        "samples": len(selected),
        "profile_switches": len(switched),
        "hard_fail": sum(1 for row in selected if str(row.get("quality_level", "")).lower() == "hard_fail"),
        **{f"mean_{key}": avg(selected, key) for key in METRIC_KEYS + SKELETON_KEYS},
        "by_profile": summarize(selected, "active_profile"),
        "by_source": summarize(selected, "source_name"),
        "by_category": summarize(selected, "category"),
        "switched_samples": [
            {
                "sample_id": row.get("sample_id", ""),
                "category": row.get("category", ""),
                "base_candidate": row.get("base_candidate", ""),
                "chosen_candidate": row.get("chosen_candidate", ""),
                "loss_delta": row.get("delta_vs_base_unified_loss", ""),
                "jump_delta": row.get("delta_vs_base_jump_count", ""),
                "coverage_delta": row.get("delta_vs_base_coverage_ratio", ""),
                "strict_precision_delta": row.get("delta_vs_base_stitch_precision_ratio", ""),
                "skeleton_coverage_delta": row.get("delta_vs_base_skeleton_coverage_ratio", ""),
                "adaptive_skeleton_coverage_delta": row.get("delta_vs_base_adaptive_skeleton_coverage_ratio", ""),
            }
            for row in switched
        ],
        "args": vars(args),
    }
    write_csv(selected, output_dir / "line_skeleton_selected_rows.csv")
    write_csv(decisions, output_dir / "line_skeleton_decision_rows.csv")
    write_csv(switched, output_dir / "line_skeleton_switched_rows.csv")
    write_csv(summarize(selected, "active_profile"), output_dir / "line_skeleton_by_profile.csv")
    write_csv(summarize(selected, "source_name"), output_dir / "line_skeleton_by_source.csv")
    write_csv(summarize(selected, "category"), output_dir / "line_skeleton_by_category.csv")
    write_json(summary, output_dir / "line_skeleton_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
