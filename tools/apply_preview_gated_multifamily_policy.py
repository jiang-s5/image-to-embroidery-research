from __future__ import annotations

import argparse
import csv
import json
import math
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
    "professional_preview_score",
    "generator_texture_score",
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
                keys.append(key)
                seen.add(key)
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


def is_nonbaseline(row: dict[str, Any]) -> bool:
    candidate = str(row.get("candidate", ""))
    family = str(row.get("deployed_family", row.get("candidate_family", "")))
    return candidate not in ("", "base_keep", "m2_81_current_profile") and family != "base_keep"


def copy_outputs(row: dict[str, Any], output_dir: Path) -> None:
    root = Path(str(row.get("candidate_root", "")))
    sample_id = str(row["sample_id"])
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "prediction.dst",
        "eval_executability.json",
        "generator_report.json",
        "coverage.json",
        "coverage_report.json",
    ):
        src = root / sample_id / filename
        if src.exists():
            shutil.copy2(src, sample_dir / filename)


def gate_candidate(row: dict[str, Any], delta: dict[str, str], args: argparse.Namespace) -> tuple[bool, str]:
    if not is_nonbaseline(row):
        return False, "baseline_or_empty_candidate"
    if str(row.get("policy_pass", "1")) in ("0", "False", "false"):
        return False, "m2_82_policy_failed"
    if safe_float(delta.get("delta_professional_preview_score")) < args.min_preview_gain:
        return False, "preview_score_regression"
    if safe_float(delta.get("delta_generator_texture_score")) < args.min_texture_gain:
        return False, "texture_score_regression"
    if safe_float(delta.get("delta_visible_connector_count")) > args.max_visible_increase:
        return False, "visible_connector_increase"
    if safe_float(delta.get("delta_off_mask_stitch_length_mm")) > args.max_off_mask_increase:
        return False, "off_mask_increase"
    return True, "preview_gate_pass"


def merge_preview_metrics(row: dict[str, Any], delta: dict[str, str], prefix: str) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "professional_preview_score",
        "generator_texture_score",
        "angle_entropy",
        "dominant_angle_mass",
        "stitch_density_per_100mm2",
        "rhythm_score",
        "visible_connector_count",
        "off_mask_stitch_length_mm",
    ):
        value = delta.get(f"{prefix}_{key}", "")
        if value != "":
            out[key] = value
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(rows),
        "preview_gate_switches": sum(1 for row in rows if str(row.get("preview_gate_selected_from", "")) == "m2_82_candidate"),
        "hard_fail": sum(
            1 for row in rows if str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower() == "hard_fail"
        ),
        "selected_candidate_counts": dict(Counter(str(row.get("candidate", "")) for row in rows)),
        "selected_family_counts": dict(Counter(str(row.get("deployed_family", row.get("candidate_family", ""))) for row in rows)),
    }
    for key in METRIC_KEYS:
        summary[f"mean_{key}"] = avg(rows, key)
    return summary


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    out = []
    for value, items in sorted(groups.items()):
        item = {key: value}
        item.update(summarize(items))
        out.append(item)
    return out


def build_policy(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {row["sample_id"]: row for row in read_csv(args.baseline_rows)}
    m2_82 = {row["sample_id"]: row for row in read_csv(args.multifamily_rows)}
    deltas = {row["sample_id"]: row for row in read_csv(args.preview_delta_rows)}
    decision_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for sample_id in sorted(baseline):
        base_row = baseline[sample_id]
        candidate = m2_82.get(sample_id, base_row)
        delta = deltas.get(sample_id, {})
        passed, reason = gate_candidate(candidate, delta, args)

        decision = dict(candidate)
        decision.update(
            {
                "preview_gate_pass": int(passed),
                "preview_gate_reason": reason,
                "delta_professional_preview_score": delta.get("delta_professional_preview_score", ""),
                "delta_generator_texture_score": delta.get("delta_generator_texture_score", ""),
                "delta_visible_connector_count": delta.get("delta_visible_connector_count", ""),
                "delta_off_mask_stitch_length_mm": delta.get("delta_off_mask_stitch_length_mm", ""),
            }
        )
        decision_rows.append(decision)

        if passed:
            selected = merge_preview_metrics(candidate, delta, "compare")
            selected["preview_gate_selected_from"] = "m2_82_candidate"
            selected["preview_gate_reason"] = reason
        else:
            selected = merge_preview_metrics(base_row, delta, "base")
            selected["preview_gate_selected_from"] = "m2_81_baseline"
            selected["preview_gate_reason"] = reason
            selected["candidate"] = "m2_81_current_profile"
            selected["deployed_family"] = "base_keep"
        selected["delta_professional_preview_score"] = delta.get("delta_professional_preview_score", "")
        selected["delta_generator_texture_score"] = delta.get("delta_generator_texture_score", "")
        selected_rows.append(selected)

    return decision_rows, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a preview-texture gate on top of M2.82 multifamily candidates.")
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv"),
    )
    parser.add_argument(
        "--multifamily-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_selected_rows.csv"),
    )
    parser.add_argument(
        "--preview-delta-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit/dst_preview_texture_delta_rows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_84_preview_gated_multifamily_policy"),
    )
    parser.add_argument("--model-id", default="m2_84_preview_gated_multifamily_policy")
    parser.add_argument("--min-preview-gain", type=float, default=0.0)
    parser.add_argument("--min-texture-gain", type=float, default=-0.005)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--copy-outputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision_rows, selected_rows = build_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_outputs:
        for row in selected_rows:
            copy_outputs(row, args.output_dir)

    switched = [row for row in selected_rows if str(row.get("preview_gate_selected_from", "")) == "m2_82_candidate"]
    rejected = [row for row in decision_rows if is_nonbaseline(row) and int(row.get("preview_gate_pass", 0)) == 0]
    write_csv(decision_rows, args.output_dir / "preview_gate_decision_rows.csv")
    write_csv(selected_rows, args.output_dir / "preview_gate_selected_rows.csv")
    write_csv(switched, args.output_dir / "preview_gate_switched_rows.csv")
    summary = {
        "model_id": args.model_id,
        "baseline_rows": str(args.baseline_rows),
        "multifamily_rows": str(args.multifamily_rows),
        "preview_delta_rows": str(args.preview_delta_rows),
        "policy": {
            "min_preview_gain": args.min_preview_gain,
            "min_texture_gain": args.min_texture_gain,
            "max_visible_increase": args.max_visible_increase,
            "max_off_mask_increase": args.max_off_mask_increase,
        },
        "selected_summary": summarize(selected_rows),
        "by_source": group_summary(selected_rows, "source_name"),
        "switch_sample_ids": [row["sample_id"] for row in switched],
        "rejected_nonbaseline_sample_ids": [row["sample_id"] for row in rejected],
        "rejection_reasons": dict(Counter(str(row.get("preview_gate_reason", "")) for row in decision_rows)),
        "interpretation": "M2.84 keeps M2.82's command-safety path, but adds the M2.83 preview-texture gate so command-safe candidates cannot replace the baseline when their DST-derived preview score or texture-family score regresses.",
    }
    write_json(summary, args.output_dir / "preview_gate_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
