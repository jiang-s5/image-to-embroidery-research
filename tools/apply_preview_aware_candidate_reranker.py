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

from audit_dst_preview_texture import audit_sample, safe_float as audit_safe_float


COMMAND_METRICS = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
)

PREVIEW_METRICS = (
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


def is_true(value: Any) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def is_nonbaseline(row: dict[str, Any]) -> bool:
    candidate = str(row.get("candidate", ""))
    family = str(row.get("deployed_family", row.get("candidate_family", "")))
    return candidate not in ("", "base_keep", "m2_81_current_profile") and family != "base_keep"


def output_exists(row: dict[str, Any]) -> bool:
    root = Path(str(row.get("candidate_root", "")))
    sample_id = str(row.get("sample_id", ""))
    return (root / sample_id / "prediction.dst").exists()


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(baseline.get(key)), 8)


def preview_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(baseline.get(key)), 8)


def preview_rerank_score(row: dict[str, Any]) -> float:
    preview_gain = safe_float(row.get("delta_professional_preview_score"))
    texture_gain = safe_float(row.get("delta_generator_texture_score"))
    precision_gain = safe_float(row.get("delta_stitch_precision_ratio"))
    coverage_gain = safe_float(row.get("delta_coverage_ratio"))
    loss_increase = max(0.0, safe_float(row.get("delta_unified_loss")))
    jump_increase = max(0.0, safe_float(row.get("delta_jump_count")))
    trim_increase = max(0.0, safe_float(row.get("delta_trim_count")))
    return round(
        6.0 * preview_gain
        + 0.20 * texture_gain
        + 0.30 * precision_gain
        + 0.20 * coverage_gain
        - 0.80 * loss_increase
        - 0.02 * jump_increase
        - 0.02 * trim_increase,
        8,
    )


def gate_candidate(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if not is_nonbaseline(row):
        return False, "baseline_or_empty_candidate"
    if not is_true(row.get("policy_pass")):
        return False, "m2_82_policy_failed"
    if not is_true(row.get("executable_candidate", 1)):
        return False, "missing_prediction_dst"
    if safe_float(row.get("delta_professional_preview_score")) < args.min_preview_gain:
        return False, "preview_gain_too_small"
    if safe_float(row.get("delta_generator_texture_score")) < args.min_texture_gain:
        return False, "texture_regression_too_large"
    if safe_float(row.get("delta_visible_connector_count")) > args.max_visible_increase:
        return False, "visible_connector_increase"
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
    return True, "preview_aware_candidate_pass"


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


def add_audit_metrics(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in PREVIEW_METRICS:
        out[key] = audit.get(key, "")
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(rows),
        "preview_aware_switches": sum(
            1 for row in rows if str(row.get("preview_aware_selected_from", "")) == "preview_positive_candidate"
        ),
        "hard_fail": sum(
            1 for row in rows if str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower() == "hard_fail"
        ),
        "selected_candidate_counts": dict(Counter(str(row.get("candidate", "")) for row in rows)),
        "selected_family_counts": dict(Counter(str(row.get("deployed_family", row.get("candidate_family", ""))) for row in rows)),
    }
    for key in COMMAND_METRICS + PREVIEW_METRICS:
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
    baseline_rows = {row["sample_id"]: row for row in read_csv(args.baseline_rows)}
    decision_rows = read_csv(args.decision_rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        grouped[str(row.get("sample_id", ""))].append(row)

    audited_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    baseline_audits: dict[str, dict[str, Any]] = {}

    for sample_id, baseline in sorted(baseline_rows.items()):
        baseline_root = Path(str(baseline.get("candidate_root", "")))
        baseline_audit = audit_sample(baseline_root / sample_id, baseline)
        baseline_audits[sample_id] = baseline_audit
        audited_baseline = add_audit_metrics(
            {
                **baseline,
                "candidate": "m2_81_current_profile",
                "deployed_family": "base_keep",
                "candidate_root": str(baseline_root),
                "preview_candidate_gate_pass": 0,
                "preview_candidate_gate_reason": "baseline",
                "preview_candidate_score": 0.0,
            },
            baseline_audit,
        )

        passing_candidates: list[dict[str, Any]] = []
        for row in grouped.get(sample_id, []):
            if not is_nonbaseline(row) or not output_exists(row):
                continue
            candidate_root = Path(str(row.get("candidate_root", "")))
            candidate_audit = audit_sample(candidate_root / sample_id, row)
            audited = add_audit_metrics(dict(row), candidate_audit)
            for key in COMMAND_METRICS:
                audited[f"delta_{key}"] = metric_delta(audited, baseline, key)
            for key in PREVIEW_METRICS:
                audited[f"delta_{key}"] = preview_delta(audited, baseline_audit, key)
            passed, reason = gate_candidate(audited, args)
            audited["preview_candidate_gate_pass"] = int(passed)
            audited["preview_candidate_gate_reason"] = reason
            audited["preview_candidate_score"] = preview_rerank_score(audited) if passed else -999.0
            audited_rows.append(audited)
            if passed:
                passing_candidates.append(audited)

        if passing_candidates:
            selected = max(passing_candidates, key=lambda item: safe_float(item.get("preview_candidate_score")))
            selected["preview_aware_selected_from"] = "preview_positive_candidate"
        else:
            selected = audited_baseline
            selected["preview_aware_selected_from"] = "m2_81_baseline"
        selected_rows.append(selected)

    return audited_rows, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select candidates by command safety plus DST preview-texture gains.")
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv"),
    )
    parser.add_argument(
        "--decision-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_decision_rows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_85_preview_aware_candidate_reranker"),
    )
    parser.add_argument("--model-id", default="m2_85_preview_aware_candidate_reranker")
    parser.add_argument("--min-preview-gain", type=float, default=0.001)
    parser.add_argument("--min-texture-gain", type=float, default=-0.05)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--max-off-mask-increase", type=float, default=0.0)
    parser.add_argument("--max-loss-increase", type=float, default=0.01)
    parser.add_argument("--max-jump-increase", type=float, default=2.0)
    parser.add_argument("--max-trim-increase", type=float, default=2.0)
    parser.add_argument("--max-coverage-drop", type=float, default=0.015)
    parser.add_argument("--max-precision-drop", type=float, default=0.03)
    parser.add_argument("--copy-outputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audited_rows, selected_rows = build_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_outputs:
        for row in selected_rows:
            copy_outputs(row, args.output_dir)

    switched = [
        row for row in selected_rows if str(row.get("preview_aware_selected_from", "")) == "preview_positive_candidate"
    ]
    write_csv(audited_rows, args.output_dir / "preview_candidate_audit_rows.csv")
    write_csv(selected_rows, args.output_dir / "preview_aware_selected_rows.csv")
    write_csv(switched, args.output_dir / "preview_aware_switched_rows.csv")
    summary = {
        "model_id": args.model_id,
        "baseline_rows": str(args.baseline_rows),
        "decision_rows": str(args.decision_rows),
        "policy": {
            "min_preview_gain": args.min_preview_gain,
            "min_texture_gain": args.min_texture_gain,
            "max_visible_increase": args.max_visible_increase,
            "max_off_mask_increase": args.max_off_mask_increase,
            "max_loss_increase": args.max_loss_increase,
            "max_jump_increase": args.max_jump_increase,
            "max_trim_increase": args.max_trim_increase,
            "max_coverage_drop": args.max_coverage_drop,
            "max_precision_drop": args.max_precision_drop,
        },
        "candidate_audit": {
            "audited_nonbaseline_rows": len(audited_rows),
            "gate_pass_rows": sum(1 for row in audited_rows if int(row.get("preview_candidate_gate_pass", 0)) == 1),
            "positive_preview_rows": sum(
                1 for row in audited_rows if safe_float(row.get("delta_professional_preview_score")) > 0.0
            ),
            "gate_reasons": dict(Counter(str(row.get("preview_candidate_gate_reason", "")) for row in audited_rows)),
            "gate_pass_by_family": dict(
                Counter(
                    str(row.get("deployed_family", row.get("candidate_family", "")))
                    for row in audited_rows
                    if int(row.get("preview_candidate_gate_pass", 0)) == 1
                )
            ),
        },
        "selected_summary": summarize(selected_rows),
        "by_source": group_summary(selected_rows, "source_name"),
        "switch_sample_ids": [row["sample_id"] for row in switched],
        "interpretation": "M2.85 audits the broader M2.82 candidate pool with M2.83's DST-derived preview texture score and selects only candidates with positive preview gains plus command-safety gates.",
    }
    write_json(summary, args.output_dir / "preview_aware_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
