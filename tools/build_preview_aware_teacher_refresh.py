from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASE_METRIC_KEYS = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
    "texture_score",
    "delta_texture_score",
    "professional_preview_score",
    "delta_professional_preview_score",
    "generator_texture_score",
    "delta_generator_texture_score",
    "preview_sweep_score",
    "preview_sweep_gate_pass",
)

HARD_REJECT_REASONS = {
    "visible_connector_increase": 1.40,
    "off_mask_increase": 1.25,
    "precision_drop": 0.90,
    "loss_increase": 0.55,
    "texture_regression_too_large": 1.00,
}


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


def label_family(value: str) -> str:
    family = value.lower()
    if family in {"base_keep", "running_line", "auto_running", "auto_fill", "fill_tatami_like", "satin_like", "outline_border"}:
        return family
    if family in {"edgewalk_fill", "nearestrow_fill", "mask_fill"}:
        return "fill_tatami_like"
    if family in {"dt_satin", "satin_rail", "satinrail", "satin"}:
        return "satin_like"
    if family in {"style_running", "style_aware_mixed"}:
        return "running_line"
    return "reject" if family == "reject" else "candidate_non_teacher"


def normalized_preview_row(row: dict[str, str], selected_keys: set[tuple[str, str]]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", ""))
    candidate_root = str(row.get("candidate_root", ""))
    key = (sample_id, candidate_root)
    gate_reason = str(row.get("preview_sweep_gate_reason", ""))
    gate_pass = safe_float(row.get("preview_sweep_gate_pass")) > 0.5
    selected = key in selected_keys
    candidate_family = label_family(str(row.get("candidate_family", "")))
    delta_preview = safe_float(row.get("delta_professional_preview_score"))

    if selected:
        teacher_family = candidate_family
        is_positive = 1
        is_hard_negative = 0
        # The selected rows are scarce and high-value. Weight them enough to survive the larger reject pool.
        teacher_weight = round(1.75 + min(1.25, max(0.0, delta_preview) * 20.0), 8)
        teacher_reason = "m2_86_preview_positive_selected"
    elif gate_pass:
        teacher_family = "candidate_non_teacher"
        is_positive = 0
        is_hard_negative = 0
        teacher_weight = 0.30
        teacher_reason = "m2_86_preview_safe_but_not_selected"
    elif gate_reason in HARD_REJECT_REASONS:
        teacher_family = "reject"
        is_positive = 0
        is_hard_negative = 1
        teacher_weight = HARD_REJECT_REASONS[gate_reason]
        teacher_reason = f"m2_86_{gate_reason}"
    else:
        teacher_family = "candidate_non_teacher"
        is_positive = 0
        is_hard_negative = 0
        teacher_weight = 0.20
        teacher_reason = f"m2_86_{gate_reason or 'weak_non_teacher'}"

    item: dict[str, Any] = {
        "seed_source": "m2_86_preview_sweep",
        "source_file": row.get("candidate_root", ""),
        "sample_id": sample_id,
        "source_name": row.get("source_name", ""),
        "category": row.get("category", ""),
        "candidate": row.get("candidate", ""),
        "candidate_family": candidate_family,
        "teacher_family": teacher_family,
        "is_positive_teacher": is_positive,
        "is_hard_negative": is_hard_negative,
        "teacher_weight": teacher_weight,
        "teacher_reason": teacher_reason,
        "target_branch": row.get("mode", ""),
        "oracle_quality_level": row.get("oracle_quality_level", row.get("quality_level", "")),
        "allowed_by_professional_gate": 1 if gate_pass or selected else 0,
        "teacher_is_texture_switch": 1 if selected else 0,
        "preview_selected": 1 if selected else 0,
        "preview_sweep_gate_reason": gate_reason,
    }
    for key_name in BASE_METRIC_KEYS:
        if key_name == "texture_score":
            item[key_name] = safe_float(row.get("generator_texture_score"))
        elif key_name == "delta_texture_score":
            item[key_name] = safe_float(row.get("delta_generator_texture_score"))
        else:
            item[key_name] = safe_float(row.get(key_name))
    item["delta_coverage_ratio"] = safe_float(row.get("delta_coverage_ratio"))
    item["delta_stitch_precision_ratio"] = safe_float(row.get("delta_stitch_precision_ratio"))
    item["candidate_root"] = row.get("candidate_root", "")
    item["candidate_dir_name"] = row.get("candidate_dir_name", "")
    return item


def selected_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if str(row.get("preview_sweep_selected_from", "")) == "preview_positive_sweep_candidate":
            keys.add((str(row.get("sample_id", "")), str(row.get("candidate_root", ""))))
    return keys


def positive_teacher_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if safe_float(row.get("is_positive_teacher")) <= 0.5:
            continue
        selected.append(
            {
                "sample_id": row.get("sample_id", ""),
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "seed_source": row.get("seed_source", ""),
                "candidate": row.get("candidate", ""),
                "candidate_family": row.get("candidate_family", ""),
                "teacher_family": row.get("teacher_family", ""),
                "teacher_weight": row.get("teacher_weight", ""),
                "professional_preview_score": row.get("professional_preview_score", ""),
                "delta_professional_preview_score": row.get("delta_professional_preview_score", ""),
                "generator_texture_score": row.get("generator_texture_score", ""),
                "delta_generator_texture_score": row.get("delta_generator_texture_score", ""),
                "unified_loss": row.get("unified_loss", ""),
                "jump_count": row.get("jump_count", ""),
                "trim_count": row.get("trim_count", ""),
                "coverage_ratio": row.get("coverage_ratio", ""),
                "stitch_precision_ratio": row.get("stitch_precision_ratio", ""),
                "teacher_reason": row.get("teacher_reason", ""),
            }
        )
    return selected


def summarize(rows: list[dict[str, Any]], added_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if safe_float(row.get("is_positive_teacher")) > 0.5]
    added_positives = [row for row in added_rows if safe_float(row.get("is_positive_teacher")) > 0.5]
    hard_negatives = [row for row in rows if safe_float(row.get("is_hard_negative")) > 0.5]
    samples = {str(row.get("sample_id", "")) for row in rows}
    return {
        "candidate_rows": len(rows),
        "added_preview_rows": len(added_rows),
        "samples": len(samples),
        "positive_teacher_rows": len(positives),
        "added_positive_teacher_rows": len(added_positives),
        "hard_negative_rows": len(hard_negatives),
        "teacher_family_counts": dict(Counter(str(row.get("teacher_family", "")) for row in rows)),
        "positive_teacher_family_counts": dict(Counter(str(row.get("teacher_family", "")) for row in positives)),
        "positive_seed_source_counts": dict(Counter(str(row.get("seed_source", "")) for row in positives)),
        "m2_86_gate_reason_counts": dict(Counter(str(row.get("preview_sweep_gate_reason", "")) for row in added_rows)),
        "mean_positive_unified_loss": avg(positives, "unified_loss"),
        "mean_positive_jump_count": avg(positives, "jump_count"),
        "mean_positive_trim_count": avg(positives, "trim_count"),
        "mean_positive_coverage_ratio": avg(positives, "coverage_ratio"),
        "mean_positive_stitch_precision_ratio": avg(positives, "stitch_precision_ratio"),
        "mean_added_positive_preview_score": avg(added_positives, "professional_preview_score"),
        "mean_added_positive_preview_delta": avg(added_positives, "delta_professional_preview_score"),
        "mean_added_positive_texture_delta": avg(added_positives, "delta_generator_texture_score"),
    }


def write_group_summary(rows: list[dict[str, Any]], key: str, path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for name, items in sorted(grouped.items()):
        positives = [row for row in items if safe_float(row.get("is_positive_teacher")) > 0.5]
        out.append(
            {
                key: name,
                "candidate_rows": len(items),
                "positive_teacher_rows": len(positives),
                "hard_negative_rows": sum(1 for row in items if safe_float(row.get("is_hard_negative")) > 0.5),
                "positive_teacher_families": json.dumps(
                    dict(Counter(str(row.get("teacher_family", "")) for row in positives)),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "mean_positive_preview_score": avg(positives, "professional_preview_score"),
                "mean_positive_preview_delta": avg(positives, "delta_professional_preview_score"),
                "mean_positive_coverage_ratio": avg(positives, "coverage_ratio"),
                "mean_positive_stitch_precision_ratio": avg(positives, "stitch_precision_ratio"),
            }
        )
    write_csv(out, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh stitch-family teacher rows with M2.86 preview-aware labels.")
    parser.add_argument(
        "--base-seed-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed/stitch_type_teacher_seed_rows.csv"),
    )
    parser.add_argument(
        "--preview-candidate-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv"),
    )
    parser.add_argument(
        "--preview-selected-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_selected_rows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh"),
    )
    parser.add_argument("--model-id", default="m2_87_preview_teacher_refresh")
    args = parser.parse_args()

    base_rows = [dict(row) for row in read_csv(args.base_seed_rows)]
    selected = selected_keys(read_csv(args.preview_selected_rows))
    added_rows = [
        normalized_preview_row(row, selected)
        for row in read_csv(args.preview_candidate_rows)
    ]
    refreshed_rows = base_rows + added_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(refreshed_rows, args.output_dir / "preview_teacher_seed_rows.csv")
    write_csv(added_rows, args.output_dir / "preview_teacher_added_rows.csv")
    write_csv(positive_teacher_rows(refreshed_rows), args.output_dir / "preview_positive_teacher_rows.csv")
    write_group_summary(refreshed_rows, "teacher_family", args.output_dir / "preview_teacher_by_teacher_family.csv")
    write_group_summary(refreshed_rows, "seed_source", args.output_dir / "preview_teacher_by_seed_source.csv")
    write_group_summary(refreshed_rows, "source_name", args.output_dir / "preview_teacher_by_source.csv")

    summary = {
        "model_id": args.model_id,
        "base_seed_rows": str(args.base_seed_rows),
        "preview_candidate_rows": str(args.preview_candidate_rows),
        "preview_selected_rows": str(args.preview_selected_rows),
        "summary": summarize(refreshed_rows, added_rows),
        "labeling_policy": {
            "selected_m2_86_switch": "positive teacher with family mapped to trainable stitch family",
            "preview_gate_pass_but_not_selected": "neutral candidate_non_teacher",
            "visible/off_mask/precision/loss/texture_regression": "reject with reason-specific weights",
            "preview_gain_too_small": "neutral candidate_non_teacher"
        },
        "interpretation": (
            "M2.87 adds DST-preview-aware supervision to the M2.78 seed. It does not promote a new output profile by itself; "
            "it prepares a learned selector to prefer candidates that improve preview texture while learning explicit reject "
            "signals from unsafe or regressing candidates."
        ),
    }
    write_json(summary, args.output_dir / "preview_teacher_refresh_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
