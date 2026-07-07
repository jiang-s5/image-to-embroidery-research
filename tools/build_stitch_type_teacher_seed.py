from __future__ import annotations

import argparse
import csv
import json
import math
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
    values = [safe_float(row.get(key)) for row in rows]
    return round(mean(values), 8) if values else 0.0


def style_family(candidate: str, row: dict[str, Any]) -> str:
    name = candidate.lower()
    if name == "base_keep":
        return "base_keep"
    if "dt_satin" in name or "dtsatin" in name or "satinrail" in name or "satin_rail" in name or "satin" in name:
        return "satin_like"
    if "styleaware" in name or "style_aware" in name:
        return "style_aware_mixed"
    if "outline" in name:
        return "outline_border"
    if "mask_fill" in name:
        return "fill_tatami_like"
    if "skeleton" in name or "skel" in name:
        return "running_line"
    if "auto" in name:
        target = str(row.get("target_branch", row.get("planner_branch", ""))).lower()
        if "skeleton" in target or "line" in target or "text" in target:
            return "auto_running"
        return "auto_fill"
    return "unknown"


def is_hard_negative(row: dict[str, Any], max_off_mask: float, max_visible: float) -> bool:
    quality = str(row.get("oracle_quality_level", "")).lower()
    gate_reason = str(row.get("gate_reason", "")).lower()
    if "hard_fail" in quality:
        return True
    if safe_float(row.get("visible_connector_count")) > max_visible:
        return True
    if safe_float(row.get("off_mask_stitch_length_mm")) > max_off_mask:
        return True
    if "off_mask" in gate_reason or "precision_drop" in gate_reason:
        return True
    return False


def normalize_candidate_selector_rows(
    rows: list[dict[str, str]],
    source_path: Path,
    max_off_mask: float,
    max_visible: float,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        candidate = str(row.get("candidate", ""))
        family = style_family(candidate, row)
        is_positive = safe_float(row.get("is_oracle_choice")) > 0.5
        hard_negative = is_hard_negative(row, max_off_mask, max_visible)
        if is_positive:
            target = family
            weight = 1.0
            reason = "oracle_choice"
        elif hard_negative:
            target = "reject"
            weight = 0.75
            reason = "hard_negative"
        else:
            target = "candidate_non_teacher"
            weight = 0.25
            reason = "non_teacher_candidate"
        item: dict[str, Any] = {
            "seed_source": "candidate_selector",
            "source_file": str(source_path),
            "sample_id": row.get("sample_id", ""),
            "source_name": row.get("source_name", ""),
            "category": row.get("category", ""),
            "candidate": candidate,
            "candidate_family": family,
            "teacher_family": target,
            "is_positive_teacher": 1 if is_positive else 0,
            "is_hard_negative": 1 if hard_negative else 0,
            "teacher_weight": weight,
            "teacher_reason": reason,
            "target_branch": row.get("target_branch", ""),
            "oracle_quality_level": row.get("oracle_quality_level", ""),
        }
        for key in METRIC_KEYS:
            item[key] = safe_float(row.get(key))
        normalized.append(item)
    return normalized


def normalize_stitch_type_rows(
    rows: list[dict[str, str]],
    source_path: Path,
    max_off_mask: float,
    max_visible: float,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        candidate = str(row.get("candidate", ""))
        family = style_family(candidate, row)
        is_positive = safe_float(row.get("is_teacher_choice")) > 0.5
        hard_negative = is_hard_negative(row, max_off_mask, max_visible)
        allowed = safe_float(row.get("allowed_by_professional_gate")) > 0.5
        if is_positive:
            target = family
            weight = 1.25 if family == "satin_like" else 1.0
            reason = "m2_74_teacher_choice"
        elif hard_negative or not allowed:
            target = "reject"
            weight = 0.75
            reason = "professional_gate_reject"
        else:
            target = "candidate_non_teacher"
            weight = 0.25
            reason = "allowed_non_teacher_candidate"
        item: dict[str, Any] = {
            "seed_source": "stitch_type_teacher",
            "source_file": str(source_path),
            "sample_id": row.get("sample_id", ""),
            "source_name": row.get("source_name", ""),
            "category": row.get("category", ""),
            "candidate": candidate,
            "candidate_family": family,
            "teacher_family": target,
            "is_positive_teacher": 1 if is_positive else 0,
            "is_hard_negative": 1 if hard_negative or not allowed else 0,
            "teacher_weight": weight,
            "teacher_reason": reason,
            "target_branch": row.get("planner_branch", ""),
            "oracle_quality_level": row.get("gate_reason", ""),
            "allowed_by_professional_gate": 1 if allowed else 0,
            "teacher_is_texture_switch": row.get("teacher_is_texture_switch", "0"),
        }
        for key in METRIC_KEYS:
            item[key] = safe_float(row.get(key))
        item["texture_score"] = safe_float(row.get("texture_score"))
        item["delta_texture_score"] = safe_float(row.get("delta_texture_score"))
        normalized.append(item)
    return normalized


def positive_teacher_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    selected: list[dict[str, Any]] = []
    for sample_id, items in sorted(grouped.items()):
        positives = [row for row in items if safe_float(row.get("is_positive_teacher")) > 0.5]
        for row in positives:
            selected.append(
                {
                    "sample_id": sample_id,
                    "source_name": row.get("source_name", ""),
                    "category": row.get("category", ""),
                    "seed_source": row.get("seed_source", ""),
                    "candidate": row.get("candidate", ""),
                    "teacher_family": row.get("teacher_family", ""),
                    "teacher_weight": row.get("teacher_weight", ""),
                    "unified_loss": row.get("unified_loss", ""),
                    "jump_count": row.get("jump_count", ""),
                    "trim_count": row.get("trim_count", ""),
                    "coverage_ratio": row.get("coverage_ratio", ""),
                    "stitch_precision_ratio": row.get("stitch_precision_ratio", ""),
                    "teacher_reason": row.get("teacher_reason", ""),
                }
            )
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if safe_float(row.get("is_positive_teacher")) > 0.5]
    hard_negatives = [row for row in rows if safe_float(row.get("is_hard_negative")) > 0.5]
    samples = {str(row.get("sample_id", "")) for row in rows}
    source_names = {str(row.get("source_name", "")) for row in rows}
    return {
        "candidate_rows": len(rows),
        "samples": len(samples),
        "source_names": sorted(source_names),
        "positive_teacher_rows": len(positives),
        "hard_negative_rows": len(hard_negatives),
        "candidate_family_counts": dict(Counter(str(row.get("candidate_family", "")) for row in rows)),
        "teacher_family_counts": dict(Counter(str(row.get("teacher_family", "")) for row in rows)),
        "positive_teacher_family_counts": dict(Counter(str(row.get("teacher_family", "")) for row in positives)),
        "positive_seed_source_counts": dict(Counter(str(row.get("seed_source", "")) for row in positives)),
        "mean_positive_unified_loss": avg(positives, "unified_loss"),
        "mean_positive_jump_count": avg(positives, "jump_count"),
        "mean_positive_trim_count": avg(positives, "trim_count"),
        "mean_positive_coverage_ratio": avg(positives, "coverage_ratio"),
        "mean_positive_stitch_precision_ratio": avg(positives, "stitch_precision_ratio"),
    }


def write_group_summary(rows: list[dict[str, Any]], key: str, path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    summary_rows = []
    for name, items in sorted(grouped.items()):
        positives = [row for row in items if safe_float(row.get("is_positive_teacher")) > 0.5]
        summary_rows.append(
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
                "mean_positive_unified_loss": avg(positives, "unified_loss"),
                "mean_positive_coverage_ratio": avg(positives, "coverage_ratio"),
                "mean_positive_stitch_precision_ratio": avg(positives, "stitch_precision_ratio"),
            }
        )
    write_csv(summary_rows, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a broader stitch-type teacher seed table from planner candidates.")
    parser.add_argument("--candidate-selector-csv", action="append", default=[])
    parser.add_argument("--stitch-type-csv", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-off-mask-mm", type=float, default=0.5)
    parser.add_argument("--max-visible-connectors", type=float, default=0.0)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for csv_path in args.candidate_selector_csv:
        path = Path(csv_path)
        all_rows.extend(normalize_candidate_selector_rows(read_csv(path), path, args.max_off_mask_mm, args.max_visible_connectors))
    for csv_path in args.stitch_type_csv:
        path = Path(csv_path)
        all_rows.extend(normalize_stitch_type_rows(read_csv(path), path, args.max_off_mask_mm, args.max_visible_connectors))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, output_dir / "stitch_type_teacher_seed_rows.csv")
    write_csv(positive_teacher_rows(all_rows), output_dir / "stitch_type_positive_teacher_rows.csv")
    write_group_summary(all_rows, "candidate_family", output_dir / "stitch_type_seed_by_candidate_family.csv")
    write_group_summary(all_rows, "teacher_family", output_dir / "stitch_type_seed_by_teacher_family.csv")
    write_group_summary(all_rows, "source_name", output_dir / "stitch_type_seed_by_source.csv")

    summary = {
        "model_id": "m2_78_stitch_type_teacher_seed",
        "status": "broader_teacher_seed_for_professional_stitch_type_planning",
        "inputs": {
            "candidate_selector_csv": args.candidate_selector_csv,
            "stitch_type_csv": args.stitch_type_csv,
        },
        "thresholds": {
            "max_off_mask_mm": args.max_off_mask_mm,
            "max_visible_connectors": args.max_visible_connectors,
        },
        "summary": summarize(all_rows),
        "interpretation": (
            "This seed table expands stitch-type supervision beyond M2.75 satin-only choices. "
            "It includes running-line, fill/tatami-like, satin-like, outline, mixed-style, reject, "
            "and non-teacher candidate rows so the next selector can learn richer style decisions."
        ),
    }
    write_json(summary, output_dir / "stitch_type_teacher_seed_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
