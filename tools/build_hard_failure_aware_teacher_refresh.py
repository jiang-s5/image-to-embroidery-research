from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


TRAINABLE_FAMILY_MAP = {
    "base_keep": "base_keep",
    "edgewalk_fill": "fill_tatami_like",
    "nearestrow_fill": "fill_tatami_like",
    "mask_fill": "fill_tatami_like",
    "dt_satin": "satin_like",
    "satin_like": "satin_like",
    "satin_rail": "satin_like",
    "outline_border": "outline_border",
    "style_running": "running_line",
    "running_line": "running_line",
}

EXTRA_RISK_FEATURES = (
    "execution_penalty_score",
    "hard_fail_probe_selected",
    "jump_explosion_ratio",
    "trim_explosion_ratio",
    "offmask_explosion_mm",
    "visible_explosion_count",
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
    if not path.exists():
        return []
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


def trainable_family(value: str) -> str:
    return TRAINABLE_FAMILY_MAP.get(value, value if value in TRAINABLE_FAMILY_MAP.values() else "reject")


def candidate_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("sample_id", "")),
        str(row.get("candidate_root", "")),
        str(row.get("candidate", "")),
    )


def hard_fail_selected_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        candidate_key(row)
        for row in rows
        if int(safe_float(row.get("differs_from_m2_86"))) == 1
        or str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower() == "hard_fail"
    }


def execution_penalty(row: dict[str, Any], selected_bad: bool) -> dict[str, float]:
    jump = safe_float(row.get("jump_count"))
    trim = safe_float(row.get("trim_count"))
    offmask = safe_float(row.get("off_mask_stitch_length_mm"))
    visible = safe_float(row.get("visible_connector_count"))
    delta_loss = max(0.0, safe_float(row.get("delta_unified_loss")))
    delta_jump = max(0.0, safe_float(row.get("delta_jump_count")))
    delta_trim = max(0.0, safe_float(row.get("delta_trim_count")))
    delta_offmask = max(0.0, safe_float(row.get("delta_off_mask_stitch_length_mm")))
    delta_visible = max(0.0, safe_float(row.get("delta_visible_connector_count")))

    jump_explosion = max(jump / 50.0, delta_jump / 10.0)
    trim_explosion = max(trim / 10.0, delta_trim / 3.0)
    offmask_explosion = max(offmask, delta_offmask)
    visible_explosion = max(visible, delta_visible)
    score = (
        1.0 * min(8.0, jump_explosion)
        + 0.9 * min(8.0, trim_explosion)
        + 1.5 * min(8.0, offmask_explosion / 10.0)
        + 2.0 * min(8.0, visible_explosion)
        + 8.0 * delta_loss
    )
    quality = str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower()
    if quality == "hard_fail":
        score += 4.0
    if selected_bad:
        score += 2.0
    return {
        "execution_penalty_score": round(score, 8),
        "hard_fail_probe_selected": 1.0 if selected_bad else 0.0,
        "jump_explosion_ratio": round(jump_explosion, 8),
        "trim_explosion_ratio": round(trim_explosion, 8),
        "offmask_explosion_mm": round(offmask_explosion, 8),
        "visible_explosion_count": round(visible_explosion, 8),
    }


def build_probe_hard_negative(row: dict[str, str], selected_bad_keys: set[tuple[str, str, str]], weight_scale: float) -> dict[str, Any]:
    selected_bad = candidate_key(row) in selected_bad_keys
    risk = execution_penalty(row, selected_bad)
    family = trainable_family(str(row.get("candidate_family", "")))
    out: dict[str, Any] = dict(row)
    out.update(risk)
    out.update(
        {
            "seed_source": "m2_88_no_gate_hard_negative",
            "source_file": row.get("candidate_root", ""),
            "candidate_family": family,
            "teacher_family": "reject",
            "is_positive_teacher": 0,
            "is_hard_negative": 1,
            "teacher_weight": round(weight_scale + min(10.0, risk["execution_penalty_score"] / 3.0), 8),
            "teacher_reason": "m2_88_no_gate_false_positive_hard_negative",
            "target_branch": row.get("mode", row.get("candidate", "")),
            "allowed_by_professional_gate": 0,
            "teacher_is_texture_switch": 0,
            "preview_selected": 0,
            "texture_score": safe_float(row.get("generator_texture_score", row.get("texture_score"))),
            "delta_texture_score": safe_float(row.get("delta_generator_texture_score", row.get("delta_texture_score"))),
        }
    )
    return out


def add_default_risk_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in EXTRA_RISK_FEATURES:
            item.setdefault(key, 0.0)
        out.append(item)
    return out


def group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for value, items in sorted(grouped.items()):
        out.append(
            {
                key: value,
                "rows": len(items),
                "positive_rows": sum(1 for row in items if safe_float(row.get("is_positive_teacher")) > 0.5),
                "hard_negative_rows": sum(1 for row in items if safe_float(row.get("is_hard_negative")) > 0.5),
                "mean_teacher_weight": avg(items, "teacher_weight"),
                "mean_execution_penalty_score": avg(items, "execution_penalty_score"),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]], added: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if safe_float(row.get("is_positive_teacher")) > 0.5]
    hard_negatives = [row for row in rows if safe_float(row.get("is_hard_negative")) > 0.5]
    return {
        "rows": len(rows),
        "added_hard_negative_rows": len(added),
        "positive_teacher_rows": len(positives),
        "hard_negative_rows": len(hard_negatives),
        "teacher_family_counts": dict(Counter(str(row.get("teacher_family", "")) for row in rows)),
        "seed_source_counts": dict(Counter(str(row.get("seed_source", "")) for row in rows)),
        "added_sample_ids": sorted({str(row.get("sample_id", "")) for row in added}),
        "added_candidate_counts": dict(Counter(str(row.get("candidate", "")) for row in added)),
        "mean_added_teacher_weight": avg(added, "teacher_weight"),
        "mean_added_execution_penalty_score": avg(added, "execution_penalty_score"),
        "mean_added_jump_explosion_ratio": avg(added, "jump_explosion_ratio"),
        "mean_added_trim_explosion_ratio": avg(added, "trim_explosion_ratio"),
        "mean_added_offmask_explosion_mm": avg(added, "offmask_explosion_mm"),
        "mean_added_visible_explosion_count": avg(added, "visible_explosion_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add M2.88 no-gate false positives as high-weight hard negatives for the preview-aware selector."
    )
    parser.add_argument(
        "--base-seed-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh/preview_teacher_seed_rows.csv"),
    )
    parser.add_argument(
        "--no-gate-decision-rows",
        type=Path,
        default=Path(
            "results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy_relaxed_nogate/"
            "preview_learned_decision_rows.csv"
        ),
    )
    parser.add_argument(
        "--no-gate-selected-rows",
        type=Path,
        default=Path(
            "results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy_relaxed_nogate/"
            "preview_learned_selected_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_89_hard_failure_teacher_refresh"),
    )
    parser.add_argument("--model-id", default="m2_89_hard_failure_teacher_refresh")
    parser.add_argument("--teacher-weight-scale", type=float, default=4.0)
    args = parser.parse_args()

    base_rows = add_default_risk_features([dict(row) for row in read_csv(args.base_seed_rows)])
    selected_bad = hard_fail_selected_keys(read_csv(args.no_gate_selected_rows))
    decision_rows = read_csv(args.no_gate_decision_rows)
    probe_rows = [
        row
        for row in decision_rows
        if str(row.get("selector_deployed_family", "")) != "reject"
        and safe_float(row.get("preview_sweep_gate_pass")) < 0.5
    ]
    added_rows = [
        build_probe_hard_negative(row, selected_bad, args.teacher_weight_scale)
        for row in probe_rows
    ]
    refreshed = base_rows + added_rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(refreshed, args.output_dir / "hard_failure_teacher_seed_rows.csv")
    write_csv(added_rows, args.output_dir / "hard_failure_added_rows.csv")
    write_csv(group_rows(refreshed, "teacher_family"), args.output_dir / "hard_failure_by_teacher_family.csv")
    write_csv(group_rows(refreshed, "seed_source"), args.output_dir / "hard_failure_by_seed_source.csv")
    summary = {
        "model_id": args.model_id,
        "base_seed_rows": str(args.base_seed_rows),
        "no_gate_decision_rows": str(args.no_gate_decision_rows),
        "no_gate_selected_rows": str(args.no_gate_selected_rows),
        "summary": summarize(refreshed, added_rows),
        "labeling_policy": {
            "source": "M2.88 no-gate rows where M2.87 deployed a non-reject family even though the M2.86 preview gate failed.",
            "teacher_family": "reject",
            "teacher_weight": "teacher_weight_scale + execution_penalty_score / 3, capped by the script",
            "risk_features": list(EXTRA_RISK_FEATURES),
        },
        "interpretation": (
            "M2.89 does not promote a new output profile by itself. It strengthens the learned selector against candidates that "
            "look preview-positive but explode in command-level execution metrics when the deterministic safety gate is removed."
        ),
    }
    write_json(summary, args.output_dir / "hard_failure_teacher_refresh_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
