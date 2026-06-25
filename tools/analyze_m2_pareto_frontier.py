from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MINIMIZE_METRICS = [
    "mean_unified_loss",
    "mean_jump_count",
    "mean_trim_count",
    "mean_off_mask_stitch_length_mm",
    "mean_visible_connector_count",
]

MAXIMIZE_METRICS = [
    "mean_coverage_ratio",
    "mean_stitch_precision_ratio",
]


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


def groups_by_profile(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("profile", ""))].append(row)
    return dict(groups)


def dominates(a: dict[str, str], b: dict[str, str], eps: float = 1e-12) -> bool:
    better_or_equal = True
    strictly_better = False
    for metric in MINIMIZE_METRICS:
        av = safe_float(a.get(metric))
        bv = safe_float(b.get(metric))
        if av > bv + eps:
            better_or_equal = False
            break
        if av + eps < bv:
            strictly_better = True
    if better_or_equal:
        for metric in MAXIMIZE_METRICS:
            av = safe_float(a.get(metric))
            bv = safe_float(b.get(metric))
            if av + eps < bv:
                better_or_equal = False
                break
            if av > bv + eps:
                strictly_better = True
    return better_or_equal and strictly_better


def pareto_front(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    front: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        if any(i != j and dominates(other, row) for j, other in enumerate(rows)):
            continue
        front.append(row)
    return sorted(front, key=lambda row: (safe_float(row.get("mean_unified_loss")), safe_float(row.get("mean_jump_count"))))


def outcome_signature(row: dict[str, str]) -> tuple[Any, ...]:
    signature_keys = [
        "hard_fail",
        "mean_unified_loss",
        "mean_jump_count",
        "mean_trim_count",
        "mean_off_mask_stitch_length_mm",
        "mean_visible_connector_count",
        "mean_coverage_ratio",
        "mean_stitch_precision_ratio",
        "coverage_gate_modes",
        "chosen_counts",
    ]
    return tuple(row.get(key, "") for key in signature_keys)


def collapse_duplicate_outcomes(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    best_by_signature: dict[tuple[Any, ...], dict[str, str]] = {}
    duplicate_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        signature = outcome_signature(row)
        duplicate_counts[signature] = duplicate_counts.get(signature, 0) + 1
        current = best_by_signature.get(signature)
        if current is None or safe_float(row.get("objective")) < safe_float(current.get("objective")):
            best_by_signature[signature] = row
    collapsed: list[dict[str, str]] = []
    for signature, row in best_by_signature.items():
        out = dict(row)
        out["equivalent_config_count"] = str(duplicate_counts[signature])
        collapsed.append(out)
    return sorted(collapsed, key=lambda row: (safe_float(row.get("objective")), safe_float(row.get("mean_jump_count"))))


def metric_ranges(rows: list[dict[str, str]], metrics: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = [safe_float(row.get(metric)) for row in rows]
        ranges[metric] = (min(values), max(values))
    return ranges


def normalized(value: float, low: float, high: float) -> float:
    span = high - low
    if abs(span) < 1e-12:
        return 0.0
    return max(0.0, min(1.0, (value - low) / span))


def balanced_knee_score(row: dict[str, str], rows: list[dict[str, str]], min_mean_coverage: float) -> float:
    ranges = metric_ranges(rows, MINIMIZE_METRICS + MAXIMIZE_METRICS)
    score = 0.0
    weights = {
        "mean_unified_loss": 0.32,
        "mean_jump_count": 0.18,
        "mean_trim_count": 0.08,
        "mean_off_mask_stitch_length_mm": 0.12,
        "mean_visible_connector_count": 0.05,
        "mean_coverage_ratio": 0.18,
        "mean_stitch_precision_ratio": 0.07,
    }
    for metric in MINIMIZE_METRICS:
        low, high = ranges[metric]
        score += weights[metric] * normalized(safe_float(row.get(metric)), low, high)
    for metric in MAXIMIZE_METRICS:
        low, high = ranges[metric]
        score += weights[metric] * (1.0 - normalized(safe_float(row.get(metric)), low, high))
    score += 0.5 * max(0.0, min_mean_coverage - safe_float(row.get("mean_coverage_ratio")))
    return score


def annotate_rows(rows: list[dict[str, str]], front: list[dict[str, str]], min_mean_coverage: float) -> list[dict[str, Any]]:
    front_ids = {id(row) for row in front}
    annotated: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = dict(row)
        out["is_pareto"] = id(row) in front_ids
        out["balanced_knee_score"] = round(balanced_knee_score(row, rows, min_mean_coverage), 8)
        annotated.append(out)
    return annotated


def pick_recommendations(front: list[dict[str, str]], min_mean_coverage: float, max_hard_fail: int) -> list[dict[str, Any]]:
    feasible = [
        row for row in front
        if int(safe_float(row.get("hard_fail"))) <= max_hard_fail
        and safe_float(row.get("mean_coverage_ratio")) + 1e-12 >= min_mean_coverage
    ]
    pool = feasible or front
    picks: list[tuple[str, dict[str, str]]] = []
    if pool:
        picks.append(("best_objective", min(pool, key=lambda row: safe_float(row.get("objective")))))
        picks.append(("best_low_jump", min(pool, key=lambda row: (safe_float(row.get("mean_jump_count")), safe_float(row.get("objective"))))))
        picks.append(("best_high_coverage", max(pool, key=lambda row: (safe_float(row.get("mean_coverage_ratio")), -safe_float(row.get("objective"))))))
        picks.append(("balanced_knee", min(pool, key=lambda row: balanced_knee_score(row, pool, min_mean_coverage))))
    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for label, row in picks:
        key = (label, json.dumps(row, sort_keys=True))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out = dict(row)
        out["recommendation"] = label
        out["balanced_knee_score"] = round(balanced_knee_score(row, pool, min_mean_coverage), 8)
        deduped.append(out)
    return deduped


def summarize_profile(profile: str, rows: list[dict[str, str]], front: list[dict[str, str]], recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "profile": profile,
        "rows": len(rows),
        "pareto_rows": len(front),
        "coverage_min": round(min(safe_float(row.get("mean_coverage_ratio")) for row in rows), 6),
        "coverage_max": round(max(safe_float(row.get("mean_coverage_ratio")) for row in rows), 6),
        "jump_min": round(min(safe_float(row.get("mean_jump_count")) for row in rows), 6),
        "jump_max": round(max(safe_float(row.get("mean_jump_count")) for row in rows), 6),
        "loss_min": round(min(safe_float(row.get("mean_unified_loss")) for row in rows), 6),
        "loss_max": round(max(safe_float(row.get("mean_unified_loss")) for row in rows), 6),
        "pareto_mean_coverage": round(mean(safe_float(row.get("mean_coverage_ratio")) for row in front), 6) if front else 0.0,
        "recommendations": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Pareto frontiers for M2 benefit-gate sweep rows.")
    parser.add_argument("--sweep-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-mean-coverage", type=float, default=0.90)
    parser.add_argument("--max-hard-fail", type=int, default=0)
    args = parser.parse_args()

    rows = read_csv(Path(args.sweep_csv))
    output_dir = Path(args.output_dir)
    all_pareto_rows: list[dict[str, Any]] = []
    all_recommendations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for profile, group in sorted(groups_by_profile(rows).items()):
        collapsed_group = collapse_duplicate_outcomes(group)
        front = pareto_front(collapsed_group)
        annotated_front = annotate_rows(front, front, args.min_mean_coverage)
        for row in annotated_front:
            row["profile"] = profile
        recommendations = pick_recommendations(front, args.min_mean_coverage, args.max_hard_fail)
        all_pareto_rows.extend(annotated_front)
        all_recommendations.extend(recommendations)
        summary = summarize_profile(profile, collapsed_group, front, recommendations)
        summary["raw_rows"] = len(group)
        summary["unique_outcome_rows"] = len(collapsed_group)
        summaries.append(summary)

    write_csv(all_pareto_rows, output_dir / "pareto_frontier_rows.csv")
    write_csv(all_recommendations, output_dir / "pareto_recommendations.csv")
    write_json(
        {
            "sweep_csv": args.sweep_csv,
            "min_mean_coverage": args.min_mean_coverage,
            "max_hard_fail": args.max_hard_fail,
            "profiles": summaries,
        },
        output_dir / "pareto_summary.json",
    )
    print(json.dumps({"profiles": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
