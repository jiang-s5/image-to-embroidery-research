from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CONFIG_KEYS = [
    "profile",
    "strength",
    "coverage_floor",
    "penalty_weight",
    "min_coverage",
    "min_jump_gain",
    "min_precision",
    "max_loss_slack",
]

METRIC_KEYS = [
    "objective",
    "mean_unified_loss",
    "mean_jump_count",
    "mean_trim_count",
    "mean_off_mask_stitch_length_mm",
    "mean_visible_connector_count",
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def parse_name_path(items: list[list[str]]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for name, path in items:
        parsed.append((name, Path(path)))
    names = [name for name, _ in parsed]
    if len(names) != len(set(names)):
        raise ValueError("Sweep names must be unique.")
    return parsed


def parse_csv_strings(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def config_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in CONFIG_KEYS)


def row_is_feasible(row: dict[str, str], min_mean_coverage: float, max_hard_fail: int) -> bool:
    hard_fail = safe_int(row.get("hard_fail"))
    coverage = safe_float(row.get("mean_coverage_ratio"))
    return hard_fail <= max_hard_fail and coverage >= min_mean_coverage


def collapse_duplicate_configs(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    best_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = config_key(row)
        current = best_by_key.get(key)
        if current is None or safe_float(row.get("objective")) < safe_float(current.get("objective")):
            best_by_key[key] = row
    return best_by_key


def normalize(value: float, low: float, high: float, invert: bool = False) -> float:
    if abs(high - low) < 1e-12:
        return 0.0
    norm = (value - low) / (high - low)
    norm = max(0.0, min(1.0, norm))
    return 1.0 - norm if invert else norm


def metric_ranges(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for metric in METRIC_KEYS:
        values = [safe_float(row.get(metric)) for row in rows]
        ranges[metric] = (min(values), max(values)) if values else (0.0, 0.0)
    return ranges


def dataset_score(row: dict[str, str], ranges: dict[str, tuple[float, float]]) -> float:
    obj_low, obj_high = ranges["objective"]
    loss_low, loss_high = ranges["mean_unified_loss"]
    jump_low, jump_high = ranges["mean_jump_count"]
    trim_low, trim_high = ranges["mean_trim_count"]
    off_low, off_high = ranges["mean_off_mask_stitch_length_mm"]
    cov_low, cov_high = ranges["mean_coverage_ratio"]
    precision_low, precision_high = ranges["mean_stitch_precision_ratio"]
    return (
        0.40 * normalize(safe_float(row.get("objective")), obj_low, obj_high)
        + 0.18 * normalize(safe_float(row.get("mean_unified_loss")), loss_low, loss_high)
        + 0.16 * normalize(safe_float(row.get("mean_jump_count")), jump_low, jump_high)
        + 0.06 * normalize(safe_float(row.get("mean_trim_count")), trim_low, trim_high)
        + 0.08 * normalize(safe_float(row.get("mean_off_mask_stitch_length_mm")), off_low, off_high)
        + 0.08 * normalize(safe_float(row.get("mean_coverage_ratio")), cov_low, cov_high, invert=True)
        + 0.04 * normalize(safe_float(row.get("mean_stitch_precision_ratio")), precision_low, precision_high, invert=True)
    )


def summarize_common(
    key: tuple[str, ...],
    rows_by_sweep: dict[str, dict[str, str]],
    ranges_by_sweep: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, Any]:
    dataset_scores = {
        name: dataset_score(row, ranges_by_sweep[name])
        for name, row in rows_by_sweep.items()
    }
    objectives = [safe_float(row.get("objective")) for row in rows_by_sweep.values()]
    losses = [safe_float(row.get("mean_unified_loss")) for row in rows_by_sweep.values()]
    jumps = [safe_float(row.get("mean_jump_count")) for row in rows_by_sweep.values()]
    trims = [safe_float(row.get("mean_trim_count")) for row in rows_by_sweep.values()]
    coverages = [safe_float(row.get("mean_coverage_ratio")) for row in rows_by_sweep.values()]
    precisions = [safe_float(row.get("mean_stitch_precision_ratio")) for row in rows_by_sweep.values()]
    score_values = list(dataset_scores.values())
    mean_score = sum(score_values) / len(score_values)
    worst_score = max(score_values)
    score_range = max(score_values) - min(score_values)
    robust_score = 0.55 * mean_score + 0.30 * worst_score + 0.15 * score_range
    row: dict[str, Any] = {name: value for name, value in zip(CONFIG_KEYS, key)}
    row.update(
        {
            "robust_score": round(robust_score, 8),
            "mean_dataset_score": round(mean_score, 8),
            "worst_dataset_score": round(worst_score, 8),
            "dataset_score_range": round(score_range, 8),
            "mean_objective": round(sum(objectives) / len(objectives), 8),
            "worst_objective": round(max(objectives), 8),
            "mean_unified_loss": round(sum(losses) / len(losses), 8),
            "worst_unified_loss": round(max(losses), 8),
            "mean_jump_count": round(sum(jumps) / len(jumps), 8),
            "worst_jump_count": round(max(jumps), 8),
            "mean_trim_count": round(sum(trims) / len(trims), 8),
            "min_coverage_ratio": round(min(coverages), 8),
            "mean_coverage_ratio": round(sum(coverages) / len(coverages), 8),
            "mean_stitch_precision_ratio": round(sum(precisions) / len(precisions), 8),
        }
    )
    for name, source_row in rows_by_sweep.items():
        row[f"{name}_objective"] = source_row.get("objective", "")
        row[f"{name}_loss"] = source_row.get("mean_unified_loss", "")
        row[f"{name}_jump"] = source_row.get("mean_jump_count", "")
        row[f"{name}_trim"] = source_row.get("mean_trim_count", "")
        row[f"{name}_coverage"] = source_row.get("mean_coverage_ratio", "")
        row[f"{name}_precision"] = source_row.get("mean_stitch_precision_ratio", "")
        row[f"{name}_hard_fail"] = source_row.get("hard_fail", "")
        row[f"{name}_dataset_score"] = round(dataset_scores[name], 8)
    return row


def pick_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    picks = [
        ("best_robust_score", min(rows, key=lambda row: safe_float(row["robust_score"]))),
        ("best_mean_objective", min(rows, key=lambda row: safe_float(row["mean_objective"]))),
        ("best_low_jump", min(rows, key=lambda row: (safe_float(row["mean_jump_count"]), safe_float(row["mean_objective"])))),
        ("best_min_coverage", max(rows, key=lambda row: (safe_float(row["min_coverage_ratio"]), -safe_float(row["mean_objective"])))),
    ]
    recommendations: list[dict[str, Any]] = []
    for label, row in picks:
        tagged = dict(row)
        tagged["recommendation"] = label
        recommendations.append(tagged)
    return recommendations


def main() -> int:
    parser = argparse.ArgumentParser(description="Select robust M2 benefit-gate policies across multiple sweep outputs.")
    parser.add_argument("--sweep", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", default="balanced,low_jump")
    parser.add_argument("--min-mean-coverage", type=float, default=0.90)
    parser.add_argument("--max-hard-fail", type=int, default=0)
    args = parser.parse_args()

    sweeps = parse_name_path(args.sweep)
    profiles = parse_csv_strings(args.profiles)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_sweep_profile: dict[str, dict[str, dict[tuple[str, ...], dict[str, str]]]] = {}
    ranges_by_sweep_profile: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    source_stats: dict[str, Any] = {}
    for sweep_name, sweep_path in sweeps:
        rows = read_csv(sweep_path)
        rows_by_profile: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            profile = str(row.get("profile", ""))
            if profile in profiles:
                rows_by_profile[profile].append(row)
        rows_by_sweep_profile[sweep_name] = {}
        ranges_by_sweep_profile[sweep_name] = {}
        source_stats[sweep_name] = {}
        for profile in profiles:
            profile_rows = rows_by_profile.get(profile, [])
            collapsed = collapse_duplicate_configs(profile_rows)
            feasible_rows = [
                row for row in collapsed.values()
                if row_is_feasible(row, args.min_mean_coverage, args.max_hard_fail)
            ]
            rows_by_sweep_profile[sweep_name][profile] = {
                config_key(row): row for row in feasible_rows
            }
            ranges_by_sweep_profile[sweep_name][profile] = metric_ranges(feasible_rows)
            source_stats[sweep_name][profile] = {
                "raw_rows": len(profile_rows),
                "unique_config_rows": len(collapsed),
                "feasible_config_rows": len(feasible_rows),
            }

    all_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    summary_profiles: list[dict[str, Any]] = []
    for profile in profiles:
        common_keys: set[tuple[str, ...]] | None = None
        for sweep_name, _ in sweeps:
            keys = set(rows_by_sweep_profile[sweep_name][profile])
            common_keys = keys if common_keys is None else common_keys & keys
        common_keys = common_keys or set()
        profile_rows: list[dict[str, Any]] = []
        for key in sorted(common_keys):
            row_map = {
                sweep_name: rows_by_sweep_profile[sweep_name][profile][key]
                for sweep_name, _ in sweeps
            }
            range_map = {
                sweep_name: ranges_by_sweep_profile[sweep_name][profile]
                for sweep_name, _ in sweeps
            }
            profile_rows.append(summarize_common(key, row_map, range_map))
        profile_rows = sorted(
            profile_rows,
            key=lambda row: (
                safe_float(row["robust_score"]),
                safe_float(row["mean_objective"]),
                safe_float(row["mean_jump_count"]),
                -safe_float(row["min_coverage_ratio"]),
            ),
        )
        for rank, row in enumerate(profile_rows, start=1):
            row["robust_rank"] = rank
        picks = pick_recommendations(profile_rows)
        all_rows.extend(profile_rows)
        recommendation_rows.extend(picks)
        summary_profiles.append(
            {
                "profile": profile,
                "common_feasible_configs": len(profile_rows),
                "best": profile_rows[0] if profile_rows else None,
                "recommendations": picks,
            }
        )

    write_csv(all_rows, output_dir / "robust_policy_rows.csv")
    write_csv(recommendation_rows, output_dir / "robust_policy_recommendations.csv")
    write_json(
        {
            "sweeps": [{"name": name, "csv": str(path)} for name, path in sweeps],
            "profiles": profiles,
            "min_mean_coverage": args.min_mean_coverage,
            "max_hard_fail": args.max_hard_fail,
            "score_formula": "0.55*mean_dataset_score + 0.30*worst_dataset_score + 0.15*dataset_score_range",
            "dataset_score_formula": "0.40*objective + 0.18*loss + 0.16*jump + 0.06*trim + 0.08*off_mask + 0.08*coverage_reward + 0.04*precision_reward, all normalized within sweep/profile",
            "source_stats": source_stats,
            "profiles_summary": summary_profiles,
        },
        output_dir / "robust_policy_summary.json",
    )
    print(json.dumps({"profiles": summary_profiles}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

