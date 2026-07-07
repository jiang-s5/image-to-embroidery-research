from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from audit_dst_preview_texture import audit_sample


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
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(baseline.get(key)), 8)


def infer_candidate_family(name: str, row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "")).lower()
    value = f"{name} {mode}".lower()
    if "dtsatin" in value or "dt_satin" in value:
        return "dt_satin"
    if "satinrail" in value or "satin_rail" in value or "satin" in value:
        return "satin_like"
    if "outline" in value:
        return "outline_border"
    if "styleaware" in value or "style_running" in value:
        return "style_running"
    if "edgewalk" in value:
        return "edgewalk_fill"
    if "nearestrow" in value or "nearest_row" in value:
        return "nearestrow_fill"
    if "mask_fill" in value:
        return "mask_fill"
    return "other"


def candidate_name(root: Path, row: dict[str, Any]) -> str:
    mode = str(row.get("mode", "")).strip()
    if mode:
        return mode
    name = root.name
    name = re.sub(r"^public_benchmark_v1_ext33_", "", name)
    return name


def discover_candidate_roots(args: argparse.Namespace) -> list[Path]:
    if args.candidate_root:
        return sorted({root for root in args.candidate_root if root.exists()})

    include = re.compile(args.include_regex, re.IGNORECASE)
    exclude = re.compile(args.exclude_regex, re.IGNORECASE) if args.exclude_regex else None
    roots: list[Path] = []
    for root in sorted(args.results_dir.iterdir() if args.results_dir.exists() else []):
        if not root.is_dir():
            continue
        if not root.name.startswith(args.root_prefix):
            continue
        if not include.search(root.name):
            continue
        if exclude and exclude.search(root.name):
            continue
        if not (root / "source_aware_hybrid_rows.csv").exists():
            continue
        if not (root / "coverage_rows.csv").exists():
            continue
        sample_count = sum(1 for sample_dir in root.iterdir() if sample_dir.is_dir() and (sample_dir / "prediction.dst").exists())
        if sample_count:
            roots.append(root)
    return roots


def load_row_maps(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    source_rows = {row["sample_id"]: row for row in read_csv(root / "source_aware_hybrid_rows.csv") if row.get("sample_id")}
    coverage_rows = {row["sample_id"]: row for row in read_csv(root / "coverage_rows.csv") if row.get("sample_id")}
    return source_rows, coverage_rows


def build_candidate_row(
    sample_id: str,
    root: Path,
    source_row: dict[str, Any],
    coverage_row: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    sample_dir = root / sample_id
    eval_report = read_json(sample_dir / "eval_executability.json").get("pred", {})
    generator = read_json(sample_dir / "generator_report.json")
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "source_name": baseline.get("source_name", source_row.get("source_name", "")),
        "category": baseline.get("category", source_row.get("category", "")),
        "candidate": candidate_name(root, source_row),
        "candidate_root": str(root),
        "candidate_dir_name": root.name,
        "candidate_family": infer_candidate_family(root.name, source_row),
        "deployed_family": infer_candidate_family(root.name, source_row),
        "mode": source_row.get("mode", ""),
        "quality_level": source_row.get("quality_level", ""),
        "oracle_quality_level": source_row.get("quality_level", ""),
        "round_trip_parse_success": bool((sample_dir / "prediction.dst").exists()),
    }
    for key in (
        "unified_loss",
        "exec_score",
        "visual_risk",
        "jump_count",
        "trim_count",
        "jump_path_mm",
        "off_mask_stitch_length_mm",
        "visible_connector_count",
        "visible_connector_length_mm",
    ):
        row[key] = source_row.get(key, eval_report.get(key, ""))
    for key in ("coverage_ratio", "stitch_precision_ratio"):
        row[key] = coverage_row.get(key, "")
    for key in (
        "fill_rows",
        "outline_points",
        "satin_columns",
        "satin_rail_segments",
        "dt_satin_segments",
        "style_running_points",
    ):
        row[key] = generator.get(key, source_row.get(key, ""))
    return row


def add_audit_metrics(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in PREVIEW_METRICS:
        out[key] = audit.get(key, "")
    return out


def gate_candidate(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if not row.get("round_trip_parse_success"):
        return False, "missing_prediction_dst"
    for key in ("unified_loss", "coverage_ratio", "stitch_precision_ratio"):
        if str(row.get(key, "")) == "":
            return False, f"missing_{key}"
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
    return True, "preview_sweep_candidate_pass"


def preview_sweep_score(row: dict[str, Any]) -> float:
    preview_gain = safe_float(row.get("delta_professional_preview_score"))
    texture_gain = safe_float(row.get("delta_generator_texture_score"))
    precision_gain = safe_float(row.get("delta_stitch_precision_ratio"))
    coverage_gain = safe_float(row.get("delta_coverage_ratio"))
    loss_increase = max(0.0, safe_float(row.get("delta_unified_loss")))
    jump_increase = max(0.0, safe_float(row.get("delta_jump_count")))
    trim_increase = max(0.0, safe_float(row.get("delta_trim_count")))
    family_bonus = 0.0
    if str(row.get("candidate_family")) in {"satin_like", "dt_satin", "outline_border"}:
        family_bonus = 0.0005
    return round(
        6.5 * preview_gain
        + 0.25 * texture_gain
        + 0.35 * precision_gain
        + 0.20 * coverage_gain
        + family_bonus
        - 0.85 * loss_increase
        - 0.02 * jump_increase
        - 0.02 * trim_increase,
        8,
    )


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(rows),
        "preview_sweep_switches": sum(
            1 for row in rows if str(row.get("preview_sweep_selected_from", "")) == "preview_positive_sweep_candidate"
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


def root_inventory_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        rows.append(
            {
                "candidate_root": str(root),
                "candidate_dir_name": root.name,
                "sample_dirs_with_prediction_dst": sum(
                    1 for sample_dir in root.iterdir() if sample_dir.is_dir() and (sample_dir / "prediction.dst").exists()
                ),
                "has_source_aware_rows": int((root / "source_aware_hybrid_rows.csv").exists()),
                "has_coverage_rows": int((root / "coverage_rows.csv").exists()),
            }
        )
    return rows


def build_policy(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = {row["sample_id"]: row for row in read_csv(args.baseline_rows)}
    roots = discover_candidate_roots(args)
    root_maps = {root: load_row_maps(root) for root in roots}
    inventory = root_inventory_rows(roots)
    audited_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for sample_id, baseline in sorted(baseline_rows.items()):
        baseline_root = Path(str(baseline.get("candidate_root", "")))
        baseline_audit = audit_sample(baseline_root / sample_id, baseline)
        audited_baseline = add_audit_metrics(
            {
                **baseline,
                "candidate": "m2_81_current_profile",
                "deployed_family": "base_keep",
                "candidate_family": "base_keep",
                "candidate_root": str(baseline_root),
                "preview_sweep_gate_pass": 0,
                "preview_sweep_gate_reason": "baseline",
                "preview_sweep_score": 0.0,
            },
            baseline_audit,
        )

        passing_candidates: list[dict[str, Any]] = []
        for root in roots:
            source_rows, coverage_rows = root_maps[root]
            source_row = source_rows.get(sample_id)
            coverage_row = coverage_rows.get(sample_id)
            if not source_row or not coverage_row:
                continue
            sample_dir = root / sample_id
            if not (sample_dir / "prediction.dst").exists():
                continue
            candidate = build_candidate_row(sample_id, root, source_row, coverage_row, baseline)
            candidate_audit = audit_sample(sample_dir, candidate)
            audited = add_audit_metrics(candidate, candidate_audit)
            for key in COMMAND_METRICS:
                audited[f"delta_{key}"] = metric_delta(audited, baseline, key)
            for key in PREVIEW_METRICS:
                audited[f"delta_{key}"] = metric_delta(audited, baseline_audit, key)
            passed, reason = gate_candidate(audited, args)
            audited["preview_sweep_gate_pass"] = int(passed)
            audited["preview_sweep_gate_reason"] = reason
            audited["preview_sweep_score"] = preview_sweep_score(audited) if passed else -999.0
            audited_rows.append(audited)
            if passed:
                passing_candidates.append(audited)

        if passing_candidates:
            selected = max(passing_candidates, key=lambda item: safe_float(item.get("preview_sweep_score")))
            selected["preview_sweep_selected_from"] = "preview_positive_sweep_candidate"
        else:
            selected = audited_baseline
            selected["preview_sweep_selected_from"] = "m2_81_baseline"
        selected_rows.append(selected)

    return inventory, audited_rows, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan existing generator result roots and select command-safe candidates with positive DST preview-texture gains."
    )
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--root-prefix", default="public_benchmark_v1_ext33_")
    parser.add_argument(
        "--include-regex",
        default=r"(satin|dtsatin|edgewalk|outline|styleaware|mask_fill|nearestrow)",
    )
    parser.add_argument(
        "--exclude-regex",
        default=r"(selector|m2_81|m2_82|m2_83|m2_84|m2_85|eval_masks|repair_veto|geometry_priors)",
    )
    parser.add_argument("--candidate-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector"),
    )
    parser.add_argument("--model-id", default="m2_86_preview_candidate_sweep_selector")
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
    inventory_rows, audited_rows, selected_rows = build_policy(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_outputs:
        for row in selected_rows:
            copy_outputs(row, args.output_dir)

    switched = [
        row
        for row in selected_rows
        if str(row.get("preview_sweep_selected_from", "")) == "preview_positive_sweep_candidate"
    ]
    write_csv(inventory_rows, args.output_dir / "preview_sweep_root_inventory.csv")
    write_csv(audited_rows, args.output_dir / "preview_sweep_candidate_rows.csv")
    write_csv(selected_rows, args.output_dir / "preview_sweep_selected_rows.csv")
    write_csv(switched, args.output_dir / "preview_sweep_switched_rows.csv")
    summary = {
        "model_id": args.model_id,
        "baseline_rows": str(args.baseline_rows),
        "policy": {
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
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
            "candidate_roots": len(inventory_rows),
            "candidate_sample_dirs": sum(int(row["sample_dirs_with_prediction_dst"]) for row in inventory_rows),
            "audited_candidate_rows": len(audited_rows),
            "gate_pass_rows": sum(1 for row in audited_rows if int(row.get("preview_sweep_gate_pass", 0)) == 1),
            "positive_preview_rows": sum(
                1 for row in audited_rows if safe_float(row.get("delta_professional_preview_score")) > 0.0
            ),
            "gate_reasons": dict(Counter(str(row.get("preview_sweep_gate_reason", "")) for row in audited_rows)),
            "gate_pass_by_family": dict(
                Counter(
                    str(row.get("candidate_family", ""))
                    for row in audited_rows
                    if int(row.get("preview_sweep_gate_pass", 0)) == 1
                )
            ),
            "gate_pass_by_root": dict(
                Counter(
                    str(row.get("candidate_dir_name", ""))
                    for row in audited_rows
                    if int(row.get("preview_sweep_gate_pass", 0)) == 1
                )
            ),
        },
        "selected_summary": summarize(selected_rows),
        "by_source": group_summary(selected_rows, "source_name"),
        "by_family": group_summary(selected_rows, "deployed_family"),
        "switch_sample_ids": [row["sample_id"] for row in switched],
        "interpretation": "M2.86 expands M2.85 from the M2.82 candidate table to all existing ext33 generator roots with command/coverage rows, while preserving strict command-safety and positive preview-texture gates.",
    }
    write_json(summary, args.output_dir / "preview_sweep_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
