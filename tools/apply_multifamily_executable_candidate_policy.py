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

from calibrate_stitch_family_deployment_policy import deploy_family
from train_stitch_family_selector import predict_row


RISK_FAMILIES = {
    "running_line",
    "auto_running",
    "auto_fill",
    "fill_tatami_like",
    "satin_like",
    "outline_border",
}

METRIC_KEYS = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
)

DEFAULT_EXTRA_CANDIDATE_DIRS = {
    "mask_fill_edgewalk_rows16_p40": "results/public_benchmark_v1_ext33_mask_fill_edgewalk_rows16_p40",
    "mask_fill_edgewalk_outline_ext_in3_rows16_p40": "results/public_benchmark_v1_ext33_mask_fill_edgewalk_outline_ext_in3_rows16_p40",
    "mask_fill_edgewalk_styleaware_r008_d45": "results/public_benchmark_v1_ext33_mask_fill_edgewalk_styleaware_r008_d45",
    "base_keep": "results/public_benchmark_v1_ext33_m2_72_line_skeleton_profile",
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def candidate_family_for_model(row: dict[str, Any]) -> str:
    family = str(row.get("candidate_family", ""))
    if family:
        return family
    kind = str(row.get("candidate_kind", ""))
    if kind in ("dt_satin", "satin_rail"):
        return "satin_like"
    if kind == "base_keep":
        return "base_keep"
    return kind


def load_candidate_dir_map(config_path: Path) -> dict[str, Path]:
    mapping = {key: Path(value) for key, value in DEFAULT_EXTRA_CANDIDATE_DIRS.items()}
    if config_path.exists():
        data = read_json(config_path)
        for item in data.get("candidate_pool", []):
            name = str(item.get("name", ""))
            result_dir = str(item.get("result_dir", ""))
            if name and result_dir:
                mapping[name] = Path(result_dir)
    return mapping


def texture_root_index(texture_rows_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not texture_rows_path.exists():
        return {}
    rows = read_csv(texture_rows_path)
    return {(row.get("sample_id", ""), row.get("candidate", "")): row for row in rows}


def resolve_candidate_root(
    row: dict[str, Any],
    candidate_dirs: dict[str, Path],
    texture_rows: dict[tuple[str, str], dict[str, str]],
) -> Path | None:
    key = (str(row.get("sample_id", "")), str(row.get("candidate", "")))
    texture_row = texture_rows.get(key)
    if texture_row and texture_row.get("candidate_root"):
        return Path(texture_row["candidate_root"])
    return candidate_dirs.get(str(row.get("candidate", "")))


def enrich_prediction(row: dict[str, Any], model: dict[str, Any], min_risky_margin: float) -> dict[str, Any]:
    model_row = dict(row)
    model_row["candidate_family"] = candidate_family_for_model(row)
    model_row.setdefault("seed_source", row.get("seed_source", "candidate_selector"))
    model_row.setdefault("target_branch", row.get("target_branch", ""))
    predicted, score, score_map = predict_row(model_row, model, float(model.get("reject_margin", 0.0)))
    out = dict(row)
    out["candidate_family"] = model_row["candidate_family"]
    out["predicted_family"] = predicted
    out["prediction_score"] = score
    for label, value in score_map.items():
        out[f"score_{label}"] = value
    deployed, reason, margin = deploy_family(out, RISK_FAMILIES, min_risky_margin)
    out["deployed_family"] = deployed
    out["deployment_reason"] = reason
    out["selector_margin_vs_reject"] = round(margin, 8)
    return out


def output_exists(root: Path | None, sample_id: str) -> bool:
    return bool(root) and (root / sample_id / "prediction.dst").exists()


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(baseline.get(key)), 8)


def is_nonbaseline_candidate(row: dict[str, Any]) -> bool:
    candidate = str(row.get("candidate", ""))
    family = str(row.get("candidate_family", ""))
    deployed_family = str(row.get("deployed_family", ""))
    return candidate != "base_keep" and family != "base_keep" and deployed_family != "base_keep"


def pass_safety_gate(candidate: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if str(candidate.get("deployed_family", "")) == "reject":
        return False, "family_rejected"
    if str(candidate.get("oracle_quality_level", candidate.get("quality_level", ""))).lower() == "hard_fail":
        return False, "hard_fail"
    if metric_delta(candidate, baseline, "visible_connector_count") > args.max_visible_increase:
        return False, "visible_increase"
    if metric_delta(candidate, baseline, "off_mask_stitch_length_mm") > args.max_off_mask_increase:
        return False, "off_mask_increase"
    if metric_delta(candidate, baseline, "unified_loss") > args.max_loss_increase:
        return False, "loss_increase"
    if metric_delta(candidate, baseline, "jump_count") > args.max_jump_increase:
        return False, "jump_increase"
    if metric_delta(candidate, baseline, "trim_count") > args.max_trim_increase:
        return False, "trim_increase"
    if metric_delta(candidate, baseline, "coverage_ratio") < -args.max_coverage_drop:
        return False, "coverage_drop"
    if metric_delta(candidate, baseline, "stitch_precision_ratio") < -args.max_precision_drop:
        return False, "precision_drop"
    return True, "safe_multifamily_candidate"


def deployment_score(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    loss_gain = -metric_delta(candidate, baseline, "unified_loss")
    jump_gain = -metric_delta(candidate, baseline, "jump_count")
    trim_gain = -metric_delta(candidate, baseline, "trim_count")
    coverage_gain = metric_delta(candidate, baseline, "coverage_ratio")
    precision_gain = metric_delta(candidate, baseline, "stitch_precision_ratio")
    margin = safe_float(candidate.get("selector_margin_vs_reject"))
    return round(
        2.0 * loss_gain
        + 0.04 * jump_gain
        + 0.05 * trim_gain
        + 0.8 * coverage_gain
        + 0.6 * precision_gain
        + 0.05 * max(0.0, margin),
        8,
    )


def copy_outputs(row: dict[str, Any], output_dir: Path) -> None:
    root = Path(str(row["candidate_root"]))
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
        "policy_switches": sum(
            1
            for row in rows
            if str(row.get("selected_from", "")) == "multifamily_candidate" and is_nonbaseline_candidate(row)
        ),
        "hard_fail": sum(
            1 for row in rows if str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower() == "hard_fail"
        ),
        "selected_family_counts": dict(Counter(str(row.get("deployed_family", "")) for row in rows)),
        "selected_candidate_counts": dict(Counter(str(row.get("candidate", "")) for row in rows)),
    }
    for key in METRIC_KEYS:
        summary[f"mean_{key}"] = avg(rows, key)
    return summary


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for value, group_rows in sorted(grouped.items()):
        item = {key: value}
        item.update(summarize(group_rows))
        out.append(item)
    return out


def build_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows = read_csv(args.seed_rows)
    baseline_rows = read_csv(args.baseline_rows)
    baseline_by_sample = {row["sample_id"]: row for row in baseline_rows}
    candidate_dirs = load_candidate_dir_map(args.candidate_pool_config)
    texture_rows = texture_root_index(args.texture_candidate_rows)
    model = read_json(args.selector_model)

    decision_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row.get("sample_id", ""))].append(row)

    for sample_id, baseline in sorted(baseline_by_sample.items()):
        baseline_root = Path(str(baseline.get("candidate_root", "")))
        baseline_item = dict(baseline)
        baseline_item.update(
            {
                "candidate": "m2_81_current_profile",
                "candidate_family": "base_keep",
                "deployed_family": "base_keep",
                "deployment_reason": "baseline_current_profile",
                "selector_margin_vs_reject": "",
                "candidate_root": str(baseline_root),
                "executable_candidate": int(output_exists(baseline_root, sample_id)),
                "selected_from": "baseline_current_profile",
                "policy_pass": 1,
                "policy_reason": "baseline",
                "policy_score": 0.0,
            }
        )

        candidates: list[dict[str, Any]] = []
        for row in grouped.get(sample_id, []):
            root = resolve_candidate_root(row, candidate_dirs, texture_rows)
            enriched = enrich_prediction(row, model, args.min_risky_margin)
            enriched["candidate_root"] = str(root) if root else ""
            enriched["executable_candidate"] = int(output_exists(root, sample_id))
            if not enriched["executable_candidate"]:
                enriched["policy_pass"] = 0
                enriched["policy_reason"] = "missing_prediction_dst"
                enriched["policy_score"] = -999.0
            else:
                passed, reason = pass_safety_gate(enriched, baseline_item, args)
                enriched["policy_pass"] = int(passed)
                enriched["policy_reason"] = reason
                enriched["policy_score"] = deployment_score(enriched, baseline_item) if passed else -999.0
            for key in METRIC_KEYS:
                enriched[f"delta_vs_baseline_{key}"] = metric_delta(enriched, baseline_item, key)
            decision_rows.append(enriched)
            if enriched["policy_pass"]:
                candidates.append(enriched)

        promoted = [
            row
            for row in candidates
            if is_nonbaseline_candidate(row) and safe_float(row.get("policy_score")) >= args.min_policy_score
        ]
        if promoted:
            selected = max(promoted, key=lambda row: safe_float(row.get("policy_score")))
            selected["selected_from"] = "multifamily_candidate"
        else:
            selected = baseline_item
        selected_rows.append(selected)

    return decision_rows, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a conservative multi-family executable candidate policy.")
    parser.add_argument(
        "--seed-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed/stitch_type_teacher_seed_rows.csv"),
    )
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv"),
    )
    parser.add_argument(
        "--selector-model",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_professional_gate_selector/stitch_family_selector_model.json"),
    )
    parser.add_argument(
        "--candidate-pool-config",
        type=Path,
        default=Path("configs/best_current_model_m2_12_maskfill_selector.json"),
    )
    parser.add_argument(
        "--texture-candidate-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset/stitch_type_candidate_rows.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy"),
    )
    parser.add_argument("--model-id", default="m2_82_multifamily_candidate_policy")
    parser.add_argument("--min-risky-margin", type=float, default=0.20)
    parser.add_argument("--min-policy-score", type=float, default=0.05)
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
    decision_rows, selected_rows = build_candidates(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_outputs:
        for row in selected_rows:
            if row.get("candidate_root"):
                copy_outputs(row, args.output_dir)

    write_csv(decision_rows, args.output_dir / "multifamily_decision_rows.csv")
    write_csv(selected_rows, args.output_dir / "multifamily_selected_rows.csv")
    switched = [
        row
        for row in selected_rows
        if str(row.get("selected_from", "")) == "multifamily_candidate" and is_nonbaseline_candidate(row)
    ]
    write_csv(switched, args.output_dir / "multifamily_switched_rows.csv")

    executable = [row for row in decision_rows if int(row.get("executable_candidate", 0)) == 1]
    passed = [row for row in executable if int(row.get("policy_pass", 0)) == 1]
    summary = {
        "model_id": args.model_id,
        "seed_rows": str(args.seed_rows),
        "baseline_rows": str(args.baseline_rows),
        "selector_model": str(args.selector_model),
        "candidate_pool_config": str(args.candidate_pool_config),
        "policy": {
            "min_risky_margin": args.min_risky_margin,
            "min_policy_score": args.min_policy_score,
            "max_visible_increase": args.max_visible_increase,
            "max_off_mask_increase": args.max_off_mask_increase,
            "max_loss_increase": args.max_loss_increase,
            "max_jump_increase": args.max_jump_increase,
            "max_trim_increase": args.max_trim_increase,
            "max_coverage_drop": args.max_coverage_drop,
            "max_precision_drop": args.max_precision_drop,
        },
        "candidate_audit": {
            "decision_rows": len(decision_rows),
            "executable_candidate_rows": len(executable),
            "policy_pass_rows": len(passed),
            "promotable_nonbaseline_rows": sum(
                1
                for row in passed
                if is_nonbaseline_candidate(row) and safe_float(row.get("policy_score")) >= args.min_policy_score
            ),
            "executable_by_family": dict(Counter(str(row.get("candidate_family", "")) for row in executable)),
            "policy_pass_by_family": dict(Counter(str(row.get("deployed_family", "")) for row in passed)),
            "rejection_reasons": dict(Counter(str(row.get("policy_reason", "")) for row in decision_rows)),
        },
        "selected_summary": summarize(selected_rows),
        "by_source": group_summary(selected_rows, "source_name"),
        "switched_sample_ids": [row["sample_id"] for row in switched],
        "interpretation": "M2.82 audits a broader executable multi-family candidate pool and only promotes candidates that pass learned family deployment plus conservative command-level gates.",
    }
    write_json(summary, args.output_dir / "multifamily_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
