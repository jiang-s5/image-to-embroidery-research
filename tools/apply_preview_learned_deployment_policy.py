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

from apply_preview_candidate_sweep_selector import COMMAND_METRICS, PREVIEW_METRICS
from train_stitch_family_selector import predict_row


LABELS = (
    "base_keep",
    "running_line",
    "auto_running",
    "auto_fill",
    "fill_tatami_like",
    "satin_like",
    "outline_border",
    "reject",
)

FAMILY_MAP = {
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

RISK_FAMILIES = {
    "running_line",
    "auto_running",
    "auto_fill",
    "fill_tatami_like",
    "satin_like",
    "outline_border",
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


def map_candidate_family(row: dict[str, Any]) -> str:
    family = str(row.get("candidate_family", "")).strip()
    return FAMILY_MAP.get(family, family)


def selector_input_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    gate_pass = int(safe_float(row.get("preview_sweep_gate_pass")))
    risk = execution_risk_features(row)
    out["raw_candidate_family"] = row.get("candidate_family", "")
    out["candidate_family"] = map_candidate_family(row)
    out["seed_source"] = "m2_86_preview_sweep"
    out["target_branch"] = row.get("mode") or row.get("candidate") or row.get("planner_branch", "")
    out["texture_score"] = row.get("generator_texture_score", row.get("texture_score", 0.0))
    out["delta_texture_score"] = row.get("delta_generator_texture_score", row.get("delta_texture_score", 0.0))
    out["allowed_by_professional_gate"] = gate_pass
    out["preview_sweep_gate_pass"] = gate_pass
    out.update(risk)
    return out


def execution_risk_features(row: dict[str, Any]) -> dict[str, float]:
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
    return {
        "execution_penalty_score": round(score, 8),
        "hard_fail_probe_selected": 0.0,
        "jump_explosion_ratio": round(jump_explosion, 8),
        "trim_explosion_ratio": round(trim_explosion, 8),
        "offmask_explosion_mm": round(offmask_explosion, 8),
        "visible_explosion_count": round(visible_explosion, 8),
    }


def prediction_margin(score_map: dict[str, float], predicted: str) -> float:
    reject = safe_float(score_map.get("reject"))
    if predicted == "reject":
        non_reject = [safe_float(score_map.get(label)) for label in LABELS if label != "reject"]
        return round(reject - max(non_reject or [0.0]), 8)
    return round(safe_float(score_map.get(predicted)) - reject, 8)


def deploy_prediction(predicted: str, margin: float, min_risky_margin: float) -> tuple[str, str]:
    if predicted in RISK_FAMILIES and margin < min_risky_margin:
        return "reject", "risk_margin_reject"
    return predicted, "keep_selector_prediction"


def score_candidate(row: dict[str, Any], margin_weight: float) -> float:
    base_score = safe_float(row.get("preview_sweep_score"), -999.0)
    margin = max(0.0, safe_float(row.get("selector_margin_vs_reject")))
    return round(base_score + margin_weight * margin, 8)


def enrich_candidate(
    row: dict[str, Any],
    model: dict[str, Any],
    reject_margin: float,
    min_risky_margin: float,
    selector_margin_weight: float,
) -> dict[str, Any]:
    model_row = selector_input_row(row)
    predicted, score, score_map = predict_row(model_row, model, reject_margin)
    margin = prediction_margin(score_map, predicted)
    deployed, reason = deploy_prediction(predicted, margin, min_risky_margin)
    out = dict(row)
    out["selector_candidate_family"] = model_row["candidate_family"]
    out["selector_seed_source"] = model_row["seed_source"]
    out["selector_target_branch"] = model_row["target_branch"]
    out["selector_predicted_family"] = predicted
    out["selector_prediction_score"] = score
    out["selector_margin_vs_reject"] = margin
    out["selector_deployed_family"] = deployed
    out["selector_deployment_reason"] = reason
    out["selector_nonreject"] = int(deployed != "reject")
    out["learned_preview_policy_score"] = score_candidate(out, selector_margin_weight)
    for key in (
        "execution_penalty_score",
        "hard_fail_probe_selected",
        "jump_explosion_ratio",
        "trim_explosion_ratio",
        "offmask_explosion_mm",
        "visible_explosion_count",
    ):
        out[key] = model_row.get(key, 0.0)
    for label in LABELS:
        out[f"selector_score_{label}"] = score_map.get(label, 0.0)
    return out


def is_eligible(row: dict[str, Any], require_preview_gate: bool) -> tuple[bool, str]:
    if require_preview_gate and int(safe_float(row.get("preview_sweep_gate_pass"))) != 1:
        return False, "preview_sweep_gate_failed"
    if str(row.get("selector_deployed_family")) == "reject":
        return False, "learned_selector_reject"
    return True, "learned_selector_accept"


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
        "learned_policy_switches": sum(
            1 for row in rows if str(row.get("learned_preview_selected_from")) == "learned_preview_candidate"
        ),
        "hard_fail": sum(
            1 for row in rows if str(row.get("oracle_quality_level", row.get("quality_level", ""))).lower() == "hard_fail"
        ),
        "selected_candidate_counts": dict(Counter(str(row.get("candidate", "")) for row in rows)),
        "selected_family_counts": dict(
            Counter(str(row.get("deployed_family", row.get("candidate_family", ""))) for row in rows)
        ),
        "selector_predicted_family_counts": dict(
            Counter(str(row.get("selector_predicted_family", "")) for row in rows)
        ),
        "selector_deployed_family_counts": dict(
            Counter(str(row.get("selector_deployed_family", "")) for row in rows)
        ),
    }
    for key in COMMAND_METRICS + PREVIEW_METRICS:
        summary[f"mean_{key}"] = avg(rows, key)
    return summary


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    out = []
    for value, items in sorted(grouped.items()):
        item = {key: value}
        item.update(summarize(items))
        out.append(item)
    return out


def metric_deltas(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    keys = [
        "samples",
        "hard_fail",
        "mean_unified_loss",
        "mean_jump_count",
        "mean_trim_count",
        "mean_off_mask_stitch_length_mm",
        "mean_visible_connector_count",
        "mean_coverage_ratio",
        "mean_stitch_precision_ratio",
        "mean_professional_preview_score",
        "mean_generator_texture_score",
    ]
    return {
        f"delta_{key}": round(safe_float(current.get(key)) - safe_float(reference.get(key)), 8)
        for key in keys
    }


def build_policy(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = read_json(args.selector_model)
    candidate_rows = read_csv(args.candidate_rows)
    baseline_rows = {row["sample_id"]: row for row in read_csv(args.baseline_rows) if row.get("sample_id")}
    m2_86_selected = {row["sample_id"]: row for row in read_csv(args.m2_86_selected_rows) if row.get("sample_id")}
    policy_rule = (
        "Require the strict M2.86 preview sweep gate, then use the supplied learned selector as a non-reject veto and light reranker. "
        "The default fallback rows are M2.86 selected rows so preview metrics stay on the same audited scale."
        if args.require_preview_gate
        else "Exploratory no-gate mode: use the supplied learned selector without the M2.86 preview sweep gate. "
        "This is intentionally audited against M2.86 before any promotion."
    )

    enriched_candidates = [
        enrich_candidate(row, model, args.reject_margin, args.min_risky_margin, args.selector_margin_weight)
        for row in candidate_rows
    ]

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_rows: list[dict[str, Any]] = []
    for row in enriched_candidates:
        eligible, reason = is_eligible(row, args.require_preview_gate)
        out = dict(row)
        out["learned_preview_eligible"] = int(eligible)
        out["learned_preview_gate_reason"] = reason
        decision_rows.append(out)
        if eligible:
            by_sample[str(row.get("sample_id"))].append(out)

    selected_rows: list[dict[str, Any]] = []
    for sample_id, baseline in sorted(baseline_rows.items()):
        candidates = by_sample.get(sample_id, [])
        if candidates:
            selected = max(candidates, key=lambda item: safe_float(item.get("learned_preview_policy_score"), -999.0))
            selected = dict(selected)
            selected["learned_preview_selected_from"] = "learned_preview_candidate"
        else:
            selected = dict(baseline)
            selected["learned_preview_selected_from"] = "m2_81_baseline"
            selected["learned_preview_policy_score"] = 0.0
            selected["selector_deployed_family"] = "base_keep"
            selected["selector_predicted_family"] = "base_keep"
            selected["selector_margin_vs_reject"] = ""
        reference = m2_86_selected.get(sample_id, {})
        selected["m2_86_candidate"] = reference.get("candidate", "")
        selected["m2_86_candidate_family"] = reference.get("candidate_family", "")
        selected["m2_86_selected_from"] = reference.get("preview_sweep_selected_from", "")
        selected["differs_from_m2_86"] = int(
            str(selected.get("candidate", "")) != str(reference.get("candidate", ""))
            or str(selected.get("candidate_root", "")) != str(reference.get("candidate_root", ""))
        )
        selected_rows.append(selected)

    summary = {
        "model_id": args.model_id,
        "candidate_rows": str(args.candidate_rows),
        "fallback_rows": str(args.baseline_rows),
        "m2_86_selected_rows": str(args.m2_86_selected_rows),
        "selector_model": str(args.selector_model),
        "policy": {
            "require_preview_gate": args.require_preview_gate,
            "reject_margin": args.reject_margin,
            "min_risky_margin": args.min_risky_margin,
            "selector_margin_weight": args.selector_margin_weight,
            "rule": policy_rule,
        },
        "candidate_audit": {
            "candidate_rows": len(enriched_candidates),
            "preview_gate_pass_rows": sum(
                1 for row in enriched_candidates if int(safe_float(row.get("preview_sweep_gate_pass"))) == 1
            ),
            "selector_nonreject_rows": sum(1 for row in enriched_candidates if int(row.get("selector_nonreject", 0)) == 1),
            "eligible_rows": sum(1 for row in decision_rows if int(row.get("learned_preview_eligible", 0)) == 1),
            "selector_predicted_family_counts": dict(
                Counter(str(row.get("selector_predicted_family", "")) for row in enriched_candidates)
            ),
            "selector_deployed_family_counts": dict(
                Counter(str(row.get("selector_deployed_family", "")) for row in enriched_candidates)
            ),
            "gate_reasons": dict(Counter(str(row.get("learned_preview_gate_reason", "")) for row in decision_rows)),
        },
        "selected_summary": summarize(selected_rows),
        "by_source": group_summary(selected_rows, "source_name"),
        "by_family": group_summary(selected_rows, "deployed_family"),
        "switch_sample_ids": [
            row["sample_id"]
            for row in selected_rows
            if str(row.get("learned_preview_selected_from")) == "learned_preview_candidate"
        ],
        "differs_from_m2_86_sample_ids": [
            row["sample_id"] for row in selected_rows if int(row.get("differs_from_m2_86", 0)) == 1
        ],
        "interpretation": "This run applies the supplied learned selector as a deployment-stage safety/texture accept signal over the M2.86 candidate pool. The strict policy keeps the deterministic M2.86 command and preview gate; no-gate mode is exploratory and must be audited before promotion.",
    }
    m2_86_summary = read_json(args.m2_86_summary).get("selected_summary", {})
    if m2_86_summary:
        summary["comparison_to_m2_86"] = metric_deltas(summary["selected_summary"], m2_86_summary)
    return decision_rows, selected_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a learned preview-aware selector as a deployment-stage policy over M2.86 candidates."
    )
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv"),
    )
    parser.add_argument(
        "--m2-86-selected-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_selected_rows.csv"),
    )
    parser.add_argument(
        "--baseline-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_selected_rows.csv"),
        help="Fallback rows used when no learned preview candidate is selected. Defaults to audited M2.86 selected rows.",
    )
    parser.add_argument(
        "--m2-86-summary",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_summary.json"),
    )
    parser.add_argument(
        "--selector-model",
        type=Path,
        default=Path(
            "results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/"
            "stitch_family_selector_model.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy"),
    )
    parser.add_argument("--model-id", default="m2_88_preview_learned_deployment_policy")
    parser.add_argument("--reject-margin", type=float, default=0.1)
    parser.add_argument("--min-risky-margin", type=float, default=0.0)
    parser.add_argument("--selector-margin-weight", type=float, default=0.001)
    parser.add_argument("--require-preview-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-outputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_rows, selected_rows, summary = build_policy(args)
    if args.copy_outputs:
        for row in selected_rows:
            copy_outputs(row, args.output_dir)
    write_csv(decision_rows, args.output_dir / "preview_learned_decision_rows.csv")
    write_csv(selected_rows, args.output_dir / "preview_learned_selected_rows.csv")
    write_csv(
        [row for row in selected_rows if str(row.get("learned_preview_selected_from")) == "learned_preview_candidate"],
        args.output_dir / "preview_learned_switched_rows.csv",
    )
    write_json(summary, args.output_dir / "preview_learned_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
