from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from calibrate_stitch_family_deployment_policy import deploy_family
from train_stitch_family_selector import predict_row


METRIC_KEYS = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
    "texture_score",
)

TEXTURE_CANDIDATE_FAMILY = {
    "base_keep": "base_keep",
    "dt_satin": "satin_like",
    "satin_rail": "satin_like",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def stitch_family_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate_family = TEXTURE_CANDIDATE_FAMILY.get(row.get("candidate_kind", ""), row.get("candidate_kind", ""))
    return {
        **row,
        "candidate_family": candidate_family,
        "seed_source": "stitch_type_teacher",
        "target_branch": row.get("planner_branch", ""),
        "delta_texture_score": row.get("delta_texture_score", ""),
    }


def policy_enrich(row: dict[str, Any], model: dict[str, Any], min_risky_margin: float) -> dict[str, Any]:
    model_row = stitch_family_row(row)
    predicted, score, score_map = predict_row(model_row, model, float(model.get("reject_margin", 0.0)))
    scored = {
        **row,
        "candidate_family": model_row["candidate_family"],
        "raw_predicted_family": predicted,
        "predicted_family": predicted,
        "prediction_score": score,
    }
    for label, value in score_map.items():
        scored[f"score_{label}"] = value
    deployed, reason, margin = deploy_family(scored, RISK_FAMILIES, min_risky_margin)
    scored["deployed_family"] = deployed
    scored["deployment_reason"] = reason
    scored["selector_margin_vs_reject"] = round(margin, 8)
    scored["eligible_texture_candidate"] = int(
        row.get("candidate_kind") != "base_keep"
        and str(row.get("allowed_by_professional_gate", "")) == "1"
        and deployed == "satin_like"
    )
    return scored


def copy_outputs(row: dict[str, Any], output_dir: Path) -> None:
    sample_id = str(row["sample_id"])
    source_root = Path(str(row["candidate_root"]))
    sample_out = output_dir / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in (
        "prediction.dst",
        "eval_executability.json",
        "generator_report.json",
        "coverage.json",
        "coverage_report.json",
    ):
        source = source_root / sample_id / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(rows),
        "texture_switches": sum(1 for row in rows if row.get("candidate_kind") != "base_keep"),
        "hard_fail": sum(1 for row in rows if str(row.get("quality_level", "")).lower() == "hard_fail"),
    }
    for key in METRIC_KEYS:
        summary[f"mean_{key}"] = avg(rows, key)
    return summary


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    output = []
    for value, group_rows in sorted(grouped.items()):
        item = {key: value}
        item.update(summarize(group_rows))
        output.append(item)
    return output


def select_rows(rows: list[dict[str, Any]], model: dict[str, Any], min_risky_margin: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)

    decision_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for sample_id, sample_rows in sorted(grouped.items()):
        enriched = [policy_enrich(row, model, min_risky_margin) for row in sample_rows]
        decision_rows.extend(enriched)
        base = next((row for row in enriched if row.get("candidate_kind") == "base_keep"), enriched[0])
        eligible = [row for row in enriched if int(row.get("eligible_texture_candidate", 0)) == 1]
        if eligible:
            chosen = max(eligible, key=lambda row: safe_float(row.get("professional_utility")))
            chosen["family_policy_selected_reason"] = "eligible_texture_highest_utility"
        else:
            chosen = dict(base)
            chosen["family_policy_selected_reason"] = "fallback_base_keep"
        chosen["sample_id"] = sample_id
        chosen["texture_profile_switched"] = str(chosen.get("candidate_kind") != "base_keep")
        selected_rows.append(chosen)
    return decision_rows, selected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a learned stitch-family deployment policy to the professional texture candidate pool."
    )
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset/stitch_type_candidate_rows.csv"),
    )
    parser.add_argument(
        "--selector-model",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_professional_gate_selector/stitch_family_selector_model.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile"),
    )
    parser.add_argument("--model-id", default="m2_81_family_policy_texture_profile")
    parser.add_argument("--min-risky-margin", type=float, default=0.20)
    parser.add_argument("--copy-outputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.candidate_rows)
    model = load_json(args.selector_model)
    decision_rows, selected_rows = select_rows(rows, model, args.min_risky_margin)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_outputs:
        for row in selected_rows:
            copy_outputs(row, args.output_dir)

    write_csv(decision_rows, args.output_dir / "family_policy_decision_rows.csv")
    write_csv(selected_rows, args.output_dir / "family_policy_selected_rows.csv")
    switched_rows = [row for row in selected_rows if row.get("candidate_kind") != "base_keep"]
    write_csv(switched_rows, args.output_dir / "family_policy_switched_rows.csv")
    summary = {
        "model_id": args.model_id,
        "candidate_rows": str(args.candidate_rows),
        "selector_model": str(args.selector_model),
        "min_risky_margin": args.min_risky_margin,
        "summary": summarize(selected_rows),
        "by_source": group_summary(selected_rows, "source_name"),
        "by_candidate_kind": group_summary(selected_rows, "candidate_kind"),
        "texture_switch_sample_ids": [row["sample_id"] for row in switched_rows],
        "interpretation": "This is the first actual texture-profile application of the learned stitch-family selector plus deployment gate.",
    }
    write_json(summary, args.output_dir / "family_policy_texture_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
