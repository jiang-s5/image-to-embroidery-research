from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

DEFAULT_RISK_FAMILIES = (
    "running_line",
    "auto_running",
    "auto_fill",
    "fill_tatami_like",
    "satin_like",
    "outline_border",
)

TEXTURE_FAMILIES = {"satin_like", "fill_tatami_like", "outline_border"}


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


def prediction_margin(row: dict[str, Any], predicted_family: str) -> float:
    if predicted_family == "reject":
        non_reject_scores = [
            safe_float(row.get(f"score_{label}"))
            for label in LABELS
            if label != "reject"
        ]
        return safe_float(row.get("score_reject")) - max(non_reject_scores or [0.0])
    return safe_float(row.get(f"score_{predicted_family}", row.get("prediction_score"))) - safe_float(
        row.get("score_reject")
    )


def deploy_family(row: dict[str, Any], risk_families: set[str], min_risky_margin: float) -> tuple[str, str, float]:
    predicted = str(row.get("predicted_family", ""))
    margin = prediction_margin(row, predicted)
    if predicted in risk_families and margin < min_risky_margin:
        return "reject", "risk_margin_reject", margin
    return predicted, "keep_selector_prediction", margin


def summarize(rows: list[dict[str, Any]], family_key: str) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "positive_accuracy": 0.0,
            "nonbase_positive_accuracy": 0.0,
            "reject_recall": 0.0,
            "false_texture_promotions": 0,
        }
    exact = [row for row in rows if row[family_key] == row["teacher_family"]]
    positive = [row for row in rows if row["teacher_family"] != "reject"]
    nonbase_positive = [row for row in positive if row["teacher_family"] != "base_keep"]
    reject = [row for row in rows if row["teacher_family"] == "reject"]
    reject_hit = [row for row in reject if row[family_key] == "reject"]
    false_texture = [
        row
        for row in rows
        if row["teacher_family"] in ("reject", "base_keep") and row[family_key] in TEXTURE_FAMILIES
    ]
    return {
        "rows": len(rows),
        "accuracy": round(len(exact) / max(1, len(rows)), 8),
        "positive_accuracy": round(
            sum(1 for row in positive if row[family_key] == row["teacher_family"]) / max(1, len(positive)),
            8,
        ),
        "nonbase_positive_accuracy": round(
            sum(1 for row in nonbase_positive if row[family_key] == row["teacher_family"])
            / max(1, len(nonbase_positive)),
            8,
        ),
        "reject_recall": round(len(reject_hit) / max(1, len(reject)), 8),
        "false_texture_promotions": len(false_texture),
        "teacher_family_counts": dict(Counter(str(row["teacher_family"]) for row in rows)),
        "deployed_family_counts": dict(Counter(str(row[family_key]) for row in rows)),
    }


def confusion_rows(rows: list[dict[str, Any]], family_key: str) -> list[dict[str, Any]]:
    counts = Counter((str(row["teacher_family"]), str(row[family_key])) for row in rows)
    return [
        {"teacher_family": teacher, "deployed_family": deployed, "count": count}
        for (teacher, deployed), count in sorted(counts.items())
    ]


def by_source_summary(rows: list[dict[str, Any]], family_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_name", ""))].append(row)
    return {source: summarize(source_rows, family_key) for source, source_rows in sorted(grouped.items())}


def apply_policy(
    rows: list[dict[str, Any]],
    risk_families: set[str],
    min_risky_margin: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        deployed, reason, margin = deploy_family(row, risk_families, min_risky_margin)
        enriched = dict(row)
        enriched["raw_predicted_family"] = row.get("predicted_family", "")
        enriched["deployed_family"] = deployed
        enriched["deployment_reason"] = reason
        enriched["selector_margin_vs_reject"] = round(margin, 8)
        enriched["deployment_correct"] = 1 if deployed == row.get("teacher_family") else 0
        out.append(enriched)
    return out


def sweep_rows(
    rows: list[dict[str, Any]],
    risk_families: set[str],
    margins: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for margin in margins:
        deployed_rows = apply_policy(rows, risk_families, margin)
        metrics = summarize(deployed_rows, "deployed_family")
        raw_metrics = summarize(deployed_rows, "predicted_family")
        out.append(
            {
                "min_risky_margin": margin,
                "risk_families": "|".join(sorted(risk_families)),
                "accuracy": metrics["accuracy"],
                "positive_accuracy": metrics["positive_accuracy"],
                "nonbase_positive_accuracy": metrics["nonbase_positive_accuracy"],
                "reject_recall": metrics["reject_recall"],
                "false_texture_promotions": metrics["false_texture_promotions"],
                "accuracy_delta_vs_raw": round(metrics["accuracy"] - raw_metrics["accuracy"], 8),
                "reject_recall_delta_vs_raw": round(metrics["reject_recall"] - raw_metrics["reject_recall"], 8),
                "false_texture_delta_vs_raw": metrics["false_texture_promotions"]
                - raw_metrics["false_texture_promotions"],
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a conservative deployment policy on top of M2.79 stitch-family predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "results/public_benchmark_v1_ext33_m2_79_stitch_family_selector_rejectw4/"
            "stitch_family_source_heldout_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/public_benchmark_v1_ext33_m2_80_stitch_family_deployment_policy"),
    )
    parser.add_argument("--model-id", default="m2_80_stitch_family_deployment_policy")
    parser.add_argument("--min-risky-margin", type=float, default=0.20)
    parser.add_argument(
        "--sweep-margins",
        default="0,0.02,0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30,0.40,0.50",
    )
    parser.add_argument("--risk-families", default=",".join(DEFAULT_RISK_FAMILIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.predictions)
    risk_families = {item.strip() for item in args.risk_families.split(",") if item.strip()}
    margins = [safe_float(item.strip()) for item in args.sweep_margins.split(",") if item.strip()]

    deployed_rows = apply_policy(rows, risk_families, args.min_risky_margin)
    raw_summary = summarize(deployed_rows, "predicted_family")
    deployed_summary = summarize(deployed_rows, "deployed_family")
    sweep = sweep_rows(rows, risk_families, margins)

    write_csv(deployed_rows, args.output_dir / "stitch_family_deployment_policy_rows.csv")
    write_csv(confusion_rows(deployed_rows, "deployed_family"), args.output_dir / "stitch_family_deployment_confusion.csv")
    write_csv(sweep, args.output_dir / "stitch_family_deployment_sweep.csv")
    write_json(
        {
            "model_id": args.model_id,
            "input_predictions": str(args.predictions),
            "policy": {
                "min_risky_margin": args.min_risky_margin,
                "risk_families": sorted(risk_families),
                "rule": "If the selector predicts a risky stitch family but its score margin over reject is below the threshold, deploy reject instead.",
            },
            "raw_selector_summary": raw_summary,
            "deployed_policy_summary": deployed_summary,
            "by_source": by_source_summary(deployed_rows, "deployed_family"),
            "sweep": sweep,
            "interpretation": "M2.80 adds a conservative deployment gate after M2.79 so complex stitch-family promotions require a stronger margin over reject.",
        },
        args.output_dir / "stitch_family_deployment_policy_summary.json",
    )

    print(json.dumps({"raw": raw_summary, "deployed": deployed_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
