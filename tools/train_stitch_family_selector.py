from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


NUMERIC_FEATURES = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
    "texture_score",
    "delta_texture_score",
)

CATEGORICAL_FEATURES = (
    "candidate_family",
    "seed_source",
    "target_branch",
)

METRIC_LABELS = (
    "base_keep",
    "running_line",
    "auto_running",
    "auto_fill",
    "fill_tatami_like",
    "satin_like",
    "outline_border",
    "reject",
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


def target_label(row: dict[str, Any]) -> str:
    label = str(row.get("teacher_family", ""))
    return "" if label == "candidate_non_teacher" else label


def labeled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if target_label(row)]


def feature_schema(
    rows: list[dict[str, Any]],
    include_source_name: bool,
    include_candidate_name: bool,
    include_gate_features: bool,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    numeric = list(NUMERIC_FEATURES)
    if include_gate_features:
        numeric.extend(["allowed_by_professional_gate", "teacher_is_texture_switch"])

    categorical_keys = list(CATEGORICAL_FEATURES)
    if include_source_name:
        categorical_keys.append("source_name")
    if include_candidate_name:
        categorical_keys.append("candidate")

    categorical: dict[str, list[str]] = {}
    for key in categorical_keys:
        categorical[key] = sorted({str(row.get(key, "")) for row in rows})

    names = list(numeric)
    for key, values in categorical.items():
        names.extend(f"{key}={value}" for value in values)
    return names, categorical, numeric


def row_vector(
    row: dict[str, Any],
    names: list[str],
    categorical: dict[str, list[str]],
    numeric: list[str],
) -> np.ndarray:
    values = {key: safe_float(row.get(key)) for key in numeric}
    for key, cats in categorical.items():
        active = str(row.get(key, ""))
        for cat in cats:
            values[f"{key}={cat}"] = 1.0 if active == cat else 0.0
    return np.asarray([values.get(name, 0.0) for name in names], dtype=np.float64)


def matrix(
    rows: list[dict[str, Any]],
    names: list[str],
    categorical: dict[str, list[str]],
    numeric: list[str],
) -> np.ndarray:
    return np.vstack([row_vector(row, names, categorical, numeric) for row in rows])


def class_weights(rows: list[dict[str, Any]], enabled: bool) -> dict[str, float]:
    if not enabled:
        return {}
    counts = Counter(target_label(row) for row in rows if target_label(row))
    if not counts:
        return {}
    total = sum(counts.values())
    return {label: math.sqrt(total / max(1, count)) for label, count in counts.items()}


def fit_ridge_classifier(
    rows: list[dict[str, Any]],
    labels: list[str],
    names: list[str],
    categorical: dict[str, list[str]],
    numeric: list[str],
    alpha: float,
    balance_classes: bool,
    reject_weight: float,
) -> dict[str, Any]:
    train_rows = labeled_rows(rows)
    if not train_rows:
        raise ValueError("No labeled rows available for training")
    x = matrix(train_rows, names, categorical, numeric)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    xs = (x - mean_x) / std_x

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = np.zeros((len(train_rows), len(labels)), dtype=np.float64)
    row_weights = np.ones(len(train_rows), dtype=np.float64)
    cweights = class_weights(train_rows, balance_classes)
    for row_idx, row in enumerate(train_rows):
        label = target_label(row)
        if label in label_to_idx:
            y[row_idx, label_to_idx[label]] = 1.0
        row_weights[row_idx] = safe_float(row.get("teacher_weight"), 1.0) * cweights.get(label, 1.0)
        if label == "reject":
            row_weights[row_idx] *= reject_weight

    xw = xs * np.sqrt(row_weights[:, None])
    yw = y * np.sqrt(row_weights[:, None])
    eye = np.eye(xs.shape[1], dtype=np.float64)
    coef = np.linalg.solve(xw.T @ xw + alpha * eye, xw.T @ yw)

    return {
        "type": "ridge_stitch_family_classifier",
        "labels": labels,
        "feature_names": names,
        "categorical_values": categorical,
        "numeric_features": numeric,
        "alpha": alpha,
        "balance_classes": balance_classes,
        "reject_weight": reject_weight,
        "mean": mean_x.tolist(),
        "std": std_x.tolist(),
        "coef": coef.tolist(),
        "training_rows": len(train_rows),
        "training_label_counts": dict(Counter(target_label(row) for row in train_rows)),
    }


def predict_row(row: dict[str, Any], model: dict[str, Any], reject_margin: float) -> tuple[str, float, dict[str, float]]:
    labels = [str(item) for item in model["labels"]]
    names = [str(item) for item in model["feature_names"]]
    categorical = {str(key): [str(item) for item in value] for key, value in model["categorical_values"].items()}
    numeric = [str(item) for item in model["numeric_features"]]
    vec = row_vector(row, names, categorical, numeric)
    mean_x = np.asarray(model["mean"], dtype=np.float64)
    std_x = np.asarray(model["std"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    scores = ((vec - mean_x) / std_x) @ coef
    best_idx = int(np.argmax(scores))
    if "reject" in labels:
        reject_idx = labels.index("reject")
        if reject_idx != best_idx and float(scores[reject_idx]) + reject_margin >= float(scores[best_idx]):
            best_idx = reject_idx
    score_map = {label: round(float(scores[idx]), 8) for idx, label in enumerate(labels)}
    return labels[best_idx], round(float(scores[best_idx]), 8), score_map


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "accuracy": 0.0,
            "positive_accuracy": 0.0,
            "nonbase_positive_accuracy": 0.0,
            "reject_recall": 0.0,
        }
    exact = [row for row in rows if row["predicted_family"] == row["teacher_family"]]
    positive = [row for row in rows if row["teacher_family"] != "reject"]
    nonbase_positive = [row for row in positive if row["teacher_family"] != "base_keep"]
    reject = [row for row in rows if row["teacher_family"] == "reject"]
    reject_hit = [row for row in reject if row["predicted_family"] == "reject"]
    false_texture = [
        row
        for row in rows
        if row["teacher_family"] in ("reject", "base_keep")
        and row["predicted_family"] in ("satin_like", "fill_tatami_like", "outline_border")
    ]
    return {
        "rows": len(rows),
        "accuracy": round(len(exact) / max(1, len(rows)), 8),
        "positive_accuracy": round(
            sum(1 for row in positive if row["predicted_family"] == row["teacher_family"]) / max(1, len(positive)),
            8,
        ),
        "nonbase_positive_accuracy": round(
            sum(1 for row in nonbase_positive if row["predicted_family"] == row["teacher_family"])
            / max(1, len(nonbase_positive)),
            8,
        ),
        "reject_recall": round(len(reject_hit) / max(1, len(reject)), 8),
        "false_texture_promotions": len(false_texture),
        "teacher_family_counts": dict(Counter(str(row["teacher_family"]) for row in rows)),
        "predicted_family_counts": dict(Counter(str(row["predicted_family"]) for row in rows)),
    }


def confusion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row["teacher_family"]), str(row["predicted_family"])) for row in rows)
    return [
        {"teacher_family": teacher, "predicted_family": pred, "count": count}
        for (teacher, pred), count in sorted(counts.items())
    ]


def source_held_out_predictions(
    rows: list[dict[str, Any]],
    labels: list[str],
    names: list[str],
    categorical: dict[str, list[str]],
    numeric: list[str],
    alpha: float,
    balance_classes: bool,
    reject_weight: float,
    reject_margin: float,
    holdout_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(holdout_column, ""))].append(row)

    predictions: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for heldout_value, heldout_rows in sorted(grouped.items()):
        train_rows = [row for row in rows if str(row.get(holdout_column, "")) != heldout_value]
        model = fit_ridge_classifier(train_rows, labels, names, categorical, numeric, alpha, balance_classes, reject_weight)
        eval_rows = labeled_rows(heldout_rows)
        for row in eval_rows:
            predicted, score, score_map = predict_row(row, model, reject_margin)
            out = {
                "validation_mode": "source_held_out",
                "heldout_group": heldout_value,
                "holdout_column": holdout_column,
                "sample_id": row.get("sample_id", ""),
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "seed_source": row.get("seed_source", ""),
                "candidate": row.get("candidate", ""),
                "candidate_family": row.get("candidate_family", ""),
                "teacher_family": target_label(row),
                "predicted_family": predicted,
                "correct": 1 if predicted == target_label(row) else 0,
                "prediction_score": score,
                "teacher_weight": row.get("teacher_weight", ""),
            }
            for label in labels:
                out[f"score_{label}"] = score_map.get(label, 0.0)
            predictions.append(out)
        fold_summary = summarize_predictions([row for row in predictions if row["heldout_group"] == heldout_value])
        fold_summary.update(
            {
                "heldout_group": heldout_value,
                "train_rows": len(labeled_rows(train_rows)),
                "eval_rows": len(eval_rows),
                "train_label_counts": json.dumps(model["training_label_counts"], sort_keys=True),
            }
        )
        fold_rows.append(fold_summary)
    return predictions, fold_rows


def by_source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_name", ""))].append(row)
    return {key: summarize_predictions(items) for key, items in sorted(grouped.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a source-held-out multi-family stitch-type selector.")
    parser.add_argument("--seed-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="m2_79_stitch_family_selector")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--holdout-column", default="source_name")
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--reject-weight", type=float, default=1.0)
    parser.add_argument("--reject-margin", type=float, default=0.0)
    parser.add_argument("--include-source-name", action="store_true")
    parser.add_argument("--include-candidate-name", action="store_true")
    parser.add_argument("--include-gate-features", action="store_true")
    args = parser.parse_args()

    rows = read_csv(Path(args.seed_rows))
    labels = [label for label in METRIC_LABELS if any(target_label(row) == label for row in rows)]
    names, categorical, numeric = feature_schema(
        rows,
        include_source_name=args.include_source_name,
        include_candidate_name=args.include_candidate_name,
        include_gate_features=args.include_gate_features,
    )
    predictions, fold_rows = source_held_out_predictions(
        rows,
        labels,
        names,
        categorical,
        numeric,
        args.alpha,
        args.balance_classes,
        args.reject_weight,
        args.reject_margin,
        args.holdout_column,
    )
    full_model = fit_ridge_classifier(rows, labels, names, categorical, numeric, args.alpha, args.balance_classes, args.reject_weight)
    summary = {
        "model_id": args.model_id,
        "seed_rows": args.seed_rows,
        "rows": len(rows),
        "labeled_rows": len(labeled_rows(rows)),
        "labels": labels,
        "feature_count": len(names),
        "alpha": args.alpha,
        "balance_classes": args.balance_classes,
        "reject_weight": args.reject_weight,
        "reject_margin": args.reject_margin,
        "include_source_name": args.include_source_name,
        "include_candidate_name": args.include_candidate_name,
        "include_gate_features": args.include_gate_features,
        "holdout_column": args.holdout_column,
        "source_held_out": summarize_predictions(predictions),
        "by_source": by_source_summary(predictions),
        "interpretation": "This evaluates whether the broader M2.78 teacher seed supports multi-family stitch-type learning under source-held-out validation.",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(predictions, output_dir / "stitch_family_source_heldout_predictions.csv")
    write_csv(fold_rows, output_dir / "stitch_family_source_heldout_folds.csv")
    write_csv(confusion_rows(predictions), output_dir / "stitch_family_confusion.csv")
    model = dict(full_model)
    model["model_id"] = args.model_id
    model["seed_rows"] = args.seed_rows
    model["holdout_column"] = args.holdout_column
    model["include_source_name"] = args.include_source_name
    model["include_candidate_name"] = args.include_candidate_name
    model["include_gate_features"] = args.include_gate_features
    model["reject_margin"] = args.reject_margin
    write_json(model, output_dir / "stitch_family_selector_model.json")
    write_json(summary, output_dir / "stitch_family_selector_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
