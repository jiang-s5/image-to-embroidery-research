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


BASE_NUMERIC_FEATURES = (
    "is_base_keep",
    "is_satin_like",
    "branch_confidence",
    "branch_line_score",
    "mask_area_ratio",
    "skeleton_area_ratio",
    "skeleton_to_mask_ratio",
    "skeleton_components",
    "large_skeleton_components",
    "largest_skeleton_component",
    "unified_loss",
    "delta_unified_loss",
    "jump_count",
    "delta_jump_count",
    "trim_count",
    "delta_trim_count",
    "off_mask_stitch_length_mm",
    "delta_off_mask_stitch_length_mm",
    "visible_connector_count",
    "delta_visible_connector_count",
    "coverage_ratio",
    "delta_coverage_ratio",
    "stitch_precision_ratio",
    "delta_stitch_precision_ratio",
    "fill_rows",
    "delta_fill_rows",
    "satin_rail_segments",
    "delta_satin_rail_segments",
    "dt_satin_segments",
    "delta_dt_satin_segments",
    "texture_score",
    "delta_texture_score",
)

GATE_NUMERIC_FEATURES = (
    "allowed_by_professional_gate",
)

CATEGORICAL_FEATURES = (
    "candidate_kind",
    "planner_branch",
    "source_name",
    "phase",
)

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
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


def grouped_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sample_id"])].append(row)
    return dict(groups)


def feature_schema(rows: list[dict[str, Any]], include_gate_features: bool) -> tuple[list[str], dict[str, list[str]]]:
    numeric = list(BASE_NUMERIC_FEATURES)
    if include_gate_features:
        numeric.extend(GATE_NUMERIC_FEATURES)
    categorical: dict[str, list[str]] = {}
    for key in CATEGORICAL_FEATURES:
        values = sorted({str(row.get(key, "")) for row in rows})
        categorical[key] = values
    names = list(numeric)
    for key, values in categorical.items():
        names.extend(f"{key}={value}" for value in values)
    return names, categorical


def row_vector(row: dict[str, Any], names: list[str], categorical: dict[str, list[str]], include_gate_features: bool) -> np.ndarray:
    numeric = list(BASE_NUMERIC_FEATURES)
    if include_gate_features:
        numeric.extend(GATE_NUMERIC_FEATURES)
    values: dict[str, float] = {key: safe_float(row.get(key)) for key in numeric}
    for key, cats in categorical.items():
        active = str(row.get(key, ""))
        for cat in cats:
            values[f"{key}={cat}"] = 1.0 if active == cat else 0.0
    return np.asarray([values.get(name, 0.0) for name in names], dtype=np.float64)


def matrix(rows: list[dict[str, Any]], names: list[str], categorical: dict[str, list[str]], include_gate_features: bool) -> np.ndarray:
    return np.vstack([row_vector(row, names, categorical, include_gate_features) for row in rows])


def fit_pairwise_ranker(
    groups: dict[str, list[dict[str, Any]]],
    names: list[str],
    categorical: dict[str, list[str]],
    include_gate_features: bool,
    alpha: float,
    texture_choice_weight: float,
) -> dict[str, Any]:
    train_rows = [row for items in groups.values() for row in items]
    x = matrix(train_rows, names, categorical, include_gate_features)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0

    diff_rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for items in groups.values():
        teachers = [row for row in items if safe_float(row.get("is_teacher_choice")) > 0.5]
        if len(teachers) != 1:
            continue
        teacher = teachers[0]
        teacher_vec = (row_vector(teacher, names, categorical, include_gate_features) - mean_x) / std_x
        base_weight = texture_choice_weight if safe_float(teacher.get("teacher_is_texture_switch")) > 0.5 else 1.0
        for row in items:
            if row is teacher:
                continue
            row_vec = (row_vector(row, names, categorical, include_gate_features) - mean_x) / std_x
            diff = teacher_vec - row_vec
            diff_rows.append(diff)
            targets.append(1.0)
            weights.append(base_weight)
            diff_rows.append(-diff)
            targets.append(-1.0)
            weights.append(base_weight)

    if not diff_rows:
        coef = np.zeros(len(names), dtype=np.float64)
    else:
        dx = np.vstack(diff_rows)
        y = np.asarray(targets, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        xw = dx * np.sqrt(w[:, None])
        yw = y * np.sqrt(w)
        eye = np.eye(dx.shape[1], dtype=np.float64)
        coef = np.linalg.solve(xw.T @ xw + alpha * eye, xw.T @ yw)

    return {
        "type": "pairwise_stitch_type_ranker",
        "feature_names": names,
        "categorical_values": categorical,
        "include_gate_features": include_gate_features,
        "alpha": alpha,
        "texture_choice_weight": texture_choice_weight,
        "mean": mean_x.tolist(),
        "std": std_x.tolist(),
        "coef": coef.tolist(),
        "pairwise_examples": len(diff_rows),
    }


def score_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    names = [str(item) for item in model["feature_names"]]
    categorical = {str(key): [str(item) for item in value] for key, value in model["categorical_values"].items()}
    vec = row_vector(row, names, categorical, bool(model["include_gate_features"]))
    mean_x = np.asarray(model["mean"], dtype=np.float64)
    std_x = np.asarray(model["std"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    return float(((vec - mean_x) / std_x) @ coef)


def predict_choice(rows: list[dict[str, Any]], model: dict[str, Any], use_professional_gate: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    for row in rows:
        score = score_row(row, model)
        blocked = False
        if use_professional_gate and row.get("candidate") != "base_keep" and safe_float(row.get("allowed_by_professional_gate")) < 0.5:
            blocked = True
            score = -1e9
        item = dict(row)
        item["selector_score"] = round(score, 8)
        item["blocked_by_professional_gate"] = 1 if blocked else 0
        scored.append(item)
    return max(scored, key=lambda item: safe_float(item.get("selector_score"), -1e9)), scored


def summarize_choices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_texture = [row for row in rows if safe_float(row.get("teacher_is_texture_switch")) > 0.5]
    predicted_texture = [row for row in rows if row.get("predicted_candidate") != "base_keep"]
    true_positive_texture = [
        row
        for row in rows
        if safe_float(row.get("teacher_is_texture_switch")) > 0.5 and row.get("predicted_candidate") != "base_keep"
    ]
    exact = [row for row in rows if row.get("predicted_candidate") == row.get("teacher_candidate")]
    return {
        "samples": len(rows),
        "exact_teacher_match": len(exact),
        "exact_teacher_match_rate": round(len(exact) / max(1, len(rows)), 8),
        "teacher_texture_switches": len(teacher_texture),
        "predicted_texture_switches": len(predicted_texture),
        "texture_switch_recall": round(len(true_positive_texture) / max(1, len(teacher_texture)), 8),
        "texture_switch_precision": round(len(true_positive_texture) / max(1, len(predicted_texture)), 8),
        "predicted_kinds": dict(Counter(str(row.get("predicted_kind", "")) for row in rows)),
        "teacher_kinds": dict(Counter(str(row.get("teacher_kind", "")) for row in rows)),
        **{f"predicted_mean_{key}": avg(rows, f"predicted_{key}") for key in METRIC_KEYS},
        **{f"teacher_mean_{key}": avg(rows, f"teacher_{key}") for key in METRIC_KEYS},
    }


def leave_one_out(
    groups: dict[str, list[dict[str, Any]]],
    names: list[str],
    categorical: dict[str, list[str]],
    include_gate_features: bool,
    alpha: float,
    texture_choice_weight: float,
    use_professional_gate: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    choice_rows: list[dict[str, Any]] = []
    candidate_score_rows: list[dict[str, Any]] = []
    for heldout_id, heldout_rows in groups.items():
        train_groups = {key: value for key, value in groups.items() if key != heldout_id}
        model = fit_pairwise_ranker(train_groups, names, categorical, include_gate_features, alpha, texture_choice_weight)
        predicted, scored = predict_choice(heldout_rows, model, use_professional_gate)
        teacher = next(row for row in heldout_rows if safe_float(row.get("is_teacher_choice")) > 0.5)
        choice = {
            "sample_id": heldout_id,
            "source_name": teacher.get("source_name", ""),
            "category": teacher.get("category", ""),
            "teacher_candidate": teacher.get("candidate", ""),
            "teacher_kind": teacher.get("candidate_kind", ""),
            "teacher_is_texture_switch": teacher.get("teacher_is_texture_switch", ""),
            "predicted_candidate": predicted.get("candidate", ""),
            "predicted_kind": predicted.get("candidate_kind", ""),
            "predicted_matches_teacher": 1 if predicted.get("candidate") == teacher.get("candidate") else 0,
            "predicted_blocked_by_gate": predicted.get("blocked_by_professional_gate", ""),
            "predicted_selector_score": predicted.get("selector_score", ""),
        }
        for key in METRIC_KEYS:
            choice[f"teacher_{key}"] = teacher.get(key, "")
            choice[f"predicted_{key}"] = predicted.get(key, "")
            choice[f"delta_predicted_vs_teacher_{key}"] = round(safe_float(predicted.get(key)) - safe_float(teacher.get(key)), 8)
        choice_rows.append(choice)
        for row in scored:
            candidate_score_rows.append(
                {
                    "heldout_sample_id": heldout_id,
                    "sample_id": row.get("sample_id", ""),
                    "candidate": row.get("candidate", ""),
                    "candidate_kind": row.get("candidate_kind", ""),
                    "is_teacher_choice": row.get("is_teacher_choice", ""),
                    "selector_score": row.get("selector_score", ""),
                    "blocked_by_professional_gate": row.get("blocked_by_professional_gate", ""),
                    "allowed_by_professional_gate": row.get("allowed_by_professional_gate", ""),
                }
            )
    return choice_rows, candidate_score_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a constrained learned stitch-type selector.")
    parser.add_argument("--candidate-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="m2_76_learned_stitch_type_selector")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--texture-choice-weight", type=float, default=4.0)
    parser.add_argument("--include-gate-features", action="store_true")
    parser.add_argument("--use-professional-gate-at-inference", action="store_true")
    args = parser.parse_args()

    rows = read_csv(Path(args.candidate_rows))
    groups = grouped_by_sample(rows)
    names, categorical = feature_schema(rows, args.include_gate_features)

    loo_choices, loo_scores = leave_one_out(
        groups,
        names,
        categorical,
        args.include_gate_features,
        args.alpha,
        args.texture_choice_weight,
        args.use_professional_gate_at_inference,
    )
    full_model = fit_pairwise_ranker(groups, names, categorical, args.include_gate_features, args.alpha, args.texture_choice_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_id": args.model_id,
        "candidate_rows": len(rows),
        "samples": len(groups),
        "feature_count": len(names),
        "include_gate_features": args.include_gate_features,
        "use_professional_gate_at_inference": args.use_professional_gate_at_inference,
        "alpha": args.alpha,
        "texture_choice_weight": args.texture_choice_weight,
        "loo": summarize_choices(loo_choices),
        "interpretation": "This is a constrained learned selector prototype. It should not replace M2.74 unless leave-one-out texture recall and exact-match quality are strong enough.",
    }
    model = dict(full_model)
    model["model_id"] = args.model_id
    model["candidate_rows"] = args.candidate_rows
    model["training_samples"] = len(groups)
    model["use_professional_gate_at_inference"] = args.use_professional_gate_at_inference

    write_csv(loo_choices, output_dir / "stitch_type_selector_loo_choices.csv")
    write_csv(loo_scores, output_dir / "stitch_type_selector_loo_candidate_scores.csv")
    write_json(summary, output_dir / "stitch_type_selector_summary.json")
    write_json(model, output_dir / "stitch_type_selector_model.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
