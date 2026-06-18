from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


METRIC_KEYS = (
    "unified_loss",
    "exec_score",
    "visual_risk",
    "jump_count",
    "trim_count",
    "jump_path_mm",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "visible_connector_length_mm",
    "stitch_count",
    "stitch_path_mm",
    "hard_score",
    "hard_rank",
    "mean_rank",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


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


def feature_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return safe_float(value)
    return None


def build_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        features = row.get("features", {})
        if not isinstance(features, dict):
            continue
        for key, value in features.items():
            if feature_value(value) is not None:
                names.add(str(key))
    return sorted(names)


def matrix_from_rows(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    index = {name: i for i, name in enumerate(feature_names)}
    for row_idx, row in enumerate(rows):
        features = row.get("features", {})
        if not isinstance(features, dict):
            continue
        for key, value in features.items():
            if key in index:
                numeric = feature_value(value)
                if numeric is not None:
                    matrix[row_idx, index[key]] = numeric
    return matrix


def target_from_row(row: dict[str, Any], target: str) -> float:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        return 0.0
    if target == "unified_loss":
        return safe_float(metrics.get("unified_loss"))
    if target == "hard_score":
        return safe_float(metrics.get("hard_score"))
    if target == "hard_rank":
        return safe_float(metrics.get("hard_rank"))
    if target == "mean_rank":
        return safe_float(metrics.get("mean_rank"))
    raise ValueError(f"Unknown target: {target}")


def row_metric(row: dict[str, Any], key: str) -> float:
    metrics = row.get("metrics", {})
    return safe_float(metrics.get(key)) if isinstance(metrics, dict) else 0.0


def fit_ridge(rows: list[dict[str, Any]], feature_names: list[str], target: str, alpha: float) -> dict[str, Any]:
    x = matrix_from_rows(rows, feature_names)
    y = np.array([target_from_row(row, target) for row in rows], dtype=np.float64)
    if len(rows) == 0:
        raise ValueError("Cannot fit with zero rows.")
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    xz = (x - mean_x) / std_x
    design = np.concatenate([np.ones((xz.shape[0], 1), dtype=np.float64), xz], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    try:
        weights = np.linalg.solve(design.T @ design + reg, design.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    method_means: dict[str, float] = {}
    by_method: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, y):
        by_method[str(row.get("method", ""))].append(float(value))
    global_mean = float(y.mean())
    for method, values in by_method.items():
        method_means[method] = float(mean(values)) if values else global_mean
    return {
        "target": target,
        "alpha": float(alpha),
        "feature_names": feature_names,
        "mean": mean_x.tolist(),
        "std": std_x.tolist(),
        "weights": weights.tolist(),
        "method_means": method_means,
        "global_mean": global_mean,
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]], shrinkage: float) -> list[float]:
    feature_names = [str(item) for item in model["feature_names"]]
    x = matrix_from_rows(rows, feature_names)
    mean_x = np.array(model["mean"], dtype=np.float64)
    std_x = np.array(model["std"], dtype=np.float64)
    std_x[std_x < 1e-8] = 1.0
    xz = (x - mean_x) / std_x
    design = np.concatenate([np.ones((xz.shape[0], 1), dtype=np.float64), xz], axis=1)
    weights = np.array(model["weights"], dtype=np.float64)
    raw = design @ weights
    method_means = model.get("method_means", {})
    global_mean = safe_float(model.get("global_mean"))
    out: list[float] = []
    for value, row in zip(raw.tolist(), rows):
        method_mean = safe_float(method_means.get(str(row.get("method", "")), global_mean))
        out.append((1.0 - shrinkage) * float(value) + shrinkage * method_mean)
    return out


def grouped_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("sample_id", ""))].append(row)
    return dict(groups)


def best_row(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    return min(rows, key=lambda row: target_from_row(row, target))


def actual_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: round(mean(row_metric(row, key) for row in rows), 8) for key in METRIC_KEYS}


def evaluate_spec(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    target: str,
    alpha: float,
    shrinkage: float,
    selection_target: str,
) -> dict[str, Any]:
    groups = grouped_by_sample(rows)
    selected: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for sample_id in sorted(groups):
        train_rows = [row for row in rows if str(row.get("sample_id", "")) != sample_id]
        held = groups[sample_id]
        model = fit_ridge(train_rows, feature_names, target, alpha)
        predictions = predict(model, held, shrinkage)
        ranked = sorted(zip(held, predictions), key=lambda item: item[1])
        chosen = ranked[0][0]
        selected.append(chosen)
        oracle = best_row(held, selection_target)
        fold_rows.append(
            {
                "sample_id": sample_id,
                "selected_method": chosen.get("method", ""),
                "oracle_method": oracle.get("method", ""),
                "predicted_score": round(ranked[0][1], 8),
                "actual_selection_target": round(target_from_row(chosen, selection_target), 8),
                "oracle_selection_target": round(target_from_row(oracle, selection_target), 8),
                "selected_unified_loss": row_metric(chosen, "unified_loss"),
                "oracle_unified_loss": row_metric(oracle, "unified_loss"),
                "selected_hard_score": row_metric(chosen, "hard_score"),
                "oracle_hard_score": row_metric(oracle, "hard_score"),
                "selected_visible_connector_count": row_metric(chosen, "visible_connector_count"),
                "selected_off_mask_stitch_length_mm": row_metric(chosen, "off_mask_stitch_length_mm"),
                "selected_jump_count": row_metric(chosen, "jump_count"),
                "selected_trim_count": row_metric(chosen, "trim_count"),
            }
        )
    summary = actual_summary(selected)
    return {
        "target": target,
        "alpha": float(alpha),
        "shrinkage": float(shrinkage),
        "selection_target": selection_target,
        "folds": fold_rows,
        "selected_summary": summary,
        "mean_selection_target": round(mean(target_from_row(row, selection_target) for row in selected), 8),
        "mean_unified_loss": summary["unified_loss"],
        "mean_hard_score": summary["hard_score"],
    }


def fixed_method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    groups = grouped_by_sample(rows)
    selected: list[dict[str, Any]] = []
    for group in groups.values():
        matches = [row for row in group if row.get("method") == method]
        if matches:
            selected.append(matches[0])
    return {"method": method, "selected_summary": actual_summary(selected), "samples": len(selected)}


def train_mean_method_summary(rows: list[dict[str, Any]], selection_target: str) -> dict[str, Any]:
    groups = grouped_by_sample(rows)
    selected: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for sample_id, held in sorted(groups.items()):
        train_rows = [row for row in rows if str(row.get("sample_id", "")) != sample_id]
        by_method: dict[str, list[float]] = defaultdict(list)
        for row in train_rows:
            by_method[str(row.get("method", ""))].append(target_from_row(row, selection_target))
        best_method = min(by_method, key=lambda method: mean(by_method[method]))
        chosen = next((row for row in held if row.get("method") == best_method), held[0])
        selected.append(chosen)
        fold_rows.append(
            {
                "sample_id": sample_id,
                "selected_method": chosen.get("method", ""),
                "actual_selection_target": target_from_row(chosen, selection_target),
            }
        )
    return {
        "method": "train_fold_best_mean_method",
        "selection_target": selection_target,
        "selected_summary": actual_summary(selected),
        "folds": fold_rows,
    }


def oracle_summary(rows: list[dict[str, Any]], selection_target: str) -> dict[str, Any]:
    selected = [best_row(group, selection_target) for group in grouped_by_sample(rows).values()]
    return {"method": f"oracle_{selection_target}", "selected_summary": actual_summary(selected)}


def write_report(
    path: Path,
    best: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    model_path: Path,
    data_path: Path,
) -> None:
    lines = [
        "# M1 Planner Selector Loop",
        "",
        "M1 is a lightweight learned planner selector. It does not generate stitches directly; it selects a planner preset from the current preset bank using pre-export image/geometry summary features plus preset parameters.",
        "",
        "The current dataset is intentionally small, so the main validation is leave-one-out by sample. Treat this as a loop-engineering checkpoint, not a final generalization claim.",
        "",
        "## Best Leave-One-Out Selector",
        "",
        f"- target: `{best['target']}`",
        f"- alpha: `{best['alpha']}`",
        f"- shrinkage-to-method-prior: `{best['shrinkage']}`",
        f"- selection target: `{best['selection_target']}`",
        f"- model artifact: `{model_path.as_posix()}`",
        f"- training rows: `{data_path.as_posix()}`",
        "",
        "| Metric | M1 Selected Mean |",
        "| --- | ---: |",
    ]
    for key, value in best["selected_summary"].items():
        lines.append(f"| {key} | {value:.6f} |")

    lines.extend(
        [
            "",
            "## Fold Decisions",
            "",
            "| Sample | M1 Method | Oracle Method | Selected Unified | Oracle Unified | Selected Hard Score | Oracle Hard Score |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best["folds"]:
        lines.append(
            "| {sample_id} | {selected_method} | {oracle_method} | {selected_unified_loss:.6f} | {oracle_unified_loss:.6f} | {selected_hard_score:.6f} | {oracle_hard_score:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Baseline | Unified | Hard Score | Visible Connectors | Off-Mask mm | Jumps | Trims |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in baselines:
        name = str(item.get("method", ""))
        summary = item.get("selected_summary", {})
        if not isinstance(summary, dict):
            continue
        lines.append(
            "| {name} | {unified:.6f} | {hard:.6f} | {visible:.3f} | {off:.3f} | {jump:.3f} | {trim:.3f} |".format(
                name=name,
                unified=safe_float(summary.get("unified_loss")),
                hard=safe_float(summary.get("hard_score")),
                visible=safe_float(summary.get("visible_connector_count")),
                off=safe_float(summary.get("off_mask_stitch_length_mm")),
                jump=safe_float(summary.get("jump_count")),
                trim=safe_float(summary.get("trim_count")),
            )
        )

    lines.extend(
        [
            "",
            "## Loop Search Leaderboard",
            "",
            "| Rank | Target | Alpha | Shrinkage | Unified | Hard Score | Visible | Off-Mask mm | Jumps |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, item in enumerate(leaderboard[:12], start=1):
        summary = item["selected_summary"]
        lines.append(
            "| {rank} | {target} | {alpha:.4g} | {shrinkage:.2f} | {unified:.6f} | {hard:.6f} | {visible:.3f} | {off:.3f} | {jump:.3f} |".format(
                rank=index,
                target=item["target"],
                alpha=float(item["alpha"]),
                shrinkage=float(item["shrinkage"]),
                unified=safe_float(summary.get("unified_loss")),
                hard=safe_float(summary.get("hard_score")),
                visible=safe_float(summary.get("visible_connector_count")),
                off=safe_float(summary.get("off_mask_stitch_length_mm")),
                jump=safe_float(summary.get("jump_count")),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This reaches the M1 stage because planner choice is now a trained selector with leave-one-out evaluation and a saved model artifact.",
            "- It is not M2: it does not predict individual graph edges or routes.",
            "- The next credible improvement is to expand the holdout set and add pre-export graph statistics from `graph_tsp_trace.json` generated before final DST export.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and validate the M1 planner selector with leave-one-out loops.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", default="results/m1_selector/latest")
    parser.add_argument("--model-output", default="checkpoints/m1_planner_selector.json")
    parser.add_argument("--selection-target", default="hard_score", choices=["hard_score", "unified_loss", "hard_rank", "mean_rank"])
    parser.add_argument("--targets", default="hard_score,unified_loss,hard_rank,mean_rank")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--shrinkages", default="0,0.25,0.5,0.75")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.dataset_jsonl))
    if not rows:
        raise SystemExit("No selector rows found.")
    samples = sorted({str(row.get("sample_id", "")) for row in rows})
    if len(samples) < 2:
        raise SystemExit("Need at least two samples for leave-one-out validation.")

    feature_names = build_feature_names(rows)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    alphas = [safe_float(item) for item in args.alphas.split(",") if item.strip()]
    shrinkages = [safe_float(item) for item in args.shrinkages.split(",") if item.strip()]

    leaderboard: list[dict[str, Any]] = []
    for target in targets:
        for alpha in alphas:
            for shrinkage in shrinkages:
                result = evaluate_spec(rows, feature_names, target, alpha, shrinkage, args.selection_target)
                leaderboard.append(result)

    leaderboard.sort(
        key=lambda item: (
            item["mean_selection_target"],
            item["selected_summary"]["unified_loss"],
            item["selected_summary"]["visible_connector_count"],
            item["selected_summary"]["jump_count"],
        )
    )
    best = leaderboard[0]
    final_model = fit_ridge(rows, feature_names, best["target"], best["alpha"])
    final_model.update(
        {
            "version": "m1_planner_selector_ridge_v1",
            "selection_target": args.selection_target,
            "shrinkage": best["shrinkage"],
            "samples": samples,
            "methods": sorted({str(row.get("method", "")) for row in rows}),
            "feature_count": len(feature_names),
            "training_rows": len(rows),
            "notes": "Small M1 loop checkpoint. Validate on larger holdout before making generalization claims.",
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(final_model, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_methods = sorted({str(row.get("method", "")) for row in rows})
    baselines: list[dict[str, Any]] = [
        oracle_summary(rows, args.selection_target),
        train_mean_method_summary(rows, args.selection_target),
    ]
    baselines.extend(fixed_method_summary(rows, method) for method in baseline_methods)

    write_csv(best["folds"], output_dir / "m1_selector_loo_decisions.csv")
    write_csv(
        [
            {
                "rank": index + 1,
                "target": item["target"],
                "alpha": item["alpha"],
                "shrinkage": item["shrinkage"],
                **{f"mean_{key}": value for key, value in item["selected_summary"].items()},
            }
            for index, item in enumerate(leaderboard)
        ],
        output_dir / "m1_selector_loop_leaderboard.csv",
    )
    result_payload = {
        "best": best,
        "baselines": baselines,
        "leaderboard_top": leaderboard[:20],
        "model_output": str(model_path),
        "dataset_jsonl": args.dataset_jsonl,
        "feature_count": len(feature_names),
        "samples": samples,
    }
    (output_dir / "m1_selector_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        output_dir / "m1_selector_report.md",
        best=best,
        leaderboard=leaderboard,
        baselines=baselines,
        model_path=model_path,
        data_path=Path(args.dataset_jsonl),
    )
    print(
        json.dumps(
            {
                "best": {
                    "target": best["target"],
                    "alpha": best["alpha"],
                    "shrinkage": best["shrinkage"],
                    "selection_target": best["selection_target"],
                    "selected_summary": best["selected_summary"],
                },
                "model_output": str(model_path),
                "report": str(output_dir / "m1_selector_report.md"),
                "decisions": str(output_dir / "m1_selector_loo_decisions.csv"),
                "feature_count": len(feature_names),
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
