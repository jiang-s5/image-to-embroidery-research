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

from auto_planner_branch import compute_branch_features, select_planner_branch
from rerank_planner_candidates import score_candidate


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


def build_candidate_rows(
    dataset_dir: Path,
    candidates: list[tuple[str, Path]],
    flat_min_coverage: float,
    line_min_coverage: float,
    coverage_weight: float,
    precision_weight: float,
    hard_fail_penalty: float = 0.04,
) -> list[dict[str, Any]]:
    manifest = read_csv(dataset_dir / "manifest.csv")
    row_by_candidate: dict[str, dict[str, dict[str, str]]] = {}
    coverage_by_candidate: dict[str, dict[str, dict[str, str]]] = {}
    for name, candidate_dir in candidates:
        row_by_candidate[name] = {row["sample_id"]: row for row in read_csv(candidate_dir / "source_aware_hybrid_rows.csv")}
        coverage_by_candidate[name] = {row["sample_id"]: row for row in read_csv(candidate_dir / "coverage_rows.csv")}

    rows: list[dict[str, Any]] = []
    for sample in manifest:
        sample_id = sample["sample_id"]
        branch = select_planner_branch(
            compute_branch_features(dataset_dir / sample["mask_path"], dataset_dir / sample["skeleton_path"])
        )
        target_branch = branch["branch"]
        per_sample: list[tuple[str, float]] = []
        pending: list[dict[str, Any]] = []
        for candidate_name, _candidate_dir in candidates:
            metrics = row_by_candidate[candidate_name][sample_id]
            coverage = coverage_by_candidate[candidate_name][sample_id]
            oracle_score, terms = score_candidate(
                metrics,
                coverage,
                target_branch,
                flat_min_coverage,
                line_min_coverage,
                coverage_weight,
                precision_weight,
                hard_fail_penalty,
            )
            per_sample.append((candidate_name, oracle_score))
            candidate_lower = candidate_name.lower()
            is_skeleton = 1.0 if "skeleton" in candidate_lower else 0.0
            is_auto = 1.0 if "auto" in candidate_lower else 0.0
            is_mask_fill = 1.0 if "mask_fill" in candidate_lower else 0.0
            is_style_aware = 1.0 if "styleaware" in candidate_lower or "style_aware" in candidate_lower else 0.0
            is_outline = 1.0 if "outline" in candidate_lower else 0.0
            is_satin = 1.0 if "satin" in candidate_lower else 0.0
            is_dt_satin = 1.0 if "dtsatin" in candidate_lower or "dt_satin" in candidate_lower else 0.0
            feature_payload = {
                "bias": 1.0,
                "candidate_is_skeleton": is_skeleton,
                "candidate_is_auto": is_auto,
                "candidate_is_mask_fill": is_mask_fill,
                "candidate_is_style_aware": is_style_aware,
                "candidate_is_outline": is_outline,
                "candidate_is_satin": is_satin,
                "candidate_is_dt_satin": is_dt_satin,
                "branch_confidence": safe_float(branch["confidence"]),
                "branch_line_score": safe_float(branch["line_score"]),
                "mask_area_ratio": safe_float(branch["features"]["mask_area_ratio"]),
                "skeleton_to_mask_ratio": safe_float(branch["features"]["skeleton_to_mask_ratio"]),
                "largest_skeleton_component": safe_float(branch["features"]["largest_skeleton_component"]),
                "large_skeleton_components": safe_float(branch["features"]["large_skeleton_components"]),
                "metric_unified_loss": safe_float(metrics.get("unified_loss")),
                "metric_jump_count": safe_float(metrics.get("jump_count")),
                "metric_trim_count": safe_float(metrics.get("trim_count")),
                "metric_off_mask_mm": safe_float(metrics.get("off_mask_stitch_length_mm")),
                "metric_visible_count": safe_float(metrics.get("visible_connector_count")),
                "coverage_ratio": safe_float(coverage.get("coverage_ratio")),
                "precision_ratio": safe_float(coverage.get("stitch_precision_ratio")),
                "interaction_skeleton_x_line_score": is_skeleton * safe_float(branch["line_score"]),
                "interaction_skeleton_x_coverage": is_skeleton * safe_float(coverage.get("coverage_ratio")),
                "interaction_auto_x_coverage": is_auto * safe_float(coverage.get("coverage_ratio")),
                "interaction_mask_fill_x_coverage": is_mask_fill * safe_float(coverage.get("coverage_ratio")),
                "interaction_mask_fill_x_line_score": is_mask_fill * safe_float(branch["line_score"]),
                "interaction_style_aware_x_coverage": is_style_aware * safe_float(coverage.get("coverage_ratio")),
                "interaction_style_aware_x_line_score": is_style_aware * safe_float(branch["line_score"]),
            }
            pending.append(
                {
                    "sample_id": sample_id,
                    "source_name": sample.get("source_name", ""),
                    "category": sample.get("category", ""),
                    "candidate": candidate_name,
                    "target_branch": target_branch,
                    "oracle_score": round(oracle_score, 8),
                    "oracle_quality_level": metrics.get("quality_level", ""),
                    "unified_loss": metrics.get("unified_loss", ""),
                    "jump_count": metrics.get("jump_count", ""),
                    "trim_count": metrics.get("trim_count", ""),
                    "off_mask_stitch_length_mm": metrics.get("off_mask_stitch_length_mm", ""),
                    "visible_connector_count": metrics.get("visible_connector_count", ""),
                    "coverage_ratio": coverage.get("coverage_ratio", ""),
                    "stitch_precision_ratio": coverage.get("stitch_precision_ratio", ""),
                    "coverage_deficit": terms["coverage_deficit"],
                    "precision_deficit": terms["precision_deficit"],
                    "hard_penalty": terms["hard_penalty"],
                    "features": feature_payload,
                }
            )
        best_candidate = min(per_sample, key=lambda item: item[1])[0]
        for row in pending:
            row["is_oracle_choice"] = 1 if row["candidate"] == best_candidate else 0
            rows.append(row)
    return rows


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        features = row.get("features", {})
        if isinstance(features, dict):
            names.update(str(key) for key in features)
    return sorted(names)


def matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    data = np.zeros((len(rows), len(names)), dtype=np.float64)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    for row_idx, row in enumerate(rows):
        features = row.get("features", {})
        if not isinstance(features, dict):
            continue
        for key, value in features.items():
            if key in name_to_idx:
                data[row_idx, name_to_idx[key]] = safe_float(value)
    return data


def fit_ridge(rows: list[dict[str, Any]], names: list[str], alpha: float) -> dict[str, Any]:
    x = matrix(rows, names)
    y = np.array([safe_float(row["oracle_score"]) for row in rows], dtype=np.float64)
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
    return {
        "type": "ridge_candidate_score_regressor",
        "target": "oracle_score",
        "alpha": alpha,
        "feature_names": names,
        "mean": mean_x.tolist(),
        "std": std_x.tolist(),
        "weights": weights.tolist(),
    }


def predict(model: dict[str, Any], rows: list[dict[str, Any]]) -> list[float]:
    names = [str(item) for item in model["feature_names"]]
    x = matrix(rows, names)
    mean_x = np.array(model["mean"], dtype=np.float64)
    std_x = np.array(model["std"], dtype=np.float64)
    std_x[std_x < 1e-8] = 1.0
    xz = (x - mean_x) / std_x
    design = np.concatenate([np.ones((xz.shape[0], 1), dtype=np.float64), xz], axis=1)
    weights = np.array(model["weights"], dtype=np.float64)
    return (design @ weights).tolist()


def groups_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["sample_id"])].append(row)
    return dict(groups)


def hard_safe_candidates(rows: list[dict[str, Any]], enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return rows
    safe_rows = [row for row in rows if row.get("oracle_quality_level") != "hard_fail"]
    return safe_rows or rows


def coverage_floor_candidates(
    rows: list[dict[str, Any]],
    enabled: bool,
    flat_min_coverage: float,
    line_min_coverage: float,
    tolerance: float = 0.0,
    mode: str = "branch",
    line_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not enabled:
        return rows
    if not rows:
        return rows
    target_branch = str(rows[0].get("target_branch", ""))
    source_name = str(rows[0].get("source_name", ""))
    line_source_names = line_sources or {"QuickDraw", "Rendered text"}
    if mode == "source":
        is_line_like = source_name in line_source_names
    else:
        is_line_like = target_branch == "line_text_skeleton"
    min_coverage = line_min_coverage if is_line_like else flat_min_coverage
    threshold = max(0.0, min_coverage - tolerance)
    covered = [row for row in rows if safe_float(row.get("coverage_ratio")) >= threshold]
    return covered or rows


def selectable_candidates(
    rows: list[dict[str, Any]],
    exclude_hard_fail: bool,
    enforce_coverage_floor: bool,
    flat_min_coverage: float,
    line_min_coverage: float,
    coverage_floor_tolerance: float,
    coverage_floor_mode: str = "branch",
    coverage_floor_line_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    candidates = hard_safe_candidates(rows, exclude_hard_fail)
    candidates = coverage_floor_candidates(
        candidates,
        enforce_coverage_floor,
        flat_min_coverage,
        line_min_coverage,
        coverage_floor_tolerance,
        coverage_floor_mode,
        coverage_floor_line_sources,
    )
    return candidates


def summarize_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return round(mean(safe_float(row.get(key)) for row in rows), 6)

    return {
        "samples": len(rows),
        "oracle_match": sum(int(row.get("learned_matches_oracle", 0)) for row in rows),
        "hard_fail": sum(1 for row in rows if row.get("oracle_quality_level") == "hard_fail"),
        "mean_oracle_score": avg("oracle_score"),
        "mean_unified_loss": avg("unified_loss"),
        "mean_jump_count": avg("jump_count"),
        "mean_trim_count": avg("trim_count"),
        "mean_off_mask_stitch_length_mm": avg("off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg("visible_connector_count"),
        "mean_coverage_ratio": avg("coverage_ratio"),
        "mean_stitch_precision_ratio": avg("stitch_precision_ratio"),
    }


def leave_one_out(
    rows: list[dict[str, Any]],
    names: list[str],
    alpha: float,
    exclude_hard_fail: bool = False,
    enforce_coverage_floor: bool = False,
    flat_min_coverage: float = 0.75,
    line_min_coverage: float = 0.45,
    coverage_floor_tolerance: float = 0.0,
    coverage_floor_mode: str = "branch",
    coverage_floor_line_sources: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = groups_by_sample(rows)
    chosen: list[dict[str, Any]] = []
    for sample_id in sorted(groups):
        train = [row for row in rows if row["sample_id"] != sample_id]
        held = groups[sample_id]
        model = fit_ridge(train, names, alpha)
        selectable = selectable_candidates(
            held,
            exclude_hard_fail,
            enforce_coverage_floor,
            flat_min_coverage,
            line_min_coverage,
            coverage_floor_tolerance,
            coverage_floor_mode,
            coverage_floor_line_sources,
        )
        preds = predict(model, selectable)
        ranked = sorted(zip(selectable, preds), key=lambda item: item[1])
        chosen_row = dict(ranked[0][0])
        oracle_pool = selectable_candidates(
            held,
            exclude_hard_fail,
            enforce_coverage_floor,
            flat_min_coverage,
            line_min_coverage,
            coverage_floor_tolerance,
            coverage_floor_mode,
            coverage_floor_line_sources,
        )
        oracle_row = min(oracle_pool, key=lambda row: safe_float(row["oracle_score"]))
        chosen_row["predicted_score"] = round(ranked[0][1], 8)
        chosen_row["oracle_candidate"] = oracle_row["candidate"]
        chosen_row["learned_matches_oracle"] = 1 if chosen_row["candidate"] == oracle_row["candidate"] else 0
        chosen.append(chosen_row)
    return chosen, summarize_selected(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and validate a learned M2 candidate selector.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument("--precision-weight", type=float, default=0.15)
    parser.add_argument("--hard-fail-penalty", type=float, default=0.04)
    parser.add_argument("--exclude-hard-fail", action="store_true")
    parser.add_argument("--enforce-coverage-floor", action="store_true")
    parser.add_argument("--coverage-floor-tolerance", type=float, default=0.0)
    parser.add_argument("--coverage-floor-mode", choices=["branch", "source"], default="branch")
    parser.add_argument("--coverage-floor-line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [(name, Path(path)) for name, path in args.candidate]
    rows = build_candidate_rows(
        Path(args.dataset_dir),
        candidates,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_weight,
        args.precision_weight,
        args.hard_fail_penalty,
    )
    names = feature_names(rows)
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}
    flat_rows = []
    for row in rows:
        out = {key: value for key, value in row.items() if key != "features"}
        features = row.get("features", {})
        if isinstance(features, dict):
            for key, value in features.items():
                out[f"feature.{key}"] = value
        flat_rows.append(out)
    write_csv(flat_rows, output_dir / "candidate_selector_dataset.csv")

    loo_rows, loo_summary = leave_one_out(
        rows,
        names,
        args.alpha,
        args.exclude_hard_fail,
        args.enforce_coverage_floor,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_floor_tolerance,
        args.coverage_floor_mode,
        coverage_floor_line_sources,
    )
    write_csv([{key: value for key, value in row.items() if key != "features"} for row in loo_rows], output_dir / "loo_selected_rows.csv")
    model = fit_ridge(rows, names, args.alpha)
    (output_dir / "m2_candidate_selector_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "samples": len(groups_by_sample(rows)),
        "candidates_per_sample": len(candidates),
        "candidate_rows": len(rows),
        "alpha": args.alpha,
        "exclude_hard_fail": args.exclude_hard_fail,
        "enforce_coverage_floor": args.enforce_coverage_floor,
        "coverage_floor_tolerance": args.coverage_floor_tolerance,
        "coverage_floor_mode": args.coverage_floor_mode,
        "coverage_floor_line_sources": sorted(coverage_floor_line_sources),
        "feature_count": len(names),
        "leave_one_out": loo_summary,
    }
    (output_dir / "m2_candidate_selector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
