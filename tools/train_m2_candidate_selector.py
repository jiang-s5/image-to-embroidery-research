from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
            is_edgewalk = 1.0 if "edgewalk" in candidate_lower else 0.0
            is_nearestrow = 1.0 if "nearestrow" in candidate_lower or "nearest_row" in candidate_lower else 0.0
            is_adaptive_inset = 1.0 if "adapt_" in candidate_lower or "_adapt" in candidate_lower else 0.0
            inset_match = re.search(r"inset(\d+)", candidate_lower)
            fixed_inset_px = float(inset_match.group(1)) if inset_match else 0.0
            is_fixed_inset = 1.0 if fixed_inset_px > 0.0 and is_adaptive_inset == 0.0 else 0.0
            is_inset1 = 1.0 if fixed_inset_px == 1.0 and is_adaptive_inset == 0.0 else 0.0
            is_inset2 = 1.0 if fixed_inset_px == 2.0 and is_adaptive_inset == 0.0 else 0.0
            is_raw_nearestrow = 1.0 if is_nearestrow and not is_fixed_inset and not is_adaptive_inset else 0.0
            adapt_match = re.search(r"adapt_t(\d+(?:p\d+)?)_m(\d+(?:p\d+)?)", candidate_lower)
            adaptive_thin_px = float(adapt_match.group(1).replace("p", ".")) if adapt_match else 0.0
            adaptive_mid_px = float(adapt_match.group(2).replace("p", ".")) if adapt_match else 0.0
            row_match = re.search(r"rows(\d+)", candidate_lower)
            path_match = re.search(r"_p(\d+)", candidate_lower)
            row_spacing_tag = float(row_match.group(1)) if row_match else 0.0
            mask_path_limit_tag = float(path_match.group(1)) if path_match else 0.0
            is_style_aware = 1.0 if "styleaware" in candidate_lower or "style_aware" in candidate_lower else 0.0
            is_outline = 1.0 if "outline" in candidate_lower else 0.0
            is_satin = 1.0 if "satin" in candidate_lower else 0.0
            is_satinrail = 1.0 if "satinrail" in candidate_lower else 0.0
            is_dt_satin = 1.0 if "dtsatin" in candidate_lower or "dt_satin" in candidate_lower else 0.0
            is_quantvalid = 1.0 if "quantvalid" in candidate_lower else 0.0
            is_segvalid = 1.0 if "segvalid" in candidate_lower else 0.0
            is_qdirect = 1.0 if "qdirect" in candidate_lower else 0.0
            is_qpath = 1.0 if "qpath" in candidate_lower else 0.0
            is_strict = 1.0 if "strict" in candidate_lower else 0.0
            mask_area_ratio = safe_float(branch["features"]["mask_area_ratio"])
            skeleton_to_mask_ratio = safe_float(branch["features"]["skeleton_to_mask_ratio"])
            branch_line_score = safe_float(branch["line_score"])
            coverage_ratio = safe_float(coverage.get("coverage_ratio"))
            precision_ratio = safe_float(coverage.get("stitch_precision_ratio"))
            feature_payload = {
                "bias": 1.0,
                "candidate_is_skeleton": is_skeleton,
                "candidate_is_auto": is_auto,
                "candidate_is_mask_fill": is_mask_fill,
                "candidate_is_edgewalk": is_edgewalk,
                "candidate_is_nearestrow": is_nearestrow,
                "candidate_is_raw_nearestrow": is_raw_nearestrow,
                "candidate_is_fixed_inset": is_fixed_inset,
                "candidate_is_inset1": is_inset1,
                "candidate_is_inset2": is_inset2,
                "candidate_fixed_inset_px": fixed_inset_px,
                "candidate_is_adaptive_inset": is_adaptive_inset,
                "candidate_adaptive_thin_px": adaptive_thin_px,
                "candidate_adaptive_mid_px": adaptive_mid_px,
                "candidate_row_spacing_tag": row_spacing_tag,
                "candidate_mask_path_limit_tag": mask_path_limit_tag,
                "candidate_is_style_aware": is_style_aware,
                "candidate_is_outline": is_outline,
                "candidate_is_satin": is_satin,
                "candidate_is_satinrail": is_satinrail,
                "candidate_is_dt_satin": is_dt_satin,
                "candidate_is_quantvalid": is_quantvalid,
                "candidate_is_segvalid": is_segvalid,
                "candidate_is_qdirect": is_qdirect,
                "candidate_is_qpath": is_qpath,
                "candidate_is_strict": is_strict,
                "branch_confidence": safe_float(branch["confidence"]),
                "branch_line_score": branch_line_score,
                "mask_area_ratio": mask_area_ratio,
                "skeleton_to_mask_ratio": skeleton_to_mask_ratio,
                "largest_skeleton_component": safe_float(branch["features"]["largest_skeleton_component"]),
                "large_skeleton_components": safe_float(branch["features"]["large_skeleton_components"]),
                "metric_unified_loss": safe_float(metrics.get("unified_loss")),
                "metric_jump_count": safe_float(metrics.get("jump_count")),
                "metric_trim_count": safe_float(metrics.get("trim_count")),
                "metric_off_mask_mm": safe_float(metrics.get("off_mask_stitch_length_mm")),
                "metric_visible_count": safe_float(metrics.get("visible_connector_count")),
                "coverage_ratio": coverage_ratio,
                "precision_ratio": precision_ratio,
                "interaction_skeleton_x_line_score": is_skeleton * branch_line_score,
                "interaction_skeleton_x_coverage": is_skeleton * coverage_ratio,
                "interaction_auto_x_coverage": is_auto * coverage_ratio,
                "interaction_mask_fill_x_coverage": is_mask_fill * coverage_ratio,
                "interaction_mask_fill_x_line_score": is_mask_fill * branch_line_score,
                "interaction_edgewalk_x_coverage": is_edgewalk * coverage_ratio,
                "interaction_nearestrow_x_coverage": is_nearestrow * coverage_ratio,
                "interaction_nearestrow_x_mask_area": is_nearestrow * mask_area_ratio,
                "interaction_fixed_inset_x_mask_area": is_fixed_inset * mask_area_ratio,
                "interaction_inset2_x_mask_area": is_inset2 * mask_area_ratio,
                "interaction_adaptive_inset_x_mask_area": is_adaptive_inset * mask_area_ratio,
                "interaction_adaptive_inset_x_skeleton_ratio": is_adaptive_inset * skeleton_to_mask_ratio,
                "interaction_style_aware_x_coverage": is_style_aware * coverage_ratio,
                "interaction_style_aware_x_line_score": is_style_aware * branch_line_score,
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


def target_value(row: dict[str, Any], target: str) -> float:
    if target == "oracle_choice":
        return safe_float(row.get("is_oracle_choice"))
    return safe_float(row["oracle_score"])


def target_select_direction(target: str) -> str:
    return "max" if target == "oracle_choice" else "min"


def fit_pairwise_ranker(
    rows: list[dict[str, Any]],
    names: list[str],
    alpha: float,
    min_score_gap: float = 0.0,
) -> dict[str, Any]:
    x = matrix(rows, names)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    xz = (x - mean_x) / std_x

    row_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_groups[str(row["sample_id"])].append(index)

    diff_rows: list[np.ndarray] = []
    diff_targets: list[float] = []
    for indices in row_groups.values():
        for left_pos, left_index in enumerate(indices):
            left_score = safe_float(rows[left_index].get("oracle_score"))
            for right_index in indices[left_pos + 1 :]:
                right_score = safe_float(rows[right_index].get("oracle_score"))
                gap = left_score - right_score
                if abs(gap) < min_score_gap:
                    continue
                diff_rows.append(xz[left_index] - xz[right_index])
                diff_targets.append(gap)
                diff_rows.append(xz[right_index] - xz[left_index])
                diff_targets.append(-gap)

    if not diff_rows:
        return fit_ridge(rows, names, alpha, target="oracle_score")

    design = np.stack(diff_rows, axis=0)
    y = np.asarray(diff_targets, dtype=np.float64)
    reg = np.eye(design.shape[1], dtype=np.float64) * alpha
    try:
        weights = np.linalg.solve(design.T @ design + reg, design.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    return {
        "type": "ridge_pairwise_candidate_ranker",
        "target": "pairwise_score_delta",
        "select_direction": "min",
        "alpha": alpha,
        "pairwise_min_score_gap": min_score_gap,
        "feature_names": names,
        "mean": mean_x.tolist(),
        "std": std_x.tolist(),
        "weights": weights.tolist(),
        "pairwise_examples": len(diff_targets),
    }


def fit_ridge(rows: list[dict[str, Any]], names: list[str], alpha: float, target: str = "oracle_score") -> dict[str, Any]:
    x = matrix(rows, names)
    y = np.array([target_value(row, target) for row in rows], dtype=np.float64)
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
        "target": target,
        "select_direction": target_select_direction(target),
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
    weights = np.array(model["weights"], dtype=np.float64)
    if model.get("type") == "ridge_pairwise_candidate_ranker":
        return (xz @ weights).tolist()
    design = np.concatenate([np.ones((xz.shape[0], 1), dtype=np.float64), xz], axis=1)
    return (design @ weights).tolist()


def fit_selector_model(
    rows: list[dict[str, Any]],
    names: list[str],
    alpha: float,
    target: str,
    pairwise_min_score_gap: float = 0.0,
) -> dict[str, Any]:
    if target == "pairwise_score_delta":
        return fit_pairwise_ranker(rows, names, alpha, pairwise_min_score_gap)
    return fit_ridge(rows, names, alpha, target)


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


def is_style_aware_candidate(row: dict[str, Any]) -> bool:
    candidate = str(row.get("candidate", "")).lower()
    return "styleaware" in candidate or "style_aware" in candidate


def is_mask_fill_candidate(row: dict[str, Any]) -> bool:
    return "mask_fill" in str(row.get("candidate", "")).lower()


def candidate_metric_gate(
    row: dict[str, Any],
    max_off_mask_mm: float,
    max_jump_count: float,
    max_trim_count: float,
    min_precision: float,
    min_coverage: float,
) -> bool:
    if safe_float(row.get("off_mask_stitch_length_mm")) > max_off_mask_mm:
        return False
    if safe_float(row.get("jump_count")) > max_jump_count:
        return False
    if safe_float(row.get("trim_count")) > max_trim_count:
        return False
    if safe_float(row.get("stitch_precision_ratio")) < min_precision:
        return False
    if safe_float(row.get("coverage_ratio")) < min_coverage:
        return False
    return True


def style_aware_gate_candidates(
    rows: list[dict[str, Any]],
    enabled: bool,
    max_off_mask_mm: float,
    max_jump_count: float,
    max_trim_count: float,
    min_precision: float,
    min_coverage: float,
) -> list[dict[str, Any]]:
    if not enabled:
        return rows
    gated: list[dict[str, Any]] = []
    for row in rows:
        if not is_style_aware_candidate(row):
            gated.append(row)
            continue
        if candidate_metric_gate(row, max_off_mask_mm, max_jump_count, max_trim_count, min_precision, min_coverage):
            gated.append(row)
    return gated or rows


def mask_fill_gate_candidates(
    rows: list[dict[str, Any]],
    enabled: bool,
    max_off_mask_mm: float,
    max_jump_count: float,
    max_trim_count: float,
    min_precision: float,
    min_coverage: float,
) -> list[dict[str, Any]]:
    if not enabled:
        return rows
    gated: list[dict[str, Any]] = []
    for row in rows:
        if not is_mask_fill_candidate(row):
            gated.append(row)
            continue
        if candidate_metric_gate(row, max_off_mask_mm, max_jump_count, max_trim_count, min_precision, min_coverage):
            gated.append(row)
    return gated or rows


def selectable_candidates(
    rows: list[dict[str, Any]],
    exclude_hard_fail: bool,
    enforce_coverage_floor: bool,
    flat_min_coverage: float,
    line_min_coverage: float,
    coverage_floor_tolerance: float,
    coverage_floor_mode: str = "branch",
    coverage_floor_line_sources: set[str] | None = None,
    style_aware_hard_gate: bool = False,
    style_aware_max_off_mask_mm: float = 0.05,
    style_aware_max_jump_count: float = 15.0,
    style_aware_max_trim_count: float = 3.0,
    style_aware_min_precision: float = 0.75,
    style_aware_min_coverage: float = 0.80,
    mask_fill_hard_gate: bool = False,
    mask_fill_max_off_mask_mm: float = 0.05,
    mask_fill_max_jump_count: float = 20.0,
    mask_fill_max_trim_count: float = 3.0,
    mask_fill_min_precision: float = 0.70,
    mask_fill_min_coverage: float = 0.80,
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
    candidates = style_aware_gate_candidates(
        candidates,
        style_aware_hard_gate,
        style_aware_max_off_mask_mm,
        style_aware_max_jump_count,
        style_aware_max_trim_count,
        style_aware_min_precision,
        style_aware_min_coverage,
    )
    candidates = mask_fill_gate_candidates(
        candidates,
        mask_fill_hard_gate,
        mask_fill_max_off_mask_mm,
        mask_fill_max_jump_count,
        mask_fill_max_trim_count,
        mask_fill_min_precision,
        mask_fill_min_coverage,
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
    target: str = "oracle_score",
    pairwise_min_score_gap: float = 0.0,
    exclude_hard_fail: bool = False,
    enforce_coverage_floor: bool = False,
    flat_min_coverage: float = 0.75,
    line_min_coverage: float = 0.45,
    coverage_floor_tolerance: float = 0.0,
    coverage_floor_mode: str = "branch",
    coverage_floor_line_sources: set[str] | None = None,
    style_aware_hard_gate: bool = False,
    style_aware_max_off_mask_mm: float = 0.05,
    style_aware_max_jump_count: float = 15.0,
    style_aware_max_trim_count: float = 3.0,
    style_aware_min_precision: float = 0.75,
    style_aware_min_coverage: float = 0.80,
    mask_fill_hard_gate: bool = False,
    mask_fill_max_off_mask_mm: float = 0.05,
    mask_fill_max_jump_count: float = 20.0,
    mask_fill_max_trim_count: float = 3.0,
    mask_fill_min_precision: float = 0.70,
    mask_fill_min_coverage: float = 0.80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = groups_by_sample(rows)
    chosen: list[dict[str, Any]] = []
    for sample_id in sorted(groups):
        train = [row for row in rows if row["sample_id"] != sample_id]
        held = groups[sample_id]
        model = fit_selector_model(train, names, alpha, target, pairwise_min_score_gap)
        selectable = selectable_candidates(
            held,
            exclude_hard_fail,
            enforce_coverage_floor,
            flat_min_coverage,
            line_min_coverage,
            coverage_floor_tolerance,
            coverage_floor_mode,
            coverage_floor_line_sources,
            style_aware_hard_gate,
            style_aware_max_off_mask_mm,
            style_aware_max_jump_count,
            style_aware_max_trim_count,
            style_aware_min_precision,
            style_aware_min_coverage,
            mask_fill_hard_gate,
            mask_fill_max_off_mask_mm,
            mask_fill_max_jump_count,
            mask_fill_max_trim_count,
            mask_fill_min_precision,
            mask_fill_min_coverage,
        )
        preds = predict(model, selectable)
        reverse = str(model.get("select_direction", target_select_direction(target))) == "max"
        ranked = sorted(zip(selectable, preds), key=lambda item: item[1], reverse=reverse)
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
            style_aware_hard_gate,
            style_aware_max_off_mask_mm,
            style_aware_max_jump_count,
            style_aware_max_trim_count,
            style_aware_min_precision,
            style_aware_min_coverage,
            mask_fill_hard_gate,
            mask_fill_max_off_mask_mm,
            mask_fill_max_jump_count,
            mask_fill_max_trim_count,
            mask_fill_min_precision,
            mask_fill_min_coverage,
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
    parser.add_argument("--style-aware-hard-gate", action="store_true")
    parser.add_argument("--style-aware-max-off-mask-mm", type=float, default=0.05)
    parser.add_argument("--style-aware-max-jump-count", type=float, default=15.0)
    parser.add_argument("--style-aware-max-trim-count", type=float, default=3.0)
    parser.add_argument("--style-aware-min-precision", type=float, default=0.75)
    parser.add_argument("--style-aware-min-coverage", type=float, default=0.80)
    parser.add_argument("--mask-fill-hard-gate", action="store_true")
    parser.add_argument("--mask-fill-max-off-mask-mm", type=float, default=0.05)
    parser.add_argument("--mask-fill-max-jump-count", type=float, default=20.0)
    parser.add_argument("--mask-fill-max-trim-count", type=float, default=3.0)
    parser.add_argument("--mask-fill-min-precision", type=float, default=0.70)
    parser.add_argument("--mask-fill-min-coverage", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--target", choices=["oracle_score", "oracle_choice", "pairwise_score_delta"], default="oracle_score")
    parser.add_argument("--pairwise-min-score-gap", type=float, default=0.0)
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
        args.target,
        args.pairwise_min_score_gap,
        args.exclude_hard_fail,
        args.enforce_coverage_floor,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_floor_tolerance,
        args.coverage_floor_mode,
        coverage_floor_line_sources,
        args.style_aware_hard_gate,
        args.style_aware_max_off_mask_mm,
        args.style_aware_max_jump_count,
        args.style_aware_max_trim_count,
        args.style_aware_min_precision,
        args.style_aware_min_coverage,
        args.mask_fill_hard_gate,
        args.mask_fill_max_off_mask_mm,
        args.mask_fill_max_jump_count,
        args.mask_fill_max_trim_count,
        args.mask_fill_min_precision,
        args.mask_fill_min_coverage,
    )
    write_csv([{key: value for key, value in row.items() if key != "features"} for row in loo_rows], output_dir / "loo_selected_rows.csv")
    model = fit_selector_model(rows, names, args.alpha, args.target, args.pairwise_min_score_gap)
    (output_dir / "m2_candidate_selector_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "samples": len(groups_by_sample(rows)),
        "candidates_per_sample": len(candidates),
        "candidate_rows": len(rows),
        "alpha": args.alpha,
        "target": args.target,
        "select_direction": target_select_direction(args.target),
        "pairwise_min_score_gap": args.pairwise_min_score_gap,
        "exclude_hard_fail": args.exclude_hard_fail,
        "enforce_coverage_floor": args.enforce_coverage_floor,
        "coverage_floor_tolerance": args.coverage_floor_tolerance,
        "coverage_floor_mode": args.coverage_floor_mode,
        "coverage_floor_line_sources": sorted(coverage_floor_line_sources),
        "style_aware_hard_gate": args.style_aware_hard_gate,
        "style_aware_max_off_mask_mm": args.style_aware_max_off_mask_mm,
        "style_aware_max_jump_count": args.style_aware_max_jump_count,
        "style_aware_max_trim_count": args.style_aware_max_trim_count,
        "style_aware_min_precision": args.style_aware_min_precision,
        "style_aware_min_coverage": args.style_aware_min_coverage,
        "mask_fill_hard_gate": args.mask_fill_hard_gate,
        "mask_fill_max_off_mask_mm": args.mask_fill_max_off_mask_mm,
        "mask_fill_max_jump_count": args.mask_fill_max_jump_count,
        "mask_fill_max_trim_count": args.mask_fill_max_trim_count,
        "mask_fill_min_precision": args.mask_fill_min_precision,
        "mask_fill_min_coverage": args.mask_fill_min_coverage,
        "feature_count": len(names),
        "leave_one_out": loo_summary,
    }
    (output_dir / "m2_candidate_selector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
