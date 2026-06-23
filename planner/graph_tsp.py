from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from planner.geometry_priors import GeometryPriors, connector_geometry_stats


Point = tuple[float, float]
Polyline = list[Point]


@dataclass
class GraphTSPConfig:
    long_jump_threshold_mm: float = 8.0
    long_jump_weight: float = 0.0
    trim_jump_threshold_mm: float = 10.0
    trim_penalty: float = 1.0
    offmask_weight: float = 8.0
    visible_connector_penalty: float = 10.0
    min_inside_fraction: float = 0.88
    two_opt_passes: int = 0
    max_two_opt_nodes: int = 80
    geometry_dt_weight: float = 0.0
    geometry_sobel_weight: float = 0.0
    geometry_canny_weight: float = 0.0
    geometry_dt_min_px: float = 1.0
    geometry_dt_q05_px: float = 1.5
    geometry_sobel_mean_max: float = 0.18
    geometry_canny_frac_max: float = 0.12
    geometry_sample_step_px: float = 1.0
    geometry_hard_filter: bool = False


def config_to_dict(config: GraphTSPConfig | None) -> dict[str, Any] | None:
    return asdict(config) if config is not None else None


def coord_to_mm(point: Point, width: int, height: int, scale_mm: float) -> Point:
    return ((point[0] - width / 2.0) * scale_mm, (point[1] - height / 2.0) * scale_mm)


def mm_to_coord(point: Point, width: int, height: int, scale_mm: float) -> Point:
    return (point[0] / scale_mm + width / 2.0, point[1] / scale_mm + height / 2.0)


def distance_mm(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def polyline_length_mm(coords: Polyline, width: int, height: int, scale_mm: float) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    last = coord_to_mm(coords[0], width, height, scale_mm)
    for point in coords[1:]:
        current = coord_to_mm(point, width, height, scale_mm)
        total += distance_mm(last, current)
        last = current
    return total


def polyline_bbox(coords: Polyline) -> list[float]:
    arr = np.asarray(coords, dtype=np.float32)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def line_inside_fraction(mask: np.ndarray | None, start_xy: Point, end_xy: Point, samples: int = 48) -> float:
    if mask is None:
        return 1.0
    height, width = mask.shape
    xs = np.linspace(start_xy[0], end_xy[0], max(2, samples))
    ys = np.linspace(start_xy[1], end_xy[1], max(2, samples))
    xi = np.clip(np.rint(xs).astype(np.int32), 0, width - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
    return float(mask[yi, xi].mean())


def endpoint_options(coords: Polyline, width: int, height: int, scale_mm: float) -> tuple[Point, Point]:
    return coord_to_mm(coords[0], width, height, scale_mm), coord_to_mm(coords[-1], width, height, scale_mm)


def transition_cost(
    current_mm: Point,
    coords: Polyline,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
) -> tuple[float, bool, dict[str, float]]:
    start_mm, end_mm = endpoint_options(coords, width, height, scale_mm)
    candidates = [(start_mm, False, coords[0]), (end_mm, True, coords[-1])]
    best_cost = float("inf")
    best_reverse = False
    best_details: dict[str, float] = {}
    current_xy = mm_to_coord(current_mm, width, height, scale_mm)
    for target_mm, reverse, target_xy in candidates:
        distance = distance_mm(current_mm, target_mm)
        long_jump = max(0.0, distance - config.long_jump_threshold_mm)
        trim_risk = 1.0 if distance >= config.trim_jump_threshold_mm else 0.0
        inside_fraction = line_inside_fraction(mask, current_xy, target_xy)
        offmask_fraction = 1.0 - inside_fraction
        visible_risk = 1.0 if inside_fraction < config.min_inside_fraction else 0.0
        gp_stats = connector_geometry_stats(
            geometry_priors,
            current_xy,
            target_xy,
            sample_step_px=config.geometry_sample_step_px,
        )
        gp_dt_min = safe_float(gp_stats.get("gp_dt_min_px"), config.geometry_dt_min_px)
        gp_dt_q05 = safe_float(gp_stats.get("gp_dt_q05_px"), config.geometry_dt_q05_px)
        gp_sobel = safe_float(gp_stats.get("gp_sobel_cross_mean"))
        gp_canny = safe_float(gp_stats.get("gp_canny_cross_frac"))
        gp_dt_penalty = 0.0
        if gp_stats:
            gp_dt_penalty = max(0.0, (config.geometry_dt_min_px - gp_dt_min) / max(1e-6, config.geometry_dt_min_px))
            gp_dt_penalty += max(0.0, (config.geometry_dt_q05_px - gp_dt_q05) / max(1e-6, config.geometry_dt_q05_px))
        gp_sobel_penalty = max(0.0, (gp_sobel - config.geometry_sobel_mean_max) / max(1e-6, config.geometry_sobel_mean_max))
        gp_canny_penalty = max(0.0, (gp_canny - config.geometry_canny_frac_max) / max(1e-6, config.geometry_canny_frac_max))
        gp_visible_risk = 0.0
        if gp_stats and not weighted_geometry_stats_are_safe(gp_stats, config):
            gp_visible_risk = 1.0
        cost = (
            distance
            + config.long_jump_weight * long_jump
            + config.trim_penalty * trim_risk
            + config.offmask_weight * offmask_fraction * max(distance, config.long_jump_threshold_mm)
            + config.visible_connector_penalty * visible_risk
            + config.geometry_dt_weight * gp_dt_penalty * max(distance, config.long_jump_threshold_mm)
            + config.geometry_sobel_weight * gp_sobel_penalty * max(distance, config.long_jump_threshold_mm)
            + config.geometry_canny_weight * gp_canny_penalty * max(distance, config.long_jump_threshold_mm)
        )
        if cost < best_cost:
            best_cost = cost
            best_reverse = reverse
            best_details = {
                "distance_mm": distance,
                "inside_fraction": inside_fraction,
                "offmask_fraction": offmask_fraction,
                "visible_risk": visible_risk,
                "geometry_visible_risk": gp_visible_risk,
                "combined_visible_risk": max(visible_risk, gp_visible_risk),
                "trim_risk": trim_risk,
                "cost": cost,
                "geometry_dt_penalty": gp_dt_penalty,
                "geometry_sobel_penalty": gp_sobel_penalty,
                "geometry_canny_penalty": gp_canny_penalty,
                **gp_stats,
            }
    return best_cost, best_reverse, best_details


def sequence_cost(
    ordered: list[Polyline],
    start_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
) -> float:
    total = 0.0
    current = start_mm
    for coords in ordered:
        cost, reverse, _details = transition_cost(current, coords, width, height, scale_mm, mask, config, geometry_priors)
        if reverse:
            coords = list(reversed(coords))
        total += cost
        current = coord_to_mm(coords[-1], width, height, scale_mm)
    return total


def two_opt(
    ordered: list[Polyline],
    start_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
) -> list[Polyline]:
    if len(ordered) < 4 or config.two_opt_passes <= 0 or len(ordered) > config.max_two_opt_nodes:
        return ordered
    best = [list(coords) for coords in ordered]
    best_cost = sequence_cost(best, start_mm, width, height, scale_mm, mask, config, geometry_priors)
    for _ in range(config.two_opt_passes):
        improved = False
        for i in range(0, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + [list(reversed(coords)) for coords in reversed(best[i : j + 1])] + best[j + 1 :]
                cost = sequence_cost(candidate, start_mm, width, height, scale_mm, mask, config, geometry_priors)
                if cost + 1e-6 < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
        if not improved:
            break
    return best


def sequence_cost_items(
    ordered: list[dict[str, Any]],
    start_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
) -> float:
    return sequence_cost([item["coords"] for item in ordered], start_mm, width, height, scale_mm, mask, config, geometry_priors)


def two_opt_items(
    ordered: list[dict[str, Any]],
    start_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
) -> list[dict[str, Any]]:
    if len(ordered) < 4 or config.two_opt_passes <= 0 or len(ordered) > config.max_two_opt_nodes:
        return ordered
    best = [{**item, "coords": list(item["coords"])} for item in ordered]
    best_cost = sequence_cost_items(best, start_mm, width, height, scale_mm, mask, config, geometry_priors)
    for _ in range(config.two_opt_passes):
        improved = False
        for i in range(0, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                reversed_slice = [{**item, "coords": list(reversed(item["coords"]))} for item in reversed(best[i : j + 1])]
                candidate = best[:i] + reversed_slice + best[j + 1 :]
                cost = sequence_cost_items(candidate, start_mm, width, height, scale_mm, mask, config, geometry_priors)
                if cost + 1e-6 < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
        if not improved:
            break
    return best


def build_node_record(node_id: int, coords: Polyline, width: int, height: int, scale_mm: float) -> dict[str, Any]:
    start_mm, end_mm = endpoint_options(coords, width, height, scale_mm)
    return {
        "node_id": int(node_id),
        "point_count": int(len(coords)),
        "length_mm": round(polyline_length_mm(coords, width, height, scale_mm), 4),
        "start_mm": [round(float(start_mm[0]), 4), round(float(start_mm[1]), 4)],
        "end_mm": [round(float(end_mm[0]), 4), round(float(end_mm[1]), 4)],
        "bbox_px": [round(value, 3) for value in polyline_bbox(coords)],
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
    if np.isnan(parsed) or np.isinf(parsed):
        return default
    return parsed


def weighted_geometry_stats_are_safe(stats: dict[str, float], config: GraphTSPConfig) -> bool:
    if not stats:
        return True
    if config.geometry_dt_weight > 0.0:
        if safe_float(stats.get("gp_dt_min_px")) < config.geometry_dt_min_px:
            return False
        if safe_float(stats.get("gp_dt_q05_px")) < config.geometry_dt_q05_px:
            return False
    if config.geometry_sobel_weight > 0.0:
        if safe_float(stats.get("gp_sobel_cross_mean")) > config.geometry_sobel_mean_max:
            return False
    if config.geometry_canny_weight > 0.0:
        if safe_float(stats.get("gp_canny_cross_frac")) > config.geometry_canny_frac_max:
            return False
    return True


def load_m2_edge_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    normalization = payload.get("normalization", {}) if isinstance(payload.get("normalization", {}), dict) else {}
    return {
        "path": str(path),
        "feature_names": [str(item) for item in payload.get("feature_names", [])],
        "mean": np.asarray(normalization.get("mean", []), dtype=np.float64),
        "std": np.asarray(normalization.get("std", []), dtype=np.float64),
        "weights": np.asarray(payload.get("weights", []), dtype=np.float64),
        "clip": float(payload.get("feature_clip_after_normalization", 5.0)),
        "version": str(payload.get("version", "")),
    }


def node_exit_after_visit(node: dict[str, Any], reverse: bool) -> Point:
    point = node.get("start_mm" if reverse else "end_mm", [0.0, 0.0])
    if not isinstance(point, list) or len(point) < 2:
        return (0.0, 0.0)
    return (safe_float(point[0]), safe_float(point[1]))


def m2_node_bbox_features(node: dict[str, Any]) -> dict[str, float]:
    bbox = node.get("bbox_px", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    width = max(0.0, safe_float(bbox[2]) - safe_float(bbox[0]))
    height = max(0.0, safe_float(bbox[3]) - safe_float(bbox[1]))
    return {
        "candidate_bbox_width_px": width,
        "candidate_bbox_height_px": height,
        "candidate_bbox_area_px": width * height,
        "candidate_bbox_aspect": width / max(1e-6, height),
    }


def m2_edge_features(
    task_context: dict[str, Any] | None,
    node: dict[str, Any],
    previous_node: dict[str, Any],
    current_mm: Point,
    step_index: int,
    remaining_count: int,
    details: dict[str, float],
) -> dict[str, float]:
    context = task_context or {}
    start_mm = node.get("start_mm", [0.0, 0.0])
    end_mm = node.get("end_mm", [0.0, 0.0])
    if not isinstance(start_mm, list) or len(start_mm) < 2:
        start_mm = [0.0, 0.0]
    if not isinstance(end_mm, list) or len(end_mm) < 2:
        end_mm = [0.0, 0.0]
    start = (safe_float(start_mm[0]), safe_float(start_mm[1]))
    end = (safe_float(end_mm[0]), safe_float(end_mm[1]))
    dist_start = distance_mm(current_mm, start)
    dist_end = distance_mm(current_mm, end)
    target = end if dist_end < dist_start else start
    dx = target[0] - current_mm[0]
    dy = target[1] - current_mm[1]
    prev_length = safe_float(previous_node.get("length_mm"))
    prev_points = safe_float(previous_node.get("point_count"))
    candidate_length = safe_float(node.get("length_mm"))
    candidate_points = safe_float(node.get("point_count"))
    task_type = int(safe_float(context.get("type_id")))
    edge_distance = safe_float(details.get("distance_mm"))
    inside_fraction = safe_float(details.get("inside_fraction"), 1.0)
    offmask_fraction = safe_float(details.get("offmask_fraction"))
    visible_risk = safe_float(details.get("visible_risk"))
    trim_risk = safe_float(details.get("trim_risk"))
    features = {
        "distance_to_start_mm": dist_start,
        "distance_to_end_mm": dist_end,
        "min_endpoint_distance_mm": min(dist_start, dist_end),
        "endpoint_distance_gap_mm": abs(dist_start - dist_end),
        "choose_reverse_by_distance": 1.0 if dist_end < dist_start else 0.0,
        "dx_best_endpoint_mm": dx,
        "dy_best_endpoint_mm": dy,
        "abs_dx_best_endpoint_mm": abs(dx),
        "abs_dy_best_endpoint_mm": abs(dy),
        "candidate_length_mm": candidate_length,
        "candidate_point_count": candidate_points,
        "previous_length_mm": prev_length,
        "previous_point_count": prev_points,
        "length_ratio_to_previous": candidate_length / max(1e-6, prev_length),
        "point_ratio_to_previous": candidate_points / max(1e-6, prev_points),
        "remaining_count": float(remaining_count),
        "step_index": float(step_index),
        "task_area_px": safe_float(context.get("area_px")),
        "task_order_score": safe_float(context.get("order_score")),
        "task_type_running": 1.0 if task_type == 1 else 0.0,
        "task_type_satin": 1.0 if task_type == 2 else 0.0,
        "task_type_fill": 1.0 if task_type == 3 else 0.0,
        "selected_edge_distance_mm": edge_distance,
        "selected_edge_cost": safe_float(details.get("cost")),
        "selected_edge_inside_fraction": inside_fraction,
        "selected_edge_offmask_fraction": offmask_fraction,
        "selected_edge_visible_risk": visible_risk,
        "selected_edge_trim_risk": trim_risk,
        "selected_edge_long_jump_margin_mm": max(0.0, edge_distance - safe_float(context.get("long_jump_threshold_mm"), 8.0)),
        "selected_edge_safe_connect_candidate": 1.0 if inside_fraction >= safe_float(context.get("safe_min_inside_fraction"), 0.92) and visible_risk <= 0.0 else 0.0,
        "selected_edge_geometry_visible_risk": safe_float(details.get("geometry_visible_risk")),
        "selected_edge_combined_visible_risk": safe_float(details.get("combined_visible_risk"), visible_risk),
        "selected_edge_gp_clean_inside_fraction": safe_float(details.get("gp_clean_inside_fraction"), 1.0),
        "selected_edge_gp_dt_min_px": safe_float(details.get("gp_dt_min_px")),
        "selected_edge_gp_dt_q05_px": safe_float(details.get("gp_dt_q05_px")),
        "selected_edge_gp_dt_mean_px": safe_float(details.get("gp_dt_mean_px")),
        "selected_edge_gp_sobel_cross_mean": safe_float(details.get("gp_sobel_cross_mean")),
        "selected_edge_gp_canny_cross_frac": safe_float(details.get("gp_canny_cross_frac")),
        "selected_edge_gp_centerline_hit_frac": safe_float(details.get("gp_centerline_hit_frac")),
    }
    features.update(m2_node_bbox_features(node))
    return features


def m2_edge_utility(policy: dict[str, Any], features: dict[str, float]) -> float:
    names = policy.get("feature_names", [])
    weights = np.asarray(policy.get("weights", []), dtype=np.float64)
    mean = np.asarray(policy.get("mean", []), dtype=np.float64)
    std = np.asarray(policy.get("std", []), dtype=np.float64)
    if len(names) == 0 or weights.shape[0] != len(names):
        return 0.0
    values = np.asarray([safe_float(features.get(str(name))) for name in names], dtype=np.float64)
    if mean.shape[0] != values.shape[0] or std.shape[0] != values.shape[0]:
        return 0.0
    std = std.copy()
    std[std < 1e-8] = 1.0
    clip = float(policy.get("clip", 5.0))
    normalized = np.clip((values - mean) / std, -clip, clip)
    return float(normalized @ weights)


def order_polylines_graph_tsp_with_trace(
    polylines: list[Polyline],
    current_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
    edge_policy: dict[str, Any] | None = None,
    edge_policy_top_k: int = 8,
    edge_policy_hard_safe_filter: bool = False,
    edge_policy_safe_min_inside_fraction: float = 0.92,
    edge_policy_safe_max_distance_mm: float = 0.0,
    edge_policy_jump_aware_weight: float = 0.0,
    edge_policy_offmask_weight: float = 0.0,
    edge_policy_visible_weight: float = 0.0,
    edge_policy_trim_weight: float = 0.0,
    task_context: dict[str, Any] | None = None,
) -> tuple[list[Polyline], dict[str, float], dict[str, Any]]:
    items = [{"node_id": index, "coords": coords} for index, coords in enumerate(polylines) if len(coords) >= 2]
    nodes = [build_node_record(int(item["node_id"]), item["coords"], width, height, scale_mm) for item in items]
    nodes_by_id = {int(node["node_id"]): node for node in nodes}
    remaining = [{**item, "coords": list(item["coords"])} for item in items]
    ordered_items: list[dict[str, Any]] = []
    selected_edges: list[dict[str, Any]] = []
    stats = {
        "nodes": float(len(remaining)),
        "edges_considered": 0.0,
        "selected_cost": 0.0,
        "selected_distance_mm": 0.0,
        "selected_offmask_fraction": 0.0,
        "visible_risk_edges": 0.0,
        "max_transition_mm": 0.0,
        "m2_policy_decisions": 0.0,
        "m2_policy_overrides": 0.0,
        "m2_policy_utility_sum": 0.0,
        "m2_hard_safe_filtered": 0.0,
        "m2_decode_penalty_sum": 0.0,
        "geometry_edges_sampled": 0.0,
        "geometry_visible_risk_edges": 0.0,
        "geometry_dt_penalty_sum": 0.0,
        "geometry_sobel_penalty_sum": 0.0,
        "geometry_canny_penalty_sum": 0.0,
        "geometry_dt_q05_sum": 0.0,
        "geometry_sobel_cross_sum": 0.0,
        "geometry_canny_cross_sum": 0.0,
    }
    start_mm = current_mm
    current = current_mm
    previous_node = -1
    while remaining:
        deterministic_best_index = 0
        deterministic_best_cost = float("inf")
        candidate_records: list[dict[str, Any]] = []
        for index, item in enumerate(remaining):
            cost, reverse, details = transition_cost(
                current,
                item["coords"],
                width,
                height,
                scale_mm,
                mask,
                config,
                geometry_priors,
            )
            stats["edges_considered"] += 1.0
            record = {"index": index, "item": item, "cost": cost, "reverse": reverse, "details": details}
            candidate_records.append(record)
            if cost < deterministic_best_cost:
                deterministic_best_cost = cost
                deterministic_best_index = index
        chosen = candidate_records[deterministic_best_index]
        if edge_policy is not None and previous_node >= 0 and len(candidate_records) > 1:
            previous_record = nodes_by_id.get(int(previous_node))
            if previous_record is not None:
                top_k = int(edge_policy_top_k or 0)
                policy_pool = sorted(candidate_records, key=lambda record: float(record["cost"]))
                if top_k > 0:
                    policy_pool = policy_pool[:top_k]
                if edge_policy_hard_safe_filter:
                    before_filter = len(policy_pool)
                    safe_pool = [
                        record
                        for record in policy_pool
                        if safe_float(record["details"].get("inside_fraction"), 1.0) >= edge_policy_safe_min_inside_fraction
                        and (not config.geometry_hard_filter or weighted_geometry_stats_are_safe(record["details"], config))
                        and (
                            edge_policy_safe_max_distance_mm <= 0.0
                            or safe_float(record["details"].get("distance_mm")) <= edge_policy_safe_max_distance_mm
                        )
                    ]
                    if safe_pool:
                        policy_pool = safe_pool
                        stats["m2_hard_safe_filtered"] += float(before_filter - len(policy_pool))
                for record in policy_pool:
                    node_record = nodes_by_id.get(int(record["item"]["node_id"]), {})
                    context = {
                        **(task_context or {}),
                        "long_jump_threshold_mm": float(config.long_jump_threshold_mm),
                        "safe_min_inside_fraction": float(edge_policy_safe_min_inside_fraction),
                    }
                    features = m2_edge_features(
                        context,
                        node_record,
                        previous_record,
                        current,
                        int(len(ordered_items)),
                        int(len(remaining)),
                        record["details"],
                    )
                    utility = m2_edge_utility(edge_policy, features)
                    details = record["details"]
                    distance = safe_float(details.get("distance_mm"))
                    decode_penalty = (
                        edge_policy_jump_aware_weight * max(0.0, distance / max(1e-6, config.long_jump_threshold_mm))
                        + edge_policy_offmask_weight * safe_float(details.get("offmask_fraction"))
                        + edge_policy_visible_weight * safe_float(details.get("combined_visible_risk"), safe_float(details.get("visible_risk")))
                        + edge_policy_trim_weight * safe_float(details.get("trim_risk"))
                    )
                    record["m2_utility"] = utility
                    record["m2_decode_penalty"] = decode_penalty
                    record["m2_decode_score"] = utility - decode_penalty
                chosen = max(
                    policy_pool,
                    key=lambda record: (
                        float(record.get("m2_decode_score", record.get("m2_utility", 0.0))),
                        float(record.get("m2_utility", 0.0)),
                        -float(record["cost"]),
                    ),
                )
                stats["m2_policy_decisions"] += 1.0
                stats["m2_policy_utility_sum"] += float(chosen.get("m2_utility", 0.0))
                stats["m2_decode_penalty_sum"] += float(chosen.get("m2_decode_penalty", 0.0))
                if int(chosen["index"]) != int(deterministic_best_index):
                    stats["m2_policy_overrides"] += 1.0
        best_index = int(chosen["index"])
        best_cost = float(chosen["cost"])
        best_reverse = bool(chosen["reverse"])
        best_details = chosen["details"]
        item = remaining.pop(best_index)
        coords = item["coords"]
        if best_reverse:
            coords = list(reversed(coords))
            item = {**item, "coords": coords}
        ordered_items.append(item)
        selected_edges.append(
            {
                "step": int(len(ordered_items) - 1),
                "from_node": int(previous_node),
                "to_node": int(item["node_id"]),
                "reverse": bool(best_reverse),
                "cost": round(float(best_details.get("cost", best_cost)), 6),
                "distance_mm": round(float(best_details.get("distance_mm", 0.0)), 4),
                "inside_fraction": round(float(best_details.get("inside_fraction", 0.0)), 6),
                "offmask_fraction": round(float(best_details.get("offmask_fraction", 0.0)), 6),
                "visible_risk": round(float(best_details.get("visible_risk", 0.0)), 4),
                "geometry_visible_risk": round(float(best_details.get("geometry_visible_risk", 0.0)), 4),
                "combined_visible_risk": round(float(best_details.get("combined_visible_risk", best_details.get("visible_risk", 0.0))), 4),
                "trim_risk": round(float(best_details.get("trim_risk", 0.0)), 4),
                "gp_dt_min_px": round(float(best_details.get("gp_dt_min_px", 0.0)), 4),
                "gp_dt_q05_px": round(float(best_details.get("gp_dt_q05_px", 0.0)), 4),
                "gp_dt_mean_px": round(float(best_details.get("gp_dt_mean_px", 0.0)), 4),
                "gp_sobel_cross_mean": round(float(best_details.get("gp_sobel_cross_mean", 0.0)), 6),
                "gp_canny_cross_frac": round(float(best_details.get("gp_canny_cross_frac", 0.0)), 6),
                "deterministic_best_node": int(candidate_records[deterministic_best_index]["item"]["node_id"]),
                "m2_policy_used": bool(edge_policy is not None and previous_node >= 0 and len(candidate_records) > 1),
                "m2_utility": round(float(chosen.get("m2_utility", 0.0)), 6),
                "m2_decode_score": round(float(chosen.get("m2_decode_score", chosen.get("m2_utility", 0.0))), 6),
                "m2_decode_penalty": round(float(chosen.get("m2_decode_penalty", 0.0)), 6),
            }
        )
        stats["selected_cost"] += float(best_details.get("cost", best_cost))
        stats["selected_distance_mm"] += float(best_details.get("distance_mm", 0.0))
        stats["selected_offmask_fraction"] += float(best_details.get("offmask_fraction", 0.0))
        stats["visible_risk_edges"] += float(best_details.get("visible_risk", 0.0))
        if "gp_dt_q05_px" in best_details:
            stats["geometry_edges_sampled"] += 1.0
            stats["geometry_visible_risk_edges"] += float(best_details.get("geometry_visible_risk", 0.0))
            stats["geometry_dt_penalty_sum"] += float(best_details.get("geometry_dt_penalty", 0.0))
            stats["geometry_sobel_penalty_sum"] += float(best_details.get("geometry_sobel_penalty", 0.0))
            stats["geometry_canny_penalty_sum"] += float(best_details.get("geometry_canny_penalty", 0.0))
            stats["geometry_dt_q05_sum"] += float(best_details.get("gp_dt_q05_px", 0.0))
            stats["geometry_sobel_cross_sum"] += float(best_details.get("gp_sobel_cross_mean", 0.0))
            stats["geometry_canny_cross_sum"] += float(best_details.get("gp_canny_cross_frac", 0.0))
        stats["max_transition_mm"] = max(stats["max_transition_mm"], float(best_details.get("distance_mm", 0.0)))
        current = coord_to_mm(coords[-1], width, height, scale_mm)
        previous_node = int(item["node_id"])
    two_opt_allowed = len(ordered_items) <= config.max_two_opt_nodes and config.two_opt_passes > 0
    ordered_items = two_opt_items(ordered_items, start_mm, width, height, scale_mm, mask, config, geometry_priors)
    if stats["nodes"] > 0:
        stats["mean_selected_cost"] = stats["selected_cost"] / stats["nodes"]
        stats["mean_offmask_fraction"] = stats["selected_offmask_fraction"] / stats["nodes"]
    else:
        stats["mean_selected_cost"] = 0.0
        stats["mean_offmask_fraction"] = 0.0
    if stats["m2_policy_decisions"] > 0:
        stats["m2_policy_mean_selected_utility"] = stats["m2_policy_utility_sum"] / stats["m2_policy_decisions"]
    else:
        stats["m2_policy_mean_selected_utility"] = 0.0
    if stats["geometry_edges_sampled"] > 0:
        denom = stats["geometry_edges_sampled"]
        stats["geometry_mean_dt_q05_px"] = stats["geometry_dt_q05_sum"] / denom
        stats["geometry_mean_sobel_cross"] = stats["geometry_sobel_cross_sum"] / denom
        stats["geometry_mean_canny_cross"] = stats["geometry_canny_cross_sum"] / denom
    else:
        stats["geometry_mean_dt_q05_px"] = 0.0
        stats["geometry_mean_sobel_cross"] = 0.0
        stats["geometry_mean_canny_cross"] = 0.0
    graph = {
        "nodes": nodes,
        "selected_edges": selected_edges,
        "route_node_ids": [int(item["node_id"]) for item in ordered_items],
        "two_opt_allowed": bool(two_opt_allowed),
        "two_opt_note": "selected_edges record greedy pre-2opt choices; route_node_ids records final route order",
    }
    return [item["coords"] for item in ordered_items], stats, graph


def order_polylines_graph_tsp(
    polylines: list[Polyline],
    current_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
    geometry_priors: GeometryPriors | None = None,
    edge_policy: dict[str, Any] | None = None,
    edge_policy_top_k: int = 8,
    edge_policy_hard_safe_filter: bool = False,
    edge_policy_safe_min_inside_fraction: float = 0.92,
    edge_policy_safe_max_distance_mm: float = 0.0,
    edge_policy_jump_aware_weight: float = 0.0,
    edge_policy_offmask_weight: float = 0.0,
    edge_policy_visible_weight: float = 0.0,
    edge_policy_trim_weight: float = 0.0,
    task_context: dict[str, Any] | None = None,
) -> tuple[list[Polyline], dict[str, float]]:
    ordered, stats, _graph = order_polylines_graph_tsp_with_trace(
        polylines,
        current_mm,
        width,
        height,
        scale_mm,
        mask,
        config,
        geometry_priors=geometry_priors,
        edge_policy=edge_policy,
        edge_policy_top_k=edge_policy_top_k,
        edge_policy_hard_safe_filter=edge_policy_hard_safe_filter,
        edge_policy_safe_min_inside_fraction=edge_policy_safe_min_inside_fraction,
        edge_policy_safe_max_distance_mm=edge_policy_safe_max_distance_mm,
        edge_policy_jump_aware_weight=edge_policy_jump_aware_weight,
        edge_policy_offmask_weight=edge_policy_offmask_weight,
        edge_policy_visible_weight=edge_policy_visible_weight,
        edge_policy_trim_weight=edge_policy_trim_weight,
        task_context=task_context,
    )
    return ordered, stats
