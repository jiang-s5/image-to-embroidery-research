from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
        cost = (
            distance
            + config.long_jump_weight * long_jump
            + config.trim_penalty * trim_risk
            + config.offmask_weight * offmask_fraction * max(distance, config.long_jump_threshold_mm)
            + config.visible_connector_penalty * visible_risk
        )
        if cost < best_cost:
            best_cost = cost
            best_reverse = reverse
            best_details = {
                "distance_mm": distance,
                "inside_fraction": inside_fraction,
                "offmask_fraction": offmask_fraction,
                "visible_risk": visible_risk,
                "trim_risk": trim_risk,
                "cost": cost,
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
) -> float:
    total = 0.0
    current = start_mm
    for coords in ordered:
        cost, reverse, _details = transition_cost(current, coords, width, height, scale_mm, mask, config)
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
) -> list[Polyline]:
    if len(ordered) < 4 or config.two_opt_passes <= 0 or len(ordered) > config.max_two_opt_nodes:
        return ordered
    best = [list(coords) for coords in ordered]
    best_cost = sequence_cost(best, start_mm, width, height, scale_mm, mask, config)
    for _ in range(config.two_opt_passes):
        improved = False
        for i in range(0, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + [list(reversed(coords)) for coords in reversed(best[i : j + 1])] + best[j + 1 :]
                cost = sequence_cost(candidate, start_mm, width, height, scale_mm, mask, config)
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
) -> float:
    return sequence_cost([item["coords"] for item in ordered], start_mm, width, height, scale_mm, mask, config)


def two_opt_items(
    ordered: list[dict[str, Any]],
    start_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
) -> list[dict[str, Any]]:
    if len(ordered) < 4 or config.two_opt_passes <= 0 or len(ordered) > config.max_two_opt_nodes:
        return ordered
    best = [{**item, "coords": list(item["coords"])} for item in ordered]
    best_cost = sequence_cost_items(best, start_mm, width, height, scale_mm, mask, config)
    for _ in range(config.two_opt_passes):
        improved = False
        for i in range(0, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                reversed_slice = [{**item, "coords": list(reversed(item["coords"]))} for item in reversed(best[i : j + 1])]
                candidate = best[:i] + reversed_slice + best[j + 1 :]
                cost = sequence_cost_items(candidate, start_mm, width, height, scale_mm, mask, config)
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


def order_polylines_graph_tsp_with_trace(
    polylines: list[Polyline],
    current_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
) -> tuple[list[Polyline], dict[str, float], dict[str, Any]]:
    items = [{"node_id": index, "coords": coords} for index, coords in enumerate(polylines) if len(coords) >= 2]
    nodes = [build_node_record(int(item["node_id"]), item["coords"], width, height, scale_mm) for item in items]
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
    }
    start_mm = current_mm
    current = current_mm
    previous_node = -1
    while remaining:
        best_index = 0
        best_cost = float("inf")
        best_reverse = False
        best_details: dict[str, float] = {}
        for index, item in enumerate(remaining):
            cost, reverse, details = transition_cost(current, item["coords"], width, height, scale_mm, mask, config)
            stats["edges_considered"] += 1.0
            if cost < best_cost:
                best_cost = cost
                best_index = index
                best_reverse = reverse
                best_details = details
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
                "trim_risk": round(float(best_details.get("trim_risk", 0.0)), 4),
            }
        )
        stats["selected_cost"] += float(best_details.get("cost", best_cost))
        stats["selected_distance_mm"] += float(best_details.get("distance_mm", 0.0))
        stats["selected_offmask_fraction"] += float(best_details.get("offmask_fraction", 0.0))
        stats["visible_risk_edges"] += float(best_details.get("visible_risk", 0.0))
        stats["max_transition_mm"] = max(stats["max_transition_mm"], float(best_details.get("distance_mm", 0.0)))
        current = coord_to_mm(coords[-1], width, height, scale_mm)
        previous_node = int(item["node_id"])
    two_opt_allowed = len(ordered_items) <= config.max_two_opt_nodes and config.two_opt_passes > 0
    ordered_items = two_opt_items(ordered_items, start_mm, width, height, scale_mm, mask, config)
    if stats["nodes"] > 0:
        stats["mean_selected_cost"] = stats["selected_cost"] / stats["nodes"]
        stats["mean_offmask_fraction"] = stats["selected_offmask_fraction"] / stats["nodes"]
    else:
        stats["mean_selected_cost"] = 0.0
        stats["mean_offmask_fraction"] = 0.0
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
) -> tuple[list[Polyline], dict[str, float]]:
    ordered, stats, _graph = order_polylines_graph_tsp_with_trace(
        polylines,
        current_mm,
        width,
        height,
        scale_mm,
        mask,
        config,
    )
    return ordered, stats
