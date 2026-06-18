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


def order_polylines_graph_tsp(
    polylines: list[Polyline],
    current_mm: Point,
    width: int,
    height: int,
    scale_mm: float,
    mask: np.ndarray | None,
    config: GraphTSPConfig,
) -> tuple[list[Polyline], dict[str, float]]:
    remaining = [coords for coords in polylines if len(coords) >= 2]
    ordered: list[Polyline] = []
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
    while remaining:
        best_index = 0
        best_cost = float("inf")
        best_reverse = False
        best_details: dict[str, float] = {}
        for index, coords in enumerate(remaining):
            cost, reverse, details = transition_cost(current, coords, width, height, scale_mm, mask, config)
            stats["edges_considered"] += 1.0
            if cost < best_cost:
                best_cost = cost
                best_index = index
                best_reverse = reverse
                best_details = details
        coords = remaining.pop(best_index)
        if best_reverse:
            coords = list(reversed(coords))
        ordered.append(coords)
        stats["selected_cost"] += float(best_details.get("cost", best_cost))
        stats["selected_distance_mm"] += float(best_details.get("distance_mm", 0.0))
        stats["selected_offmask_fraction"] += float(best_details.get("offmask_fraction", 0.0))
        stats["visible_risk_edges"] += float(best_details.get("visible_risk", 0.0))
        stats["max_transition_mm"] = max(stats["max_transition_mm"], float(best_details.get("distance_mm", 0.0)))
        current = coord_to_mm(coords[-1], width, height, scale_mm)
    ordered = two_opt(ordered, start_mm, width, height, scale_mm, mask, config)
    if stats["nodes"] > 0:
        stats["mean_selected_cost"] = stats["selected_cost"] / stats["nodes"]
        stats["mean_offmask_fraction"] = stats["selected_offmask_fraction"] / stats["nodes"]
    else:
        stats["mean_selected_cost"] = 0.0
        stats["mean_offmask_fraction"] = 0.0
    return ordered, stats
