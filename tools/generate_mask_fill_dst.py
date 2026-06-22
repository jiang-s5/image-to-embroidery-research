from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pyembroidery import END, JUMP, STITCH, EmbPattern, EmbThread, write_dst

Pixel = tuple[int, int]

NEIGHBORS_8 = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def add_thread(pattern: EmbPattern, color: tuple[int, int, int]) -> None:
    thread = EmbThread()
    thread.set_color(*color)
    pattern.add_thread(thread)


def pixel_to_mm_xy(x: float, y: float, shape: tuple[int, int], target_width_mm: float) -> tuple[float, float]:
    height, width = shape
    scale = target_width_mm / max(1, width)
    return ((x - width / 2.0) * scale, (y - height / 2.0) * scale)


def add_abs(pattern: EmbPattern, command: int, point_mm: tuple[float, float]) -> None:
    pattern.add_stitch_absolute(command, int(round(point_mm[0] * 10)), int(round(point_mm[1] * 10)))


def dst_quantized_mm(point_mm: tuple[float, float]) -> tuple[float, float]:
    return (round(point_mm[0] * 10) / 10.0, round(point_mm[1] * 10) / 10.0)


def output_segment_inside_fraction(
    mask: np.ndarray,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    target_width_mm: float,
    validate_quantized_segments: bool,
) -> float:
    if validate_quantized_segments:
        start_mm = dst_quantized_mm(start_mm)
        end_mm = dst_quantized_mm(end_mm)
    return segment_inside_fraction(mask, start_mm, end_mm, target_width_mm)


def add_segment(pattern: EmbPattern, start_mm: tuple[float, float], end_mm: tuple[float, float], max_stitch_mm: float) -> int:
    distance = math.dist(start_mm, end_mm)
    if distance <= 1e-6:
        return 0
    steps = max(1, int(math.ceil(distance / max(0.1, max_stitch_mm))))
    for index in range(1, steps + 1):
        t = index / steps
        point = (
            start_mm[0] + (end_mm[0] - start_mm[0]) * t,
            start_mm[1] + (end_mm[1] - start_mm[1]) * t,
        )
        add_abs(pattern, STITCH, point)
    return steps


def path_length_px(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(math.dist(path[index - 1], path[index]) for index in range(1, len(path)))


def compress_grid_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    compressed = [path[0]]
    last_direction: tuple[int, int] | None = None
    for index in range(1, len(path)):
        prev = path[index - 1]
        current = path[index]
        dx = current[0] - prev[0]
        dy = current[1] - prev[1]
        direction = (0 if dx == 0 else dx // abs(dx), 0 if dy == 0 else dy // abs(dy))
        if last_direction is not None and direction != last_direction:
            compressed.append(prev)
        last_direction = direction
    compressed.append(path[-1])
    return compressed


def add_polyline_px(
    pattern: EmbPattern,
    path: list[tuple[int, int]],
    shape: tuple[int, int],
    target_width_mm: float,
    max_stitch_mm: float,
) -> int:
    if len(path) < 2:
        return 0
    points = compress_grid_path(path)
    stitch_segments = 0
    current_mm = pixel_to_mm_xy(points[0][0], points[0][1], shape, target_width_mm)
    for point in points[1:]:
        next_mm = pixel_to_mm_xy(point[0], point[1], shape, target_width_mm)
        stitch_segments += add_segment(pattern, current_mm, next_mm, max_stitch_mm)
        current_mm = next_mm
    return stitch_segments


def polyline_min_inside_fraction(
    mask: np.ndarray,
    path: list[tuple[int, int]],
    shape: tuple[int, int],
    target_width_mm: float,
    validate_quantized_segments: bool,
) -> float:
    if len(path) < 2:
        return 1.0
    points = compress_grid_path(path)
    min_inside = 1.0
    current_mm = pixel_to_mm_xy(points[0][0], points[0][1], shape, target_width_mm)
    for point in points[1:]:
        next_mm = pixel_to_mm_xy(point[0], point[1], shape, target_width_mm)
        min_inside = min(
            min_inside,
            output_segment_inside_fraction(mask, current_mm, next_mm, target_width_mm, validate_quantized_segments),
        )
        current_mm = next_mm
    return min_inside


def contour_to_path(contour: np.ndarray, stride_px: int) -> list[tuple[int, int]]:
    raw = contour.reshape(-1, 2)
    if raw.shape[0] == 0:
        return []
    stride = max(1, stride_px)
    points = [(int(point[0]), int(point[1])) for point in raw[::stride]]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def component_outline_paths(
    component_mask: np.ndarray,
    min_area_px: int,
    stride_px: int,
    include_holes: bool,
    inset_px: int,
) -> list[list[tuple[int, int]]]:
    if inset_px > 0:
        kernel_size = max(1, inset_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        component_mask = cv2.erode(component_mask.astype(np.uint8), kernel, iterations=1)
        if int(component_mask.sum()) <= 0:
            return []
    mode = cv2.RETR_CCOMP if include_holes else cv2.RETR_EXTERNAL
    contours, _hierarchy = cv2.findContours((component_mask > 0).astype(np.uint8) * 255, mode, cv2.CHAIN_APPROX_NONE)
    paths: list[list[tuple[int, int]]] = []
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area_px:
            continue
        path = contour_to_path(contour, stride_px)
        if len(path) >= 3:
            paths.append(path)
    return sorted(paths, key=lambda path: len(path), reverse=True)


def erode_mask(mask: np.ndarray, inset_px: int) -> np.ndarray:
    if inset_px <= 0:
        return mask.astype(np.uint8)
    kernel_size = max(1, inset_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def point_inside(mask: np.ndarray, point: tuple[int, int]) -> bool:
    x, y = point
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0


def inward_point(
    mask: np.ndarray,
    point: tuple[int, int],
    unit: tuple[float, float],
    preferred_distance_px: int,
    min_distance_px: int = 2,
) -> tuple[int, int] | None:
    for distance in range(max(preferred_distance_px, min_distance_px), min_distance_px - 1, -1):
        candidate = (
            int(round(point[0] + unit[0] * distance)),
            int(round(point[1] + unit[1] * distance)),
        )
        if point_inside(mask, candidate):
            return candidate
    return None


def component_satin_columns(
    component_mask: np.ndarray,
    min_area_px: int,
    step_px: int,
    width_px: int,
    outer_inset_px: int,
    max_columns: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    work_mask = erode_mask(component_mask, outer_inset_px)
    if int(work_mask.sum()) <= 0:
        return []
    contours, _hierarchy = cv2.findContours((work_mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ys, xs = np.where(component_mask > 0)
    if xs.size == 0:
        return []
    centroid = (float(xs.mean()), float(ys.mean()))
    columns: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area_px:
            continue
        raw = contour.reshape(-1, 2)
        for raw_point in raw[:: max(1, step_px)]:
            outer = (int(raw_point[0]), int(raw_point[1]))
            if not point_inside(component_mask, outer):
                outer_near = nearest_foreground(component_mask, outer, radius=max(3, outer_inset_px + 2))
                if outer_near is None:
                    continue
                outer = outer_near
            dx = centroid[0] - outer[0]
            dy = centroid[1] - outer[1]
            norm = math.hypot(dx, dy)
            if norm <= 1e-6:
                continue
            unit = (dx / norm, dy / norm)
            inner = inward_point(component_mask, outer, unit, width_px)
            if inner is None or math.dist(outer, inner) < 2.0:
                continue
            columns.append((outer, inner))
            if len(columns) >= max_columns:
                return columns
    return columns


def _contour_points_from_distance_band(region: np.ndarray, min_area_px: int, step_px: int) -> list[np.ndarray]:
    contours, _hierarchy = cv2.findContours((region > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sampled: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = abs(float(cv2.contourArea(contour)))
        if area < min_area_px:
            continue
        raw = contour.reshape(-1, 2)
        if raw.shape[0] < 3:
            continue
        sampled.append(raw[:: max(1, step_px)])
    return sampled


def component_dt_satin_pairs(
    component_mask: np.ndarray,
    min_area_px: int,
    step_px: int,
    outer_distance_px: int,
    inner_distance_px: int,
    max_pair_px: int,
    max_pairs: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Build satin-like outer/inner rail pairs from distance-transform level sets."""
    binary = (component_mask > 0).astype(np.uint8)
    if int(binary.sum()) <= 0:
        return []
    outer_distance = max(1.0, float(outer_distance_px))
    inner_distance = max(outer_distance + 1.0, float(inner_distance_px))
    max_pair_distance = max(2.0, float(max_pair_px))

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) < inner_distance:
        return []

    outer_region = ((distance >= outer_distance) & (binary > 0)).astype(np.uint8)
    inner_region = ((distance >= inner_distance) & (binary > 0)).astype(np.uint8)
    outer_contours = _contour_points_from_distance_band(outer_region, min_area_px, step_px)
    inner_contours = _contour_points_from_distance_band(inner_region, max(4, min_area_px // 4), max(1, step_px // 2))
    if not outer_contours or not inner_contours:
        return []

    inner_points = np.concatenate(inner_contours, axis=0).astype(np.float32)
    if inner_points.shape[0] == 0:
        return []

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for outer_contour in outer_contours:
        for raw_outer in outer_contour:
            outer = (int(raw_outer[0]), int(raw_outer[1]))
            if not point_inside(component_mask, outer):
                continue
            deltas = inner_points - np.asarray(outer, dtype=np.float32)
            distances = np.einsum("ij,ij->i", deltas, deltas)
            nearest_index = int(np.argmin(distances))
            pair_distance = math.sqrt(float(distances[nearest_index]))
            if pair_distance < 2.0 or pair_distance > max_pair_distance:
                continue
            raw_inner = inner_points[nearest_index]
            inner = (int(round(float(raw_inner[0]))), int(round(float(raw_inner[1]))))
            if not point_inside(component_mask, inner):
                continue
            pairs.append((outer, inner))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def component_skeleton_paths(skeleton: np.ndarray, component_mask: np.ndarray, min_pixels: int) -> list[list[tuple[int, int]]]:
    component_skeleton = ((skeleton > 0) & (component_mask > 0)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(component_skeleton, connectivity=8)
    paths: list[list[tuple[int, int]]] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_pixels:
            continue
        ys, xs = np.where(labels == label)
        pixels = list(zip(xs.tolist(), ys.tolist()))
        graph = build_skeleton_graph(pixels)
        if not graph:
            continue
        start = choose_skeleton_start(graph)
        walk = dfs_skeleton_edge_walk(graph, start)
        if len(walk) >= 2:
            paths.append(walk)
    return sorted(paths, key=len, reverse=True)


def build_skeleton_graph(pixels: list[Pixel]) -> dict[Pixel, list[Pixel]]:
    pixel_set = set(pixels)
    graph: dict[Pixel, list[Pixel]] = {}
    for x, y in pixels:
        neighbors: list[Pixel] = []
        for dx, dy in NEIGHBORS_8:
            candidate = (x + dx, y + dy)
            if candidate in pixel_set:
                neighbors.append(candidate)
        graph[(x, y)] = sorted(neighbors)
    return graph


def choose_skeleton_start(graph: dict[Pixel, list[Pixel]]) -> Pixel:
    endpoints = [pixel for pixel, neighbors in graph.items() if len(neighbors) <= 1]
    return min(endpoints or list(graph.keys()))


def dfs_skeleton_edge_walk(graph: dict[Pixel, list[Pixel]], start: Pixel) -> list[Pixel]:
    path: list[Pixel] = [start]
    seen_edges: set[tuple[Pixel, Pixel]] = set()

    def edge_key(a: Pixel, b: Pixel) -> tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    def visit(node: Pixel) -> None:
        for neighbor in graph[node]:
            key = edge_key(node, neighbor)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            path.append(neighbor)
            visit(neighbor)
            path.append(node)

    visit(start)
    return path


def classify_component_style(
    component_mask: np.ndarray,
    component_skeleton: np.ndarray | None,
    running_skeleton_ratio: float,
    running_max_distance_px: float,
    running_min_skeleton_pixels: int,
) -> dict[str, Any]:
    area = int(component_mask.sum())
    skeleton_pixels = int(component_skeleton.sum()) if component_skeleton is not None else 0
    skeleton_ratio = skeleton_pixels / max(1, area)
    distance = cv2.distanceTransform((component_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_distance = float(distance.max()) if distance.size else 0.0
    mean_distance = float(distance[component_mask > 0].mean()) if area > 0 else 0.0
    style = "fill"
    reasons: list[str] = []
    if component_skeleton is not None and skeleton_pixels >= running_min_skeleton_pixels:
        if skeleton_ratio >= running_skeleton_ratio:
            style = "running"
            reasons.append("high_skeleton_ratio")
        if max_distance <= running_max_distance_px:
            style = "running"
            reasons.append("thin_distance_transform")
    return {
        "style": style,
        "area": area,
        "skeleton_pixels": skeleton_pixels,
        "skeleton_ratio": round(skeleton_ratio, 6),
        "max_distance_px": round(max_distance, 6),
        "mean_distance_px": round(mean_distance, 6),
        "reasons": reasons,
    }


def stitch_point_sequence(
    pattern: EmbPattern,
    sequence: list[tuple[int, int]],
    current_mm: tuple[float, float] | None,
    current_px: tuple[int, int] | None,
    connector_mask: np.ndarray,
    shape: tuple[int, int],
    target_width_mm: float,
    max_stitch_mm: float,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    use_mask_path_connectors: bool,
    max_mask_path_px: float,
    max_mask_path_expansions: int,
    validate_mask_path_segments: bool,
    mask_path_min_segment_inside_fraction: float,
    validate_quantized_segments: bool,
) -> tuple[tuple[float, float] | None, tuple[int, int] | None, dict[str, int]]:
    stats = {
        "jumps": 0,
        "safe_connects": 0,
        "mask_path_connects": 0,
        "rejected_connects": 0,
        "rejected_mask_paths": 0,
        "stitch_segments": 0,
        "sequence_segments": 0,
        "sequence_rejected": 0,
    }
    if not sequence:
        return current_mm, current_px, stats

    current_mm, current_px, connect_stats = connect_to_point(
        pattern,
        current_mm,
        current_px,
        sequence[0],
        connector_mask,
        shape,
        target_width_mm,
        max_stitch_mm,
        max_connect_mm,
        min_connect_inside_fraction,
        use_mask_path_connectors,
        max_mask_path_px,
        max_mask_path_expansions,
        validate_mask_path_segments,
        mask_path_min_segment_inside_fraction,
        validate_quantized_segments,
    )
    merge_connect_stats(stats, connect_stats)

    for next_px in sequence[1:]:
        if current_mm is None:
            break
        next_mm = pixel_to_mm_xy(next_px[0], next_px[1], shape, target_width_mm)
        distance = math.dist(current_mm, next_mm)
        inside = output_segment_inside_fraction(
            connector_mask,
            current_mm,
            next_mm,
            target_width_mm,
            validate_quantized_segments,
        )
        if distance <= max_connect_mm and inside >= min_connect_inside_fraction:
            stats["stitch_segments"] += add_segment(pattern, current_mm, next_mm, max_stitch_mm)
            stats["sequence_segments"] += 1
            current_mm = next_mm
            current_px = next_px
            continue
        current_mm, current_px, connect_stats = connect_to_point(
            pattern,
            current_mm,
            current_px,
            next_px,
            connector_mask,
            shape,
            target_width_mm,
            max_stitch_mm,
            max_connect_mm,
            min_connect_inside_fraction,
            use_mask_path_connectors,
            max_mask_path_px,
            max_mask_path_expansions,
            validate_mask_path_segments,
            mask_path_min_segment_inside_fraction,
            validate_quantized_segments,
        )
        merge_connect_stats(stats, connect_stats)
        stats["sequence_rejected"] += 1
    return current_mm, current_px, stats


def connect_to_point(
    pattern: EmbPattern,
    current_mm: tuple[float, float] | None,
    current_px: tuple[int, int] | None,
    target_px: tuple[int, int],
    connector_mask: np.ndarray,
    shape: tuple[int, int],
    target_width_mm: float,
    max_stitch_mm: float,
    max_connect_mm: float,
    min_connect_inside_fraction: float,
    use_mask_path_connectors: bool,
    max_mask_path_px: float,
    max_mask_path_expansions: int,
    validate_mask_path_segments: bool,
    mask_path_min_segment_inside_fraction: float,
    validate_quantized_segments: bool,
) -> tuple[tuple[float, float], tuple[int, int], dict[str, int]]:
    target_mm = pixel_to_mm_xy(target_px[0], target_px[1], shape, target_width_mm)
    stats = {
        "jumps": 0,
        "safe_connects": 0,
        "mask_path_connects": 0,
        "rejected_connects": 0,
        "rejected_mask_paths": 0,
        "stitch_segments": 0,
    }
    if current_mm is None:
        add_abs(pattern, JUMP, target_mm)
        stats["jumps"] += 1
        return target_mm, target_px, stats

    distance = math.dist(current_mm, target_mm)
    inside = output_segment_inside_fraction(
        connector_mask,
        current_mm,
        target_mm,
        target_width_mm,
        validate_quantized_segments,
    )
    if distance <= max_connect_mm and inside >= min_connect_inside_fraction:
        stats["stitch_segments"] += add_segment(pattern, current_mm, target_mm, max_stitch_mm)
        stats["safe_connects"] += 1
        return target_mm, target_px, stats

    if use_mask_path_connectors and current_px is not None:
        path = astar_mask_path(
            connector_mask,
            current_px,
            target_px,
            max_mask_path_px,
            max_mask_path_expansions,
        )
        if path is not None:
            if validate_mask_path_segments:
                path_inside = polyline_min_inside_fraction(
                    connector_mask,
                    path,
                    shape,
                    target_width_mm,
                    validate_quantized_segments,
                )
                if path_inside < mask_path_min_segment_inside_fraction:
                    stats["rejected_mask_paths"] += 1
                    add_abs(pattern, JUMP, target_mm)
                    stats["jumps"] += 1
                    stats["rejected_connects"] += 1
                    return target_mm, target_px, stats
            stats["stitch_segments"] += add_polyline_px(pattern, path, shape, target_width_mm, max_stitch_mm)
            stats["mask_path_connects"] += 1
            return target_mm, target_px, stats
        stats["rejected_mask_paths"] += 1

    add_abs(pattern, JUMP, target_mm)
    stats["jumps"] += 1
    stats["rejected_connects"] += 1
    return target_mm, target_px, stats


def merge_connect_stats(totals: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        totals[key] = totals.get(key, 0) + int(value)


def nearest_foreground(mask: np.ndarray, point: tuple[int, int], radius: int = 3) -> tuple[int, int] | None:
    x, y = point
    height, width = mask.shape
    if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
        return (x, y)
    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if mask[yy, xx] <= 0:
                continue
            distance = math.dist((x, y), (xx, yy))
            if distance < best_distance:
                best = (xx, yy)
                best_distance = distance
    return best


def astar_mask_path(
    mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    max_path_px: float,
    max_expansions: int,
) -> list[tuple[int, int]] | None:
    start_fg = nearest_foreground(mask, start)
    end_fg = nearest_foreground(mask, end)
    if start_fg is None or end_fg is None:
        return None
    start = start_fg
    end = end_fg
    if start == end:
        return [start, end]

    height, width = mask.shape
    margin = int(math.ceil(max_path_px)) + 4
    x0 = max(0, min(start[0], end[0]) - margin)
    x1 = min(width - 1, max(start[0], end[0]) + margin)
    y0 = max(0, min(start[1], end[1]) - margin)
    y1 = min(height - 1, max(start[1], end[1]) + margin)

    if start[0] < x0 or start[0] > x1 or end[0] < x0 or end[0] > x1:
        return None
    if start[1] < y0 or start[1] > y1 or end[1] < y0 or end[1] > y1:
        return None

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    queue: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(queue, (math.dist(start, end), 0.0, start))
    best_cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    expansions = 0

    while queue and expansions < max_expansions:
        _priority, cost, current = heapq.heappop(queue)
        if cost > best_cost.get(current, float("inf")) + 1e-6:
            continue
        expansions += 1
        if current == end:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path if path_length_px(path) <= max_path_px else None
        for dx, dy, step_cost in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < x0 or nx > x1 or ny < y0 or ny > y1:
                continue
            if mask[ny, nx] <= 0:
                continue
            next_cost = cost + step_cost
            if next_cost > max_path_px:
                continue
            candidate = (nx, ny)
            if next_cost + 1e-6 >= best_cost.get(candidate, float("inf")):
                continue
            best_cost[candidate] = next_cost
            parent[candidate] = current
            heapq.heappush(queue, (next_cost + math.dist(candidate, end), next_cost, candidate))
    return None


def segment_inside_fraction(
    mask: np.ndarray,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    target_width_mm: float,
    sample_step_mm: float = 0.5,
) -> float:
    distance = math.dist(start_mm, end_mm)
    if distance <= 1e-6:
        return 1.0
    height, width = mask.shape
    scale = target_width_mm / max(1, width)
    steps = max(1, int(math.ceil(distance / max(0.1, sample_step_mm))))
    inside = 0
    total = steps + 1
    for index in range(total):
        t = index / steps
        x_mm = start_mm[0] + (end_mm[0] - start_mm[0]) * t
        y_mm = start_mm[1] + (end_mm[1] - start_mm[1]) * t
        x = int(round(x_mm / scale + width / 2.0))
        y = int(round(y_mm / scale + height / 2.0))
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            inside += 1
    return inside / total


def row_runs(component_mask: np.ndarray, y: int, min_run_px: int) -> list[tuple[int, int]]:
    xs = np.where(component_mask[y] > 0)[0]
    if xs.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(xs[0])
    prev = int(xs[0])
    for raw_x in xs[1:]:
        x = int(raw_x)
        if x == prev + 1:
            prev = x
            continue
        if prev - start + 1 >= min_run_px:
            runs.append((start, prev))
        start = prev = x
    if prev - start + 1 >= min_run_px:
        runs.append((start, prev))
    return runs


def component_fill_rows(
    component_mask: np.ndarray,
    row_spacing_px: int,
    min_run_px: int,
    reverse_first: bool = False,
    row_order_strategy: str = "serpentine",
    entry_px: tuple[int, int] | None = None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ys, xs = np.where(component_mask > 0)
    if ys.size == 0:
        return []
    y_min = int(ys.min())
    y_max = int(ys.max())
    raw_rows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    rows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    reverse = reverse_first
    for y in range(y_min, y_max + 1, max(1, row_spacing_px)):
        runs = row_runs(component_mask, y, min_run_px)
        for x0, x1 in runs:
            raw_rows.append(((x0, y), (x1, y)))
        if row_order_strategy != "nearest_endpoint":
            if reverse:
                runs = list(reversed(runs))
            for x0, x1 in runs:
                if reverse:
                    rows.append(((x1, y), (x0, y)))
                else:
                    rows.append(((x0, y), (x1, y)))
                reverse = not reverse
    if row_order_strategy != "nearest_endpoint":
        return rows

    remaining = list(raw_rows)
    current = entry_px
    while remaining:
        if current is None:
            index = 0
            start, end = remaining.pop(index)
            if reverse:
                start, end = end, start
            rows.append((start, end))
            current = end
            reverse = not reverse
            continue
        index, start, end = min(
            (
                (index, segment[0], segment[1])
                for index, segment in enumerate(remaining)
            ),
            key=lambda item: min(math.dist(current, item[1]), math.dist(current, item[2])),
        )
        remaining.pop(index)
        if math.dist(current, end) < math.dist(current, start):
            start, end = end, start
        rows.append((start, end))
        current = end
    return rows


def component_order(labels: np.ndarray, stats: np.ndarray, min_component_pixels: int, strategy: str = "area") -> list[int]:
    labels_out = []
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_pixels:
            labels_out.append(label)
    if strategy == "nearest_centroid" and labels_out:
        remaining = set(labels_out)
        current = max(remaining, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
        ordered = [current]
        remaining.remove(current)
        while remaining:
            current_center = (
                float(stats[current, cv2.CC_STAT_LEFT]) + float(stats[current, cv2.CC_STAT_WIDTH]) / 2.0,
                float(stats[current, cv2.CC_STAT_TOP]) + float(stats[current, cv2.CC_STAT_HEIGHT]) / 2.0,
            )
            current = min(
                remaining,
                key=lambda label: (
                    math.dist(
                        current_center,
                        (
                            float(stats[label, cv2.CC_STAT_LEFT]) + float(stats[label, cv2.CC_STAT_WIDTH]) / 2.0,
                            float(stats[label, cv2.CC_STAT_TOP]) + float(stats[label, cv2.CC_STAT_HEIGHT]) / 2.0,
                        ),
                    ),
                    -int(stats[label, cv2.CC_STAT_AREA]),
                ),
            )
            ordered.append(current)
            remaining.remove(current)
        return ordered
    return sorted(labels_out, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)


def generate_mask_fill_dst(
    mask_path: Path,
    output_dst: Path,
    connector_mask_path: Path | None = None,
    skeleton_path: Path | None = None,
    target_width_mm: float = 90.0,
    row_spacing_mm: float = 1.2,
    max_stitch_mm: float = 3.2,
    max_connect_mm: float = 6.0,
    min_connect_inside_fraction: float = 0.95,
    min_component_pixels: int = 64,
    min_run_mm: float = 1.0,
    fill_inset_px: int = 0,
    use_mask_path_connectors: bool = False,
    max_mask_path_mm: float = 24.0,
    max_mask_path_expansions: int = 8000,
    validate_mask_path_segments: bool = False,
    mask_path_min_segment_inside_fraction: float = 1.0,
    validate_quantized_segments: bool = False,
    add_outline: bool = False,
    outline_stride_px: int = 2,
    outline_min_area_px: int = 64,
    outline_include_holes: bool = True,
    outline_inset_px: int = 0,
    add_satin_border: bool = False,
    satin_width_px: int = 6,
    satin_step_px: int = 5,
    satin_outer_inset_px: int = 2,
    satin_min_area_px: int = 64,
    satin_max_columns_per_component: int = 180,
    add_satin_rail_border: bool = False,
    satin_rail_width_px: int = 8,
    satin_rail_step_px: int = 7,
    satin_rail_outer_inset_px: int = 4,
    satin_rail_min_area_px: int = 64,
    satin_rail_max_pairs_per_component: int = 180,
    add_dt_satin_border: bool = False,
    dt_satin_outer_distance_px: int = 3,
    dt_satin_inner_distance_px: int = 9,
    dt_satin_step_px: int = 7,
    dt_satin_min_area_px: int = 64,
    dt_satin_max_pair_px: int = 18,
    dt_satin_max_pairs_per_component: int = 180,
    use_style_aware_components: bool = False,
    style_running_skeleton_ratio: float = 0.08,
    style_running_max_distance_px: float = 4.5,
    style_running_min_skeleton_pixels: int = 4,
    component_order_strategy: str = "area",
    row_order_strategy: str = "serpentine",
    thread_rgb: tuple[int, int, int] = (30, 120, 200),
) -> dict[str, Any]:
    mask = load_mask(mask_path)
    connector_mask = load_mask(connector_mask_path) if connector_mask_path is not None and connector_mask_path.exists() else mask
    skeleton = load_mask(skeleton_path) if skeleton_path is not None and skeleton_path.exists() else None
    height, width = mask.shape
    scale = target_width_mm / max(1, width)
    row_spacing_px = max(1, int(round(row_spacing_mm / max(scale, 1e-6))))
    min_run_px = max(1, int(round(min_run_mm / max(scale, 1e-6))))
    max_mask_path_px = max(1.0, max_mask_path_mm / max(scale, 1e-6))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    pattern = EmbPattern()
    add_thread(pattern, thread_rgb)

    current_mm: tuple[float, float] | None = None
    jumps = 0
    safe_connects = 0
    rejected_connects = 0
    mask_path_connects = 0
    rejected_mask_paths = 0
    stitch_segments = 0
    fill_rows = 0
    outline_paths = 0
    outline_points = 0
    satin_columns = 0
    satin_skipped = 0
    satin_rail_pairs = 0
    satin_rail_segments = 0
    satin_rail_rejected = 0
    dt_satin_pairs = 0
    dt_satin_segments = 0
    dt_satin_rejected = 0
    style_running_components = 0
    style_fill_components = 0
    style_running_paths = 0
    style_running_points = 0
    style_fallback_fill_components = 0
    style_component_reports: list[dict[str, Any]] = []
    components_used = 0
    current_px: tuple[int, int] | None = None

    ordered_labels = component_order(labels, stats, min_component_pixels, component_order_strategy)
    for component_index, label in enumerate(ordered_labels):
        component_mask = (labels == label).astype(np.uint8)
        fill_component_mask = erode_mask(component_mask, fill_inset_px)
        component_skeleton = ((skeleton > 0) & (component_mask > 0)).astype(np.uint8) if skeleton is not None else None
        style_report = classify_component_style(
            component_mask,
            component_skeleton,
            style_running_skeleton_ratio,
            style_running_max_distance_px,
            style_running_min_skeleton_pixels,
        ) if use_style_aware_components else {"style": "fill"}
        style = str(style_report["style"])
        skeleton_paths: list[list[tuple[int, int]]] = []
        rows: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if use_style_aware_components and style == "running" and component_skeleton is not None:
            skeleton_paths = component_skeleton_paths(component_skeleton, component_mask, style_running_min_skeleton_pixels)
            if skeleton_paths:
                style_running_components += 1
                style_component_reports.append({"label": int(label), **style_report, "paths": len(skeleton_paths)})
            else:
                rows = component_fill_rows(
                    fill_component_mask,
                    row_spacing_px,
                    min_run_px,
                    reverse_first=component_index % 2 == 1,
                    row_order_strategy=row_order_strategy,
                    entry_px=current_px,
                )
                style = "fill"
                style_fallback_fill_components += 1
        else:
            rows = component_fill_rows(
                fill_component_mask,
                row_spacing_px,
                min_run_px,
                reverse_first=component_index % 2 == 1,
                row_order_strategy=row_order_strategy,
                entry_px=current_px,
            )

        if style != "running" and not rows:
            continue
        if style == "running" and not skeleton_paths:
            continue
        components_used += 1
        if style == "running":
            for path in skeleton_paths:
                current_mm, current_px, path_stats = stitch_point_sequence(
                    pattern,
                    path,
                    current_mm,
                    current_px,
                    connector_mask,
                    mask.shape,
                    target_width_mm,
                    max_stitch_mm,
                    max_connect_mm,
                    min_connect_inside_fraction,
                    use_mask_path_connectors,
                    max_mask_path_px,
                    max_mask_path_expansions,
                    validate_mask_path_segments,
                    mask_path_min_segment_inside_fraction,
                    validate_quantized_segments,
                )
                jumps += path_stats["jumps"]
                safe_connects += path_stats["safe_connects"]
                mask_path_connects += path_stats["mask_path_connects"]
                rejected_connects += path_stats["rejected_connects"]
                rejected_mask_paths += path_stats["rejected_mask_paths"]
                stitch_segments += path_stats["stitch_segments"]
                style_running_paths += 1
                style_running_points += len(path)
            continue

        if use_style_aware_components:
            style_fill_components += 1
            style_component_reports.append({"label": int(label), **style_report, "rows": len(rows)})

        for start_px, end_px in rows:
            start_mm = pixel_to_mm_xy(start_px[0], start_px[1], mask.shape, target_width_mm)
            end_mm = pixel_to_mm_xy(end_px[0], end_px[1], mask.shape, target_width_mm)
            current_mm, current_px, connect_stats = connect_to_point(
                pattern,
                current_mm,
                current_px,
                start_px,
                connector_mask,
                mask.shape,
                target_width_mm,
                max_stitch_mm,
                max_connect_mm,
                min_connect_inside_fraction,
                use_mask_path_connectors,
                max_mask_path_px,
                max_mask_path_expansions,
                validate_mask_path_segments,
                mask_path_min_segment_inside_fraction,
                validate_quantized_segments,
            )
            jumps += connect_stats["jumps"]
            safe_connects += connect_stats["safe_connects"]
            mask_path_connects += connect_stats["mask_path_connects"]
            rejected_connects += connect_stats["rejected_connects"]
            rejected_mask_paths += connect_stats["rejected_mask_paths"]
            stitch_segments += connect_stats["stitch_segments"]
            stitch_segments += add_segment(pattern, start_mm, end_mm, max_stitch_mm)
            fill_rows += 1
            current_mm = end_mm
            current_px = end_px

        if add_outline:
            for path in component_outline_paths(component_mask, outline_min_area_px, outline_stride_px, outline_include_holes, outline_inset_px):
                start_px = path[0]
                current_mm, current_px, connect_stats = connect_to_point(
                    pattern,
                    current_mm,
                    current_px,
                    start_px,
                    connector_mask,
                    mask.shape,
                    target_width_mm,
                    max_stitch_mm,
                    max_connect_mm,
                    min_connect_inside_fraction,
                    use_mask_path_connectors,
                    max_mask_path_px,
                    max_mask_path_expansions,
                    validate_mask_path_segments,
                    mask_path_min_segment_inside_fraction,
                    validate_quantized_segments,
                )
                jumps += connect_stats["jumps"]
                safe_connects += connect_stats["safe_connects"]
                mask_path_connects += connect_stats["mask_path_connects"]
                rejected_connects += connect_stats["rejected_connects"]
                rejected_mask_paths += connect_stats["rejected_mask_paths"]
                stitch_segments += connect_stats["stitch_segments"]
                stitch_segments += add_polyline_px(pattern, path, mask.shape, target_width_mm, max_stitch_mm)
                current_px = path[-1]
                current_mm = pixel_to_mm_xy(current_px[0], current_px[1], mask.shape, target_width_mm)
                outline_paths += 1
                outline_points += len(path)

        if add_satin_border:
            columns = component_satin_columns(
                component_mask,
                satin_min_area_px,
                satin_step_px,
                satin_width_px,
                satin_outer_inset_px,
                satin_max_columns_per_component,
            )
            prefer_outer = True
            for outer_px, inner_px in columns:
                start_px = outer_px if prefer_outer else inner_px
                end_px = inner_px if prefer_outer else outer_px
                start_mm = pixel_to_mm_xy(start_px[0], start_px[1], mask.shape, target_width_mm)
                end_mm = pixel_to_mm_xy(end_px[0], end_px[1], mask.shape, target_width_mm)
                if output_segment_inside_fraction(
                    connector_mask,
                    start_mm,
                    end_mm,
                    target_width_mm,
                    validate_quantized_segments,
                ) < min_connect_inside_fraction:
                    satin_skipped += 1
                    continue
                current_mm, current_px, connect_stats = connect_to_point(
                    pattern,
                    current_mm,
                    current_px,
                    start_px,
                    connector_mask,
                    mask.shape,
                    target_width_mm,
                    max_stitch_mm,
                    max_connect_mm,
                    min_connect_inside_fraction,
                    use_mask_path_connectors,
                    max_mask_path_px,
                    max_mask_path_expansions,
                    validate_mask_path_segments,
                    mask_path_min_segment_inside_fraction,
                    validate_quantized_segments,
                )
                jumps += connect_stats["jumps"]
                safe_connects += connect_stats["safe_connects"]
                mask_path_connects += connect_stats["mask_path_connects"]
                rejected_connects += connect_stats["rejected_connects"]
                rejected_mask_paths += connect_stats["rejected_mask_paths"]
                stitch_segments += connect_stats["stitch_segments"]
                stitch_segments += add_segment(pattern, start_mm, end_mm, max_stitch_mm)
                current_px = end_px
                current_mm = end_mm
                satin_columns += 1
                prefer_outer = not prefer_outer

        if add_satin_rail_border:
            rail_columns = component_satin_columns(
                component_mask,
                satin_rail_min_area_px,
                satin_rail_step_px,
                satin_rail_width_px,
                satin_rail_outer_inset_px,
                satin_rail_max_pairs_per_component,
            )
            rail_sequence: list[tuple[int, int]] = []
            for outer_px, inner_px in rail_columns:
                outer_mm = pixel_to_mm_xy(outer_px[0], outer_px[1], mask.shape, target_width_mm)
                inner_mm = pixel_to_mm_xy(inner_px[0], inner_px[1], mask.shape, target_width_mm)
                if output_segment_inside_fraction(
                    connector_mask,
                    outer_mm,
                    inner_mm,
                    target_width_mm,
                    validate_quantized_segments,
                ) < min_connect_inside_fraction:
                    satin_rail_rejected += 1
                    continue
                rail_sequence.extend([outer_px, inner_px])
                satin_rail_pairs += 1
            current_mm, current_px, rail_stats = stitch_point_sequence(
                pattern,
                rail_sequence,
                current_mm,
                current_px,
                connector_mask,
                mask.shape,
                target_width_mm,
                max_stitch_mm,
                max_connect_mm,
                min_connect_inside_fraction,
                use_mask_path_connectors,
                max_mask_path_px,
                max_mask_path_expansions,
                validate_mask_path_segments,
                mask_path_min_segment_inside_fraction,
                validate_quantized_segments,
            )
            jumps += rail_stats["jumps"]
            safe_connects += rail_stats["safe_connects"]
            mask_path_connects += rail_stats["mask_path_connects"]
            rejected_connects += rail_stats["rejected_connects"]
            rejected_mask_paths += rail_stats["rejected_mask_paths"]
            stitch_segments += rail_stats["stitch_segments"]
            satin_rail_segments += rail_stats["sequence_segments"]
            satin_rail_rejected += rail_stats["sequence_rejected"]

        if add_dt_satin_border:
            dt_columns = component_dt_satin_pairs(
                component_mask,
                dt_satin_min_area_px,
                dt_satin_step_px,
                dt_satin_outer_distance_px,
                dt_satin_inner_distance_px,
                dt_satin_max_pair_px,
                dt_satin_max_pairs_per_component,
            )
            dt_sequence: list[tuple[int, int]] = []
            for outer_px, inner_px in dt_columns:
                outer_mm = pixel_to_mm_xy(outer_px[0], outer_px[1], mask.shape, target_width_mm)
                inner_mm = pixel_to_mm_xy(inner_px[0], inner_px[1], mask.shape, target_width_mm)
                if output_segment_inside_fraction(
                    connector_mask,
                    outer_mm,
                    inner_mm,
                    target_width_mm,
                    validate_quantized_segments,
                ) < min_connect_inside_fraction:
                    dt_satin_rejected += 1
                    continue
                dt_sequence.extend([outer_px, inner_px])
                dt_satin_pairs += 1
            current_mm, current_px, dt_stats = stitch_point_sequence(
                pattern,
                dt_sequence,
                current_mm,
                current_px,
                connector_mask,
                mask.shape,
                target_width_mm,
                max_stitch_mm,
                max_connect_mm,
                min_connect_inside_fraction,
                use_mask_path_connectors,
                max_mask_path_px,
                max_mask_path_expansions,
                validate_mask_path_segments,
                mask_path_min_segment_inside_fraction,
                validate_quantized_segments,
            )
            jumps += dt_stats["jumps"]
            safe_connects += dt_stats["safe_connects"]
            mask_path_connects += dt_stats["mask_path_connects"]
            rejected_connects += dt_stats["rejected_connects"]
            rejected_mask_paths += dt_stats["rejected_mask_paths"]
            stitch_segments += dt_stats["stitch_segments"]
            dt_satin_segments += dt_stats["sequence_segments"]
            dt_satin_rejected += dt_stats["sequence_rejected"]

    if current_mm is None:
        current_mm = (0.0, 0.0)
        add_abs(pattern, JUMP, current_mm)
        jumps += 1
    add_abs(pattern, END, current_mm)
    output_dst.parent.mkdir(parents=True, exist_ok=True)
    write_dst(pattern, str(output_dst))
    return {
        "mask_path": str(mask_path),
        "connector_mask_path": str(connector_mask_path) if connector_mask_path else "",
        "skeleton_path": str(skeleton_path) if skeleton_path else "",
        "output_dst": str(output_dst),
        "components_total": int(count - 1),
        "components_used": components_used,
        "component_order_strategy": component_order_strategy,
        "row_order_strategy": row_order_strategy,
        "fill_rows": fill_rows,
        "jump_commands_added": jumps,
        "safe_connects": safe_connects,
        "mask_path_connects": mask_path_connects,
        "rejected_connects": rejected_connects,
        "rejected_mask_paths": rejected_mask_paths,
        "stitch_segments_added": stitch_segments,
        "target_width_mm": target_width_mm,
        "row_spacing_mm": row_spacing_mm,
        "row_spacing_px": row_spacing_px,
        "max_stitch_mm": max_stitch_mm,
        "max_connect_mm": max_connect_mm,
        "min_connect_inside_fraction": min_connect_inside_fraction,
        "min_component_pixels": min_component_pixels,
        "min_run_mm": min_run_mm,
        "min_run_px": min_run_px,
        "fill_inset_px": fill_inset_px,
        "use_mask_path_connectors": use_mask_path_connectors,
        "max_mask_path_mm": max_mask_path_mm,
        "max_mask_path_px": max_mask_path_px,
        "max_mask_path_expansions": max_mask_path_expansions,
        "validate_mask_path_segments": validate_mask_path_segments,
        "mask_path_min_segment_inside_fraction": mask_path_min_segment_inside_fraction,
        "validate_quantized_segments": validate_quantized_segments,
        "add_outline": add_outline,
        "outline_paths": outline_paths,
        "outline_points": outline_points,
        "outline_stride_px": outline_stride_px,
        "outline_min_area_px": outline_min_area_px,
        "outline_include_holes": outline_include_holes,
        "outline_inset_px": outline_inset_px,
        "add_satin_border": add_satin_border,
        "satin_columns": satin_columns,
        "satin_skipped": satin_skipped,
        "satin_width_px": satin_width_px,
        "satin_step_px": satin_step_px,
        "satin_outer_inset_px": satin_outer_inset_px,
        "satin_min_area_px": satin_min_area_px,
        "satin_max_columns_per_component": satin_max_columns_per_component,
        "add_satin_rail_border": add_satin_rail_border,
        "satin_rail_pairs": satin_rail_pairs,
        "satin_rail_segments": satin_rail_segments,
        "satin_rail_rejected": satin_rail_rejected,
        "satin_rail_width_px": satin_rail_width_px,
        "satin_rail_step_px": satin_rail_step_px,
        "satin_rail_outer_inset_px": satin_rail_outer_inset_px,
        "satin_rail_min_area_px": satin_rail_min_area_px,
        "satin_rail_max_pairs_per_component": satin_rail_max_pairs_per_component,
        "add_dt_satin_border": add_dt_satin_border,
        "dt_satin_pairs": dt_satin_pairs,
        "dt_satin_segments": dt_satin_segments,
        "dt_satin_rejected": dt_satin_rejected,
        "dt_satin_outer_distance_px": dt_satin_outer_distance_px,
        "dt_satin_inner_distance_px": dt_satin_inner_distance_px,
        "dt_satin_step_px": dt_satin_step_px,
        "dt_satin_min_area_px": dt_satin_min_area_px,
        "dt_satin_max_pair_px": dt_satin_max_pair_px,
        "dt_satin_max_pairs_per_component": dt_satin_max_pairs_per_component,
        "use_style_aware_components": use_style_aware_components,
        "style_running_components": style_running_components,
        "style_fill_components": style_fill_components,
        "style_running_paths": style_running_paths,
        "style_running_points": style_running_points,
        "style_fallback_fill_components": style_fallback_fill_components,
        "style_running_skeleton_ratio": style_running_skeleton_ratio,
        "style_running_max_distance_px": style_running_max_distance_px,
        "style_running_min_skeleton_pixels": style_running_min_skeleton_pixels,
        "style_component_reports": style_component_reports[:80],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a simple segment-level fill DST from a binary mask.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--connector-mask", default="")
    parser.add_argument("--skeleton", default="")
    parser.add_argument("--output-dst", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--row-spacing-mm", type=float, default=1.2)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--max-connect-mm", type=float, default=6.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.95)
    parser.add_argument("--min-component-pixels", type=int, default=64)
    parser.add_argument("--min-run-mm", type=float, default=1.0)
    parser.add_argument("--fill-inset-px", type=int, default=0)
    parser.add_argument("--use-mask-path-connectors", action="store_true")
    parser.add_argument("--max-mask-path-mm", type=float, default=24.0)
    parser.add_argument("--max-mask-path-expansions", type=int, default=8000)
    parser.add_argument("--validate-mask-path-segments", action="store_true")
    parser.add_argument("--mask-path-min-segment-inside-fraction", type=float, default=1.0)
    parser.add_argument("--validate-quantized-segments", action="store_true")
    parser.add_argument("--add-outline", action="store_true")
    parser.add_argument("--outline-stride-px", type=int, default=2)
    parser.add_argument("--outline-min-area-px", type=int, default=64)
    parser.add_argument("--outline-external-only", action="store_true")
    parser.add_argument("--outline-inset-px", type=int, default=0)
    parser.add_argument("--add-satin-border", action="store_true")
    parser.add_argument("--satin-width-px", type=int, default=6)
    parser.add_argument("--satin-step-px", type=int, default=5)
    parser.add_argument("--satin-outer-inset-px", type=int, default=2)
    parser.add_argument("--satin-min-area-px", type=int, default=64)
    parser.add_argument("--satin-max-columns-per-component", type=int, default=180)
    parser.add_argument("--add-satin-rail-border", action="store_true")
    parser.add_argument("--satin-rail-width-px", type=int, default=8)
    parser.add_argument("--satin-rail-step-px", type=int, default=7)
    parser.add_argument("--satin-rail-outer-inset-px", type=int, default=4)
    parser.add_argument("--satin-rail-min-area-px", type=int, default=64)
    parser.add_argument("--satin-rail-max-pairs-per-component", type=int, default=180)
    parser.add_argument("--add-dt-satin-border", action="store_true")
    parser.add_argument("--dt-satin-outer-distance-px", type=int, default=3)
    parser.add_argument("--dt-satin-inner-distance-px", type=int, default=9)
    parser.add_argument("--dt-satin-step-px", type=int, default=7)
    parser.add_argument("--dt-satin-min-area-px", type=int, default=64)
    parser.add_argument("--dt-satin-max-pair-px", type=int, default=18)
    parser.add_argument("--dt-satin-max-pairs-per-component", type=int, default=180)
    parser.add_argument("--use-style-aware-components", action="store_true")
    parser.add_argument("--style-running-skeleton-ratio", type=float, default=0.08)
    parser.add_argument("--style-running-max-distance-px", type=float, default=4.5)
    parser.add_argument("--style-running-min-skeleton-pixels", type=int, default=4)
    parser.add_argument("--component-order", choices=["area", "nearest_centroid"], default="area")
    parser.add_argument("--row-order", choices=["serpentine", "nearest_endpoint"], default="serpentine")
    args = parser.parse_args()

    report = generate_mask_fill_dst(
        Path(args.mask),
        Path(args.output_dst),
        connector_mask_path=Path(args.connector_mask) if args.connector_mask else None,
        skeleton_path=Path(args.skeleton) if args.skeleton else None,
        target_width_mm=args.target_width_mm,
        row_spacing_mm=args.row_spacing_mm,
        max_stitch_mm=args.max_stitch_mm,
        max_connect_mm=args.max_connect_mm,
        min_connect_inside_fraction=args.min_connect_inside_fraction,
        min_component_pixels=args.min_component_pixels,
        min_run_mm=args.min_run_mm,
        fill_inset_px=args.fill_inset_px,
        use_mask_path_connectors=args.use_mask_path_connectors,
        max_mask_path_mm=args.max_mask_path_mm,
        max_mask_path_expansions=args.max_mask_path_expansions,
        validate_mask_path_segments=args.validate_mask_path_segments,
        mask_path_min_segment_inside_fraction=args.mask_path_min_segment_inside_fraction,
        validate_quantized_segments=args.validate_quantized_segments,
        add_outline=args.add_outline,
        outline_stride_px=args.outline_stride_px,
        outline_min_area_px=args.outline_min_area_px,
        outline_include_holes=not args.outline_external_only,
        outline_inset_px=args.outline_inset_px,
        add_satin_border=args.add_satin_border,
        satin_width_px=args.satin_width_px,
        satin_step_px=args.satin_step_px,
        satin_outer_inset_px=args.satin_outer_inset_px,
        satin_min_area_px=args.satin_min_area_px,
        satin_max_columns_per_component=args.satin_max_columns_per_component,
        add_satin_rail_border=args.add_satin_rail_border,
        satin_rail_width_px=args.satin_rail_width_px,
        satin_rail_step_px=args.satin_rail_step_px,
        satin_rail_outer_inset_px=args.satin_rail_outer_inset_px,
        satin_rail_min_area_px=args.satin_rail_min_area_px,
        satin_rail_max_pairs_per_component=args.satin_rail_max_pairs_per_component,
        add_dt_satin_border=args.add_dt_satin_border,
        dt_satin_outer_distance_px=args.dt_satin_outer_distance_px,
        dt_satin_inner_distance_px=args.dt_satin_inner_distance_px,
        dt_satin_step_px=args.dt_satin_step_px,
        dt_satin_min_area_px=args.dt_satin_min_area_px,
        dt_satin_max_pair_px=args.dt_satin_max_pair_px,
        dt_satin_max_pairs_per_component=args.dt_satin_max_pairs_per_component,
        use_style_aware_components=args.use_style_aware_components,
        style_running_skeleton_ratio=args.style_running_skeleton_ratio,
        style_running_max_distance_px=args.style_running_max_distance_px,
        style_running_min_skeleton_pixels=args.style_running_min_skeleton_pixels,
        component_order_strategy=args.component_order,
        row_order_strategy=args.row_order,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
