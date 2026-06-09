from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from infer_image_to_stitch_label import (
    axis_sorted_path,
    boundary_path,
    component_orientation,
    connected_components,
    export_color_planner_as_dst,
    grayscale_image,
    hatch_component,
    image_color_layers,
    overlay_mask,
    resize_with_pad,
    stitch_type_image,
)
from pyembroidery import COLOR_CHANGE, END, JUMP, STITCH, TRIM, EmbPattern, EmbThread, write_dst, write_pes
from train_image_to_stitch_label import TinyUNet
from train_model7_geometry_planner import Model7GeometryPlannerCascade
from train_model8_joint_segment import Model8JointSegmentPlanner
from train_model10_vector_continuity import Model10VectorContinuityPlanner


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))[None]


def direction_from_axis(raw_axis: np.ndarray) -> np.ndarray:
    angle = 0.5 * np.arctan2(raw_axis[1], raw_axis[0])
    return np.stack([np.cos(angle), np.sin(angle)], axis=0).astype(np.float32)


def direction_image(direction: np.ndarray, mask: np.ndarray) -> Image.Image:
    dx = (direction[0] + 1.0) * 0.5
    dy = (direction[1] + 1.0) * 0.5
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(dx * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(dy * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(mask * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def heatmap_preview(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (arr * 255).astype(np.uint8)
    rgb[..., 1] = (np.sqrt(arr) * 190).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def grayscale_from_float(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")


def largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas)) + 1
    return labels == largest


def portrait_foreground_mask(image: Image.Image, threshold: float, close_px: int) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = rgb.shape
    border = max(8, int(min(h, w) * 0.08))
    samples = np.concatenate(
        [
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0

    fg = (dist > threshold) | ((value < 0.42) & (dist > threshold * 0.45)) | ((saturation > 0.28) & (dist > threshold * 0.55))
    kernel_size = max(3, close_px | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = largest_component(fg)
    fg = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    return fg


def save_panel(
    input_image: Image.Image,
    mask: np.ndarray,
    density: np.ndarray,
    foreground: np.ndarray,
    hybrid: np.ndarray,
    stitch_type: np.ndarray,
    endpoint: np.ndarray,
    direction: np.ndarray,
    output_path: Path,
) -> None:
    items = [
        ("input", input_image),
        ("model mask", grayscale_image(mask)),
        ("density", grayscale_image(density)),
        ("foreground", grayscale_image(foreground.astype(np.float32))),
        ("hybrid", grayscale_image(hybrid.astype(np.float32))),
        ("stitch type", stitch_type_image(stitch_type)),
        ("endpoints", heatmap_preview(endpoint)),
        ("direction", direction_image(direction, hybrid.astype(np.float32))),
        ("overlay", overlay_mask(input_image, hybrid.astype(np.float32), 0.5)),
    ]
    width, height = input_image.size
    panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for index, (label, item) in enumerate(items):
        x = index * width
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        panel.paste(item, (x, 24))
    panel.save(output_path)


def coord_to_mm(point: tuple[float, float], width: int, height: int, scale_mm: float) -> tuple[float, float]:
    return ((point[0] - width / 2.0) * scale_mm, (point[1] - height / 2.0) * scale_mm)


def mm_to_coord(point: tuple[float, float], width: int, height: int, scale_mm: float) -> tuple[float, float]:
    return (point[0] / scale_mm + width / 2.0, point[1] / scale_mm + height / 2.0)


def distance2_mm(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def bbox_distance2_mm(
    current: tuple[float, float],
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    scale_mm: float,
) -> float:
    x0, y0 = coord_to_mm((bbox[0], bbox[1]), width, height, scale_mm)
    x1, y1 = coord_to_mm((bbox[2], bbox[3]), width, height, scale_mm)
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    dx = max(0.0, xmin - current[0], current[0] - xmax)
    dy = max(0.0, ymin - current[1], current[1] - ymax)
    return dx * dx + dy * dy


def component_bbox(component: np.ndarray) -> tuple[float, float, float, float]:
    yy, xx = np.nonzero(component)
    if yy.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xx.min()), float(yy.min()), float(xx.max()), float(yy.max()))


def polyline_endpoints_mm(
    coords: list[tuple[float, float]],
    width: int,
    height: int,
    scale_mm: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return coord_to_mm(coords[0], width, height, scale_mm), coord_to_mm(coords[-1], width, height, scale_mm)


def best_polyline_distance2(
    current: tuple[float, float],
    coords: list[tuple[float, float]],
    width: int,
    height: int,
    scale_mm: float,
) -> tuple[float, bool]:
    start, end = polyline_endpoints_mm(coords, width, height, scale_mm)
    ds = distance2_mm(current, start)
    de = distance2_mm(current, end)
    return (de, True) if de < ds else (ds, False)


def transition_cost_mm(distance_mm: float, long_jump_threshold_mm: float, long_jump_weight: float) -> float:
    excess = max(0.0, distance_mm - long_jump_threshold_mm)
    return distance_mm + long_jump_weight * excess


def line_map_mean(
    score_map: np.ndarray | None,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    samples: int = 32,
) -> float:
    if score_map is None:
        return 0.0
    h, w = score_map.shape
    xs = np.linspace(start_xy[0], end_xy[0], max(2, samples))
    ys = np.linspace(start_xy[1], end_xy[1], max(2, samples))
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return float(score_map[yi, xi].mean())


def continuity_adjusted_transition(
    current: tuple[float, float],
    coords: list[tuple[float, float]],
    width: int,
    height: int,
    scale_mm: float,
    long_jump_threshold_mm: float,
    long_jump_weight: float,
    continuity_near: np.ndarray | None = None,
    continuity_jump: np.ndarray | None = None,
    continuity_order_weight: float = 0.0,
    jump_endpoint_weight: float = 0.0,
) -> tuple[float, bool, float]:
    base_cost, reverse, distance = best_polyline_transition(
        current,
        coords,
        width,
        height,
        scale_mm,
        long_jump_threshold_mm,
        long_jump_weight,
    )
    first = coords[-1] if reverse else coords[0]
    current_px = mm_to_coord(current, width, height, scale_mm)
    near_signal = line_map_mean(continuity_near, current_px, first)
    jump_signal = line_map_mean(continuity_jump, current_px, first)
    adjusted = base_cost - continuity_order_weight * near_signal * long_jump_threshold_mm
    adjusted += jump_endpoint_weight * jump_signal * max(distance, long_jump_threshold_mm)
    return adjusted, reverse, distance


def best_polyline_transition(
    current: tuple[float, float],
    coords: list[tuple[float, float]],
    width: int,
    height: int,
    scale_mm: float,
    long_jump_threshold_mm: float,
    long_jump_weight: float,
) -> tuple[float, bool, float]:
    start, end = polyline_endpoints_mm(coords, width, height, scale_mm)
    start_distance = float(np.sqrt(distance2_mm(current, start)))
    end_distance = float(np.sqrt(distance2_mm(current, end)))
    start_cost = transition_cost_mm(start_distance, long_jump_threshold_mm, long_jump_weight)
    end_cost = transition_cost_mm(end_distance, long_jump_threshold_mm, long_jump_weight)
    return (end_cost, True, end_distance) if end_cost < start_cost else (start_cost, False, start_distance)


def polyline_sequence_cost(
    polylines: list[list[tuple[float, float]]],
    current: tuple[float, float],
    width: int,
    height: int,
    scale_mm: float,
    long_jump_threshold_mm: float,
    long_jump_weight: float,
) -> float:
    total = 0.0
    last = current
    for coords in polylines:
        if len(coords) < 2:
            continue
        first = coord_to_mm(coords[0], width, height, scale_mm)
        distance = float(np.sqrt(distance2_mm(last, first)))
        total += transition_cost_mm(distance, long_jump_threshold_mm, long_jump_weight)
        last = coord_to_mm(coords[-1], width, height, scale_mm)
    return total


def two_opt_polylines(
    polylines: list[list[tuple[float, float]]],
    current: tuple[float, float],
    width: int,
    height: int,
    scale_mm: float,
    long_jump_threshold_mm: float,
    long_jump_weight: float,
    max_passes: int,
) -> list[list[tuple[float, float]]]:
    if len(polylines) < 4 or max_passes <= 0:
        return polylines
    best = [list(coords) for coords in polylines]
    best_cost = polyline_sequence_cost(best, current, width, height, scale_mm, long_jump_threshold_mm, long_jump_weight)
    for _ in range(max_passes):
        improved = False
        for i in range(0, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + [list(reversed(coords)) for coords in reversed(best[i : j + 1])] + best[j + 1 :]
                cost = polyline_sequence_cost(candidate, current, width, height, scale_mm, long_jump_threshold_mm, long_jump_weight)
                if cost + 1e-6 < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
        if not improved:
            break
    return best


def order_polylines_serpentine(
    polylines: list[list[tuple[float, float]]],
    current: tuple[float, float],
    angle: float,
    width: int,
    height: int,
    scale_mm: float,
    long_jump_threshold_mm: float,
    long_jump_weight: float,
    two_opt_passes: int,
) -> list[list[tuple[float, float]]]:
    remaining = [coords for coords in polylines if len(coords) >= 2]
    if not remaining:
        return []
    normal = np.array([-np.sin(angle), np.cos(angle)], dtype=np.float32)

    def row_key(coords: list[tuple[float, float]]) -> float:
        arr = np.asarray(coords, dtype=np.float32)
        return float((arr @ normal).mean())

    ordered = sorted(remaining, key=row_key)
    oriented: list[list[tuple[float, float]]] = []
    last = current
    for coords in ordered:
        _cost, reverse, _dist = best_polyline_transition(
            last,
            coords,
            width,
            height,
            scale_mm,
            long_jump_threshold_mm,
            long_jump_weight,
        )
        if reverse:
            coords = list(reversed(coords))
        oriented.append(coords)
        last = coord_to_mm(coords[-1], width, height, scale_mm)
    return two_opt_polylines(oriented, current, width, height, scale_mm, long_jump_threshold_mm, long_jump_weight, two_opt_passes)


def add_polyline_nearest(
    pattern: EmbPattern,
    coords: list[tuple[float, float]],
    current: tuple[float, float],
    width: int,
    height: int,
    scale_mm: float,
    connect_near_mm: float = 0.0,
    continuity_near: np.ndarray | None = None,
    continuity_connect_threshold: float = 0.0,
    continuity_connect_max_mm: float = 0.0,
    max_stitch_mm: float = 4.0,
    max_jump_mm: float = 7.5,
    trim_jump_threshold_mm: float = 10.0,
) -> tuple[int, int, int, int, float, float, tuple[float, float]]:
    if len(coords) < 2:
        return 0, 0, 0, 0, 0.0, 0.0, current
    _dist2, reverse = best_polyline_distance2(current, coords, width, height, scale_mm)
    if reverse:
        coords = list(reversed(coords))
    first = coord_to_mm(coords[0], width, height, scale_mm)
    jump_mm = float(np.sqrt(distance2_mm(current, first)))
    continuity_connect = False
    if continuity_near is not None and continuity_connect_threshold > 0.0 and continuity_connect_max_mm > 0.0:
        current_px = mm_to_coord(current, width, height, scale_mm)
        near_signal = line_map_mean(continuity_near, current_px, coords[0])
        continuity_connect = bool(jump_mm <= continuity_connect_max_mm and near_signal >= continuity_connect_threshold)
    stitch_count = 0
    jump_count = 0
    trim_count = 0
    connector_count = 0
    if (connect_near_mm > 0.0 and jump_mm <= connect_near_mm) or continuity_connect:
        added, _distance = add_stitch_segment(pattern, current, first, max_stitch_mm)
        stitch_count += added
        connector_count += 1
    else:
        added_jumps, added_trims, _distance = add_jump_segment(
            pattern,
            current,
            first,
            max_jump_mm=max_jump_mm,
            trim_jump_threshold_mm=trim_jump_threshold_mm,
        )
        jump_count += added_jumps
        trim_count += added_trims
    path_mm = jump_mm if connector_count else 0.0
    last = first
    current_mm = first
    for point in coords:
        current_mm = coord_to_mm(point, width, height, scale_mm)
        added, distance = add_stitch_segment(pattern, last, current_mm, max_stitch_mm)
        path_mm += distance
        stitch_count += added
        last = current_mm
    return stitch_count, jump_count, trim_count, connector_count, jump_mm, path_mm, current_mm


def add_stitch_segment(
    pattern: EmbPattern,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    max_stitch_mm: float,
) -> tuple[int, float]:
    distance = float(np.sqrt(distance2_mm(start_mm, end_mm)))
    if distance <= 1e-6:
        return 0, 0.0
    steps = max(1, int(np.ceil(distance / max(0.1, max_stitch_mm))))
    for step in range(1, steps + 1):
        t = step / steps
        x = start_mm[0] + (end_mm[0] - start_mm[0]) * t
        y = start_mm[1] + (end_mm[1] - start_mm[1]) * t
        pattern.add_stitch_absolute(STITCH, int(round(x * 10)), int(round(y * 10)))
    return steps, distance


def add_jump_segment(
    pattern: EmbPattern,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    max_jump_mm: float,
    trim_jump_threshold_mm: float,
) -> tuple[int, int, float]:
    distance = float(np.sqrt(distance2_mm(start_mm, end_mm)))
    if distance <= 1e-6:
        return 0, 0, 0.0
    trim_count = 0
    if trim_jump_threshold_mm > 0.0 and distance >= trim_jump_threshold_mm:
        pattern.add_stitch_absolute(TRIM, int(round(start_mm[0] * 10)), int(round(start_mm[1] * 10)))
        trim_count = 1
    steps = max(1, int(np.ceil(distance / max(0.1, max_jump_mm))))
    for step in range(1, steps + 1):
        t = step / steps
        x = start_mm[0] + (end_mm[0] - start_mm[0]) * t
        y = start_mm[1] + (end_mm[1] - start_mm[1]) * t
        pattern.add_stitch_absolute(JUMP, int(round(x * 10)), int(round(y * 10)))
    return steps, trim_count, distance


def order_polylines_nearest(
    polylines: list[list[tuple[float, float]]],
    current: tuple[float, float],
    width: int,
    height: int,
    scale_mm: float,
    long_jump_threshold_mm: float = 8.0,
    long_jump_weight: float = 1.2,
    two_opt_passes: int = 0,
    continuity_near: np.ndarray | None = None,
    continuity_jump: np.ndarray | None = None,
    continuity_order_weight: float = 0.0,
    jump_endpoint_weight: float = 0.0,
) -> list[list[tuple[float, float]]]:
    remaining = [coords for coords in polylines if len(coords) >= 2]
    ordered: list[list[tuple[float, float]]] = []
    start_current = current
    while remaining:
        best_index = 0
        best_reverse = False
        best_cost = float("inf")
        for index, coords in enumerate(remaining):
            cost, reverse, _distance = continuity_adjusted_transition(
                current,
                coords,
                width,
                height,
                scale_mm,
                long_jump_threshold_mm,
                long_jump_weight,
                continuity_near,
                continuity_jump,
                continuity_order_weight,
                jump_endpoint_weight,
            )
            if cost < best_cost:
                best_cost = cost
                best_index = index
                best_reverse = reverse
        coords = remaining.pop(best_index)
        if best_reverse:
            coords = list(reversed(coords))
        ordered.append(coords)
        current = coord_to_mm(coords[-1], width, height, scale_mm)
    return two_opt_polylines(ordered, start_current, width, height, scale_mm, long_jump_threshold_mm, long_jump_weight, two_opt_passes)


def export_color_planner_geometry_as_dst(
    mask: np.ndarray,
    density: np.ndarray,
    direction: np.ndarray,
    stitch_type: np.ndarray,
    centerline: np.ndarray,
    path_order: np.ndarray | None,
    input_image: Image.Image,
    output_base: Path,
    threshold: float,
    target_width_mm: float,
    row_step_px: int,
    point_step_px: int,
    min_run_px: int,
    min_component_px: int,
    max_components: int,
    max_colors: int,
    min_color_px: int,
    add_outline: bool,
    outline_step_px: int,
    long_jump_threshold_mm: float = 8.0,
    long_jump_weight: float = 0.0,
    two_opt_passes: int = 0,
    serpentine_fill: bool = False,
    connect_near_mm: float = 0.0,
    continuity_near: np.ndarray | None = None,
    continuity_jump: np.ndarray | None = None,
    continuity_order_weight: float = 0.0,
    jump_endpoint_weight: float = 0.0,
    continuity_connect_threshold: float = 0.0,
    continuity_connect_max_mm: float = 0.0,
    max_stitch_mm: float = 4.0,
    max_jump_mm: float = 7.5,
    trim_jump_threshold_mm: float = 10.0,
) -> dict[str, int | float | str | dict[str, int]]:
    active = mask >= threshold
    height, width = active.shape
    scale_mm = target_width_mm / width
    layers = image_color_layers(input_image, active, max_colors=max_colors, min_color_px=min_color_px)

    pattern = EmbPattern()
    if not layers:
        thread = EmbThread()
        thread.set_color(5, 85, 150)
        pattern.add_thread(thread)
    for _, color, _area in layers:
        thread = EmbThread()
        thread.set_color(*color)
        pattern.add_thread(thread)

    current = (0.0, 0.0)
    stitch_count = 0
    jump_count = 0
    jump_distance_mm = 0.0
    observed_max_jump_mm = 0.0
    stitch_path_mm = 0.0
    color_changes = 0
    trim_count = 0
    component_count = 0
    outline_count = 0
    connector_count = 0
    type_counts = {"running": 0, "satin": 0, "fill": 0}

    for color_index, (layer_mask, _color, _area) in enumerate(layers):
        if color_index > 0:
            pattern.add_stitch_absolute(COLOR_CHANGE, int(round(current[0] * 10)), int(round(current[1] * 10)))
            color_changes += 1

        tasks: list[dict[str, object]] = []
        for type_id, type_name in ((3, "fill"), (2, "satin"), (1, "running")):
            type_mask = layer_mask & (stitch_type == type_id)
            if not np.any(type_mask):
                continue
            for component in connected_components(type_mask, min_component_px):
                angle = component_orientation(component, direction, density)
                polylines: list[list[tuple[float, float]]] = []
                if type_id == 1:
                    run_source = component & (centerline >= 0.35)
                    if int(run_source.sum()) < max(8, min_run_px):
                        run_source = component
                    path = axis_sorted_path(run_source, step_px=max(1, point_step_px))
                    if path:
                        polylines.append(path)
                elif type_id == 2:
                    polylines.extend(
                        hatch_component(
                            component,
                            angle=angle + np.pi / 2.0,
                            spacing_px=max(1, row_step_px - 1),
                            point_step_px=max(1, point_step_px),
                            min_run_px=max(2, min_run_px - 1),
                        )
                    )
                else:
                    polylines.extend(
                        hatch_component(
                            component,
                            angle=angle,
                            spacing_px=row_step_px,
                            point_step_px=point_step_px,
                            min_run_px=min_run_px,
                        )
                    )
                if add_outline and type_id != 1:
                    outline = boundary_path(component, step_px=outline_step_px)
                    if outline:
                        polylines.append(outline)
                        outline_count += len(outline)
                polylines = [coords for coords in polylines if len(coords) >= 2]
                if not polylines:
                    continue
                tasks.append(
                    {
                        "type_name": type_name,
                        "type_id": type_id,
                        "area": int(component.sum()),
                        "angle": float(angle),
                        "order_score": float(path_order[component].mean()) if path_order is not None and np.any(component) else 1.0,
                        "bbox": component_bbox(component),
                        "polylines": polylines,
                    }
                )
        if path_order is not None:
            tasks.sort(key=lambda item: (float(item["order_score"]), -int(item["area"])))
        else:
            tasks.sort(key=lambda item: int(item["area"]), reverse=True)
        if max_components > 0:
            remaining_slots = max(0, max_components - component_count)
            tasks = tasks[:remaining_slots]

        while tasks:
            best_index = 0
            best_distance = float("inf")
            if path_order is not None:
                search_window = min(len(tasks), 8)
                candidates = range(search_window)
            else:
                candidates = range(len(tasks))
            for index in candidates:
                task = tasks[index]
                bbox_d2 = bbox_distance2_mm(current, task["bbox"], width, height, scale_mm)
                if bbox_d2 > best_distance:
                    continue
                task_best = float("inf")
                for coords in task["polylines"]:
                    cost, _reverse, _distance = continuity_adjusted_transition(
                        current,
                        coords,
                        width,
                        height,
                        scale_mm,
                        long_jump_threshold_mm,
                        long_jump_weight,
                        continuity_near,
                        continuity_jump,
                        continuity_order_weight,
                        jump_endpoint_weight,
                    )
                    if cost < task_best:
                        task_best = cost
                if task_best < best_distance:
                    best_distance = task_best
                    best_index = index
            task = tasks.pop(best_index)
            type_name = str(task["type_name"])
            type_counts[type_name] += 1
            component_count += 1
            if serpentine_fill and int(task["type_id"]) in (2, 3):
                ordered_polylines = order_polylines_serpentine(
                    task["polylines"],
                    current,
                    float(task["angle"]),
                    width,
                    height,
                    scale_mm,
                    long_jump_threshold_mm,
                    long_jump_weight,
                    two_opt_passes,
                )
            else:
                ordered_polylines = order_polylines_nearest(
                    task["polylines"],
                    current,
                    width,
                    height,
                    scale_mm,
                    long_jump_threshold_mm,
                    long_jump_weight,
                    two_opt_passes,
                    continuity_near=continuity_near,
                    continuity_jump=continuity_jump,
                    continuity_order_weight=continuity_order_weight,
                    jump_endpoint_weight=jump_endpoint_weight,
                )
            for coords in ordered_polylines:
                sc, jc, tc, cc, jump_mm, path_mm, current = add_polyline_nearest(
                    pattern,
                    coords,
                    current,
                    width,
                    height,
                    scale_mm,
                    connect_near_mm=connect_near_mm,
                    continuity_near=continuity_near,
                    continuity_connect_threshold=continuity_connect_threshold,
                    continuity_connect_max_mm=continuity_connect_max_mm,
                    max_stitch_mm=max_stitch_mm,
                    max_jump_mm=max_jump_mm,
                    trim_jump_threshold_mm=trim_jump_threshold_mm,
                )
                stitch_count += sc
                jump_count += jc
                trim_count += tc
                connector_count += cc
                if jc:
                    jump_distance_mm += jump_mm
                    observed_max_jump_mm = max(observed_max_jump_mm, jump_mm)
                stitch_path_mm += path_mm

    pattern.add_stitch_absolute(END, int(round(current[0] * 10)), int(round(current[1] * 10)))
    write_dst(pattern, str(output_base.with_suffix(".dst")))
    write_pes(pattern, str(output_base.with_suffix(".pes")))
    return {
        "dst": str(output_base.with_suffix(".dst")),
        "pes": str(output_base.with_suffix(".pes")),
        "stitches": stitch_count,
        "jumps": jump_count,
        "jump_distance_mm": round(jump_distance_mm, 3),
        "max_jump_mm": round(observed_max_jump_mm, 3),
        "stitch_path_mm": round(stitch_path_mm, 3),
        "color_layers": len(layers),
        "color_changes": color_changes,
        "trims": trim_count,
        "components": component_count,
        "continuity_connectors": connector_count,
        "outline_stitches": outline_count,
        "type_component_counts": type_counts,
        "target_width_mm": target_width_mm,
        "threshold": threshold,
        "export_mode": "geometry_longjump_serpentine_2opt_planner",
        "model_guided_path_order": path_order is not None,
        "long_jump_threshold_mm": long_jump_threshold_mm,
        "long_jump_weight": long_jump_weight,
        "two_opt_passes": two_opt_passes,
        "serpentine_fill": serpentine_fill,
        "connect_near_mm": connect_near_mm,
        "continuity_order_weight": continuity_order_weight,
        "jump_endpoint_weight": jump_endpoint_weight,
        "continuity_connect_threshold": continuity_connect_threshold,
        "continuity_connect_max_mm": continuity_connect_max_mm,
        "max_stitch_mm": max_stitch_mm,
        "max_jump_mm_limit": max_jump_mm,
        "trim_jump_threshold_mm": trim_jump_threshold_mm,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    target_channels = checkpoint["target_channels"]
    model_name = checkpoint.get("model", "TinyUNet")
    if model_name == "Model7GeometryPlannerCascade":
        model = Model7GeometryPlannerCascade(
            base_channels=int(checkpoint["base_channels"]),
            geometry_channels=int(checkpoint.get("geometry_channels", 13)),
            detach_planner_geometry=False,
        ).to(device)
    elif model_name in {"Model8JointSegmentPlanner", "Model9HardJointSegmentPlanner"}:
        model = Model8JointSegmentPlanner(
            base_channels=int(checkpoint["base_channels"]),
            geometry_channels=int(checkpoint.get("geometry_channels", 13)),
            detach_planner_geometry=False,
        ).to(device)
    elif model_name == "Model10VectorContinuityPlanner":
        model = Model10VectorContinuityPlanner(
            base_channels=int(checkpoint["base_channels"]),
            geometry_channels=int(checkpoint.get("geometry_channels", 13)),
            detach_planner_geometry=False,
        ).to(device)
    else:
        model = TinyUNet(out_channels=len(target_channels), base_channels=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_image = resize_with_pad(input_path, args.size)
    with torch.no_grad():
        output = model(image_to_tensor(input_image).to(device))[0].detach().cpu()

    mask = torch.sigmoid(output[0]).numpy()
    density = torch.sigmoid(output[1]).numpy()
    direction = direction_from_axis(torch.tanh(output[2:4]).numpy())
    boundary = torch.sigmoid(output[5]).numpy()
    centerline = torch.sigmoid(output[6]).numpy()
    stitch_type = output[7:11].argmax(dim=0).numpy().astype(np.uint8)
    endpoint = np.maximum(torch.sigmoid(output[11]).numpy(), torch.sigmoid(output[12]).numpy())
    path_order = torch.sigmoid(output[13]).numpy() if output.shape[0] > 13 and args.model_path_order else None
    continuity = torch.sigmoid(output[17:20]).numpy() if output.shape[0] >= 20 else None

    foreground = portrait_foreground_mask(input_image, threshold=args.foreground_threshold, close_px=args.foreground_close_px)
    active = foreground & (density >= args.density_threshold)
    if int(active.sum()) < args.min_active_px:
        active = foreground & (mask >= args.mask_threshold)
    stitch_type_hybrid = stitch_type.copy()
    stitch_type_hybrid[~active] = 0
    stitch_type_hybrid[active & (stitch_type_hybrid == 0)] = 3

    input_image.save(output_dir / "input_256.png")
    grayscale_image(mask).save(output_dir / "pred_mask.png")
    grayscale_image(density).save(output_dir / "pred_density.png")
    grayscale_image(boundary).save(output_dir / "pred_boundary.png")
    grayscale_image(centerline).save(output_dir / "pred_centerline.png")
    grayscale_image(foreground.astype(np.float32)).save(output_dir / "foreground_mask.png")
    grayscale_image(active.astype(np.float32)).save(output_dir / "hybrid_export_mask.png")
    stitch_type_image(stitch_type).save(output_dir / "pred_stitch_type_raw.png")
    stitch_type_image(stitch_type_hybrid).save(output_dir / "pred_stitch_type.png")
    heatmap_preview(endpoint).save(output_dir / "pred_endpoint_heatmap.png")
    if path_order is not None:
        grayscale_from_float(path_order).save(output_dir / "pred_path_order.png")
    if continuity is not None:
        grayscale_from_float(continuity[0]).save(output_dir / "pred_stitch_trace.png")
        grayscale_from_float(continuity[1]).save(output_dir / "pred_near_connect.png")
        grayscale_from_float(continuity[2]).save(output_dir / "pred_jump_endpoint.png")
    direction_image(direction, active.astype(np.float32)).save(output_dir / "pred_direction_hybrid.png")
    overlay_mask(input_image, active.astype(np.float32), 0.5).save(output_dir / "hybrid_overlay.png")
    save_panel(
        input_image,
        mask=mask,
        density=density,
        foreground=foreground,
        hybrid=active,
        stitch_type=stitch_type_hybrid,
        endpoint=endpoint,
        direction=direction,
        output_path=output_dir / "prediction_panel.png",
    )

    exporter = export_color_planner_geometry_as_dst if args.geometry_planner else export_color_planner_as_dst
    export_kwargs = {
        "mask": active.astype(np.float32),
        "density": density,
        "direction": direction,
        "stitch_type": stitch_type_hybrid,
        "centerline": centerline,
        "input_image": input_image,
        "output_base": output_dir / args.output_prefix,
        "threshold": 0.5,
        "target_width_mm": args.target_width_mm,
        "row_step_px": args.row_step_px,
        "point_step_px": args.point_step_px,
        "min_run_px": args.min_run_px,
        "min_component_px": args.min_component_px,
        "max_components": args.max_components,
        "max_colors": args.max_colors,
        "min_color_px": args.min_color_px,
        "add_outline": args.add_outline,
        "outline_step_px": args.outline_step_px,
    }
    if args.geometry_planner:
        export_kwargs.update(
            {
                "path_order": path_order,
                "long_jump_threshold_mm": args.long_jump_threshold_mm,
                "long_jump_weight": args.long_jump_weight,
                "two_opt_passes": args.two_opt_passes,
                "serpentine_fill": args.serpentine_fill and not args.no_serpentine_fill,
                "connect_near_mm": args.connect_near_mm,
                "continuity_near": continuity[1] if continuity is not None and args.use_continuity_planner else None,
                "continuity_jump": continuity[2] if continuity is not None and args.use_continuity_planner else None,
                "continuity_order_weight": args.continuity_order_weight if args.use_continuity_planner else 0.0,
                "jump_endpoint_weight": args.jump_endpoint_weight if args.use_continuity_planner else 0.0,
                "continuity_connect_threshold": args.continuity_connect_threshold if args.use_continuity_planner else 0.0,
                "continuity_connect_max_mm": args.continuity_connect_max_mm if args.use_continuity_planner else 0.0,
                "max_stitch_mm": args.max_stitch_mm,
                "max_jump_mm": args.max_jump_mm,
                "trim_jump_threshold_mm": args.trim_jump_threshold_mm,
            }
        )
    dst_summary = exporter(**export_kwargs)
    counts = {str(k): int(v) for k, v in zip(*np.unique(stitch_type_hybrid, return_counts=True))}
    summary = {
        "input": str(input_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "foreground_pixels": int(foreground.sum()),
        "hybrid_pixels": int(active.sum()),
        "mask_mean": float(mask.mean()),
        "density_mean": float(density.mean()),
        "stitch_type_counts": counts,
        "dst_summary": dst_summary,
        "files": [
            "input_256.png",
            "prediction_panel.png",
            "hybrid_overlay.png",
            "hybrid_export_mask.png",
            "pred_stitch_type.png",
            "pred_endpoint_heatmap.png",
            "pred_path_order.png" if path_order is not None else "",
            f"{args.output_prefix}.dst",
            f"{args.output_prefix}.pes",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model3 portrait hybrid inference.")
    parser.add_argument("input")
    parser.add_argument("--checkpoint", default="models/model3_geometry_graph_unet_gpu_e20_init_model2/best_model3_geometry_graph_unet.pt")
    parser.add_argument("--output-dir", default="outputs/model3_gpu_portrait_hybrid")
    parser.add_argument("--output-prefix", default="embroidery_output")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--foreground-threshold", type=float, default=0.115)
    parser.add_argument("--foreground-close-px", type=int, default=9)
    parser.add_argument("--density-threshold", type=float, default=0.52)
    parser.add_argument("--mask-threshold", type=float, default=0.62)
    parser.add_argument("--min-active-px", type=int, default=8000)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--row-step-px", type=int, default=2)
    parser.add_argument("--point-step-px", type=int, default=2)
    parser.add_argument("--min-run-px", type=int, default=2)
    parser.add_argument("--min-component-px", type=int, default=28)
    parser.add_argument("--max-components", type=int, default=60)
    parser.add_argument("--max-colors", type=int, default=6)
    parser.add_argument("--min-color-px", type=int, default=80)
    parser.add_argument("--add-outline", action="store_true")
    parser.add_argument("--outline-step-px", type=int, default=3)
    parser.add_argument("--geometry-planner", action="store_true")
    parser.add_argument("--model-path-order", action="store_true")
    parser.add_argument("--long-jump-threshold-mm", type=float, default=8.0)
    parser.add_argument("--long-jump-weight", type=float, default=0.0)
    parser.add_argument("--two-opt-passes", type=int, default=0)
    parser.add_argument("--connect-near-mm", type=float, default=0.0)
    parser.add_argument("--use-continuity-planner", action="store_true")
    parser.add_argument("--continuity-order-weight", type=float, default=0.7)
    parser.add_argument("--jump-endpoint-weight", type=float, default=0.35)
    parser.add_argument("--continuity-connect-threshold", type=float, default=0.28)
    parser.add_argument("--continuity-connect-max-mm", type=float, default=5.0)
    parser.add_argument("--max-stitch-mm", type=float, default=4.0)
    parser.add_argument("--max-jump-mm", type=float, default=7.5)
    parser.add_argument("--trim-jump-threshold-mm", type=float, default=10.0)
    parser.add_argument("--serpentine-fill", action="store_true")
    parser.add_argument("--no-serpentine-fill", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
