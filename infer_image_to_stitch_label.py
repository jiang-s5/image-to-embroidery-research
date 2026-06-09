from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from pyembroidery import COLOR_CHANGE, END, JUMP, STITCH, EmbPattern, EmbThread, write_dst, write_pes

from train_image_to_stitch_label import TinyUNet


STITCH_TYPE_RGB = np.asarray(
    [
        (0, 0, 0),
        (80, 190, 255),
        (255, 190, 70),
        (180, 120, 230),
    ],
    dtype=np.uint8,
)
STITCH_TYPE_NAMES = ["no_stitch", "running", "satin", "fill"]


def resize_with_pad(image_path: Path, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))[None]


def grayscale_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def direction_image(direction: np.ndarray, mask: np.ndarray) -> Image.Image:
    dx = (direction[0] + 1.0) * 0.5
    dy = (direction[1] + 1.0) * 0.5
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(dx * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(dy * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(mask * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def stitch_type_image(stitch_type: np.ndarray) -> Image.Image:
    labels = np.clip(stitch_type, 0, len(STITCH_TYPE_RGB) - 1)
    return Image.fromarray(STITCH_TYPE_RGB[labels], mode="RGB")


def overlay_mask(image: Image.Image, mask: np.ndarray, threshold: float) -> Image.Image:
    base = image.convert("RGBA")
    alpha = np.where(mask >= threshold, 120, 0).astype(np.uint8)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = 10
    overlay[..., 1] = 105
    overlay[..., 2] = 180
    overlay[..., 3] = alpha
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def save_panel(
    input_image: Image.Image,
    mask: np.ndarray,
    density: np.ndarray,
    direction: np.ndarray,
    overlay_source: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    items = [
        ("input", input_image),
        ("pred mask", grayscale_image(mask)),
        ("pred density", grayscale_image(density)),
        ("pred direction", direction_image(direction, mask)),
        ("overlay", overlay_mask(input_image, overlay_source, threshold)),
    ]
    width, height = input_image.size
    panel = Image.new("RGB", (width * len(items), height + 24), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for index, (label, item) in enumerate(items):
        x = index * width
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
        panel.paste(item, (x, 24))
    panel.save(output_path)


def export_mask_as_dst(
    mask: np.ndarray,
    output_base: Path,
    threshold: float,
    target_width_mm: float,
    row_step_px: int,
    point_step_px: int,
    min_run_px: int,
) -> dict[str, int | float | str]:
    active = mask >= threshold
    height, width = active.shape
    scale_mm = target_width_mm / width

    pattern = EmbPattern()
    thread = EmbThread()
    thread.set_color(5, 85, 150)
    pattern.add_thread(thread)

    stitch_count = 0
    jump_count = 0
    current_x = 0.0
    current_y = 0.0
    reverse = False
    for y in range(0, height, row_step_px):
        xs = np.where(active[y])[0]
        if xs.size == 0:
            continue
        runs: list[tuple[int, int]] = []
        start = int(xs[0])
        previous = int(xs[0])
        for value in xs[1:]:
            value = int(value)
            if value == previous + 1:
                previous = value
            else:
                if previous - start + 1 >= min_run_px:
                    runs.append((start, previous))
                start = previous = value
        if previous - start + 1 >= min_run_px:
            runs.append((start, previous))

        if reverse:
            runs = list(reversed(runs))
        for start, end in runs:
            coords = list(range(start, end + 1, point_step_px))
            if not coords or coords[-1] != end:
                coords.append(end)
            if reverse:
                coords = list(reversed(coords))
            first_x = (coords[0] - width / 2.0) * scale_mm
            first_y = (y - height / 2.0) * scale_mm
            pattern.add_stitch_absolute(JUMP, int(round(first_x * 10)), int(round(first_y * 10)))
            jump_count += 1
            current_x, current_y = first_x, first_y
            for x_px in coords:
                current_x = (x_px - width / 2.0) * scale_mm
                current_y = (y - height / 2.0) * scale_mm
                pattern.add_stitch_absolute(STITCH, int(round(current_x * 10)), int(round(current_y * 10)))
                stitch_count += 1
        reverse = not reverse

    pattern.add_stitch_absolute(END, int(round(current_x * 10)), int(round(current_y * 10)))
    write_dst(pattern, str(output_base.with_suffix(".dst")))
    write_pes(pattern, str(output_base.with_suffix(".pes")))
    return {
        "dst": str(output_base.with_suffix(".dst")),
        "pes": str(output_base.with_suffix(".pes")),
        "stitches": stitch_count,
        "jumps": jump_count,
        "target_width_mm": target_width_mm,
        "threshold": threshold,
        "export_mode": "scanline",
    }


def connected_components(active: np.ndarray, min_area_px: int) -> list[np.ndarray]:
    height, width = active.shape
    visited = np.zeros_like(active, dtype=bool)
    components: list[np.ndarray] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    ys, xs = np.nonzero(active)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in neighbors:
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= height or xx < 0 or xx >= width:
                    continue
                if visited[yy, xx] or not active[yy, xx]:
                    continue
                visited[yy, xx] = True
                queue.append((yy, xx))
        if len(pixels) >= min_area_px:
            component = np.zeros_like(active, dtype=bool)
            py, px = zip(*pixels)
            component[np.asarray(py), np.asarray(px)] = True
            components.append(component)
    components.sort(key=lambda item: int(item.sum()), reverse=True)
    return components


def component_orientation(
    component: np.ndarray,
    direction: np.ndarray,
    density: np.ndarray,
) -> float:
    yy, xx = np.nonzero(component)
    if yy.size == 0:
        return 0.0

    dx = direction[0, yy, xx]
    dy = direction[1, yy, xx]
    weight = np.clip(density[yy, xx], 0.05, 1.0)
    magnitude = np.hypot(dx, dy)
    valid = magnitude > 0.05
    if np.any(valid):
        # Direction is sign-invariant for fill: left-to-right and
        # right-to-left should count as the same stitch orientation.
        dx = dx[valid] / magnitude[valid]
        dy = dy[valid] / magnitude[valid]
        weight = weight[valid]
        cos2 = np.sum(weight * (dx * dx - dy * dy))
        sin2 = np.sum(weight * (2.0 * dx * dy))
        if math.hypot(float(cos2), float(sin2)) > 1e-4:
            return 0.5 * math.atan2(float(sin2), float(cos2))

    centered = np.stack([xx - xx.mean(), yy - yy.mean()], axis=1).astype(np.float32)
    if centered.shape[0] < 3:
        return 0.0
    covariance = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    vector = eigvecs[:, int(np.argmax(eigvals))]
    return math.atan2(float(vector[1]), float(vector[0]))


def hatch_component(
    component: np.ndarray,
    angle: float,
    spacing_px: int,
    point_step_px: int,
    min_run_px: int,
) -> list[list[tuple[float, float]]]:
    height, width = component.shape
    yy, xx = np.nonzero(component)
    if yy.size == 0:
        return []

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    px = xx.astype(np.float32) - cx
    py = yy.astype(np.float32) - cy
    ux = math.cos(angle)
    uy = math.sin(angle)
    vx = -uy
    vy = ux
    t_values = px * ux + py * uy
    s_values = px * vx + py * vy

    spacing = max(1, spacing_px)
    strip_half_width = max(0.6, spacing * 0.42)
    min_s = math.floor(float(s_values.min()) / spacing) * spacing
    max_s = math.ceil(float(s_values.max()) / spacing) * spacing
    lines: list[list[tuple[float, float]]] = []
    reverse = False

    for s in np.arange(min_s, max_s + spacing, spacing, dtype=np.float32):
        in_strip = np.abs(s_values - float(s)) <= strip_half_width
        if int(in_strip.sum()) < min_run_px:
            continue
        local_t = t_values[in_strip]
        order = np.argsort(local_t)
        sorted_t = local_t[order]
        if sorted_t.size < min_run_px:
            continue

        runs: list[tuple[float, float]] = []
        start = float(sorted_t[0])
        previous = float(sorted_t[0])
        max_gap = max(2.75, spacing * 1.25)
        for value in sorted_t[1:]:
            value = float(value)
            if value - previous <= max_gap:
                previous = value
            else:
                if previous - start >= min_run_px:
                    runs.append((start, previous))
                start = previous = value
        if previous - start >= min_run_px:
            runs.append((start, previous))

        for start_t, end_t in runs:
            length = max(1e-6, end_t - start_t)
            steps = max(1, int(math.ceil(length / max(1, point_step_px))))
            coords: list[tuple[float, float]] = []
            for index in range(steps + 1):
                t = start_t + length * index / steps
                x = cx + ux * t + vx * float(s)
                y = cy + uy * t + vy * float(s)
                coords.append((x, y))
            if reverse:
                coords = list(reversed(coords))
            lines.append(coords)
            reverse = not reverse
    return lines


def boundary_path(component: np.ndarray, step_px: int) -> list[tuple[float, float]]:
    padded = np.pad(component, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    boundary = center & (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    yy, xx = np.nonzero(boundary)
    if yy.size < 8:
        return []
    cx = float(xx.mean())
    cy = float(yy.mean())
    angles = np.arctan2(yy - cy, xx - cx)
    order = np.argsort(angles)
    stride = max(1, step_px)
    path = [(float(xx[i]), float(yy[i])) for i in order[::stride]]
    if len(path) >= 3:
        path.append(path[0])
    return path


def axis_sorted_path(component: np.ndarray, step_px: int) -> list[tuple[float, float]]:
    yy, xx = np.nonzero(component)
    if yy.size < 2:
        return []
    coords = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=1)
    centered = coords - coords.mean(axis=0, keepdims=True)
    if coords.shape[0] >= 3:
        covariance = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        score = centered @ axis
    else:
        score = coords[:, 0] + coords[:, 1]
    order = np.argsort(score)
    stride = max(1, step_px)
    return [(float(coords[i, 0]), float(coords[i, 1])) for i in order[::stride]]


def add_polyline(
    pattern: EmbPattern,
    coords: list[tuple[float, float]],
    width: int,
    height: int,
    scale_mm: float,
) -> tuple[int, int, float, float]:
    if len(coords) < 2:
        return 0, 0, 0.0, 0.0
    first_x = (coords[0][0] - width / 2.0) * scale_mm
    first_y = (coords[0][1] - height / 2.0) * scale_mm
    pattern.add_stitch_absolute(JUMP, int(round(first_x * 10)), int(round(first_y * 10)))
    stitch_count = 0
    current_x, current_y = first_x, first_y
    for x_px, y_px in coords:
        current_x = (x_px - width / 2.0) * scale_mm
        current_y = (y_px - height / 2.0) * scale_mm
        pattern.add_stitch_absolute(STITCH, int(round(current_x * 10)), int(round(current_y * 10)))
        stitch_count += 1
    return stitch_count, 1, current_x, current_y


def export_directional_hatch_as_dst(
    mask: np.ndarray,
    density: np.ndarray,
    direction: np.ndarray,
    output_base: Path,
    threshold: float,
    target_width_mm: float,
    row_step_px: int,
    point_step_px: int,
    min_run_px: int,
    min_component_px: int,
    max_components: int,
) -> dict[str, int | float | str]:
    active = mask >= threshold
    height, width = active.shape
    scale_mm = target_width_mm / width

    pattern = EmbPattern()
    thread = EmbThread()
    thread.set_color(5, 85, 150)
    pattern.add_thread(thread)

    components = connected_components(active, min_component_px)
    if max_components > 0:
        components = components[:max_components]

    stitch_count = 0
    jump_count = 0
    current_x = 0.0
    current_y = 0.0
    angles: list[float] = []
    for component in components:
        angle = component_orientation(component, direction, density)
        angles.append(angle)
        for coords in hatch_component(
            component,
            angle=angle,
            spacing_px=row_step_px,
            point_step_px=point_step_px,
            min_run_px=min_run_px,
        ):
            if len(coords) < 2:
                continue
            first_x = (coords[0][0] - width / 2.0) * scale_mm
            first_y = (coords[0][1] - height / 2.0) * scale_mm
            pattern.add_stitch_absolute(JUMP, int(round(first_x * 10)), int(round(first_y * 10)))
            jump_count += 1
            current_x, current_y = first_x, first_y
            for x_px, y_px in coords:
                current_x = (x_px - width / 2.0) * scale_mm
                current_y = (y_px - height / 2.0) * scale_mm
                pattern.add_stitch_absolute(STITCH, int(round(current_x * 10)), int(round(current_y * 10)))
                stitch_count += 1

    pattern.add_stitch_absolute(END, int(round(current_x * 10)), int(round(current_y * 10)))
    write_dst(pattern, str(output_base.with_suffix(".dst")))
    write_pes(pattern, str(output_base.with_suffix(".pes")))
    mean_angle_deg = float(np.rad2deg(np.mean(angles))) if angles else 0.0
    return {
        "dst": str(output_base.with_suffix(".dst")),
        "pes": str(output_base.with_suffix(".pes")),
        "stitches": stitch_count,
        "jumps": jump_count,
        "components": len(components),
        "mean_component_angle_deg": mean_angle_deg,
        "target_width_mm": target_width_mm,
        "threshold": threshold,
        "export_mode": "directional_hatch",
    }


def image_color_layers(
    input_image: Image.Image,
    active: np.ndarray,
    max_colors: int,
    min_color_px: int,
) -> list[tuple[np.ndarray, tuple[int, int, int], int]]:
    rgb = np.asarray(input_image.convert("RGB"), dtype=np.uint8)
    quantized = input_image.convert("RGB").quantize(colors=max(1, max_colors), method=Image.Quantize.MEDIANCUT)
    labels = np.asarray(quantized, dtype=np.int32)
    layers: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    for label in np.unique(labels[active]):
        layer = active & (labels == int(label))
        area = int(layer.sum())
        if area < min_color_px:
            continue
        mean = rgb[layer].mean(axis=0)
        color = tuple(int(max(0, min(255, round(channel)))) for channel in mean)
        layers.append((layer, color, area))
    layers.sort(key=lambda item: item[2], reverse=True)
    return layers


def export_color_directional_hatch_as_dst(
    mask: np.ndarray,
    density: np.ndarray,
    direction: np.ndarray,
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
) -> dict[str, int | float | str]:
    active = mask >= threshold
    height, width = active.shape
    scale_mm = target_width_mm / width
    layers = image_color_layers(input_image, active, max_colors=max_colors, min_color_px=min_color_px)

    pattern = EmbPattern()
    if not layers:
        thread = EmbThread()
        thread.set_color(5, 85, 150)
        pattern.add_thread(thread)
    for _, color, _ in layers:
        thread = EmbThread()
        thread.set_color(*color)
        pattern.add_thread(thread)

    stitch_count = 0
    jump_count = 0
    color_changes = 0
    component_count = 0
    outline_count = 0
    current_x = 0.0
    current_y = 0.0
    angles: list[float] = []

    for color_index, (layer_mask, _color, _area) in enumerate(layers):
        if color_index > 0:
            pattern.add_stitch_absolute(COLOR_CHANGE, int(round(current_x * 10)), int(round(current_y * 10)))
            color_changes += 1
        components = connected_components(layer_mask, min_component_px)
        if max_components > 0:
            remaining = max(0, max_components - component_count)
            components = components[:remaining]
        for component in components:
            component_count += 1
            angle = component_orientation(component, direction, density)
            angles.append(angle)
            for coords in hatch_component(
                component,
                angle=angle,
                spacing_px=row_step_px,
                point_step_px=point_step_px,
                min_run_px=min_run_px,
            ):
                if len(coords) < 2:
                    continue
                first_x = (coords[0][0] - width / 2.0) * scale_mm
                first_y = (coords[0][1] - height / 2.0) * scale_mm
                pattern.add_stitch_absolute(JUMP, int(round(first_x * 10)), int(round(first_y * 10)))
                jump_count += 1
                current_x, current_y = first_x, first_y
                for x_px, y_px in coords:
                    current_x = (x_px - width / 2.0) * scale_mm
                    current_y = (y_px - height / 2.0) * scale_mm
                    pattern.add_stitch_absolute(STITCH, int(round(current_x * 10)), int(round(current_y * 10)))
                    stitch_count += 1
            if add_outline:
                outline = boundary_path(component, step_px=outline_step_px)
                if len(outline) >= 3:
                    first_x = (outline[0][0] - width / 2.0) * scale_mm
                    first_y = (outline[0][1] - height / 2.0) * scale_mm
                    pattern.add_stitch_absolute(JUMP, int(round(first_x * 10)), int(round(first_y * 10)))
                    jump_count += 1
                    for x_px, y_px in outline:
                        current_x = (x_px - width / 2.0) * scale_mm
                        current_y = (y_px - height / 2.0) * scale_mm
                        pattern.add_stitch_absolute(STITCH, int(round(current_x * 10)), int(round(current_y * 10)))
                        stitch_count += 1
                        outline_count += 1
        if max_components > 0 and component_count >= max_components:
            break

    pattern.add_stitch_absolute(END, int(round(current_x * 10)), int(round(current_y * 10)))
    write_dst(pattern, str(output_base.with_suffix(".dst")))
    write_pes(pattern, str(output_base.with_suffix(".pes")))
    mean_angle_deg = float(np.rad2deg(np.mean(angles))) if angles else 0.0
    return {
        "dst": str(output_base.with_suffix(".dst")),
        "pes": str(output_base.with_suffix(".pes")),
        "stitches": stitch_count,
        "jumps": jump_count,
        "color_layers": len(layers),
        "color_changes": color_changes,
        "components": component_count,
        "outline_stitches": outline_count,
        "mean_component_angle_deg": mean_angle_deg,
        "target_width_mm": target_width_mm,
        "threshold": threshold,
        "export_mode": "color_directional_hatch",
    }


def export_color_planner_as_dst(
    mask: np.ndarray,
    density: np.ndarray,
    direction: np.ndarray,
    stitch_type: np.ndarray,
    centerline: np.ndarray,
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
    for _, color, _ in layers:
        thread = EmbThread()
        thread.set_color(*color)
        pattern.add_thread(thread)

    stitch_count = 0
    jump_count = 0
    color_changes = 0
    component_count = 0
    outline_count = 0
    current_x = 0.0
    current_y = 0.0
    type_counts = {"running": 0, "satin": 0, "fill": 0}

    for color_index, (layer_mask, _color, _area) in enumerate(layers):
        if color_index > 0:
            pattern.add_stitch_absolute(COLOR_CHANGE, int(round(current_x * 10)), int(round(current_y * 10)))
            color_changes += 1

        for type_id, type_name in ((3, "fill"), (2, "satin"), (1, "running")):
            type_mask = layer_mask & (stitch_type == type_id)
            if not np.any(type_mask):
                continue
            components = connected_components(type_mask, min_component_px)
            for component in components:
                if max_components > 0 and component_count >= max_components:
                    break
                component_count += 1
                type_counts[type_name] += 1
                angle = component_orientation(component, direction, density)
                if type_id == 1:
                    run_source = component & (centerline >= 0.35)
                    if int(run_source.sum()) < max(8, min_run_px):
                        run_source = component
                    path = axis_sorted_path(run_source, step_px=max(1, point_step_px))
                    sc, jc, current_x, current_y = add_polyline(pattern, path, width, height, scale_mm)
                    stitch_count += sc
                    jump_count += jc
                elif type_id == 2:
                    # Satin-like regions use denser short strokes across the local long axis.
                    for coords in hatch_component(
                        component,
                        angle=angle + math.pi / 2.0,
                        spacing_px=max(1, row_step_px - 1),
                        point_step_px=max(1, point_step_px),
                        min_run_px=max(2, min_run_px - 1),
                    ):
                        sc, jc, current_x, current_y = add_polyline(pattern, coords, width, height, scale_mm)
                        stitch_count += sc
                        jump_count += jc
                else:
                    for coords in hatch_component(
                        component,
                        angle=angle,
                        spacing_px=row_step_px,
                        point_step_px=point_step_px,
                        min_run_px=min_run_px,
                    ):
                        sc, jc, current_x, current_y = add_polyline(pattern, coords, width, height, scale_mm)
                        stitch_count += sc
                        jump_count += jc

                if add_outline and type_id != 1:
                    outline = boundary_path(component, step_px=outline_step_px)
                    sc, jc, current_x, current_y = add_polyline(pattern, outline, width, height, scale_mm)
                    stitch_count += sc
                    jump_count += jc
                    outline_count += sc
            if max_components > 0 and component_count >= max_components:
                break
        if max_components > 0 and component_count >= max_components:
            break

    pattern.add_stitch_absolute(END, int(round(current_x * 10)), int(round(current_y * 10)))
    write_dst(pattern, str(output_base.with_suffix(".dst")))
    write_pes(pattern, str(output_base.with_suffix(".pes")))
    return {
        "dst": str(output_base.with_suffix(".dst")),
        "pes": str(output_base.with_suffix(".pes")),
        "stitches": stitch_count,
        "jumps": jump_count,
        "color_layers": len(layers),
        "color_changes": color_changes,
        "components": component_count,
        "outline_stitches": outline_count,
        "type_component_counts": type_counts,
        "target_width_mm": target_width_mm,
        "threshold": threshold,
        "export_mode": "color_planner_stitch_type",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    target_channels = checkpoint.get("target_channels", ["mask", "density", "direction_x", "direction_y"])
    model = TinyUNet(out_channels=len(target_channels), base_channels=int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_image = resize_with_pad(input_path, args.size)
    with torch.no_grad():
        output = model(image_to_tensor(input_image).to(device))[0].detach().cpu()
    mask = torch.sigmoid(output[0]).numpy()
    density = torch.sigmoid(output[1]).numpy()
    raw_direction = torch.tanh(output[2:4]).numpy()
    if len(target_channels) >= 7 and target_channels[2:4] == ["axis_x", "axis_y"]:
        # Rich labels store sign-invariant orientation as cos(2a), sin(2a).
        # Convert back to a usable hatch vector cos(a), sin(a).
        angle = 0.5 * np.arctan2(raw_direction[1], raw_direction[0])
        direction = np.stack([np.cos(angle), np.sin(angle)], axis=0).astype(np.float32)
    else:
        direction = raw_direction
    centerline = torch.sigmoid(output[6]).numpy() if output.shape[0] >= 7 else np.zeros_like(mask)
    stitch_type = None
    if len(target_channels) >= 11 and all(name.startswith("stitch_type_") for name in target_channels[7:11]):
        stitch_type = output[7:11].argmax(dim=0).numpy().astype(np.uint8)
    dst_map = density if args.dst_source == "density" else mask

    input_image.save(output_dir / "input_256.png")
    grayscale_image(mask).save(output_dir / "pred_mask.png")
    grayscale_image(density).save(output_dir / "pred_density.png")
    direction_image(direction, mask).save(output_dir / "pred_direction.png")
    if output.shape[0] >= 7:
        grayscale_image(torch.sigmoid(output[5]).numpy()).save(output_dir / "pred_boundary.png")
        grayscale_image(centerline).save(output_dir / "pred_centerline.png")
    if stitch_type is not None:
        stitch_type_image(stitch_type).save(output_dir / "pred_stitch_type.png")
    overlay_mask(input_image, mask, args.threshold).save(output_dir / "pred_mask_overlay.png")
    overlay_mask(input_image, dst_map, args.threshold).save(output_dir / "pred_overlay.png")
    save_panel(input_image, mask, density, direction, dst_map, args.threshold, output_dir / "prediction_panel.png")
    if args.export_mode == "planner":
        if stitch_type is None:
            raise ValueError("--export-mode planner requires a checkpoint with stitch_type output channels")
        dst_summary = export_color_planner_as_dst(
            dst_map,
            density=density,
            direction=direction,
            stitch_type=stitch_type,
            centerline=centerline,
            input_image=input_image,
            output_base=output_dir / "model_predicted",
            threshold=args.threshold,
            target_width_mm=args.target_width_mm,
            row_step_px=args.row_step_px,
            point_step_px=args.point_step_px,
            min_run_px=args.min_run_px,
            min_component_px=args.min_component_px,
            max_components=args.max_components,
            max_colors=args.max_colors,
            min_color_px=args.min_color_px,
            add_outline=args.add_outline,
            outline_step_px=args.outline_step_px,
        )
    elif args.export_mode == "directional" and args.color_mode == "input":
        dst_summary = export_color_directional_hatch_as_dst(
            dst_map,
            density=density,
            direction=direction,
            input_image=input_image,
            output_base=output_dir / "model_predicted",
            threshold=args.threshold,
            target_width_mm=args.target_width_mm,
            row_step_px=args.row_step_px,
            point_step_px=args.point_step_px,
            min_run_px=args.min_run_px,
            min_component_px=args.min_component_px,
            max_components=args.max_components,
            max_colors=args.max_colors,
            min_color_px=args.min_color_px,
            add_outline=args.add_outline,
            outline_step_px=args.outline_step_px,
        )
    elif args.export_mode == "directional":
        dst_summary = export_directional_hatch_as_dst(
            dst_map,
            density=density,
            direction=direction,
            output_base=output_dir / "model_predicted",
            threshold=args.threshold,
            target_width_mm=args.target_width_mm,
            row_step_px=args.row_step_px,
            point_step_px=args.point_step_px,
            min_run_px=args.min_run_px,
            min_component_px=args.min_component_px,
            max_components=args.max_components,
        )
    else:
        dst_summary = export_mask_as_dst(
            dst_map,
            output_base=output_dir / "model_predicted",
            threshold=args.threshold,
            target_width_mm=args.target_width_mm,
            row_step_px=args.row_step_px,
            point_step_px=args.point_step_px,
            min_run_px=args.min_run_px,
        )
    summary = {
        "input": str(input_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "mask_mean": float(mask.mean()),
        "mask_max": float(mask.max()),
        "density_mean": float(density.mean()),
        "has_stitch_type": stitch_type is not None,
        "stitch_type_names": STITCH_TYPE_NAMES if stitch_type is not None else [],
        "dst_source": args.dst_source,
        "files": [
            "input_256.png",
            "pred_mask.png",
            "pred_density.png",
            "pred_direction.png",
            "pred_mask_overlay.png",
            "pred_overlay.png",
            "pred_stitch_type.png" if stitch_type is not None else "",
            "prediction_panel.png",
            "model_predicted.dst",
            "model_predicted.pes",
        ],
        "dst_summary": dst_summary,
        "note": (
            "Experimental DST generated from learned raster labels. "
            "Color layers come from input-image quantization when --color-mode=input; "
            "planner mode uses stitch_type channels when available."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the image-to-stitch-label model on one image.")
    parser.add_argument("input")
    parser.add_argument("--checkpoint", default="models/image_to_stitch_label_unet/best_image_to_stitch_label_unet.pt")
    parser.add_argument("--output-dir", default="outputs/model_inference")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dst-source", choices=["mask", "density"], default="mask")
    parser.add_argument("--target-width-mm", type=float, default=80.0)
    parser.add_argument("--row-step-px", type=int, default=2)
    parser.add_argument("--point-step-px", type=int, default=2)
    parser.add_argument("--min-run-px", type=int, default=2)
    parser.add_argument("--export-mode", choices=["scanline", "directional", "planner"], default="scanline")
    parser.add_argument("--color-mode", choices=["single", "input"], default="single")
    parser.add_argument("--max-colors", type=int, default=6)
    parser.add_argument("--min-color-px", type=int, default=96)
    parser.add_argument("--add-outline", action="store_true")
    parser.add_argument("--outline-step-px", type=int, default=3)
    parser.add_argument("--min-component-px", type=int, default=24)
    parser.add_argument("--max-components", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
