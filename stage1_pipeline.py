from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from pyembroidery import COLOR_CHANGE, END, JUMP, STITCH, TRIM, EmbPattern, EmbThread
from pyembroidery import write_dst, write_pes


@dataclass
class Region:
    id: int
    color_index: int
    rgb: tuple[int, int, int]
    polygon: Polygon
    area_mm2: float
    width_mm: float
    height_mm: float
    stitch_type: str


@dataclass
class StitchPoint:
    x: float
    y: float
    command: str


@dataclass
class StitchBlock:
    region_id: int
    color_index: int
    rgb: tuple[int, int, int]
    stitch_type: str
    points: list[StitchPoint]


THREAD_COLOR_NAMES = [
    ((255, 255, 255), "white"),
    ((0, 0, 0), "black"),
    ((32, 32, 32), "dark charcoal"),
    ((255, 0, 0), "red"),
    ((220, 50, 50), "warm red"),
    ((255, 180, 0), "gold yellow"),
    ((255, 230, 80), "light yellow"),
    ((0, 120, 190), "medium blue"),
    ((0, 180, 215), "cyan blue"),
    ((20, 80, 120), "deep blue"),
    ((40, 150, 110), "green teal"),
    ((120, 70, 160), "purple"),
]


def load_rgb(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def quantize_method(name: str) -> Image.Quantize:
    methods = {
        "mediancut": Image.Quantize.MEDIANCUT,
        "maxcoverage": Image.Quantize.MAXCOVERAGE,
        "fastoctree": Image.Quantize.FASTOCTREE,
    }
    return methods[name]


def quantize_image(
    image: Image.Image,
    colors: int,
    method_name: str,
) -> tuple[np.ndarray, list[tuple[int, int, int]], Image.Image]:
    quantized = image.quantize(colors=colors, method=quantize_method(method_name))
    label_map = np.asarray(quantized, dtype=np.uint8)
    palette = quantized.getpalette()[: colors * 3]
    rgb_palette = [
        (int(palette[i]), int(palette[i + 1]), int(palette[i + 2]))
        for i in range(0, len(palette), 3)
    ]
    preview = quantized.convert("RGB")
    return label_map, rgb_palette, preview


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def find_background_labels(
    label_map: np.ndarray,
    palette: Sequence[tuple[int, int, int]],
    include_background: bool,
    light_threshold: int,
) -> set[int]:
    if include_background:
        return set()
    h, w = label_map.shape
    border = np.concatenate(
        [
            label_map[0, :],
            label_map[h - 1, :],
            label_map[:, 0],
            label_map[:, w - 1],
        ]
    )
    background: set[int] = set()
    counts = np.bincount(label_map.reshape(-1), minlength=len(palette))
    border_counts = np.bincount(border, minlength=len(palette))
    dominant_border = int(np.argmax(border_counts))
    if border_counts[dominant_border] / max(1, border.size) > 0.30:
        background.add(dominant_border)
    for index, rgb in enumerate(palette):
        if luminance(rgb) >= light_threshold and counts[index] / label_map.size > 0.05:
            background.add(index)
    return background


def mask_to_polygons(mask: np.ndarray, min_area_px: int, simplify_px: float) -> list[np.ndarray]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.75, simplify_px * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) >= 3:
            polygons.append(approx[:, 0, :])
    return polygons


def remove_border_connected_components(mask: np.ndarray) -> np.ndarray:
    count, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    keep = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if label not in border_labels:
            keep[labels == label] = 255
    return keep


def pixel_polygon_to_mm(points: np.ndarray, width_px: int, height_px: int, scale_mm: float) -> Polygon | None:
    coords = [
        ((float(x) - width_px / 2.0) * scale_mm, (float(y) - height_px / 2.0) * scale_mm)
        for x, y in points
    ]
    poly = Polygon(coords).buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda item: item.area)
    if not isinstance(poly, Polygon) or poly.area <= 0:
        return None
    return poly


def classify_stitch(poly: Polygon, area_mm2: float, satin_max_width_mm: float) -> str:
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    short = max(0.01, min(width, height))
    long = max(width, height)
    aspect = long / short
    if short < 1.2 or (area_mm2 < 8.0 and aspect > 2.0):
        return "run"
    if short <= satin_max_width_mm and aspect >= 1.45:
        return "satin"
    return "fill"


def extract_regions(
    label_map: np.ndarray,
    palette: Sequence[tuple[int, int, int]],
    target_width_mm: float,
    background_labels: set[int],
    min_area_px: int,
    simplify_px: float,
    satin_max_width_mm: float,
) -> list[Region]:
    h, w = label_map.shape
    scale_mm = target_width_mm / w
    regions: list[Region] = []
    region_id = 1
    for color_index, rgb in enumerate(palette):
        mask = np.where(label_map == color_index, 255, 0).astype(np.uint8)
        if color_index in background_labels:
            mask = remove_border_connected_components(mask)
            if not np.any(mask):
                continue
        for points in mask_to_polygons(mask, min_area_px=min_area_px, simplify_px=simplify_px):
            poly = pixel_polygon_to_mm(points, w, h, scale_mm)
            if poly is None:
                continue
            minx, miny, maxx, maxy = poly.bounds
            area = float(poly.area)
            stitch_type = classify_stitch(poly, area, satin_max_width_mm=satin_max_width_mm)
            regions.append(
                Region(
                    id=region_id,
                    color_index=color_index,
                    rgb=tuple(rgb),
                    polygon=poly,
                    area_mm2=area,
                    width_mm=maxx - minx,
                    height_mm=maxy - miny,
                    stitch_type=stitch_type,
                )
            )
            region_id += 1
    return regions


def angle_from_min_rect(poly: Polygon) -> float:
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    if len(coords) < 2:
        return 0.0
    best_angle = 0.0
    best_len = -1.0
    for a, b in zip(coords, coords[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_angle = math.atan2(dy, dx)
    return best_angle


def interpolate_segment(a: tuple[float, float], b: tuple[float, float], max_step: float) -> list[tuple[float, float]]:
    ax, ay = a
    bx, by = b
    distance = math.hypot(bx - ax, by - ay)
    steps = max(1, int(math.ceil(distance / max_step)))
    return [
        (ax + (bx - ax) * i / steps, ay + (by - ay) * i / steps)
        for i in range(1, steps + 1)
    ]


def add_path(
    points: list[StitchPoint],
    coords: Sequence[tuple[float, float]],
    max_step: float,
    start_command: str = "JUMP",
) -> None:
    if not coords:
        return
    start = coords[0]
    if points and start_command == "STITCH":
        previous = (points[-1].x, points[-1].y)
        for x, y in interpolate_segment(previous, start, max_step=max_step):
            points.append(StitchPoint(x, y, "STITCH"))
    else:
        points.append(StitchPoint(start[0], start[1], start_command))
    last = start
    for coord in coords[1:]:
        for x, y in interpolate_segment(last, coord, max_step=max_step):
            points.append(StitchPoint(x, y, "STITCH"))
        last = coord


def generate_run(poly: Polygon, max_step: float) -> list[StitchPoint]:
    coords = list(poly.exterior.coords)
    points: list[StitchPoint] = []
    add_path(points, coords, max_step=max_step)
    return points


def line_segments(geometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0 else []
    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length > 0]
    if isinstance(geometry, GeometryCollection):
        items: list[LineString] = []
        for part in geometry.geoms:
            items.extend(line_segments(part))
        return items
    return []


def scan_segments(poly: Polygon, angle: float, spacing: float, padding: float = 5.0) -> list[list[tuple[float, float]]]:
    vx, vy = math.cos(angle), math.sin(angle)
    nx, ny = -vy, vx
    coords = list(poly.exterior.coords)
    proj_v = [x * vx + y * vy for x, y in coords]
    proj_n = [x * nx + y * ny for x, y in coords]
    min_v, max_v = min(proj_v) - padding, max(proj_v) + padding
    min_n, max_n = min(proj_n) - padding, max(proj_n) + padding
    offset = min_n
    rows: list[list[tuple[float, float]]] = []
    while offset <= max_n:
        p1 = (vx * min_v + nx * offset, vy * min_v + ny * offset)
        p2 = (vx * max_v + nx * offset, vy * max_v + ny * offset)
        inter = poly.intersection(LineString([p1, p2]))
        segments = line_segments(inter)
        segments.sort(key=lambda segment: segment.length, reverse=True)
        row: list[tuple[float, float]] = []
        for segment in segments:
            coords_segment = list(segment.coords)
            if len(coords_segment) >= 2 and segment.length >= 0.3:
                row.extend([coords_segment[0], coords_segment[-1]])
        if row:
            rows.append(row)
        offset += spacing
    return rows


def generate_fill(poly: Polygon, angle: float, spacing: float, max_step: float) -> list[StitchPoint]:
    rows = scan_segments(poly, angle, spacing)
    points: list[StitchPoint] = []
    reverse = False
    first = True
    for row in rows:
        if reverse:
            row = list(reversed(row))
        add_path(points, row, max_step=max_step, start_command="JUMP" if first else "STITCH")
        reverse = not reverse
        first = False
    return points


def generate_satin(poly: Polygon, angle: float, spacing: float, max_step: float) -> list[StitchPoint]:
    # Cross-sections perpendicular to the long axis produce a simple satin-like zig-zag.
    rows = scan_segments(poly, angle + math.pi / 2.0, spacing)
    points: list[StitchPoint] = []
    reverse = False
    first = True
    for row in rows:
        if len(row) < 2:
            continue
        a, b = row[0], row[1]
        segment = [a, b] if not reverse else [b, a]
        add_path(points, segment, max_step=max_step, start_command="JUMP" if first else "STITCH")
        reverse = not reverse
        first = False
    return points


def generate_blocks(
    regions: Sequence[Region],
    fill_spacing_mm: float,
    satin_spacing_mm: float,
    run_step_mm: float,
    max_stitch_mm: float,
) -> list[StitchBlock]:
    blocks: list[StitchBlock] = []
    for region in regions:
        angle = angle_from_min_rect(region.polygon)
        if region.stitch_type == "run":
            points = generate_run(region.polygon, max_step=run_step_mm)
        elif region.stitch_type == "satin":
            points = generate_satin(region.polygon, angle=angle, spacing=satin_spacing_mm, max_step=max_stitch_mm)
        else:
            points = generate_fill(region.polygon, angle=angle, spacing=fill_spacing_mm, max_step=max_stitch_mm)
        if len(points) >= 2:
            blocks.append(
                StitchBlock(
                    region_id=region.id,
                    color_index=region.color_index,
                    rgb=region.rgb,
                    stitch_type=region.stitch_type,
                    points=points,
                )
            )
    return blocks


def block_start(block: StitchBlock) -> tuple[float, float]:
    first = block.points[0]
    return first.x, first.y


def block_end(block: StitchBlock) -> tuple[float, float]:
    last = block.points[-1]
    return last.x, last.y


def route_blocks(blocks: Sequence[StitchBlock]) -> list[StitchBlock]:
    color_order = sorted(
        {block.color_index for block in blocks},
        key=lambda color: -sum(len(block.points) for block in blocks if block.color_index == color),
    )
    routed: list[StitchBlock] = []
    cursor = (0.0, 0.0)
    for color in color_order:
        remaining = [block for block in blocks if block.color_index == color]
        while remaining:
            best_index = min(
                range(len(remaining)),
                key=lambda i: math.dist(cursor, block_start(remaining[i])),
            )
            block = remaining.pop(best_index)
            routed.append(block)
            cursor = block_end(block)
    return routed


def command_to_pyembroidery(command: str) -> int:
    if command == "JUMP":
        return JUMP
    if command == "TRIM":
        return TRIM
    return STITCH


def export_machine_files(blocks: Sequence[StitchBlock], output_base: Path) -> None:
    pattern = EmbPattern()
    unique_colors: list[tuple[int, tuple[int, int, int]]] = []
    for block in blocks:
        entry = (block.color_index, block.rgb)
        if entry not in unique_colors:
            unique_colors.append(entry)
    for _, rgb in unique_colors:
        thread = EmbThread()
        thread.set_color(*rgb)
        pattern.add_thread(thread)

    current_color = None
    current_x = 0.0
    current_y = 0.0
    for block in blocks:
        if current_color is None:
            current_color = block.color_index
        elif block.color_index != current_color:
            pattern.add_stitch_absolute(COLOR_CHANGE, int(round(current_x * 10)), int(round(current_y * 10)))
            current_color = block.color_index
        for point in block.points:
            command = command_to_pyembroidery(point.command)
            pattern.add_stitch_absolute(command, int(round(point.x * 10)), int(round(point.y * 10)))
            current_x, current_y = point.x, point.y
    pattern.add_stitch_absolute(END, int(round(current_x * 10)), int(round(current_y * 10)))
    write_dst(pattern, str(output_base.with_suffix(".dst")))
    write_pes(pattern, str(output_base.with_suffix(".pes")))


def export_tsv(blocks: Sequence[StitchBlock], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["region_id", "color_index", "stitch_type", "command", "x_mm", "y_mm"])
        for block in blocks:
            for point in block.points:
                writer.writerow(
                    [
                        block.region_id,
                        block.color_index,
                        block.stitch_type,
                        point.command,
                        f"{point.x:.3f}",
                        f"{point.y:.3f}",
                    ]
                )


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def nearest_thread_name(rgb: tuple[int, int, int]) -> str:
    def distance(item: tuple[tuple[int, int, int], str]) -> float:
        color, _ = item
        return math.dist(rgb, color)

    return min(THREAD_COLOR_NAMES, key=distance)[1]


def export_thread_palette(blocks: Sequence[StitchBlock], path: Path) -> None:
    seen: set[int] = set()
    rows = []
    for block in blocks:
        if block.color_index in seen:
            continue
        seen.add(block.color_index)
        same_color = [item for item in blocks if item.color_index == block.color_index]
        rows.append(
            {
                "thread_order": len(rows) + 1,
                "color_index": block.color_index,
                "hex": rgb_hex(block.rgb),
                "rgb": f"{block.rgb[0]},{block.rgb[1]},{block.rgb[2]}",
                "approx_color_name": nearest_thread_name(block.rgb),
                "region_count": len(same_color),
                "stitch_count": sum(1 for item in same_color for point in item.points if point.command == "STITCH"),
                "note": "Approximate RGB color. Match to your real thread brand manually.",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "thread_order",
                "color_index",
                "hex",
                "rgb",
                "approx_color_name",
                "region_count",
                "stitch_count",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_svg(regions: Sequence[Region], blocks: Sequence[StitchBlock], path: Path, margin: float = 6.0) -> None:
    all_polys = [region.polygon for region in regions]
    if not all_polys:
        return
    bounds = unary_union(all_polys).bounds
    minx, miny, maxx, maxy = bounds
    width = maxx - minx + 2 * margin
    height = maxy - miny + 2 * margin

    def tx(x: float) -> float:
        return x - minx + margin

    def ty(y: float) -> float:
        return y - miny + margin

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}mm" '
            f'height="{height:.2f}mm" viewBox="0 0 {width:.2f} {height:.2f}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for region in regions:
        coords = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in region.polygon.exterior.coords)
        lines.append(
            f'<polygon points="{coords}" fill="{rgb_hex(region.rgb)}" fill-opacity="0.18" '
            f'stroke="{rgb_hex(region.rgb)}" stroke-width="0.18"/>'
        )
    for block in blocks:
        visible = [point for point in block.points if point.command == "STITCH"]
        if len(visible) < 2:
            continue
        coords = " ".join(f"{tx(point.x):.2f},{ty(point.y):.2f}" for point in visible)
        lines.append(
            f'<polyline points="{coords}" fill="none" stroke="{rgb_hex(block.rgb)}" '
            f'stroke-width="0.22" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def export_json(
    image_path: Path,
    image_size: tuple[int, int],
    regions: Sequence[Region],
    blocks: Sequence[StitchBlock],
    path: Path,
) -> None:
    total_stitches = sum(1 for block in blocks for point in block.points if point.command == "STITCH")
    total_jumps = sum(1 for block in blocks for point in block.points if point.command == "JUMP")
    payload = {
        "input": str(image_path),
        "image_size_px": image_size,
        "region_count": len(regions),
        "block_count": len(blocks),
        "total_stitches": total_stitches,
        "total_jumps": total_jumps,
        "regions": [
            {
                "id": region.id,
                "color_index": region.color_index,
                "rgb": region.rgb,
                "area_mm2": round(region.area_mm2, 3),
                "width_mm": round(region.width_mm, 3),
                "height_mm": round(region.height_mm, 3),
                "stitch_type": region.stitch_type,
                "bounds": [round(v, 3) for v in region.polygon.bounds],
            }
            for region in regions
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(input_path, max_side=args.max_side)
    label_map, palette, quantized_preview = quantize_image(
        image,
        colors=args.max_colors,
        method_name=args.quantize_method,
    )
    background_labels = find_background_labels(
        label_map,
        palette,
        include_background=args.include_background,
        light_threshold=args.background_luminance,
    )
    regions = extract_regions(
        label_map,
        palette,
        target_width_mm=args.target_width_mm,
        background_labels=background_labels,
        min_area_px=args.min_area_px,
        simplify_px=args.simplify,
        satin_max_width_mm=args.satin_max_width_mm,
    )
    blocks = generate_blocks(
        regions,
        fill_spacing_mm=args.fill_spacing_mm,
        satin_spacing_mm=args.satin_spacing_mm,
        run_step_mm=args.run_step_mm,
        max_stitch_mm=args.max_stitch_mm,
    )
    routed = route_blocks(blocks)

    quantized_preview.save(output_dir / "01_quantized.png")
    export_svg(regions, routed, output_dir / "02_stitch_preview.svg")
    export_thread_palette(routed, output_dir / "thread_palette.csv")
    export_tsv(routed, output_dir / "stitch_plan.tsv")
    export_json(input_path, image.size, regions, routed, output_dir / "metadata.json")
    export_machine_files(routed, output_dir / "machine")

    summary = {
        "output_dir": str(output_dir),
        "regions": len(regions),
        "blocks": len(routed),
        "stitches": sum(1 for block in routed for point in block.points if point.command == "STITCH"),
        "jumps": sum(1 for block in routed for point in block.points if point.command == "JUMP"),
        "files": [
            "01_quantized.png",
            "02_stitch_preview.svg",
            "thread_palette.csv",
            "stitch_plan.tsv",
            "metadata.json",
            "machine.dst",
            "machine.pes",
        ],
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-1 rule-based image-to-embroidery prototype."
    )
    parser.add_argument("input", help="Input PNG/JPG image.")
    parser.add_argument("-o", "--output", default="outputs/stage1_run", help="Output directory.")
    parser.add_argument("--max-colors", type=int, default=6, help="Palette size for color quantization.")
    parser.add_argument(
        "--quantize-method",
        choices=["mediancut", "maxcoverage", "fastoctree"],
        default="mediancut",
        help="Color quantization method. fastoctree often preserves small saturated regions better.",
    )
    parser.add_argument("--target-width-mm", type=float, default=80.0, help="Physical design width.")
    parser.add_argument("--max-side", type=int, default=768, help="Largest image side used for processing.")
    parser.add_argument("--min-area-px", type=int, default=80, help="Minimum connected region area in pixels.")
    parser.add_argument("--simplify", type=float, default=0.006, help="Contour simplification fraction.")
    parser.add_argument("--include-background", action="store_true", help="Do not remove light/border background.")
    parser.add_argument("--background-luminance", type=int, default=242, help="Luminance threshold for white background.")
    parser.add_argument("--satin-max-width-mm", type=float, default=7.0, help="Max short side for satin rule.")
    parser.add_argument("--fill-spacing-mm", type=float, default=0.85, help="Row spacing for fill stitch.")
    parser.add_argument("--satin-spacing-mm", type=float, default=1.25, help="Cross-section spacing for satin stitch.")
    parser.add_argument("--run-step-mm", type=float, default=1.8, help="Max step for run stitch outlines.")
    parser.add_argument("--max-stitch-mm", type=float, default=3.2, help="Max interpolated stitch length.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
