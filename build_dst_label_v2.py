from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyembroidery import COLOR_CHANGE, COMMAND_MASK, END, JUMP, STITCH, TRIM, read_dst, read_pes


LABEL_V2_CHANNELS = [
    "stitch_trace",
    "same_color_near_connect",
    "long_jump_endpoint",
    "trim_endpoint",
    "color_change_endpoint",
    "closure_gap_endpoint",
    "path_order",
]


COMMAND_NAMES = {
    STITCH: "STITCH",
    JUMP: "JUMP",
    TRIM: "TRIM",
    COLOR_CHANGE: "COLOR_CHANGE",
    END: "END",
}


@dataclass
class StitchSegment:
    segment_id: int
    color_layer: int
    entry_mm: tuple[float, float]
    exit_mm: tuple[float, float]
    entry_xy: tuple[float, float]
    exit_xy: tuple[float, float]
    length_mm: float
    source_order: int
    preceding_commands: list[str]


def read_pattern(path: Path):
    if path.suffix.lower() == ".pes":
        return read_pes(str(path))
    return read_dst(str(path))


def fit_points(points: list[tuple[float, float]], size: int, padding: int) -> tuple[dict[str, float], list[tuple[float, float]]]:
    if not points:
        info = {"min_x": 0.0, "min_y": 0.0, "width_mm": 0.0, "height_mm": 0.0, "scale_px_per_mm": 1.0}
        return info, []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    scale = (size - 2 * padding) / max(width, height)
    ox = (size - width * scale) / 2.0
    oy = (size - height * scale) / 2.0
    fitted = [(ox + (x - min_x) * scale, oy + (y - min_y) * scale) for x, y in points]
    info = {"min_x": min_x, "min_y": min_y, "width_mm": width, "height_mm": height, "scale_px_per_mm": scale}
    return info, fitted


def read_stitch_segments(path: Path, size: int, padding: int) -> tuple[list[StitchSegment], dict[str, float], dict[str, int]]:
    pattern = read_pattern(path)
    raw: list[dict[str, object]] = []
    last: tuple[float, float] | None = None
    color_layer = 1
    pending: list[str] = []
    command_counts = {"stitch": 0, "jump": 0, "trim": 0, "color_change": 0}

    for order, (x_raw, y_raw, command_raw) in enumerate(pattern.stitches):
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        command = command_raw & COMMAND_MASK
        command_name = COMMAND_NAMES.get(command, f"CMD_{command}")
        if command == COLOR_CHANGE:
            color_layer += 1
            command_counts["color_change"] += 1
            pending.append(command_name)
        elif command == JUMP:
            command_counts["jump"] += 1
            pending.append(command_name)
        elif command == TRIM:
            command_counts["trim"] += 1
            pending.append(command_name)
        elif command == STITCH:
            command_counts["stitch"] += 1
            if last is not None:
                raw.append(
                    {
                        "color_layer": color_layer,
                        "entry_mm": last,
                        "exit_mm": (x, y),
                        "source_order": order,
                        "preceding_commands": pending,
                    }
                )
            pending = []
        else:
            pending.append(command_name)
        last = (x, y)

    points: list[tuple[float, float]] = []
    for item in raw:
        points.append(item["entry_mm"])  # type: ignore[arg-type]
        points.append(item["exit_mm"])  # type: ignore[arg-type]
    info, fitted = fit_points(points, size, padding)
    segments: list[StitchSegment] = []
    for i, item in enumerate(raw):
        entry_xy = fitted[2 * i]
        exit_xy = fitted[2 * i + 1]
        entry_mm = item["entry_mm"]  # type: ignore[assignment]
        exit_mm = item["exit_mm"]  # type: ignore[assignment]
        length = math.dist(entry_mm, exit_mm)
        segments.append(
            StitchSegment(
                segment_id=i + 1,
                color_layer=int(item["color_layer"]),
                entry_mm=entry_mm,
                exit_mm=exit_mm,
                entry_xy=entry_xy,
                exit_xy=exit_xy,
                length_mm=float(length),
                source_order=int(item["source_order"]),
                preceding_commands=list(item["preceding_commands"]),  # type: ignore[arg-type]
            )
        )
    return segments, info, command_counts


def draw_line(arr: np.ndarray, a: tuple[float, float], b: tuple[float, float], width: int, value: float) -> None:
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255))
    draw = ImageDraw.Draw(image)
    draw.line([(a[0], a[1]), (b[0], b[1])], fill=int(value * 255), width=max(1, width))
    np.maximum(arr, np.asarray(image, dtype=np.float32) / 255.0, out=arr)


def draw_point(arr: np.ndarray, p: tuple[float, float], radius: int, value: float) -> None:
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255))
    draw = ImageDraw.Draw(image)
    x, y = p
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=int(value * 255))
    np.maximum(arr, np.asarray(image, dtype=np.float32) / 255.0, out=arr)


def build_maps(
    segments: list[StitchSegment],
    size: int,
    line_width: int,
    point_radius: int,
    near_connect_mm: float,
    long_jump_mm: float,
    closure_gap_mm: float,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    maps = np.zeros((len(LABEL_V2_CHANNELS), size, size), dtype=np.float32)
    path_events: list[dict[str, object]] = []
    if not segments:
        return maps, path_events, {"segments": 0}

    total = max(1, len(segments) - 1)
    for index, segment in enumerate(segments):
        draw_line(maps[0], segment.entry_xy, segment.exit_xy, line_width, 1.0)
        draw_line(maps[6], segment.entry_xy, segment.exit_xy, max(1, line_width), index / total)

    same_color_near = 0
    long_jumps = 0
    trim_edges = 0
    color_change_edges = 0
    for prev, cur in zip(segments, segments[1:]):
        gap_mm = math.dist(prev.exit_mm, cur.entry_mm)
        same_color = prev.color_layer == cur.color_layer
        commands = cur.preceding_commands
        is_trim = "TRIM" in commands
        is_color_change = "COLOR_CHANGE" in commands or not same_color
        is_long = gap_mm >= long_jump_mm
        is_near = same_color and gap_mm <= near_connect_mm
        if is_near:
            draw_line(maps[1], prev.exit_xy, cur.entry_xy, line_width, 1.0)
            same_color_near += 1
        if is_long:
            draw_point(maps[2], prev.exit_xy, point_radius, 1.0)
            draw_point(maps[2], cur.entry_xy, point_radius, 1.0)
            long_jumps += 1
        if is_trim:
            draw_point(maps[3], prev.exit_xy, point_radius, 1.0)
            draw_point(maps[3], cur.entry_xy, point_radius, 1.0)
            trim_edges += 1
        if is_color_change:
            draw_point(maps[4], prev.exit_xy, point_radius, 0.85)
            draw_point(maps[4], cur.entry_xy, point_radius, 0.85)
            color_change_edges += 1
        path_events.append(
            {
                "from_segment": prev.segment_id,
                "to_segment": cur.segment_id,
                "same_color": same_color,
                "distance_mm": round(gap_mm, 4),
                "commands": commands,
                "near_connect_label": is_near,
                "long_jump_label": is_long,
                "trim_label": is_trim,
                "color_change_label": is_color_change,
            }
        )

    closure_gap_count = 0
    by_color: dict[int, list[StitchSegment]] = {}
    for segment in segments:
        by_color.setdefault(segment.color_layer, []).append(segment)
    for color_layer, color_segments in by_color.items():
        first = color_segments[0]
        last = color_segments[-1]
        gap = math.dist(first.entry_mm, last.exit_mm)
        if gap >= closure_gap_mm and len(color_segments) >= 4:
            draw_point(maps[5], first.entry_xy, point_radius, 1.0)
            draw_point(maps[5], last.exit_xy, point_radius, 1.0)
            closure_gap_count += 1
            path_events.append(
                {
                    "from_segment": last.segment_id,
                    "to_segment": first.segment_id,
                    "same_color": True,
                    "distance_mm": round(gap, 4),
                    "commands": ["CLOSURE_CHECK"],
                    "closure_gap_label": True,
                    "color_layer": color_layer,
                }
            )

    stats = {
        "segments": len(segments),
        "same_color_near_connect_edges": same_color_near,
        "long_jump_edges": long_jumps,
        "trim_edges": trim_edges,
        "color_change_edges": color_change_edges,
        "closure_gap_edges": closure_gap_count,
        "channels": LABEL_V2_CHANNELS,
    }
    return np.clip(maps, 0.0, 1.0).astype(np.float32), path_events, stats


def preview_label(maps: np.ndarray) -> Image.Image:
    rgb = np.zeros((maps.shape[1], maps.shape[2], 3), dtype=np.uint8)
    rgb[..., 0] = np.maximum(maps[2], maps[3]) * 255
    rgb[..., 1] = np.maximum(maps[0] * 0.65, maps[1]) * 255
    rgb[..., 2] = np.maximum(maps[4], maps[5]) * 255
    return Image.fromarray(rgb)


def segment_to_json(segment: StitchSegment) -> dict[str, object]:
    return {
        "segment_id": segment.segment_id,
        "color_layer": segment.color_layer,
        "source_order": segment.source_order,
        "entry_mm": [round(segment.entry_mm[0], 4), round(segment.entry_mm[1], 4)],
        "exit_mm": [round(segment.exit_mm[0], 4), round(segment.exit_mm[1], 4)],
        "entry_xy": [round(segment.entry_xy[0], 3), round(segment.entry_xy[1], 3)],
        "exit_xy": [round(segment.exit_xy[0], 3), round(segment.exit_xy[1], 3)],
        "length_mm": round(segment.length_mm, 4),
        "preceding_commands": segment.preceding_commands,
    }


def process_file(dst_path: Path, pair_id: str, output_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    labels_dir = output_dir / "labels"
    graphs_dir = output_dir / "graphs"
    previews_dir = output_dir / "previews"
    for directory in (labels_dir, graphs_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    segments, transform, command_counts = read_stitch_segments(dst_path, args.size, args.padding)
    maps, path_events, stats = build_maps(
        segments,
        args.size,
        args.line_width,
        args.point_radius,
        args.near_connect_mm,
        args.long_jump_mm,
        args.closure_gap_mm,
    )

    label_path = labels_dir / f"{pair_id}_dst_label_v2.npy"
    segment_path = graphs_dir / f"{pair_id}_segments_v2.json"
    events_path = graphs_dir / f"{pair_id}_path_events_v2.json"
    preview_path = previews_dir / f"{pair_id}_dst_label_v2_preview.png"
    np.save(label_path, maps)
    segment_path.write_text(json.dumps([segment_to_json(s) for s in segments], ensure_ascii=False, indent=2), encoding="utf-8")
    events_path.write_text(json.dumps(path_events, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_label(maps).save(preview_path)

    return {
        "pair_id": pair_id,
        "source_file": str(dst_path),
        "dst_label_v2_npy": str(label_path.relative_to(output_dir)),
        "segments_v2_json": str(segment_path.relative_to(output_dir)),
        "path_events_v2_json": str(events_path.relative_to(output_dir)),
        "dst_label_v2_preview_png": str(preview_path.relative_to(output_dir)),
        "segment_count_v2": stats["segments"],
        "long_jump_edges_v2": stats["long_jump_edges"],
        "trim_edges_v2": stats["trim_edges"],
        "color_change_edges_v2": stats["color_change_edges"],
        "closure_gap_edges_v2": stats["closure_gap_edges"],
        "width_mm": round(float(transform["width_mm"]), 4),
        "height_mm": round(float(transform["height_mm"]), 4),
        "stitch_commands": command_counts["stitch"],
        "jump_commands": command_counts["jump"],
        "trim_commands": command_counts["trim"],
        "color_change_commands": command_counts["color_change"],
    }


def iter_input_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.dst_dir:
        files = sorted([p for p in Path(args.dst_dir).rglob("*") if p.suffix.lower() in {".dst", ".pes"}])
        return [(f"sample_{i:05d}_{path.stem[:40]}", path) for i, path in enumerate(files, start=1)]
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        manifest_path = dataset_dir / args.manifest
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        pairs = []
        for row in rows:
            raw = row.get("source_dst") or row.get("source_file") or ""
            path = Path(raw)
            if not path.is_absolute():
                path = dataset_dir / raw
            pairs.append((row.get("pair_id") or path.stem, path))
        return pairs
    raise ValueError("Provide --dst-dir or --dataset-dir.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build enhanced DST-derived label v2 maps and path-event annotations.")
    parser.add_argument("--dst-dir", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--manifest", default="manifest_dataset2.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=3)
    parser.add_argument("--near-connect-mm", type=float, default=2.5)
    parser.add_argument("--long-jump-mm", type=float, default=8.0)
    parser.add_argument("--closure-gap-mm", type=float, default=6.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, (pair_id, path) in enumerate(iter_input_files(args), start=1):
        if args.limit and index > args.limit:
            break
        if not path.exists():
            failures.append({"pair_id": pair_id, "path": str(path), "error": "missing file"})
            continue
        try:
            rows.append(process_file(path, pair_id, output_dir, args))
        except Exception as exc:  # pragma: no cover - batch builder should continue.
            failures.append({"pair_id": pair_id, "path": str(path), "error": str(exc)})
        if index % 100 == 0:
            print(json.dumps({"index": index, "processed": len(rows), "failed": len(failures)}, ensure_ascii=False), flush=True)

    manifest_path = output_dir / "manifest_dst_label_v2.csv"
    if rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "processed": len(rows),
        "failed": len(failures),
        "channels": LABEL_V2_CHANNELS,
        "manifest": str(manifest_path),
        "failures": failures[:50],
    }
    (output_dir / "dst_label_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
