from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pyembroidery import COLOR_CHANGE, COMMAND_MASK, STITCH, read_dst


THREAD_PALETTE = [
    (20, 95, 150),
    (28, 120, 70),
    (180, 40, 45),
    (230, 165, 35),
    (95, 70, 150),
    (35, 35, 35),
    (220, 90, 35),
    (25, 135, 160),
]


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    color_index: int


def safe_id(path: Path, index: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")
    return f"dst3322_{index:05d}_{stem[:48] or 'design'}"


def read_dst_segments(path: Path) -> list[Segment]:
    pattern = read_dst(str(path))
    segments: list[Segment] = []
    last: tuple[float, float] | None = None
    color_index = 0
    for x_raw, y_raw, command_raw in pattern.stitches:
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        command = command_raw & COMMAND_MASK
        if command == COLOR_CHANGE:
            color_index += 1
        if last is not None and command == STITCH:
            segments.append(Segment(last[0], last[1], x, y, color_index))
        last = (x, y)
    return segments


def fit_segments(
    segments: list[Segment],
    size: int,
    padding: int,
) -> tuple[list[Segment], dict[str, float]]:
    points = [(seg.x0, seg.y0) for seg in segments] + [(seg.x1, seg.y1) for seg in segments]
    if not points:
        return [], {"scale": 1.0, "width_mm": 0.0, "height_mm": 0.0, "min_x": 0.0, "min_y": 0.0}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    scale = (size - 2 * padding) / max(width, height)
    offset_x = (size - width * scale) / 2.0
    offset_y = (size - height * scale) / 2.0

    fitted = [
        Segment(
            offset_x + (seg.x0 - min_x) * scale,
            offset_y + (seg.y0 - min_y) * scale,
            offset_x + (seg.x1 - min_x) * scale,
            offset_y + (seg.y1 - min_y) * scale,
            seg.color_index,
        )
        for seg in segments
    ]
    return fitted, {
        "scale": float(scale),
        "width_mm": float(width),
        "height_mm": float(height),
        "min_x": float(min_x),
        "min_y": float(min_y),
    }


def rasterize_labels(
    segments: list[Segment],
    size: int,
    line_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.zeros((size, size), dtype=np.float32)
    dir_x = np.zeros((size, size), dtype=np.float32)
    dir_y = np.zeros((size, size), dtype=np.float32)
    radius = max(0, line_width // 2)
    offsets = [
        (ox, oy)
        for oy in range(-radius, radius + 1)
        for ox in range(-radius, radius + 1)
        if ox * ox + oy * oy <= radius * radius
    ] or [(0, 0)]

    for seg in segments:
        dx = seg.x1 - seg.x0
        dy = seg.y1 - seg.y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
        xs = np.rint(np.linspace(seg.x0, seg.x1, steps + 1)).astype(np.int32)
        ys = np.rint(np.linspace(seg.y0, seg.y1, steps + 1)).astype(np.int32)
        ux = dx / length
        uy = dy / length
        for ox, oy in offsets:
            xx = np.clip(xs + ox, 0, size - 1)
            yy = np.clip(ys + oy, 0, size - 1)
            np.add.at(count, (yy, xx), 1.0)
            np.add.at(dir_x, (yy, xx), ux)
            np.add.at(dir_y, (yy, xx), uy)

    touched = count > 0
    direction = np.zeros((2, size, size), dtype=np.float32)
    direction[0, touched] = dir_x[touched] / count[touched]
    direction[1, touched] = dir_y[touched] / count[touched]
    if np.any(touched):
        denom = max(1.0, float(np.percentile(count[touched], 99)))
        density = np.clip(np.log1p(count) / math.log1p(denom), 0.0, 1.0).astype(np.float32)
    else:
        density = count
    return touched.astype(np.float32), density, direction


def fabric_background(size: int, rng: np.random.Generator) -> Image.Image:
    base = np.full((size, size, 3), (237, 234, 220), dtype=np.uint8)
    noise = rng.normal(0, 5.0, (size, size, 1)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)
    for y in range(0, size, 5):
        shade = int(222 + rng.integers(-4, 5))
        draw.line([(0, y), (size, y)], fill=(shade, shade, shade - 8), width=1)
    for x in range(0, size, 7):
        shade = int(226 + rng.integers(-4, 5))
        draw.line([(x, 0), (x, size)], fill=(shade, shade, shade - 8), width=1)
    return image.filter(ImageFilter.GaussianBlur(radius=0.25))


def shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * factor))) for channel in color)


def render_effect(
    segments: list[Segment],
    size: int,
    line_width: int,
    seed: int,
) -> Image.Image:
    scale = 3
    high_size = size * scale
    rng = np.random.default_rng(seed)
    image = fabric_background(size, rng).resize((high_size, high_size), Image.Resampling.BICUBIC).convert("RGBA")
    shadow = Image.new("RGBA", (high_size, high_size), (0, 0, 0, 0))
    thread = Image.new("RGBA", (high_size, high_size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    thread_draw = ImageDraw.Draw(thread)
    width = max(1, line_width * scale)
    shadow_width = max(width + scale, width)

    for index, seg in enumerate(segments):
        color = THREAD_PALETTE[seg.color_index % len(THREAD_PALETTE)]
        local = shade(color, 0.82 + 0.28 * ((index % 7) / 6.0))
        xy = [
            (seg.x0 * scale, seg.y0 * scale),
            (seg.x1 * scale, seg.y1 * scale),
        ]
        shadow_xy = [(xy[0][0] + 1.2 * scale, xy[0][1] + 1.2 * scale), (xy[1][0] + 1.2 * scale, xy[1][1] + 1.2 * scale)]
        shadow_draw.line(shadow_xy, fill=(30, 25, 20, 38), width=shadow_width)
        thread_draw.line(xy, fill=(*local, 235), width=width)
        if width >= 3:
            highlight = shade(color, 1.22)
            thread_draw.line(xy, fill=(*highlight, 75), width=1)

    image = Image.alpha_composite(image, shadow.filter(ImageFilter.GaussianBlur(radius=0.45 * scale)))
    image = Image.alpha_composite(image, thread.filter(ImageFilter.GaussianBlur(radius=0.12 * scale)))
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def label_preview(mask: np.ndarray, density: np.ndarray) -> Image.Image:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (density * 255).astype(np.uint8)
    rgb[..., 1] = (mask * 120).astype(np.uint8)
    rgb[..., 2] = (mask * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: int = 128, cols: int = 8) -> None:
    if not image_paths:
        return
    rows = int(math.ceil(len(image_paths) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + 18)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb_size
        y = (i // cols) * (thumb_size + 18)
        sheet.paste(image, (x + (thumb_size - image.width) // 2, y))
        draw.text((x + 4, y + thumb_size + 2), path.stem[:18], fill=(0, 0, 0))
    sheet.save(output_path)


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def prepare(args: argparse.Namespace) -> dict[str, object]:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    previews_dir = output_dir / "previews"
    for directory in (images_dir, labels_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dst_paths = sorted([path for path in input_dir.rglob("*") if path.suffix.lower() == ".dst"])
    rng = random.Random(args.seed)
    rng.shuffle(dst_paths)
    if args.max_files:
        dst_paths = dst_paths[: args.max_files]

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    contact_paths: list[Path] = []
    total_candidates = len(dst_paths)
    for candidate_index, dst_path in enumerate(dst_paths):
        try:
            raw_segments = read_dst_segments(dst_path)
        except Exception as exc:
            skipped.append({"source_dst": str(dst_path), "reason": f"read failed: {exc}"})
            continue
        if len(raw_segments) < args.min_segments:
            skipped.append({"source_dst": str(dst_path), "reason": f"too few stitch segments: {len(raw_segments)}"})
            continue
        if args.max_segments and len(raw_segments) > args.max_segments:
            skipped.append({"source_dst": str(dst_path), "reason": f"too many stitch segments: {len(raw_segments)}"})
            continue

        pair_id = safe_id(dst_path, len(rows) + 1)
        image_path = images_dir / f"{pair_id}.png"
        mask_path = labels_dir / f"{pair_id}_mask.png"
        density_path = labels_dir / f"{pair_id}_density.npy"
        direction_path = labels_dir / f"{pair_id}_direction.npy"
        preview_path = previews_dir / f"{pair_id}_label_preview.png"

        fitted, transform = fit_segments(raw_segments, size=args.size, padding=args.padding)
        effect = render_effect(fitted, size=args.size, line_width=args.render_line_width, seed=args.seed + len(rows))
        mask, density, direction = rasterize_labels(fitted, size=args.size, line_width=args.label_line_width)

        effect.save(image_path)
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)
        np.save(density_path, density)
        np.save(direction_path, direction)
        label_preview(mask, density).save(preview_path)
        if len(contact_paths) < args.contact_count:
            contact_paths.append(image_path)

        split = split_name(len(rows), total_candidates, args.train_ratio, args.val_ratio)
        rows.append(
            {
                "pair_id": pair_id,
                "split": split,
                "source_kind": "dst_rendered",
                "input_png": str(image_path.relative_to(output_dir)),
                "mask_png": str(mask_path.relative_to(output_dir)),
                "density_npy": str(density_path.relative_to(output_dir)),
                "direction_npy": str(direction_path.relative_to(output_dir)),
                "source_dst": str(dst_path),
                "segments": len(raw_segments),
                "mask_pixels": int(mask.sum()),
                "width_mm": round(transform["width_mm"], 3),
                "height_mm": round(transform["height_mm"], 3),
                "note": "Rendered stitch-effect image from DST; label is decoded from the same DST.",
            }
        )
        if args.progress_every and len(rows) % args.progress_every == 0:
            print(f"prepared {len(rows)} usable DST designs")

    write_csv(
        output_dir / "manifest.csv",
        rows,
        [
            "pair_id",
            "split",
            "source_kind",
            "input_png",
            "mask_png",
            "density_npy",
            "direction_npy",
            "source_dst",
            "segments",
            "mask_pixels",
            "width_mm",
            "height_mm",
            "note",
        ],
    )
    write_csv(output_dir / "skipped.csv", skipped, ["source_dst", "reason"])
    make_contact_sheet(contact_paths, output_dir / "contact_sheet.png")

    split_counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "candidate_dst_files": total_candidates,
        "usable_designs": len(rows),
        "skipped": len(skipped),
        "split_counts": split_counts,
        "canvas_size": args.size,
        "label_channels": ["mask_png", "density_npy", "direction_npy"],
        "note": "DST-only corpus: rendered DST image is input; the same DST is decoded as supervision.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# DST Rendered Supervision 3322\n\n"
        "Input images are rendered from DST files. Labels are decoded from the same DST files:\n\n"
        "- `mask_png`: stitch path mask\n"
        "- `density_npy`: stitch-density raster\n"
        "- `direction_npy`: two-channel stitch direction map\n\n"
        "This is useful for pretraining on real DST stitch structure. It is not yet a paired natural-photo-to-DST dataset.\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render DST files and build image-to-stitch supervision.")
    parser.add_argument("--input-dir", default="datasets/dst_corpus_3322/raw")
    parser.add_argument("--output-dir", default="datasets/dst_rendered_supervision_3322")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=14)
    parser.add_argument("--render-line-width", type=int, default=2)
    parser.add_argument("--label-line-width", type=int, default=2)
    parser.add_argument("--min-segments", type=int, default=50)
    parser.add_argument("--max-segments", type=int, default=50000)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--contact-count", type=int, default=48)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3830)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = prepare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
