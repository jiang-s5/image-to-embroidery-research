from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from pyembroidery import COLOR_CHANGE, COMMAND_MASK, JUMP, STITCH, read_dst


COLORS = [
    (16, 96, 150),
    (28, 132, 78),
    (190, 46, 44),
    (222, 160, 35),
    (92, 72, 150),
    (35, 35, 35),
]


def read_segments(path: Path) -> list[tuple[float, float, float, float, int, int]]:
    pattern = read_dst(str(path))
    segments: list[tuple[float, float, float, float, int, int]] = []
    last: tuple[float, float] | None = None
    color_index = 0
    for x_raw, y_raw, cmd_raw in pattern.stitches:
        x = float(x_raw) / 10.0
        y = float(y_raw) / 10.0
        cmd = cmd_raw & COMMAND_MASK
        if cmd == COLOR_CHANGE:
            color_index += 1
        if last is not None and cmd in {STITCH, JUMP}:
            segments.append((last[0], last[1], x, y, color_index, cmd))
        last = (x, y)
    return segments


def fit(
    segments: list[tuple[float, float, float, float, int, int]],
    size: int,
    padding: int,
) -> tuple[list[tuple[float, float, float, float, int, int]], dict[str, float]]:
    stitch_segments = [seg for seg in segments if seg[5] == STITCH]
    if not stitch_segments:
        stitch_segments = segments
    xs = [seg[0] for seg in stitch_segments] + [seg[2] for seg in stitch_segments]
    ys = [seg[1] for seg in stitch_segments] + [seg[3] for seg in stitch_segments]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(1e-6, max_x - min_x)
    height = max(1e-6, max_y - min_y)
    scale = (size - 2 * padding) / max(width, height)
    ox = (size - width * scale) / 2.0
    oy = (size - height * scale) / 2.0
    fitted = [
        (
            ox + (x0 - min_x) * scale,
            oy + (y0 - min_y) * scale,
            ox + (x1 - min_x) * scale,
            oy + (y1 - min_y) * scale,
            color,
            cmd,
        )
        for x0, y0, x1, y1, color, cmd in segments
    ]
    return fitted, {"width_mm": width, "height_mm": height, "scale": scale}


def fabric(size: int) -> Image.Image:
    image = Image.new("RGB", (size, size), (238, 235, 222))
    draw = ImageDraw.Draw(image)
    for y in range(0, size, 6):
        draw.line([(0, y), (size, y)], fill=(226, 224, 211), width=1)
    for x in range(0, size, 7):
        draw.line([(x, 0), (x, size)], fill=(229, 227, 214), width=1)
    return image.filter(ImageFilter.GaussianBlur(0.2)).convert("RGBA")


def render(path: Path, output: Path, size: int, padding: int, line_width: int) -> dict[str, object]:
    raw = read_segments(path)
    segments, info = fit(raw, size=size, padding=padding)
    scale = 3
    canvas_size = size * scale
    base = fabric(size).resize((canvas_size, canvas_size), Image.Resampling.BICUBIC)
    shadow = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    thread = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    td = ImageDraw.Draw(thread)
    stitch_count = 0
    jump_count = 0
    for i, (x0, y0, x1, y1, color_index, cmd) in enumerate(segments):
        if cmd == JUMP:
            jump_count += 1
            continue
        stitch_count += 1
        color = COLORS[color_index % len(COLORS)]
        factor = 0.82 + 0.24 * ((i % 9) / 8.0)
        shaded = tuple(max(0, min(255, int(c * factor))) for c in color)
        xy = [(x0 * scale, y0 * scale), (x1 * scale, y1 * scale)]
        shadow_xy = [(xy[0][0] + 2.0, xy[0][1] + 2.0), (xy[1][0] + 2.0, xy[1][1] + 2.0)]
        sd.line(shadow_xy, fill=(20, 18, 15, 50), width=max(2, line_width * scale + 2))
        td.line(xy, fill=(*shaded, 238), width=max(1, line_width * scale))
        td.line(xy, fill=(255, 255, 255, 42), width=1)
    result = Image.alpha_composite(base, shadow.filter(ImageFilter.GaussianBlur(1.0)))
    result = Image.alpha_composite(result, thread.filter(ImageFilter.GaussianBlur(0.22)))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").resize((size, size), Image.Resampling.LANCZOS).save(output)
    return {
        "input_dst": str(path),
        "output_png": str(output),
        "stitch_segments": stitch_count,
        "jump_segments": jump_count,
        **info,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a DST file into a fabric-style PNG preview.")
    parser.add_argument("input_dst")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--padding", type=int, default=40)
    parser.add_argument("--line-width", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = render(
        Path(args.input_dst),
        Path(args.output),
        size=args.size,
        padding=args.padding,
        line_width=args.line_width,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
