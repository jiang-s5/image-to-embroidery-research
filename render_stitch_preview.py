from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = [
    (0, 119, 182),
    (255, 183, 3),
    (16, 78, 112),
    (235, 70, 70),
    (49, 151, 149),
    (42, 49, 120),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a stitch_plan.tsv file to a PNG preview.")
    parser.add_argument("stitch_plan", help="Path to stitch_plan.tsv.")
    parser.add_argument("-o", "--output", default="", help="Output PNG path.")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=620)
    parser.add_argument("--margin", type=int, default=48)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = Path(args.stitch_plan)
    output = Path(args.output) if args.output else path.with_name("03_stitch_preview.png")

    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "region_id": int(row["region_id"]),
                    "color_index": int(row["color_index"]),
                    "command": row["command"],
                    "x": float(row["x_mm"]),
                    "y": float(row["y_mm"]),
                }
            )
    if not rows:
        raise SystemExit("stitch plan is empty")

    xs = [row["x"] for row in rows]
    ys = [row["y"] for row in rows]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(1e-6, maxx - minx)
    spany = max(1e-6, maxy - miny)
    scale = min((args.width - 2 * args.margin) / spanx, (args.height - 2 * args.margin) / spany)

    def project(x: float, y: float) -> tuple[int, int]:
        px = args.margin + int((x - minx) * scale)
        py = args.margin + int((y - miny) * scale)
        return px, py

    image = Image.new("RGB", (args.width, args.height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((args.margin, args.margin, args.width - args.margin, args.height - args.margin), outline=(230, 234, 238))

    last = None
    last_color = None
    for row in rows:
        color = COLORS[row["color_index"] % len(COLORS)]
        point = project(row["x"], row["y"])
        if row["command"] == "JUMP" or last is None or last_color != row["color_index"]:
            if last is not None:
                draw.line((last, point), fill=(205, 210, 216), width=1)
            last = point
            last_color = row["color_index"]
            continue
        draw.line((last, point), fill=color, width=2)
        last = point
        last_color = row["color_index"]

    image.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
