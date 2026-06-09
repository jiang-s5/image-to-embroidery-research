from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a contact sheet for generated logos.")
    parser.add_argument("--input-dir", default="datasets/logos/images")
    parser.add_argument("--output", default="datasets/logos/contact_sheet.png")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--thumb-height", type=int, default=150)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no PNG files found in {input_dir}")

    cols = max(1, args.cols)
    rows = (len(files) + cols - 1) // cols
    label_h = 24
    pad = 14
    cell_w = args.thumb_width + pad * 2
    cell_h = args.thumb_height + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        image.thumbnail((args.thumb_width, args.thumb_height), Image.Resampling.LANCZOS)
        col = i % cols
        row = i // cols
        x0 = col * cell_w
        y0 = row * cell_h
        ix = x0 + pad + (args.thumb_width - image.width) // 2
        iy = y0 + pad
        sheet.paste(image, (ix, iy))
        label = path.stem.replace("logo_", "")
        draw.text((x0 + pad, y0 + pad + args.thumb_height + 4), label, fill=(32, 36, 40), font=font)

    sheet.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
