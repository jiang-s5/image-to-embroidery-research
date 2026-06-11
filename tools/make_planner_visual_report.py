from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_dst_preview import render


DEFAULT_METHODS = [
    ("A0", "A0_continuity"),
    ("A2 fixed", "A2_fixed_params"),
    ("A2 retrieval", "A2_retrieval_relation"),
]


def parse_methods(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_METHODS
    result = []
    for value in values:
        if ":" in value:
            label, method = value.split(":", 1)
        else:
            label, method = value, value
        result.append((label.strip(), method.strip()))
    return result


def sample_dirs(run_dir: Path, limit: int) -> list[Path]:
    samples = sorted(path for path in run_dir.iterdir() if path.is_dir())
    return samples[:limit] if limit else samples


def build_sheet(
    run_dir: Path,
    preprocess_dir: Path,
    output_dir: Path,
    methods: list[tuple[str, str]],
    limit: int,
    size: int,
) -> Path:
    render_dir = output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_dirs(run_dir, limit)
    cols = [("Input", "input")] + methods
    thumb = 210
    label_h = 38
    sample_w = 230
    header_h = 36
    pad = 10
    width = sample_w + len(cols) * (thumb + pad) + pad
    height = header_h + len(samples) * (thumb + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for col_index, (label, _method) in enumerate(cols):
        x = sample_w + col_index * (thumb + pad) + pad
        draw.text((x, 12), label, fill=(20, 20, 20), font=font)

    for row_index, sample_dir in enumerate(samples):
        sample_id = sample_dir.name
        y = header_h + row_index * (thumb + label_h + pad) + pad
        short = sample_id.replace("dst3322_", "").replace("_model10_continuity_prediction", "")
        draw.text((pad, y + 6), short[:32], fill=(20, 20, 20), font=font)
        for col_index, (_label, method) in enumerate(cols):
            if method == "input":
                image_path = preprocess_dir / sample_id / "input_resized.png"
            else:
                dst_path = sample_dir / method / "embroidery_output.dst"
                if not dst_path.exists():
                    continue
                image_path = render_dir / f"{sample_id}_{method}.png"
                if not image_path.exists():
                    render(dst_path, image_path, size=size, padding=28, line_width=2)
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
            cell_x = sample_w + col_index * (thumb + pad) + pad
            x = cell_x + (thumb - image.width) // 2
            image_y = y + (thumb - image.height) // 2
            sheet.paste(image, (x, image_y))
            draw.rectangle([cell_x, y, cell_x + thumb, y + thumb], outline=(220, 220, 220))

    output = output_dir / "planner_visual_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render planner DST outputs into an input/A0/A2 visual contact sheet.")
    parser.add_argument("--ablation-dir", default="outputs/planner_ablation_mini_demo")
    parser.add_argument("--output-dir", default="outputs/planner_visual_report")
    parser.add_argument("--method", action="append", default=[], help="Method or label:method pair. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ablation_dir = Path(args.ablation_dir)
    output = build_sheet(
        run_dir=ablation_dir / "runs",
        preprocess_dir=ablation_dir / "preprocess",
        output_dir=Path(args.output_dir),
        methods=parse_methods(args.method),
        limit=args.limit,
        size=args.size,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
