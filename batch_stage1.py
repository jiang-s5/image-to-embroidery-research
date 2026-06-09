from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stage1_pipeline import build_parser as build_stage1_parser
from stage1_pipeline import run_pipeline


def image_files(path: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run the Stage 1 pipeline on a logo folder.")
    parser.add_argument("--input-dir", default="datasets/logos/images", help="Folder of logo images.")
    parser.add_argument("--output-dir", default="outputs/logo_baseline", help="Folder for per-logo outputs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process.")
    parser.add_argument("--max-colors", type=int, default=5)
    parser.add_argument("--target-width-mm", type=float, default=80.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = image_files(input_dir)
    if args.limit > 0:
        files = files[: args.limit]

    stage1_parser = build_stage1_parser()
    rows = []
    for i, image_path in enumerate(files, start=1):
        logo_out = output_dir / image_path.stem
        stage_args = stage1_parser.parse_args(
            [
                str(image_path),
                "-o",
                str(logo_out),
                "--max-colors",
                str(args.max_colors),
                "--target-width-mm",
                str(args.target_width_mm),
            ]
        )
        summary = run_pipeline(stage_args)
        rows.append(
            {
                "filename": image_path.name,
                "output_dir": summary["output_dir"],
                "regions": summary["regions"],
                "blocks": summary["blocks"],
                "stitches": summary["stitches"],
                "jumps": summary["jumps"],
            }
        )
        print(f"[{i}/{len(files)}] {image_path.name}: {summary['regions']} regions, {summary['stitches']} stitches")

    summary_path = output_dir / "batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "output_dir", "regions", "blocks", "stitches", "jumps"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
