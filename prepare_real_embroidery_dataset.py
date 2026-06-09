from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from pyembroidery import COLOR_CHANGE, COMMAND_MASK, END, JUMP, STITCH, TRIM, read_dst, read_pes


def command_counts(pattern) -> dict[str, int]:
    commands = [stitch[2] & COMMAND_MASK for stitch in pattern.stitches]
    return {
        "command_count": len(commands),
        "stitch_commands": sum(cmd == STITCH for cmd in commands),
        "jump_commands": sum(cmd == JUMP for cmd in commands),
        "trim_commands": sum(cmd == TRIM for cmd in commands),
        "color_changes": sum(cmd == COLOR_CHANGE for cmd in commands),
        "end_commands": sum(cmd == END for cmd in commands),
    }


def movement_stats(pattern) -> dict[str, float]:
    points = [(x / 10.0, y / 10.0, cmd & COMMAND_MASK) for x, y, cmd in pattern.stitches]
    if not points:
        return {
            "width_mm": 0.0,
            "height_mm": 0.0,
            "max_move_mm": 0.0,
            "total_jump_mm": 0.0,
        }
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    max_move = 0.0
    total_jump = 0.0
    last = None
    for x, y, cmd in points:
        pt = (x, y)
        if last is not None and cmd in {STITCH, JUMP}:
            distance = math.dist(last, pt)
            max_move = max(max_move, distance)
            if cmd == JUMP:
                total_jump += distance
        if cmd in {STITCH, JUMP}:
            last = pt
    return {
        "width_mm": round(max(xs) - min(xs), 3),
        "height_mm": round(max(ys) - min(ys), 3),
        "max_move_mm": round(max_move, 3),
        "total_jump_mm": round(total_jump, 3),
    }


def render_pattern(pattern, output: Path, size: int = 900, margin: int = 44) -> None:
    stitch_points = [(x / 10.0, y / 10.0, cmd & COMMAND_MASK) for x, y, cmd in pattern.stitches if (cmd & COMMAND_MASK) in {STITCH, JUMP}]
    if not stitch_points:
        return
    xs = [x for x, _, _ in stitch_points]
    ys = [y for _, y, _ in stitch_points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = max(1e-6, maxx - minx)
    spany = max(1e-6, maxy - miny)
    scale = min((size - 2 * margin) / spanx, (size - 2 * margin) / spany)

    def project(x: float, y: float) -> tuple[int, int]:
        return margin + int((x - minx) * scale), margin + int((y - miny) * scale)

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    last = None
    for x, y, cmd in stitch_points:
        point = project(x, y)
        if last is not None:
            if cmd == JUMP:
                draw.line((last, point), fill=(205, 210, 216), width=1)
            else:
                draw.line((last, point), fill=(18, 92, 145), width=2)
        last = point
    image.save(output)


def find_design_dirs(root: Path) -> list[Path]:
    dirs = set()
    for ext in ["*.dst", "*.pes"]:
        for path in root.rglob(ext):
            dirs.add(path.parent)
    return sorted(dirs)


def first_image(folder: Path) -> Path | None:
    for pattern in ["*.png", "*.jpg", "*.jpeg"]:
        found = sorted(folder.glob(pattern))
        if found:
            return found[0]
    return None


def prepare(root: Path, output: Path) -> list[dict[str, str | int | float]]:
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    image_dir = output / "source_images"
    image_dir.mkdir(exist_ok=True)

    rows: list[dict[str, str | int | float]] = []
    for folder in find_design_dirs(root):
        design_id = folder.name
        dst_path = next(iter(sorted(folder.glob("*.dst"))), None)
        pes_path = next(iter(sorted(folder.glob("*.pes"))), None)
        source_image = first_image(folder)

        preferred_path = dst_path or pes_path
        if preferred_path is None:
            continue
        reader = read_dst if preferred_path.suffix.lower() == ".dst" else read_pes
        pattern = reader(str(preferred_path))

        preview_path = preview_dir / f"{design_id}_stitch_preview.png"
        render_pattern(pattern, preview_path)

        copied_image = ""
        if source_image is not None:
            target = image_dir / f"{design_id}{source_image.suffix.lower()}"
            shutil.copy2(source_image, target)
            copied_image = str(target)

        row = {
            "design_id": design_id,
            "folder": str(folder),
            "source_image": copied_image,
            "dst_path": str(dst_path) if dst_path else "",
            "pes_path": str(pes_path) if pes_path else "",
            "preview_path": str(preview_path),
            "threads_in_preferred_file": pattern.count_threads(),
        }
        row.update(command_counts(pattern))
        row.update(movement_stats(pattern))
        rows.append(row)

    manifest = output / "manifest.csv"
    if rows:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an external embroidery repository as a real-sample dataset.")
    parser.add_argument("--repo-root", required=True, help="Path to an embroidery design repository.")
    parser.add_argument("--output", default="datasets/real_embroidery", help="Prepared dataset output folder.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = prepare(Path(args.repo_root), Path(args.output))
    print(f"prepared {len(rows)} designs")
    if rows:
        print(f"manifest: {Path(args.output) / 'manifest.csv'}")
    if len(rows) < 20:
        print("warning: fewer than 20 designs; this is useful for parsing/evaluation, not meaningful model training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
