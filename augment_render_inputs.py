from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def fabric_texture(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    base = np.full((height, width, 3), (236, 233, 220), dtype=np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 5.0, (height, width, 1))
    base += noise
    for y in range(0, height, rng.choice([5, 6, 7])):
        base[y : y + 1, :, :] -= rng.uniform(3.0, 8.0)
    for x in range(0, width, rng.choice([6, 7, 8])):
        base[:, x : x + 1, :] -= rng.uniform(2.0, 6.0)
    return Image.fromarray(np.uint8(np.clip(base, 0, 255))).filter(ImageFilter.GaussianBlur(0.25))


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    texture = fabric_texture(image.size, rng)
    alpha = rng.uniform(0.08, 0.24)
    image = Image.blend(image, texture, alpha)
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.82, 1.22))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.86, 1.18))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
    if rng.random() < 0.45:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.75)))
    arr = np.asarray(image, dtype=np.float32)
    yy = np.linspace(-1.0, 1.0, arr.shape[0], dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, arr.shape[1], dtype=np.float32)[None, :]
    light = 1.0 + rng.uniform(-0.12, 0.12) * xx + rng.uniform(-0.12, 0.12) * yy
    arr *= light[..., None]
    arr += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.0, 4.0), arr.shape)
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))


def iter_image_rows(args: argparse.Namespace) -> tuple[Path, list[dict[str, str]]]:
    if args.dataset_dir:
        root = Path(args.dataset_dir)
        manifest_path = root / args.manifest
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return root, rows
    if args.input_dir:
        root = Path(args.input_dir)
        rows = [{"pair_id": path.stem, "input_png": str(path.relative_to(root))} for path in sorted(root.rglob("*.png"))]
        return root, rows
    raise ValueError("Provide --dataset-dir or --input-dir.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create photometric render augmentations while preserving geometry labels.")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--manifest", default="manifest_dataset2.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    root, rows = iter_image_rows(args)
    if args.limit:
        rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    out_rows: list[dict[str, str]] = []
    for row in rows:
        input_rel = row.get("input_png", "")
        if not input_rel:
            continue
        src = root / input_rel
        if not src.exists():
            continue
        image = Image.open(src).convert("RGB")
        for variant in range(args.variants):
            pair_id = f"{row.get('pair_id', src.stem)}_aug{variant + 1:02d}"
            out_name = f"{pair_id}.png"
            out_path = image_dir / out_name
            augment_image(image, rng).save(out_path)
            out_row = dict(row)
            out_row["pair_id"] = pair_id
            out_row["input_png"] = str(out_path.relative_to(output_dir))
            out_row["source_kind"] = "render_augmented"
            out_rows.append(out_row)

    manifest_path = output_dir / "manifest_render_augmented.csv"
    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
    summary = {"output_dir": str(output_dir), "augmented": len(out_rows), "manifest": str(manifest_path)}
    (output_dir / "render_augmentation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
