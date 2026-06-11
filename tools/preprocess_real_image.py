from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def resize_with_pad(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    new_size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def crop_panel_input(image: Image.Image, enabled: bool) -> Image.Image:
    if not enabled or image.width < image.height * 2:
        return image
    tile_width = min(image.height, max(1, int(round(image.width / 7.0)))) if image.width > image.height * 4 else image.height
    header = max(0, image.height - tile_width)
    return image.crop((0, header, tile_width, header + tile_width))


def border_background_color(arr: np.ndarray, border: int = 12) -> np.ndarray:
    strips = [
        arr[:border, :, :],
        arr[-border:, :, :],
        arr[:, :border, :],
        arr[:, -border:, :],
    ]
    pixels = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    return np.median(pixels, axis=0)


def foreground_mask(arr: np.ndarray, close_px: int, min_component_px: int) -> np.ndarray:
    arr_f = arr.astype(np.float32)
    bg = border_background_color(arr_f)
    dist = np.linalg.norm(arr_f - bg[None, None, :], axis=2)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)
    mask = (dist > 22.0) | ((sat > 35.0) & (value < 248.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, close_px), max(1, close_px)))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_px:
            clean[labels == label] = 1
    return clean.astype(bool)


def quantize_colors(arr: np.ndarray, mask: np.ndarray, colors: int, seed: int) -> np.ndarray:
    work = cv2.bilateralFilter(arr, d=7, sigmaColor=45, sigmaSpace=45)
    pixels = work[mask].reshape(-1, 3).astype(np.float32)
    if len(pixels) < colors:
        result = np.full_like(arr, 255)
        result[mask] = work[mask]
        return result
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers = cv2.kmeans(pixels, colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quant = np.full_like(arr, 255)
    quant_pixels = np.clip(centers[labels.flatten()], 0, 255).astype(np.uint8)
    quant[mask] = quant_pixels
    return quant


def edge_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 45, 120)
    boundary = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    merged = np.maximum(edges, boundary)
    merged[~mask] = 0
    return cv2.dilate(merged, np.ones((2, 2), np.uint8), iterations=1)


def canonicalize(
    input_path: Path,
    output_dir: Path,
    size: int,
    colors: int,
    close_px: int,
    min_component_px: int,
    seed: int,
    auto_crop_panel: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(input_path).convert("RGB")
    cropped = crop_panel_input(source, auto_crop_panel)
    image = resize_with_pad(cropped, size)
    arr = np.asarray(image, dtype=np.uint8)
    mask = foreground_mask(arr, close_px=close_px, min_component_px=min_component_px)
    quant = quantize_colors(arr, mask, colors=colors, seed=seed)
    edges = edge_map(quant, mask)
    canonical = quant.copy()
    canonical[~mask] = 255
    edge_pixels = edges > 0
    canonical[edge_pixels] = np.maximum(0, canonical[edge_pixels].astype(np.int16) - 90).astype(np.uint8)

    image.save(output_dir / "input_resized.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_dir / "foreground_mask.png")
    Image.fromarray(quant).save(output_dir / "color_quantized.png")
    Image.fromarray(edges).save(output_dir / "edge_map.png")
    Image.fromarray(canonical).save(output_dir / "canonical_input.png")
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "size": size,
        "colors": colors,
        "auto_crop_panel": auto_crop_panel,
        "source_size": [source.width, source.height],
        "cropped_size": [cropped.width, cropped.height],
        "foreground_pixels": int(mask.sum()),
        "foreground_ratio": float(mask.mean()),
        "files": [
            "input_resized.png",
            "foreground_mask.png",
            "color_quantized.png",
            "edge_map.png",
            "canonical_input.png",
        ],
    }
    (output_dir / "preprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize a real image into an embroidery-design-like input.")
    parser.add_argument("input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--colors", type=int, default=8)
    parser.add_argument("--close-px", type=int, default=9)
    parser.add_argument("--min-component-px", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--no-auto-crop-panel", action="store_true")
    args = parser.parse_args()
    summary = canonicalize(
        Path(args.input),
        Path(args.output_dir),
        size=args.size,
        colors=args.colors,
        close_px=args.close_px,
        min_component_px=args.min_component_px,
        seed=args.seed,
        auto_crop_panel=not args.no_auto_crop_panel,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
