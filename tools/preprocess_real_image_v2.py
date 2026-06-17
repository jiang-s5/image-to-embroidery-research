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


def border_pixels(arr: np.ndarray, border: int) -> np.ndarray:
    strips = [
        arr[:border, :, :],
        arr[-border:, :, :],
        arr[:, :border, :],
        arr[:, -border:, :],
    ]
    return np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)


def remove_small_components(mask: np.ndarray, min_area: int, max_components: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    components: list[tuple[int, int]] = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    clean = np.zeros(mask.shape, dtype=np.uint8)
    for _area, label in components[:max_components]:
        clean[labels == label] = 1
    return clean.astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    work = mask.astype(np.uint8)
    h, w = work.shape
    flood = work.copy()
    pad = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 1)
    holes = (flood == 0) & (work == 0)
    return (work | holes.astype(np.uint8)).astype(bool)


def remove_border_slivers(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask.astype(bool)
    h, w = mask.shape
    clean = mask.astype(np.uint8).copy()
    for label in range(1, num):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_vertical_border = x <= 2 or x + bw >= w - 2
        touches_horizontal_border = y <= 2 or y + bh >= h - 2
        near_vertical_border = x <= int(w * 0.10) or x + bw >= int(w * 0.90)
        near_horizontal_border = y <= int(h * 0.10) or y + bh >= int(h * 0.90)
        very_thin_vertical = bh > h * 0.16 and bw <= max(14, int(w * 0.12))
        very_thin_horizontal = bw > w * 0.16 and bh <= max(14, int(h * 0.12))
        tiny_border_noise = area < max(12, int(h * w * 0.002)) and (touches_vertical_border or touches_horizontal_border)
        if (near_vertical_border and very_thin_vertical) or (near_horizontal_border and very_thin_horizontal) or tiny_border_noise:
            clean[labels == label] = 0
    return clean.astype(bool)


def remove_detached_region_noise(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    clean = mask.astype(np.uint8).copy()
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1, min(bw, bh))
        small = area < max(96, int(largest_area * 0.055))
        thin = aspect >= 4.0
        if small and thin:
            clean[labels == label] = 0
    return clean.astype(bool)


def estimate_initial_mask(arr: np.ndarray, color_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = cv2.bilateralFilter(arr, d=7, sigmaColor=45, sigmaSpace=45)
    border = max(8, int(round(min(arr.shape[:2]) * 0.08)))
    bg_rgb = np.median(border_pixels(smooth.astype(np.float32), border), axis=0)
    dist_rgb = np.linalg.norm(smooth.astype(np.float32) - bg_rgb[None, None, :], axis=2)

    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    local = cv2.GaussianBlur(gray, (0, 0), 7)
    local_dark = gray.astype(np.float32) < (local.astype(np.float32) - 16.0)

    probable = (
        (dist_rgb > color_distance)
        | ((saturation > 42.0) & (value < 248.0))
        | ((value < 92.0) & (dist_rgb > color_distance * 0.35))
        | (local_dark & (dist_rgb > color_distance * 0.25))
    )
    sure_fg = (
        (dist_rgb > color_distance * 1.75)
        | ((saturation > 72.0) & (value < 244.0))
        | ((value < 70.0) & (dist_rgb > color_distance * 0.45))
    )
    sure_bg = np.zeros(probable.shape, dtype=bool)
    sure_bg[:border, :] = True
    sure_bg[-border:, :] = True
    sure_bg[:, :border] = True
    sure_bg[:, -border:] = True
    sure_bg |= (dist_rgb < color_distance * 0.42) & (saturation < 32.0) & (value > 120.0)
    return probable, sure_fg, sure_bg


def grabcut_mask(arr: np.ndarray, probable: np.ndarray, sure_fg: np.ndarray, sure_bg: np.ndarray) -> np.ndarray:
    gc_mask = np.full(probable.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable] = cv2.GC_PR_FGD
    gc_mask[sure_fg] = cv2.GC_FGD
    gc_mask[sure_bg] = cv2.GC_BGD
    if int((gc_mask == cv2.GC_FGD).sum()) < 16 or int((gc_mask == cv2.GC_PR_FGD).sum()) < 64:
        return probable
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(arr, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return probable
    return (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)


def refine_mask(mask: np.ndarray, min_component_px: int, max_components: int) -> np.ndarray:
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    work = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel5)
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel3)
    clean = remove_small_components(work > 0, min_area=min_component_px, max_components=max_components)
    clean = fill_holes(clean)
    clean = cv2.morphologyEx(clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel3)
    return clean.astype(bool)


def estimate_thread_mask(arr: np.ndarray, color_distance: float, min_component_px: int, max_components: int) -> np.ndarray:
    smooth = cv2.bilateralFilter(arr, d=5, sigmaColor=35, sigmaSpace=35)
    border = max(8, int(round(min(arr.shape[:2]) * 0.08)))
    bg_rgb = np.median(border_pixels(smooth.astype(np.float32), border), axis=0)
    dist_rgb = np.linalg.norm(smooth.astype(np.float32) - bg_rgb[None, None, :], axis=2)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    local = cv2.GaussianBlur(gray, (0, 0), 5)
    local_dark = gray.astype(np.float32) < (local.astype(np.float32) - 13.0)
    local_bright = gray.astype(np.float32) > (local.astype(np.float32) + 18.0)

    thread = (
        ((saturation > 34.0) & (value < 248.0))
        | ((dist_rgb > color_distance * 0.85) & (saturation > 20.0))
        | (local_dark & (dist_rgb > color_distance * 0.25))
        | (local_bright & (dist_rgb > color_distance * 0.45) & (saturation > 18.0))
    )
    border_mask = np.zeros(thread.shape, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    thread &= ~(border_mask & (dist_rgb < color_distance * 0.7) & (saturation < 28.0))

    work = cv2.morphologyEx(thread.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    clean = remove_small_components(work > 0, min_area=max(8, min_component_px // 3), max_components=max_components * 4)
    return remove_border_slivers(clean)


def reconstruct_region_mask(
    thread_mask: np.ndarray,
    object_mask: np.ndarray,
    min_component_px: int,
    max_components: int,
) -> np.ndarray:
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    work = cv2.morphologyEx(thread_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel7, iterations=1)
    work = cv2.dilate(work, kernel7, iterations=1)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel13, iterations=1)
    region = fill_holes(work > 0)
    # If the object-level mask is reasonably selective, keep reconstructed
    # regions inside it. This avoids turning cloth/background into stitch fill.
    if object_mask.mean() < 0.78:
        guard = cv2.dilate(object_mask.astype(np.uint8), kernel7, iterations=1).astype(bool)
        region &= guard
    region = remove_small_components(region, min_area=max(min_component_px, 64), max_components=max_components * 2)
    region = remove_border_slivers(region)
    region = remove_detached_region_noise(region)
    if region.mean() > 0.72:
        return thread_mask
    if region.sum() < thread_mask.sum() * 1.15:
        return thread_mask
    return region


def quantize_masked(arr: np.ndarray, mask: np.ndarray, colors: int, seed: int) -> np.ndarray:
    filtered = cv2.bilateralFilter(arr, d=7, sigmaColor=50, sigmaSpace=50)
    result = np.full_like(arr, 255)
    if int(mask.sum()) < max(32, colors):
        result[mask] = filtered[mask]
        return result
    pixels = filtered[mask].reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.8)
    cv2.setRNGSeed(seed)
    _compactness, labels, centers = cv2.kmeans(pixels, colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    result[mask] = np.clip(centers[labels.flatten()], 0, 255).astype(np.uint8)
    return result


def make_design_input(arr: np.ndarray, mask: np.ndarray, quant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(quant, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    boundary = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edges = np.maximum(edges, boundary)
    edges[~mask] = 0
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    design = quant.copy()
    design[~mask] = 255
    edge_pixels = edges > 0
    design[edge_pixels] = np.maximum(0, design[edge_pixels].astype(np.int16) - 95).astype(np.uint8)

    cutout = arr.copy()
    cutout[~mask] = 255
    return cutout, design


def select_inference_mask(thread_mask: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, str, float]:
    ratio = float(region_mask.sum()) / max(1.0, float(thread_mask.sum()))
    # Region reconstruction helps filled motifs, but it can merge text/logo
    # strokes into blobs. The ratio gate keeps region only for moderate fill
    # expansion and falls back to thread-only for sparse line/text motifs.
    if 1.15 <= ratio <= 1.55 and float(region_mask.mean()) <= 0.50:
        return region_mask, "region_reconstruction", ratio
    return thread_mask, "thread_fallback", ratio


def preprocess(
    input_path: Path,
    output_dir: Path,
    size: int,
    colors: int,
    color_distance: float,
    min_component_px: int,
    max_components: int,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(input_path).convert("RGB")
    resized = resize_with_pad(source, size)
    arr = np.asarray(resized, dtype=np.uint8)

    probable, sure_fg, sure_bg = estimate_initial_mask(arr, color_distance=color_distance)
    raw_mask = grabcut_mask(arr, probable=probable, sure_fg=sure_fg, sure_bg=sure_bg)
    refined = refine_mask(raw_mask, min_component_px=min_component_px, max_components=max_components)
    if refined.mean() < 0.01 or refined.mean() > 0.82:
        refined = refine_mask(probable, min_component_px=min_component_px, max_components=max_components)

    thread_mask = estimate_thread_mask(arr, color_distance=color_distance, min_component_px=min_component_px, max_components=max_components)
    if refined.mean() < 0.82:
        thread_mask &= cv2.dilate(refined.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    if thread_mask.mean() < 0.005:
        thread_mask = refined
    region_mask = reconstruct_region_mask(
        thread_mask,
        refined,
        min_component_px=min_component_px,
        max_components=max_components,
    )

    quant = quantize_masked(arr, refined, colors=colors, seed=seed)
    thread_quant = quantize_masked(arr, thread_mask, colors=colors, seed=seed + 1)
    region_quant = quantize_masked(arr, region_mask, colors=colors, seed=seed + 2)
    cutout, design = make_design_input(arr, refined, quant)
    thread_cutout, thread_design = make_design_input(arr, thread_mask, thread_quant)
    region_cutout, region_design = make_design_input(arr, region_mask, region_quant)
    selected_mask, selected_source, region_thread_ratio = select_inference_mask(thread_mask, region_mask)
    selected_quant = quantize_masked(arr, selected_mask, colors=colors, seed=seed + 3)
    selected_cutout, selected_design = make_design_input(arr, selected_mask, selected_quant)

    Image.fromarray(arr).save(output_dir / "input.png")
    Image.fromarray((probable.astype(np.uint8) * 255)).save(output_dir / "init_mask.png")
    Image.fromarray((refined.astype(np.uint8) * 255)).save(output_dir / "mask.png")
    Image.fromarray((thread_mask.astype(np.uint8) * 255)).save(output_dir / "thread_mask.png")
    Image.fromarray((region_mask.astype(np.uint8) * 255)).save(output_dir / "region_mask.png")
    Image.fromarray(cutout).save(output_dir / "cutout.png")
    Image.fromarray(quant).save(output_dir / "quant.png")
    Image.fromarray(design).save(output_dir / "design.png")
    Image.fromarray(thread_cutout).save(output_dir / "thread_cutout.png")
    Image.fromarray(thread_design).save(output_dir / "thread_design.png")
    Image.fromarray(region_cutout).save(output_dir / "region_cutout.png")
    Image.fromarray(region_design).save(output_dir / "region_design.png")
    Image.fromarray((selected_mask.astype(np.uint8) * 255)).save(output_dir / "selected_mask.png")
    Image.fromarray(selected_cutout).save(output_dir / "selected_cutout.png")
    Image.fromarray(selected_design).save(output_dir / "selected_design.png")

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "source_size": [source.width, source.height],
        "size": size,
        "colors": colors,
        "color_distance": color_distance,
        "min_component_px": min_component_px,
        "max_components": max_components,
        "initial_foreground_pixels": int(probable.sum()),
        "initial_foreground_ratio": float(probable.mean()),
        "refined_foreground_pixels": int(refined.sum()),
        "refined_foreground_ratio": float(refined.mean()),
        "thread_foreground_pixels": int(thread_mask.sum()),
        "thread_foreground_ratio": float(thread_mask.mean()),
        "region_foreground_pixels": int(region_mask.sum()),
        "region_foreground_ratio": float(region_mask.mean()),
        "region_thread_area_ratio": region_thread_ratio,
        "selected_mask_source": selected_source,
        "selected_foreground_pixels": int(selected_mask.sum()),
        "selected_foreground_ratio": float(selected_mask.mean()),
        "files": [
            "input.png",
            "init_mask.png",
            "mask.png",
            "thread_mask.png",
            "region_mask.png",
            "cutout.png",
            "quant.png",
            "design.png",
            "thread_cutout.png",
            "thread_design.png",
            "region_cutout.png",
            "region_design.png",
            "selected_mask.png",
            "selected_cutout.png",
            "selected_design.png",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-photo foreground extraction for image-to-embroidery inference.")
    parser.add_argument("input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--colors", type=int, default=10)
    parser.add_argument("--color-distance", type=float, default=28.0)
    parser.add_argument("--min-component-px", type=int, default=48)
    parser.add_argument("--max-components", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()
    summary = preprocess(
        Path(args.input),
        Path(args.output_dir),
        size=args.size,
        colors=args.colors,
        color_distance=args.color_distance,
        min_component_px=args.min_component_px,
        max_components=args.max_components,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
