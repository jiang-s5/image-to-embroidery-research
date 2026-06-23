from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class GeometryPriors:
    clean_mask: np.ndarray
    dt_map: np.ndarray
    sobel_mag: np.ndarray
    canny_edges: np.ndarray
    centerline_map: np.ndarray | None
    meta: dict[str, Any]


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def normalize_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    maximum = float(arr.max()) if arr.size else 0.0
    if maximum <= 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / maximum * 255.0, 0, 255).astype(np.uint8)


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode PNG: {path}")
    encoded.tofile(str(path))


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(bool)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(labels_count, dtype=bool)
    keep[0] = False
    for label in range(1, labels_count):
        keep[label] = int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area)
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return mask.astype(bool)
    mask_bool = mask.astype(bool)
    inv = (~mask_bool).astype(np.uint8)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = mask_bool.copy()
    height, width = mask_bool.shape
    border_labels = set(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[height - 1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, width - 1]).tolist())
    for label in range(1, labels_count):
        if label in border_labels:
            continue
        if int(stats[label, cv2.CC_STAT_AREA]) <= int(max_area):
            filled[labels == label] = True
    return filled


def clean_binary_mask(
    mask_u8: np.ndarray,
    morph_open: int = 3,
    morph_close: int = 5,
    remove_small_objects: int = 64,
    remove_small_holes: int = 128,
) -> np.ndarray:
    mask = np.asarray(mask_u8)
    if mask.ndim != 2:
        raise ValueError("mask_u8 must be a 2D array")
    clean = mask > 0
    clean = remove_small_components(clean, remove_small_objects)
    clean = fill_small_holes(clean, remove_small_holes)
    open_size = odd_kernel(morph_open)
    close_size = odd_kernel(morph_close)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    clean_u8 = (clean.astype(np.uint8) * 255)
    if open_size > 1:
        clean_u8 = cv2.morphologyEx(clean_u8, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if close_size > 1:
        clean_u8 = cv2.morphologyEx(clean_u8, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return (clean_u8 > 0).astype(np.uint8)


def image_to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    # The project generally uses PIL/RGB arrays. The exact color order is not
    # important for grayscale edge priors, but RGB conversion is the least
    # surprising default here.
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def build_sobel_canny(
    image: np.ndarray,
    gaussian_sigma: float = 1.0,
    sobel_ksize: int = 3,
    canny_low: float | None = None,
    canny_high: float | None = None,
    canny_low_ratio: float = 0.4,
    canny_high_quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    gray = image_to_gray(image)
    if gaussian_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), float(gaussian_sigma))
    sobel_ksize = int(sobel_ksize)
    if sobel_ksize != -1:
        sobel_ksize = odd_kernel(sobel_ksize)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_ksize)
    sobel = cv2.magnitude(gx, gy)
    sobel_norm = sobel / (float(sobel.max()) + 1e-6)
    nonzero = sobel[sobel > 0]
    if canny_high is None:
        canny_high = float(np.quantile(nonzero, float(canny_high_quantile))) if nonzero.size else 150.0
    if canny_low is None:
        canny_low = float(canny_low_ratio) * float(canny_high)
    canny = cv2.Canny(
        gray,
        threshold1=float(canny_low),
        threshold2=float(canny_high),
        apertureSize=3,
        L2gradient=True,
    )
    return sobel_norm.astype(np.float32), (canny > 0).astype(np.uint8), {
        "canny_low": float(canny_low),
        "canny_high": float(canny_high),
    }


def build_geometry_priors(
    image: np.ndarray,
    mask_u8: np.ndarray,
    dt_method: str = "opencv_precise_l2",
    gaussian_sigma: float = 1.0,
    sobel_ksize: int = 3,
    canny_low: float | None = None,
    canny_high: float | None = None,
    canny_low_ratio: float = 0.4,
    canny_high_quantile: float = 0.90,
    morph_open: int = 3,
    morph_close: int = 5,
    remove_small_objects: int = 64,
    remove_small_holes: int = 128,
    build_centerline: bool = False,
) -> GeometryPriors:
    clean_mask = clean_binary_mask(
        mask_u8,
        morph_open=morph_open,
        morph_close=morph_close,
        remove_small_objects=remove_small_objects,
        remove_small_holes=remove_small_holes,
    )
    clean_u8 = clean_mask.astype(np.uint8) * 255
    if dt_method == "scipy_edt":
        try:
            from scipy import ndimage as ndi  # type: ignore

            dt_map = ndi.distance_transform_edt(clean_mask.astype(bool)).astype(np.float32)
        except Exception:
            dt_method = "opencv_precise_l2"
            dt_map = cv2.distanceTransform(clean_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    else:
        dt_map = cv2.distanceTransform(clean_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    sobel_mag, canny_edges, edge_meta = build_sobel_canny(
        image,
        gaussian_sigma=gaussian_sigma,
        sobel_ksize=sobel_ksize,
        canny_low=canny_low,
        canny_high=canny_high,
        canny_low_ratio=canny_low_ratio,
        canny_high_quantile=canny_high_quantile,
    )

    centerline_map = None
    centerline_method = ""
    if build_centerline:
        try:
            from skimage.morphology import medial_axis  # type: ignore

            centerline_map = medial_axis(clean_mask.astype(bool)).astype(np.uint8)
            centerline_method = "skimage_medial_axis"
        except Exception:
            # OpenCV thinning lives in opencv-contrib, so the fallback is an
            # empty map instead of adding another hard dependency.
            centerline_map = np.zeros_like(clean_mask, dtype=np.uint8)
            centerline_method = "unavailable"

    meta = {
        "version": "geometry_priors_v1",
        "dt_method": dt_method,
        "gaussian_sigma": float(gaussian_sigma),
        "sobel_ksize": int(sobel_ksize),
        "morph_open": int(morph_open),
        "morph_close": int(morph_close),
        "remove_small_objects": int(remove_small_objects),
        "remove_small_holes": int(remove_small_holes),
        "clean_mask_pixels": int(clean_mask.sum()),
        "dt_max_px": float(dt_map.max()) if dt_map.size else 0.0,
        "dt_mean_px": float(dt_map[clean_mask > 0].mean()) if np.any(clean_mask > 0) else 0.0,
        "canny_edge_pixels": int(canny_edges.sum()),
        "centerline_method": centerline_method,
        **edge_meta,
    }
    return GeometryPriors(
        clean_mask=clean_mask.astype(np.uint8),
        dt_map=dt_map.astype(np.float32),
        sobel_mag=sobel_mag.astype(np.float32),
        canny_edges=canny_edges.astype(np.uint8),
        centerline_map=centerline_map,
        meta=meta,
    )


def sample_line_points(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    shape: tuple[int, int],
    sample_step_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    distance = float(np.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]))
    steps = max(1, int(np.ceil(distance / max(0.25, float(sample_step_px)))))
    xs = np.linspace(start_xy[0], end_xy[0], steps + 1)
    ys = np.linspace(start_xy[1], end_xy[1], steps + 1)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, width - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
    return yi, xi


def connector_geometry_stats(
    priors: GeometryPriors | None,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    sample_step_px: float = 1.0,
) -> dict[str, float]:
    if priors is None:
        return {}
    yi, xi = sample_line_points(start_xy, end_xy, priors.clean_mask.shape, sample_step_px=sample_step_px)
    dt_values = priors.dt_map[yi, xi].astype(np.float32)
    sobel_values = priors.sobel_mag[yi, xi].astype(np.float32)
    canny_values = priors.canny_edges[yi, xi].astype(np.float32)
    clean_values = priors.clean_mask[yi, xi].astype(np.float32)
    centerline_frac = 0.0
    if priors.centerline_map is not None:
        centerline_frac = float(priors.centerline_map[yi, xi].mean())
    return {
        "gp_clean_inside_fraction": float(clean_values.mean()) if clean_values.size else 1.0,
        "gp_dt_min_px": float(dt_values.min()) if dt_values.size else 0.0,
        "gp_dt_q05_px": float(np.quantile(dt_values, 0.05)) if dt_values.size else 0.0,
        "gp_dt_mean_px": float(dt_values.mean()) if dt_values.size else 0.0,
        "gp_sobel_cross_mean": float(sobel_values.mean()) if sobel_values.size else 0.0,
        "gp_canny_cross_frac": float(canny_values.mean()) if canny_values.size else 0.0,
        "gp_centerline_hit_frac": centerline_frac,
    }


def geometry_stats_are_safe(
    stats: dict[str, float],
    dt_min_px: float = 1.0,
    dt_q05_px: float = 1.5,
    canny_frac_max: float = 0.12,
    sobel_mean_max: float = 0.18,
) -> bool:
    if not stats:
        return True
    return (
        float(stats.get("gp_dt_min_px", 0.0)) >= float(dt_min_px)
        and float(stats.get("gp_dt_q05_px", 0.0)) >= float(dt_q05_px)
        and float(stats.get("gp_canny_cross_frac", 0.0)) <= float(canny_frac_max)
        and float(stats.get("gp_sobel_cross_mean", 0.0)) <= float(sobel_mean_max)
    )


def save_geometry_prior_artifacts(priors: GeometryPriors, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "clean_mask": "clean_mask.png",
        "dt_vis": "dt_vis.png",
        "sobel_vis": "sobel_vis.png",
        "canny": "canny_edges.png",
        "meta": "geometry_priors_meta.json",
        "dt_map": "dt_map.npy",
        "sobel_mag": "sobel_mag.npy",
    }
    write_png(output_dir / files["clean_mask"], priors.clean_mask.astype(np.uint8) * 255)
    write_png(output_dir / files["dt_vis"], normalize_u8(priors.dt_map))
    write_png(output_dir / files["sobel_vis"], normalize_u8(priors.sobel_mag))
    write_png(output_dir / files["canny"], priors.canny_edges.astype(np.uint8) * 255)
    np.save(output_dir / files["dt_map"], priors.dt_map)
    np.save(output_dir / files["sobel_mag"], priors.sobel_mag)
    if priors.centerline_map is not None:
        files["centerline"] = "centerline.png"
        write_png(output_dir / files["centerline"], priors.centerline_map.astype(np.uint8) * 255)
    (output_dir / files["meta"]).write_text(json.dumps(priors.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(output_dir / value) for key, value in files.items()}
