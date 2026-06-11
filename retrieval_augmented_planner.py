from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image


FEATURE_KEYS = [
    "area_ratio",
    "bbox_aspect",
    "density_mean",
    "centerline_ratio",
    "boundary_ratio",
    "running_ratio",
    "satin_ratio",
    "fill_ratio",
]


PLANNER_KEYS = [
    "row_step_px",
    "point_step_px",
    "min_component_px",
    "max_components",
    "long_jump_threshold_mm",
    "long_jump_weight",
    "two_opt_passes",
    "connect_near_mm",
    "continuity_order_weight",
    "jump_endpoint_weight",
    "continuity_connect_threshold",
    "continuity_connect_max_mm",
    "max_stitch_mm",
    "max_jump_mm",
    "trim_jump_threshold_mm",
    "serpentine_fill",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox_aspect(mask: np.ndarray) -> float:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return 1.0
    width = max(1, int(xx.max() - xx.min() + 1))
    height = max(1, int(yy.max() - yy.min() + 1))
    return float(width / height)


def image_feature_vector(path: Path, size: int = 256) -> dict[str, float]:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    mask = (gray < 0.94) | (sat > 0.08)
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx * gx + gy * gy)
    fill = float(mask.mean())
    return {
        "area_ratio": fill,
        "bbox_aspect": _bbox_aspect(mask),
        "density_mean": float((1.0 - gray)[mask].mean()) if np.any(mask) else 0.0,
        "centerline_ratio": float((edge > 0.08).mean()),
        "boundary_ratio": float((edge > 0.12).mean()),
        "running_ratio": 0.25,
        "satin_ratio": 0.25,
        "fill_ratio": 0.50,
    }


def prediction_feature_vector(
    mask: np.ndarray,
    density: np.ndarray,
    stitch_type: np.ndarray,
    centerline: np.ndarray,
    boundary: np.ndarray,
) -> dict[str, float]:
    active = mask > 0.5
    total = max(1, mask.size)
    active_count = max(1, int(active.sum()))
    return {
        "area_ratio": float(active.sum() / total),
        "bbox_aspect": _bbox_aspect(active),
        "density_mean": float(density[active].mean()) if np.any(active) else float(density.mean()),
        "centerline_ratio": float((centerline[active] > 0.35).sum() / active_count),
        "boundary_ratio": float((boundary[active] > 0.35).sum() / active_count),
        "running_ratio": float(((stitch_type == 1) & active).sum() / active_count),
        "satin_ratio": float(((stitch_type == 2) & active).sum() / active_count),
        "fill_ratio": float(((stitch_type == 3) & active).sum() / active_count),
    }


def planner_prior_from_features(features: dict[str, float], stats: dict[str, float] | None = None) -> dict[str, float | bool | int]:
    stats = stats or {}
    area = features.get("area_ratio", 0.0)
    fill = features.get("fill_ratio", 0.5)
    boundary = features.get("boundary_ratio", 0.0)
    center = features.get("centerline_ratio", 0.0)
    nodes = stats.get("path_graph_nodes", 0.0)
    long_ratio = stats.get("long_jump_ratio", 0.0)
    complex_design = nodes >= 20 or boundary > 0.16 or center > 0.10
    dense_fill = fill >= 0.45 or area >= 0.18
    return {
        "row_step_px": 2 if dense_fill else 3,
        "point_step_px": 1 if complex_design else 2,
        "min_component_px": 18 if complex_design else 28,
        "max_components": 90 if complex_design else 60,
        "long_jump_threshold_mm": 7.0 if long_ratio > 0.08 else 8.0,
        "long_jump_weight": 1.1 if long_ratio > 0.05 else 0.45,
        "two_opt_passes": 2 if complex_design else 1,
        "connect_near_mm": 2.0 if center > 0.06 else 1.2,
        "continuity_order_weight": 0.85,
        "jump_endpoint_weight": 0.45 if long_ratio > 0.04 else 0.35,
        "continuity_connect_threshold": 0.25,
        "continuity_connect_max_mm": 5.0,
        "max_stitch_mm": 3.6 if dense_fill else 4.0,
        "max_jump_mm": 6.5 if long_ratio > 0.04 else 7.5,
        "trim_jump_threshold_mm": 8.5 if long_ratio > 0.04 else 10.0,
        "serpentine_fill": True,
    }


def _distance(query: dict[str, float], item: dict[str, object]) -> float:
    features = item.get("features", {})
    if not isinstance(features, dict):
        return float("inf")
    total = 0.0
    used = 0
    for key in FEATURE_KEYS:
        q = query.get(key)
        v = features.get(key)
        if q is None or v is None:
            continue
        scale = 1.0 if key != "bbox_aspect" else 2.0
        total += ((float(q) - float(v)) / scale) ** 2
        used += 1
    return float(total / max(1, used)) ** 0.5


def retrieve_planner_prior(
    index_path: Path,
    query_features: dict[str, float],
    top_k: int = 5,
) -> dict[str, object]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    ranked = sorted(items, key=lambda item: _distance(query_features, item))[: max(1, top_k)]
    if not ranked:
        return {"matches": [], "planner_prior": planner_prior_from_features(query_features)}
    priors: list[dict[str, object]] = []
    for item in ranked:
        prior = item.get("planner_prior")
        if isinstance(prior, dict):
            priors.append(prior)
    if not priors:
        priors = [planner_prior_from_features(query_features)]
    blended: dict[str, float | bool | int] = {}
    for key in PLANNER_KEYS:
        values = [prior[key] for prior in priors if key in prior]
        if not values:
            continue
        if isinstance(values[0], bool):
            blended[key] = sum(bool(v) for v in values) >= (len(values) / 2)
        elif isinstance(values[0], int):
            blended[key] = int(round(mean(float(v) for v in values)))
        else:
            blended[key] = float(mean(float(v) for v in values))
    matches = [
        {
            "id": item.get("id"),
            "distance": round(_distance(query_features, item), 6),
            "source": item.get("source"),
        }
        for item in ranked
    ]
    return {"matches": matches, "planner_prior": blended, "query_features": query_features}


def build_index_from_images(image_dir: Path, output: Path, limit: int = 0) -> dict[str, object]:
    files = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if limit:
        files = files[:limit]
    items = []
    for index, path in enumerate(files, start=1):
        features = image_feature_vector(path)
        items.append(
            {
                "id": f"image_{index:05d}_{path.stem[:40]}",
                "source": path.as_posix(),
                "features": features,
                "planner_prior": planner_prior_from_features(features),
            }
        )
    payload = {"version": 1, "kind": "image_retrieval_planner_index", "items": items, "feature_keys": FEATURE_KEYS, "planner_keys": PLANNER_KEYS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_index_from_dataset(dataset_dir: Path, output: Path, manifest: str, limit: int = 0) -> dict[str, object]:
    with (dataset_dir / manifest).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]
    items = []
    for row in rows:
        features = {
            "area_ratio": _safe_float(row.get("mask_ratio")),
            "bbox_aspect": _safe_float(row.get("aspect_ratio"), 1.0),
            "density_mean": _safe_float(row.get("density_mean"), _safe_float(row.get("mask_ratio"))),
            "centerline_ratio": _safe_float(row.get("axis_valid_coverage_on_mask"), 0.0),
            "boundary_ratio": _safe_float(row.get("boundary_ratio"), 0.0),
            "running_ratio": 0.25,
            "satin_ratio": 0.25,
            "fill_ratio": 0.50,
        }
        stats = {
            "path_graph_nodes": _safe_float(row.get("path_graph_nodes")),
            "long_jump_ratio": _safe_float(row.get("long_jump_edges_v2")) / max(1.0, _safe_float(row.get("segment_count_v2"), 1.0)),
        }
        items.append(
            {
                "id": row.get("pair_id"),
                "source": row.get("source_dst") or row.get("input_png"),
                "features": features,
                "planner_prior": planner_prior_from_features(features, stats),
                "stats": stats,
            }
        )
    payload = {"version": 1, "kind": "dataset_retrieval_planner_index", "items": items, "feature_keys": FEATURE_KEYS, "planner_keys": PLANNER_KEYS}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or query a retrieval-augmented embroidery planner index.")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--manifest", default="manifest_dataset2.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.image_dir:
        payload = build_index_from_images(Path(args.image_dir), Path(args.output), args.limit)
    elif args.dataset_dir:
        payload = build_index_from_dataset(Path(args.dataset_dir), Path(args.output), args.manifest, args.limit)
    else:
        raise SystemExit("Provide --image-dir or --dataset-dir.")
    print(json.dumps({"output": args.output, "items": len(payload["items"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
