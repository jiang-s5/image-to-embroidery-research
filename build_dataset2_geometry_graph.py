from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from prepare_dst_rendered_supervision_dataset import fit_segments, read_dst_segments


STITCH_TYPE_RGB = np.asarray(
    [
        (0, 0, 0),
        (80, 190, 255),
        (255, 190, 70),
        (180, 120, 230),
    ],
    dtype=np.int32,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stable_unit(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(0xFFFFFFFFFFFF)


def bin_numeric(value: float, cuts: list[float], labels: list[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def load_stitch_type(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    distances = ((rgb[:, :, None, :] - STITCH_TYPE_RGB[None, None, :, :]) ** 2).sum(axis=3)
    return np.argmin(distances, axis=2).astype(np.uint8)


def load_color_layer(path: Path, background_threshold: int = 8) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    non_background = np.any(flat > background_threshold, axis=1)
    colors, counts = np.unique(flat[non_background], axis=0, return_counts=True)
    if colors.shape[0] == 0:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)
    order = np.argsort(-counts)
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for label_id, color_index in enumerate(order.tolist(), start=1):
        if label_id >= 255:
            break
        color = colors[color_index]
        labels[np.all(rgb == color, axis=2)] = label_id
    return labels


def save_gray(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(path)


def save_label_preview(arr: np.ndarray, path: Path) -> None:
    palette = np.asarray(
        [
            (0, 0, 0),
            (58, 145, 220),
            (240, 170, 45),
            (110, 185, 120),
            (220, 85, 80),
            (150, 100, 210),
            (40, 180, 180),
            (230, 120, 190),
            (135, 95, 55),
            (230, 220, 80),
        ],
        dtype=np.uint8,
    )
    rgb = palette[np.clip(arr.astype(np.int32), 0, len(palette) - 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def connected_components(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    active = mask.astype(bool)
    h, w = active.shape
    seen = np.zeros_like(active, dtype=bool)
    comps: list[np.ndarray] = []
    ys, xs = np.nonzero(active)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        queue: deque[tuple[int, int]] = deque([(sy, sx)])
        seen[sy, sx] = True
        pts: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                if seen[yy, xx] or not active[yy, xx]:
                    continue
                seen[yy, xx] = True
                queue.append((yy, xx))
        if len(pts) >= min_area:
            comp = np.zeros_like(active, dtype=bool)
            py, px = zip(*pts)
            comp[np.asarray(py), np.asarray(px)] = True
            comps.append(comp)
    comps.sort(key=lambda item: int(item.sum()), reverse=True)
    return comps


def boundary_points(component: np.ndarray) -> np.ndarray:
    padded = np.pad(component.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    boundary = center & (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
        | ~padded[:-2, :-2]
        | ~padded[:-2, 2:]
        | ~padded[2:, :-2]
        | ~padded[2:, 2:]
    )
    yy, xx = np.nonzero(boundary)
    return np.stack([xx, yy], axis=1).astype(np.float32) if yy.size else np.zeros((0, 2), dtype=np.float32)


def aabb_from_component(component: np.ndarray) -> list[int]:
    yy, xx = np.nonzero(component)
    if yy.size == 0:
        return [0, 0, 0, 0]
    return [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]


def obb_from_points(points: np.ndarray) -> list[float]:
    if points.shape[0] < 3:
        if points.shape[0] == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        x0, y0 = points.mean(axis=0)
        return [float(x0), float(y0), 1.0, 1.0, 0.0]
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    axis0 = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.atan2(float(axis0[1]), float(axis0[0]))
    c = math.cos(-angle)
    s = math.sin(-angle)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    rp = centered @ rot.T
    min_xy = rp.min(axis=0)
    max_xy = rp.max(axis=0)
    width, height = max_xy - min_xy
    center_local = (min_xy + max_xy) * 0.5
    inv = rot
    center = mean + center_local @ inv
    return [float(center[0]), float(center[1]), float(width), float(height), float(angle)]


def polygon_from_boundary(points: np.ndarray, max_points: int = 64) -> list[list[float]]:
    if points.shape[0] == 0:
        return []
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = points[order]
    if ordered.shape[0] > max_points:
        step = max(1, int(math.ceil(ordered.shape[0] / max_points)))
        ordered = ordered[::step]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in ordered.tolist()]


def clean_color_layer(color_layer: np.ndarray, min_island_area: int) -> tuple[np.ndarray, dict[str, int]]:
    cleaned = color_layer.copy()
    removed = 0
    components_total = 0
    for color_id in [int(x) for x in np.unique(color_layer) if int(x) > 0]:
        comps = connected_components(color_layer == color_id, min_area=1)
        components_total += len(comps)
        large = [comp for comp in comps if int(comp.sum()) >= min_island_area]
        if not large:
            continue
        for comp in comps:
            area = int(comp.sum())
            if area >= min_island_area:
                continue
            cleaned[comp] = 0
            removed += area
    return cleaned, {"removed_pixels": removed, "components_total": components_total}


def rasterize_segment_ids(source_dst: Path, size: int, padding: int, line_width: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    segments, _ = fit_segments(read_dst_segments(source_dst), size=size, padding=padding)
    segment_id_map = np.zeros((size, size), dtype=np.uint16)
    segment_records: list[dict[str, object]] = []
    radius = max(0, line_width // 2)
    offsets = [
        (ox, oy)
        for oy in range(-radius, radius + 1)
        for ox in range(-radius, radius + 1)
        if ox * ox + oy * oy <= radius * radius
    ] or [(0, 0)]
    for index, seg in enumerate(segments, start=1):
        if index >= np.iinfo(np.uint16).max:
            break
        dx = seg.x1 - seg.x0
        dy = seg.y1 - seg.y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
        xs = np.rint(np.linspace(seg.x0, seg.x1, steps + 1)).astype(np.int32)
        ys = np.rint(np.linspace(seg.y0, seg.y1, steps + 1)).astype(np.int32)
        for ox, oy in offsets:
            xx = np.clip(xs + ox, 0, size - 1)
            yy = np.clip(ys + oy, 0, size - 1)
            segment_id_map[yy, xx] = index
        segment_records.append(
            {
                "segment_id": index,
                "color_layer": int(seg.color_index) + 1,
                "entry_xy": [round(float(seg.x0), 3), round(float(seg.y0), 3)],
                "exit_xy": [round(float(seg.x1), 3), round(float(seg.y1), 3)],
                "length_px": round(float(length), 3),
            }
        )
    return segment_id_map, segment_records


def endpoint_heatmaps(segment_records: list[dict[str, object]], size: int, sigma: float = 1.8) -> np.ndarray:
    heat = np.zeros((2, size, size), dtype=np.uint8)
    for record in segment_records:
        for channel, key in enumerate(("entry_xy", "exit_xy")):
            x, y = record[key]
            cx = int(round(float(x)))
            cy = int(round(float(y)))
            if 0 <= cx < size and 0 <= cy < size:
                heat[channel, cy, cx] = 255
    blurred = []
    for channel in range(2):
        image = Image.fromarray(heat[channel], mode="L").filter(ImageFilter.GaussianBlur(radius=sigma))
        arr = np.asarray(image, dtype=np.float32)
        max_value = float(arr.max())
        blurred.append(arr / max_value if max_value > 0 else arr)
    return np.stack(blurred, axis=0).astype(np.float16)


def graph_from_components(row_id: str, color_layer: np.ndarray, stitch_type: np.ndarray, boundary: np.ndarray) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    node_masks: list[np.ndarray] = []
    node_id = 0
    for color_id in [int(x) for x in np.unique(color_layer) if int(x) > 0]:
        for comp in connected_components(color_layer == color_id, min_area=8):
            node_id += 1
            yy, xx = np.nonzero(comp)
            counts = np.bincount(stitch_type[comp].reshape(-1), minlength=4)
            if counts.sum() == 0:
                type_id = 0
            else:
                counts[0] = 0
                type_id = int(np.argmax(counts))
            pts = np.stack([xx, yy], axis=1).astype(np.float32)
            bpts = boundary_points(comp)
            node_masks.append(comp)
            nodes.append(
                {
                    "id": node_id,
                    "color_layer": color_id,
                    "stitch_type": type_id,
                    "area_px": int(comp.sum()),
                    "centroid_xy": [round(float(xx.mean()), 3), round(float(yy.mean()), 3)],
                    "aabb_xyxy": aabb_from_component(comp),
                    "obb_cxcywh_angle": [round(v, 5) for v in obb_from_points(pts)],
                    "contour_poly": polygon_from_boundary(bpts, max_points=64),
                    "boundary_mean": round(float(boundary[comp].mean() / 255.0), 5) if np.any(comp) else 0.0,
                }
            )
    edges: list[dict[str, object]] = []
    centroids = np.asarray([node["centroid_xy"] for node in nodes], dtype=np.float32) if nodes else np.zeros((0, 2))
    for i, left in enumerate(nodes):
        distances: list[tuple[float, int]] = []
        for j, right in enumerate(nodes):
            if i == j:
                continue
            distance = float(np.linalg.norm(centroids[i] - centroids[j]))
            same_color = left["color_layer"] == right["color_layer"]
            cost = distance * (0.65 if same_color else 1.0)
            distances.append((cost, j))
        for cost, j in sorted(distances)[: min(4, len(distances))]:
            right = nodes[j]
            edges.append(
                {
                    "source": int(left["id"]),
                    "target": int(right["id"]),
                    "cost": round(float(cost), 4),
                    "same_color": bool(left["color_layer"] == right["color_layer"]),
                }
            )
    return {"pair_id": row_id, "nodes": nodes, "edges": edges}


def make_geometry_preview(input_img: Image.Image, mask: np.ndarray, nodes: list[dict[str, object]], output: Path) -> None:
    preview = input_img.convert("RGB")
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for node in nodes[:128]:
        x0, y0, x1, y1 = node["aabb_xyxy"]
        color = (255, 180, 40, 190) if node["stitch_type"] == 2 else (80, 210, 255, 180) if node["stitch_type"] == 1 else (190, 120, 230, 160)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
        cx, cy = node["centroid_xy"]
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 60, 60, 200))
    alpha = np.where(mask > 0, 42, 0).astype(np.uint8)
    fill = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    fill[..., 1] = 80
    fill[..., 2] = 180
    fill[..., 3] = alpha
    base = Image.alpha_composite(preview.convert("RGBA"), Image.fromarray(fill, mode="RGBA"))
    result = Image.alpha_composite(base, overlay).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)


def build_canonical_split(rows: list[dict[str, str]], old_rows: list[dict[str, str]] | None) -> list[dict[str, object]]:
    old_by_stem: dict[str, dict[str, str]] = {}
    if old_rows:
        old_by_stem = {Path(row["source_dst"]).stem: row for row in old_rows}
    canonical: list[dict[str, object]] = []
    for row in rows:
        score = stable_unit(row["source_family"] + "::" + row["source_dst"])
        if score < 0.15:
            split = "test"
        elif score < 0.27:
            split = "val"
        else:
            split = "train"
        segments = int(float(row["segments"]))
        mask_ratio = float(row["mask_ratio"])
        color_layer = load_color_layer(Path(row["_dataset_dir"]) / row["color_layer_png"])
        color_count = len([x for x in np.unique(color_layer) if int(x) > 0])
        stitch_type = load_stitch_type(Path(row["_dataset_dir"]) / row["stitch_type_heuristic_png"])
        counts = np.bincount(stitch_type.reshape(-1), minlength=4).astype(np.float64)
        fg = counts[1:].sum()
        probs = counts[1:] / max(1.0, fg)
        entropy = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
        old = old_by_stem.get(Path(row["source_dst"]).stem, {})
        canonical.append(
            {
                "pair_id": row["pair_id"],
                "source_family": row["source_family"],
                "canonical_split": split,
                "rich_split_stratified": row["split_stratified"],
                "old_4ch_split": old.get("split", ""),
                "leakage_if_using_old_pretrain": bool(old.get("split", "") == "train" and split in {"val", "test"}),
                "segment_bin": bin_numeric(segments, [1000, 3000, 8000, 20000], ["seg_lt1k", "seg_1k_3k", "seg_3k_8k", "seg_8k_20k", "seg_ge20k"]),
                "mask_ratio_bin": bin_numeric(mask_ratio, [0.05, 0.10, 0.20, 0.40], ["mask_lt05", "mask_05_10", "mask_10_20", "mask_20_40", "mask_ge40"]),
                "color_layers": color_count,
                "stitch_type_entropy": round(entropy, 6),
                "qa_flags": row["qa_flags"],
                "source_dst": row["source_dst"],
            }
        )
    return canonical


def build(args: argparse.Namespace) -> dict[str, object]:
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    graph_dir = output_dir / "graphs"
    preview_dir = output_dir / "previews"
    for directory in (labels_dir, graph_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_rows(source_dir / "manifest_rich.csv")
    for row in rows:
        row["_dataset_dir"] = str(source_dir)
    old_manifest = Path(args.old_manifest)
    old_rows = read_rows(old_manifest) if old_manifest.exists() else []
    canonical = build_canonical_split(rows, old_rows)
    split_by_pair = {row["pair_id"]: row for row in canonical}
    write_rows(
        output_dir / "canonical_split.csv",
        canonical,
        [
            "pair_id",
            "source_family",
            "canonical_split",
            "rich_split_stratified",
            "old_4ch_split",
            "leakage_if_using_old_pretrain",
            "segment_bin",
            "mask_ratio_bin",
            "color_layers",
            "stitch_type_entropy",
            "qa_flags",
            "source_dst",
        ],
    )

    manifest_rows: list[dict[str, object]] = []
    split_counts = Counter()
    leakage_count = 0
    for index, row in enumerate(rows, start=1):
        pair_id = row["pair_id"]
        split_info = split_by_pair[pair_id]
        split_counts[str(split_info["canonical_split"])] += 1
        leakage_count += int(bool(split_info["leakage_if_using_old_pretrain"]))

        input_path = source_dir / row["input_png"]
        mask = np.asarray(Image.open(source_dir / row["mask_png"]).convert("L"), dtype=np.uint8)
        boundary = np.asarray(Image.open(source_dir / row["boundary_png"]).convert("L"), dtype=np.uint8)
        color_layer = load_color_layer(source_dir / row["color_layer_png"])
        stitch_type = load_stitch_type(source_dir / row["stitch_type_heuristic_png"])
        confidence = np.load(source_dir / row["direction_confidence_npy"]).astype(np.float32)
        axis_valid = ((mask > 0) & (confidence >= args.axis_conf_threshold)).astype(np.uint8) * 255
        cleaned_color, clean_stats = clean_color_layer(color_layer, args.min_color_island_area)
        segment_id_map, segment_records = rasterize_segment_ids(Path(row["source_dst"]), args.canvas_size, args.padding, args.segment_line_width)
        endpoint = endpoint_heatmaps(segment_records, args.canvas_size, sigma=args.endpoint_sigma)
        graph = graph_from_components(pair_id, cleaned_color, stitch_type, boundary)

        axis_valid_path = labels_dir / f"{pair_id}_axis_valid_mask.png"
        clean_color_path = labels_dir / f"{pair_id}_cleaned_color_layer.png"
        clean_color_preview_path = preview_dir / f"{pair_id}_cleaned_color_layer_preview.png"
        segment_id_path = labels_dir / f"{pair_id}_segment_id_map.npy"
        endpoint_path = labels_dir / f"{pair_id}_endpoint_heatmap.npy"
        segment_records_path = graph_dir / f"{pair_id}_entry_exit_per_segment.json"
        path_graph_path = graph_dir / f"{pair_id}_path_graph.json"
        geometry_preview_path = preview_dir / f"{pair_id}_geometry_graph_preview.png"

        save_gray(axis_valid, axis_valid_path)
        save_gray(cleaned_color, clean_color_path)
        if index <= args.preview_count:
            save_label_preview(cleaned_color, clean_color_preview_path)
        np.save(segment_id_path, segment_id_map)
        np.save(endpoint_path, endpoint)
        segment_records_path.write_text(json.dumps(segment_records, ensure_ascii=False, indent=2), encoding="utf-8")
        path_graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        if index <= args.preview_count:
            make_geometry_preview(Image.open(input_path), mask, graph["nodes"], geometry_preview_path)

        qa_flags = [flag for flag in row["qa_flags"].split("|") if flag]
        if int(axis_valid.sum() / 255) < max(16, int((mask > 0).sum() * 0.05)):
            qa_flags.append("low_axis_valid_coverage")
        if clean_stats["removed_pixels"] > 0:
            qa_flags.append("color_islands_cleaned")
        if graph["nodes"] and any(min(node["aabb_xyxy"][0], node["aabb_xyxy"][1], args.canvas_size - 1 - node["aabb_xyxy"][2], args.canvas_size - 1 - node["aabb_xyxy"][3]) <= 2 for node in graph["nodes"]):
            qa_flags.append("edge_truncation_risk")

        manifest_rows.append(
            {
                **{key: value for key, value in row.items() if key != "_dataset_dir"},
                "canonical_split": split_info["canonical_split"],
                "leakage_if_using_old_pretrain": split_info["leakage_if_using_old_pretrain"],
                "axis_valid_mask_png": str(axis_valid_path.relative_to(output_dir)),
                "cleaned_color_layer_png": str(clean_color_path.relative_to(output_dir)),
                "cleaned_color_layer_preview_png": str(clean_color_preview_path.relative_to(output_dir)) if clean_color_preview_path.exists() else "",
                "segment_id_map_npy": str(segment_id_path.relative_to(output_dir)),
                "endpoint_heatmap_npy": str(endpoint_path.relative_to(output_dir)),
                "entry_exit_per_segment_json": str(segment_records_path.relative_to(output_dir)),
                "path_graph_json": str(path_graph_path.relative_to(output_dir)),
                "geometry_graph_preview_png": str(geometry_preview_path.relative_to(output_dir)) if geometry_preview_path.exists() else "",
                "axis_valid_pixels": int(axis_valid.sum() / 255),
                "axis_valid_coverage_on_mask": round(float((axis_valid > 0).sum() / max(1, (mask > 0).sum())), 6),
                "path_graph_nodes": len(graph["nodes"]),
                "path_graph_edges": len(graph["edges"]),
                "color_island_removed_pixels": clean_stats["removed_pixels"],
                "dataset2_qa_flags": "|".join(sorted(set(qa_flags))),
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"[dataset2] processed {index}/{len(rows)} samples; latest={pair_id}; nodes={len(graph['nodes'])}; split={split_info['canonical_split']}", flush=True)

    fieldnames = list(manifest_rows[0].keys())
    write_rows(output_dir / "manifest_dataset2.csv", manifest_rows, fieldnames)
    for name in ("README.md", "summary_rich.json"):
        copy_or_link(source_dir / name, output_dir / f"source_{name}")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "samples": len(manifest_rows),
        "canonical_split_counts": dict(split_counts),
        "old_pretrain_leakage_flagged_samples": leakage_count,
        "new_labels": [
            "axis_valid_mask_png",
            "cleaned_color_layer_png",
            "cleaned_color_layer_preview_png",
            "segment_id_map_npy",
            "endpoint_heatmap_npy",
            "entry_exit_per_segment_json",
            "path_graph_json",
            "geometry_graph_preview_png",
        ],
        "note": "Dataset2 adds geometry/graph supervision and a canonical split for future model3 training. Previous model1/model2 metrics remain non-canonical.",
    }
    (output_dir / "summary_dataset2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# dataset2_geometry_graph

This dataset is an optimized derivative of `dst_rendered_supervision_3322_rich_v2`.

Main changes:

- `canonical_split.csv`: future train/val/test split shared by all stages.
- `axis_valid_mask`: masks low-confidence direction labels.
- `cleaned_color_layer`: removes tiny color islands.
- `segment_id_map`: rasterized segment IDs from source DST.
- `endpoint_heatmap`: entry/exit heatmaps for segment-level supervision.
- `entry_exit_per_segment`: segment-level entry/exit records.
- `path_graph`: component graph for deterministic path planning.

Important: model1/model2 were trained before this canonical split existed, so use this split for model3 and later experiments.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build geometry/graph optimized dataset2 from rich_v2.")
    parser.add_argument("--source-dir", default="datasets/dst_rendered_supervision_3322_rich_v2")
    parser.add_argument("--output-dir", default="datasets/dataset2_geometry_graph")
    parser.add_argument("--old-manifest", default="datasets/dst_rendered_supervision_3322_1000/manifest.csv")
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=16)
    parser.add_argument("--axis-conf-threshold", type=float, default=0.2)
    parser.add_argument("--min-color-island-area", type=int, default=4)
    parser.add_argument("--segment-line-width", type=int, default=1)
    parser.add_argument("--endpoint-sigma", type=float, default=1.8)
    parser.add_argument("--preview-count", type=int, default=72)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
