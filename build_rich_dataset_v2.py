from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_dst_rendered_supervision_dataset import fit_segments, read_dst_segments


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_family(source_dst: str) -> str:
    stem = Path(source_dst).stem
    match = re.match(r"([A-Za-z]+)", stem)
    return match.group(1) if match else stem[:3]


def stable_unit(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def stratified_split(row: dict[str, str]) -> str:
    score = stable_unit(row["source_dst"])
    if score < 0.10:
        return "test"
    if score < 0.20:
        return "val"
    return "train"


def family_ood_split(family: str, family_rank: int) -> str:
    # Keep the very large families available for in-distribution training, and
    # hold out smaller families for a stricter OOD check.
    if family_rank % 10 in {0, 3}:
        return "ood_test"
    if family_rank % 10 == 6:
        return "ood_val"
    return "train"


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    active = mask > 0
    padded = np.pad(active, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    boundary = center & (
        ~padded[:-2, 1:-1]
        | ~padded[2:, 1:-1]
        | ~padded[1:-1, :-2]
        | ~padded[1:-1, 2:]
    )
    return boundary.astype(np.uint8) * 255


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    image = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            remove: list[tuple[int, int]] = []
            h, w = image.shape
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if image[y, x] == 0:
                        continue
                    p2 = image[y - 1, x]
                    p3 = image[y - 1, x + 1]
                    p4 = image[y, x + 1]
                    p5 = image[y + 1, x + 1]
                    p6 = image[y + 1, x]
                    p7 = image[y + 1, x - 1]
                    p8 = image[y, x - 1]
                    p9 = image[y - 1, x - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = int(sum(neighbors))
                    if count < 2 or count > 6:
                        continue
                    transitions = sum(
                        1
                        for a, b in zip(neighbors, neighbors[1:] + neighbors[:1])
                        if a == 0 and b == 1
                    )
                    if transitions != 1:
                        continue
                    if step == 0:
                        if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                            continue
                    remove.append((y, x))
            if remove:
                changed = True
                for y, x in remove:
                    image[y, x] = 0
    return image.astype(np.uint8) * 255


def axis_direction(direction_xy: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = direction_xy[0]
    dy = direction_xy[1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    axis = np.zeros_like(direction_xy, dtype=np.float32)
    valid = (mask > 0) & (magnitude > 1e-6)
    axis[0, valid] = np.cos(2.0 * angle[valid])
    axis[1, valid] = np.sin(2.0 * angle[valid])
    confidence = np.zeros(mask.shape, dtype=np.float32)
    confidence[valid] = np.clip(magnitude[valid], 0.0, 1.0)
    return axis, confidence


def png_from_float(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")


def direction_axis_preview(axis: np.ndarray, confidence: np.ndarray) -> Image.Image:
    rgb = np.zeros((axis.shape[1], axis.shape[2], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((axis[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip((axis[1] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def rasterize_dst_structure(
    source_dst: Path,
    size: int,
    padding: int,
    line_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    raw_segments = read_dst_segments(source_dst)
    segments, _ = fit_segments(raw_segments, size=size, padding=padding)
    color_layer = np.zeros((size, size), dtype=np.uint16)
    path_order = np.zeros((size, size), dtype=np.float32)
    stitch_type = np.zeros((size, size), dtype=np.uint8)
    radius = max(0, line_width // 2)
    offsets = [
        (ox, oy)
        for oy in range(-radius, radius + 1)
        for ox in range(-radius, radius + 1)
        if ox * ox + oy * oy <= radius * radius
    ] or [(0, 0)]

    total = max(1, len(segments) - 1)
    entry_exit: dict[int, dict[str, object]] = {}
    for index, seg in enumerate(segments):
        dx = seg.x1 - seg.x0
        dy = seg.y1 - seg.y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        color_id = int(seg.color_index) + 1
        if color_id not in entry_exit:
            entry_exit[color_id] = {
                "color_layer": color_id,
                "entry_xy": [round(float(seg.x0), 3), round(float(seg.y0), 3)],
                "exit_xy": [round(float(seg.x1), 3), round(float(seg.y1), 3)],
                "segment_start": index,
                "segment_end": index,
            }
        else:
            entry_exit[color_id]["exit_xy"] = [round(float(seg.x1), 3), round(float(seg.y1), 3)]
            entry_exit[color_id]["segment_end"] = index

        # Heuristic only: DST does not preserve object-level stitch type.
        if length <= 1.8:
            type_id = 1  # short/run-like
        elif length <= 6.0:
            type_id = 2  # satin-like local strokes
        else:
            type_id = 3  # fill-like longer strokes

        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
        xs = np.rint(np.linspace(seg.x0, seg.x1, steps + 1)).astype(np.int32)
        ys = np.rint(np.linspace(seg.y0, seg.y1, steps + 1)).astype(np.int32)
        for ox, oy in offsets:
            xx = np.clip(xs + ox, 0, size - 1)
            yy = np.clip(ys + oy, 0, size - 1)
            color_layer[yy, xx] = color_id
            path_order[yy, xx] = index / total
            stitch_type[yy, xx] = type_id
    return color_layer, path_order, stitch_type, list(entry_exit.values())


def color_layer_preview(color_layer: np.ndarray) -> Image.Image:
    palette = np.asarray(
        [
            (0, 0, 0),
            (30, 115, 180),
            (50, 160, 90),
            (210, 70, 60),
            (230, 170, 45),
            (120, 90, 180),
            (40, 40, 40),
            (235, 110, 40),
            (30, 160, 170),
            (190, 80, 160),
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[color_layer % len(palette)], mode="RGB")


def stitch_type_preview(stitch_type: np.ndarray) -> Image.Image:
    palette = np.asarray(
        [
            (0, 0, 0),
            (80, 190, 255),
            (255, 190, 70),
            (180, 120, 230),
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[np.clip(stitch_type, 0, len(palette) - 1)], mode="RGB")


def keypoint_preview(size: int, keypoints: list[dict[str, object]]) -> Image.Image:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for item in keypoints:
        entry = item["entry_xy"]
        exit_ = item["exit_xy"]
        ex, ey = float(entry[0]), float(entry[1])
        xx, xy = float(exit_[0]), float(exit_[1])
        draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(60, 210, 90), outline=(255, 255, 255))
        draw.ellipse([xx - 3, xy - 3, xx + 3, xy + 3], fill=(230, 65, 65), outline=(255, 255, 255))
        draw.line([(ex, ey), (xx, xy)], fill=(120, 120, 120), width=1)
    return image


def make_panel(
    output: Path,
    title: str,
    input_png: Path,
    mask_png: Path,
    density_png: Image.Image,
    boundary_png: Path,
    centerline_png: Path,
    axis_preview: Image.Image,
    color_preview: Image.Image,
    order_preview: Image.Image,
    keypoint_img: Image.Image,
) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    items = [
        ("input", Image.open(input_png).convert("RGB")),
        ("mask", Image.open(mask_png).convert("L").convert("RGB")),
        ("density", density_png.convert("RGB")),
        ("boundary", Image.open(boundary_png).convert("L").convert("RGB")),
        ("centerline", Image.open(centerline_png).convert("L").convert("RGB")),
        ("axis/conf", axis_preview),
        ("color layer", color_preview),
        ("path order", order_preview.convert("RGB")),
        ("entry/exit", keypoint_img),
    ]
    tile_w = 128
    tile_h = 128
    panel = Image.new("RGB", (tile_w * len(items), tile_h + 42), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 5), title[:90], fill=(0, 0, 0), font=font)
    for index, (label, image) in enumerate(items):
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
        tile.paste(image, ((tile_w - image.width) // 2, (tile_h - image.height) // 2))
        x = index * tile_w
        panel.paste(tile, (x, 24))
        draw.rectangle([x, 24, x + tile_w - 1, 24 + tile_h - 1], outline=(220, 220, 220))
        draw.text((x + 5, tile_h + 28), label, fill=(30, 30, 30), font=small)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def build(args: argparse.Namespace) -> dict[str, object]:
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    review_dir = Path(args.review_dir).resolve()

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if review_dir.exists() and args.overwrite:
        shutil.rmtree(review_dir)
    for directory in [
        output_dir / "images",
        output_dir / "labels",
        output_dir / "previews",
        output_dir / "qa",
        review_dir / "pair_panels_first_120",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_rows(source_dir / "manifest.csv")
    family_counts: dict[str, int] = {}
    for row in rows:
        family = source_family(row["source_dst"])
        family_counts[family] = family_counts.get(family, 0) + 1
    family_rank = {family: idx for idx, (family, _) in enumerate(sorted(family_counts.items()))}

    rich_rows: list[dict[str, object]] = []
    qa_rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    ood_counts: dict[str, int] = {}
    low_conf_count = 0

    for index, row in enumerate(rows, start=1):
        pair_id = row["pair_id"]
        family = source_family(row["source_dst"])
        input_src = source_dir / row["input_png"]
        mask_src = source_dir / row["mask_png"]
        density_src = source_dir / row["density_npy"]
        direction_src = source_dir / row["direction_npy"]

        input_dst = output_dir / "images" / f"{pair_id}.png"
        mask_dst = output_dir / "labels" / f"{pair_id}_mask.png"
        density_dst = output_dir / "labels" / f"{pair_id}_density.npy"
        direction_dst = output_dir / "labels" / f"{pair_id}_direction_xy.npy"
        copy_or_link(input_src, input_dst)
        copy_or_link(mask_src, mask_dst)
        copy_or_link(density_src, density_dst)
        copy_or_link(direction_src, direction_dst)

        axis_path = output_dir / "labels" / f"{pair_id}_direction_axis.npy"
        conf_path = output_dir / "labels" / f"{pair_id}_direction_confidence.npy"
        boundary_path = output_dir / "labels" / f"{pair_id}_boundary.png"
        centerline_path = output_dir / "labels" / f"{pair_id}_centerline.png"
        color_layer_path = output_dir / "labels" / f"{pair_id}_color_layer.png"
        path_order_path = output_dir / "labels" / f"{pair_id}_path_order.npy"
        path_order_preview_path = output_dir / "previews" / f"{pair_id}_path_order_preview.png"
        stitch_type_path = output_dir / "labels" / f"{pair_id}_stitch_type_heuristic.png"
        keypoints_path = output_dir / "labels" / f"{pair_id}_entry_exit_keypoints.json"
        keypoints_preview_path = output_dir / "previews" / f"{pair_id}_entry_exit_preview.png"
        axis_preview_path = output_dir / "previews" / f"{pair_id}_direction_axis_preview.png"
        rich_preview_path = output_dir / "previews" / f"{pair_id}_rich_panel.png"

        mask = np.asarray(Image.open(mask_src).convert("L"), dtype=np.uint8)
        mask_bool = mask > 0
        required_outputs = [
            axis_path,
            conf_path,
            boundary_path,
            centerline_path,
            color_layer_path,
            path_order_path,
            path_order_preview_path,
            stitch_type_path,
            keypoints_path,
            keypoints_preview_path,
            axis_preview_path,
        ]
        density: np.ndarray | None = None
        if all(path.exists() for path in required_outputs):
            confidence = np.load(conf_path).astype(np.float32)
        else:
            density = np.load(density_src).astype(np.float32)
            direction_xy = np.load(direction_src).astype(np.float32)
            axis, confidence = axis_direction(direction_xy, mask_bool)
            boundary = boundary_mask(mask)
            centerline = zhang_suen_thinning(mask)

            np.save(axis_path, axis)
            np.save(conf_path, confidence)
            Image.fromarray(boundary, mode="L").save(boundary_path)
            Image.fromarray(centerline, mode="L").save(centerline_path)
            axis_preview = direction_axis_preview(axis, confidence)
            axis_preview.save(axis_preview_path)
            color_layer, path_order, stitch_type, keypoints = rasterize_dst_structure(
                Path(row["source_dst"]),
                size=args.canvas_size,
                padding=args.padding,
                line_width=args.structure_line_width,
            )
            color_preview = color_layer_preview(color_layer)
            order_preview = png_from_float(path_order)
            keypoint_img = keypoint_preview(args.canvas_size, keypoints)
            color_preview.save(color_layer_path)
            np.save(path_order_path, path_order)
            order_preview.save(path_order_preview_path)
            stitch_type_preview(stitch_type).save(stitch_type_path)
            keypoints_path.write_text(json.dumps(keypoints, ensure_ascii=False, indent=2), encoding="utf-8")
            keypoint_img.save(keypoints_preview_path)

        if index <= args.panel_count:
            if not rich_preview_path.exists():
                if density is None:
                    density = np.load(density_src).astype(np.float32)
                make_panel(
                    rich_preview_path,
                    pair_id,
                    input_dst,
                    mask_dst,
                    png_from_float(density),
                    boundary_path,
                    centerline_path,
                    Image.open(axis_preview_path).convert("RGB"),
                    Image.open(color_layer_path).convert("RGB"),
                    Image.open(path_order_preview_path).convert("RGB"),
                    Image.open(keypoints_preview_path).convert("RGB"),
                )
            copy_or_link(rich_preview_path, review_dir / "pair_panels_first_120" / rich_preview_path.name)

        width_mm = float(row["width_mm"])
        height_mm = float(row["height_mm"])
        aspect = max(width_mm, height_mm) / max(1e-6, min(width_mm, height_mm))
        mask_ratio = float(mask_bool.mean())
        confidence_on_mask = confidence[mask_bool]
        mean_conf = float(confidence_on_mask.mean()) if confidence_on_mask.size else 0.0
        low_conf_ratio = float((confidence_on_mask < 0.2).mean()) if confidence_on_mask.size else 1.0
        if low_conf_ratio > 0.5:
            low_conf_count += 1
        mm_per_px_est = max(width_mm, height_mm) / max(1, args.canvas_size - 2 * args.padding)
        flags = []
        if mask_ratio < 0.05:
            flags.append("very_sparse_mask")
        elif mask_ratio < 0.10:
            flags.append("sparse_mask")
        if aspect > 8.0:
            flags.append("extreme_aspect")
        if low_conf_ratio > 0.5:
            flags.append("weak_direction_label")
        if int(row["segments"]) > 30000:
            flags.append("high_complexity")

        split = stratified_split(row)
        ood_split = family_ood_split(family, family_rank[family])
        split_counts[split] = split_counts.get(split, 0) + 1
        ood_counts[ood_split] = ood_counts.get(ood_split, 0) + 1

        rich_rows.append(
            {
                "pair_id": pair_id,
                "split_original": row["split"],
                "split_stratified": split,
                "split_family_ood": ood_split,
                "source_family": family,
                "source_kind": row["source_kind"],
                "input_png": str(input_dst.relative_to(output_dir)),
                "mask_png": str(mask_dst.relative_to(output_dir)),
                "density_npy": str(density_dst.relative_to(output_dir)),
                "direction_xy_npy": str(direction_dst.relative_to(output_dir)),
                "direction_axis_npy": str(axis_path.relative_to(output_dir)),
                "direction_confidence_npy": str(conf_path.relative_to(output_dir)),
                "boundary_png": str(boundary_path.relative_to(output_dir)),
                "centerline_png": str(centerline_path.relative_to(output_dir)),
                "color_layer_png": str(color_layer_path.relative_to(output_dir)),
                "path_order_npy": str(path_order_path.relative_to(output_dir)),
                "path_order_preview_png": str(path_order_preview_path.relative_to(output_dir)),
                "stitch_type_heuristic_png": str(stitch_type_path.relative_to(output_dir)),
                "entry_exit_keypoints_json": str(keypoints_path.relative_to(output_dir)),
                "entry_exit_preview_png": str(keypoints_preview_path.relative_to(output_dir)),
                "direction_axis_preview_png": str(axis_preview_path.relative_to(output_dir)),
                "source_dst": row["source_dst"],
                "segments": row["segments"],
                "mask_pixels": row["mask_pixels"],
                "mask_ratio": round(mask_ratio, 6),
                "mean_direction_confidence": round(mean_conf, 6),
                "low_direction_confidence_ratio": round(low_conf_ratio, 6),
                "width_mm": row["width_mm"],
                "height_mm": row["height_mm"],
                "aspect_ratio": round(aspect, 6),
                "mm_per_px_est": round(mm_per_px_est, 6),
                "qa_flags": "|".join(flags),
            }
        )
        qa_rows.append(
            {
                "pair_id": pair_id,
                "source_family": family,
                "split_stratified": split,
                "split_family_ood": ood_split,
                "mask_ratio": round(mask_ratio, 6),
                "segments": row["segments"],
                "aspect_ratio": round(aspect, 6),
                "mean_direction_confidence": round(mean_conf, 6),
                "low_direction_confidence_ratio": round(low_conf_ratio, 6),
                "qa_flags": "|".join(flags),
            }
        )
        if args.progress_every and index % args.progress_every == 0:
            print(f"processed {index}/{len(rows)}")

    fieldnames = list(rich_rows[0].keys())
    write_rows(output_dir / "manifest_rich.csv", rich_rows, fieldnames)
    write_rows(output_dir / "qa" / "qa_flags.csv", qa_rows, list(qa_rows[0].keys()))
    shutil.copy2(source_dir / "summary.json", output_dir / "source_summary.json")
    shutil.copy2(source_dir / "skipped.csv", output_dir / "source_skipped.csv")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "samples": len(rich_rows),
        "new_labels": [
            "boundary_png",
            "centerline_png",
            "direction_axis_npy",
            "direction_confidence_npy",
            "color_layer_png",
            "path_order_npy",
            "stitch_type_heuristic_png",
            "entry_exit_keypoints_json",
        ],
        "split_stratified_counts": split_counts,
        "split_family_ood_counts": ood_counts,
        "source_families": len(family_counts),
        "top_source_families": sorted(family_counts.items(), key=lambda item: item[1], reverse=True)[:10],
        "qa": {
            "weak_direction_label_samples": low_conf_count,
            "qa_flags_csv": str((output_dir / "qa" / "qa_flags.csv").resolve()),
        },
        "note": "V2 keeps original labels and adds richer structural supervision. Existing files are hard-linked where possible.",
    }
    (output_dir / "summary_rich.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (review_dir / "README.md").write_text(
        "# Rich Dataset V2 Review\n\n"
        "This review folder shows the first 120 rich-label panels.\n\n"
        "Each panel contains: input, mask, density, boundary, centerline, and axis/confidence preview.\n\n"
        f"Full rich dataset: `{output_dir}`\n",
        encoding="utf-8",
    )
    (review_dir / "summary_rich.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rich-label V2 dataset from DST rendered supervision.")
    parser.add_argument("--source-dir", default="datasets/dst_rendered_supervision_3322_1000")
    parser.add_argument("--output-dir", default="datasets/dst_rendered_supervision_3322_rich_v2")
    parser.add_argument("--review-dir", default="dataset_review_package_v2")
    parser.add_argument("--canvas-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=14)
    parser.add_argument("--panel-count", type=int, default=120)
    parser.add_argument("--structure-line-width", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
