from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def draw_point(canvas: np.ndarray, xy: list[float] | tuple[float, float], radius: int, value: float = 1.0) -> None:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
        cv2.circle(canvas, (x, y), radius, float(value), thickness=-1, lineType=cv2.LINE_AA)


def draw_line(
    canvas: np.ndarray,
    a: list[float] | tuple[float, float],
    b: list[float] | tuple[float, float],
    thickness: int,
    value: float = 1.0,
) -> None:
    ax = int(round(float(a[0])))
    ay = int(round(float(a[1])))
    bx = int(round(float(b[0])))
    by = int(round(float(b[1])))
    cv2.line(canvas, (ax, ay), (bx, by), float(value), thickness=thickness, lineType=cv2.LINE_AA)


def distance(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def build_label(
    segments: list[dict[str, object]],
    shape: tuple[int, int],
    connect_px: float,
    long_jump_px: float,
    line_thickness: int,
    point_radius: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    h, w = shape
    stitch_trace = np.zeros((h, w), dtype=np.float32)
    near_connect = np.zeros((h, w), dtype=np.float32)
    jump_endpoint = np.zeros((h, w), dtype=np.float32)

    real_segments = [item for item in segments if item.get("entry_xy") is not None and item.get("exit_xy") is not None]
    real_segments.sort(key=lambda item: int(item.get("segment_id", 0)))

    for item in real_segments:
        entry = item["entry_xy"]
        exit_ = item["exit_xy"]
        draw_line(stitch_trace, entry, exit_, line_thickness, 1.0)

    near_count = 0
    long_count = 0
    color_change_count = 0
    for prev, cur in zip(real_segments, real_segments[1:]):
        prev_exit = prev["exit_xy"]
        cur_entry = cur["entry_xy"]
        gap = distance(prev_exit, cur_entry)
        same_color = int(prev.get("color_layer", -1)) == int(cur.get("color_layer", -2))
        if same_color and gap <= connect_px:
            draw_line(near_connect, prev_exit, cur_entry, line_thickness, 1.0)
            near_count += 1
        elif gap >= long_jump_px:
            draw_point(jump_endpoint, prev_exit, point_radius, 1.0)
            draw_point(jump_endpoint, cur_entry, point_radius, 1.0)
            long_count += 1
        elif not same_color:
            draw_point(jump_endpoint, prev_exit, point_radius, 0.75)
            draw_point(jump_endpoint, cur_entry, point_radius, 0.75)
            color_change_count += 1

    stacked = np.stack(
        [
            np.clip(stitch_trace, 0.0, 1.0),
            np.clip(near_connect, 0.0, 1.0),
            np.clip(jump_endpoint, 0.0, 1.0),
        ],
        axis=0,
    )
    stats = {
        "segments": len(real_segments),
        "near_connect_edges": near_count,
        "long_jump_edges": long_count,
        "color_change_edges": color_change_count,
        "stitch_trace_pixels": int((stacked[0] > 0).sum()),
        "near_connect_pixels": int((stacked[1] > 0).sum()),
        "jump_endpoint_pixels": int((stacked[2] > 0).sum()),
    }
    return stacked.astype(np.float32), stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build vector-continuity supervision from DST-derived segment entry/exit labels.")
    parser.add_argument("--dataset-dir", default="datasets/dataset2_collection_20260512_geometry_graph")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--connect-mm", type=float, default=2.5)
    parser.add_argument("--long-jump-mm", type=float, default=8.0)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest_dataset2.csv"
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    summary: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "connect_mm": args.connect_mm,
        "long_jump_mm": args.long_jump_mm,
        "rows_total": len(rows),
        "processed": 0,
        "skipped_existing": 0,
        "failed": 0,
        "totals": {
            "segments": 0,
            "near_connect_edges": 0,
            "long_jump_edges": 0,
            "color_change_edges": 0,
        },
    }

    for index, row in enumerate(rows, start=1):
        if args.limit and index > args.limit:
            break
        pair_id = row["pair_id"]
        out_path = labels_dir / f"{pair_id}_vector_continuity.npy"
        if out_path.exists() and not args.overwrite:
            summary["skipped_existing"] = int(summary["skipped_existing"]) + 1
            continue
        try:
            segment_map = np.load(dataset_dir / row["segment_id_map_npy"])
            graph_path = dataset_dir / row["entry_exit_per_segment_json"]
            segments = json.loads(graph_path.read_text(encoding="utf-8"))
            mm_per_px = max(float(row.get("mm_per_px_est") or 1.0), 1e-6)
            label, stats = build_label(
                segments,
                segment_map.shape,
                connect_px=args.connect_mm / mm_per_px,
                long_jump_px=args.long_jump_mm / mm_per_px,
                line_thickness=args.line_thickness,
                point_radius=args.point_radius,
            )
            np.save(out_path, label)
            summary["processed"] = int(summary["processed"]) + 1
            totals = summary["totals"]
            assert isinstance(totals, dict)
            for key in ("segments", "near_connect_edges", "long_jump_edges", "color_change_edges"):
                totals[key] = int(totals.get(key, 0)) + int(stats[key])
        except Exception as exc:  # pragma: no cover - batch builder should continue.
            summary["failed"] = int(summary["failed"]) + 1
            print(json.dumps({"pair_id": pair_id, "error": str(exc)}, ensure_ascii=False), flush=True)

        if index % 100 == 0:
            print(json.dumps({"index": index, "processed": summary["processed"], "failed": summary["failed"]}, ensure_ascii=False), flush=True)

    (dataset_dir / "vector_continuity_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
