from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from eval_stitch_coverage import coverage_metrics


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, ""))].append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        output.append(
            {
                group_key: key,
                "samples": len(items),
                "mean_skeleton_coverage_ratio": round(mean(safe_float(row.get("skeleton_coverage_ratio")) for row in items), 8),
                "mean_skeleton_precision_ratio": round(mean(safe_float(row.get("skeleton_precision_ratio")) for row in items), 8),
                "mean_adaptive_skeleton_coverage_ratio": round(
                    mean(safe_float(row.get("adaptive_skeleton_coverage_ratio")) for row in items),
                    8,
                ),
                "mean_adaptive_skeleton_precision_ratio": round(
                    mean(safe_float(row.get("adaptive_skeleton_precision_ratio")) for row in items),
                    8,
                ),
                "mean_skeleton_overfill_outside_adaptive_ratio": round(
                    mean(safe_float(row.get("skeleton_overfill_outside_adaptive_ratio")) for row in items),
                    8,
                ),
                "low_skeleton_coverage_samples": sum(
                    1 for row in items if safe_float(row.get("skeleton_coverage_ratio")) < safe_float(row.get("skeleton_coverage_threshold"))
                ),
                "low_adaptive_skeleton_coverage_samples": sum(
                    1
                    for row in items
                    if safe_float(row.get("adaptive_skeleton_coverage_ratio"))
                    < safe_float(row.get("adaptive_skeleton_coverage_threshold"))
                ),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit line-art DST outputs against skeleton/centerline targets.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--selection", action="append", nargs=2, metavar=("NAME", "ROWS_CSV"), required=True)
    parser.add_argument("--output-root", action="append", nargs=2, metavar=("NAME", "OUTPUT_DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-filter", default="QuickDraw")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--line-radius-px", type=int, default=1)
    parser.add_argument("--adaptive-precision-radius-px", type=int, default=1)
    parser.add_argument("--skeleton-coverage-threshold", type=float, default=0.95)
    parser.add_argument("--adaptive-skeleton-coverage-threshold", type=float, default=0.95)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["sample_id"]: row for row in read_csv(dataset_dir / "manifest.csv")}
    roots = {name: Path(path) for name, path in args.output_root}
    source_filter = {item.strip() for item in args.source_filter.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []

    for selection_name, csv_path in args.selection:
        root = roots.get(selection_name)
        if root is None:
            raise ValueError(f"Missing --output-root for selection {selection_name!r}")
        for selected in read_csv(Path(csv_path)):
            sample_id = selected["sample_id"]
            sample = manifest[sample_id]
            if source_filter and sample.get("source_name", "") not in source_filter:
                continue
            skeleton_rel = sample.get("skeleton_path", "")
            if not skeleton_rel:
                continue
            dst_path = root / sample_id / "prediction.dst"
            if not dst_path.exists():
                raise FileNotFoundError(dst_path)
            metrics = coverage_metrics(
                dst_path,
                dataset_dir / skeleton_rel,
                target_width_mm=args.target_width_mm,
                line_radius_px=args.line_radius_px,
                adaptive_precision_radius_px=args.adaptive_precision_radius_px,
            )
            rows.append(
                {
                    "selection": selection_name,
                    "sample_id": sample_id,
                    "source_name": sample.get("source_name", ""),
                    "category": sample.get("category", ""),
                    "dst_path": str(dst_path),
                    "skeleton_path": str(dataset_dir / skeleton_rel),
                    "skeleton_coverage_threshold": args.skeleton_coverage_threshold,
                    "adaptive_skeleton_coverage_threshold": args.adaptive_skeleton_coverage_threshold,
                    "skeleton_coverage_ratio": metrics["coverage_ratio"],
                    "skeleton_precision_ratio": metrics["stitch_precision_ratio"],
                    "adaptive_skeleton_coverage_ratio": metrics["adaptive_coverage_ratio"],
                    "adaptive_skeleton_precision_ratio": metrics["adaptive_stitch_precision_ratio"],
                    "skeleton_overfill_outside_adaptive_ratio": metrics["overfill_outside_adaptive_ratio"],
                    "target_pixels": metrics["target_pixels"],
                    "stitch_pixels": metrics["stitch_pixels"],
                    "overlap_pixels": metrics["overlap_pixels"],
                    "adaptive_overlap_pixels": metrics["adaptive_overlap_pixels"],
                    "low_skeleton_coverage": metrics["coverage_ratio"] < args.skeleton_coverage_threshold,
                    "low_adaptive_skeleton_coverage": metrics["adaptive_coverage_ratio"] < args.adaptive_skeleton_coverage_threshold,
                }
            )

    by_selection = summarize(rows, "selection")
    by_category = summarize(rows, "category")
    review_rows = [
        row
        for row in rows
        if str(row.get("low_skeleton_coverage", "")).lower() == "true"
        or str(row.get("low_adaptive_skeleton_coverage", "")).lower() == "true"
    ]
    write_csv(rows, output_dir / "line_skeleton_coverage_rows.csv")
    write_csv(by_selection, output_dir / "line_skeleton_coverage_by_selection.csv")
    write_csv(by_category, output_dir / "line_skeleton_coverage_by_category.csv")
    write_csv(review_rows, output_dir / "line_skeleton_coverage_review_queue.csv")
    write_json(
        {
            "dataset_dir": args.dataset_dir,
            "source_filter": sorted(source_filter),
            "samples": len(rows),
            "review_samples": len(review_rows),
            "by_selection": by_selection,
            "by_category": by_category,
            "args": vars(args),
        },
        output_dir / "line_skeleton_coverage_summary.json",
    )
    print(json.dumps({"samples": len(rows), "review_samples": len(review_rows), "by_selection": by_selection}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
