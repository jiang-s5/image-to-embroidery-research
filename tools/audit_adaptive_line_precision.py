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
                "mean_coverage_ratio": round(mean(safe_float(row.get("coverage_ratio")) for row in items), 8),
                "mean_stitch_precision_ratio": round(mean(safe_float(row.get("stitch_precision_ratio")) for row in items), 8),
                "mean_adaptive_coverage_ratio": round(mean(safe_float(row.get("adaptive_coverage_ratio")) for row in items), 8),
                "mean_adaptive_stitch_precision_ratio": round(
                    mean(safe_float(row.get("adaptive_stitch_precision_ratio")) for row in items),
                    8,
                ),
                "mean_overfill_outside_adaptive_ratio": round(
                    mean(safe_float(row.get("overfill_outside_adaptive_ratio")) for row in items),
                    8,
                ),
                "low_strict_precision_samples": sum(
                    1 for row in items if safe_float(row.get("stitch_precision_ratio")) < safe_float(row.get("strict_precision_threshold"))
                ),
                "low_adaptive_precision_samples": sum(
                    1 for row in items if safe_float(row.get("adaptive_stitch_precision_ratio")) < safe_float(row.get("adaptive_precision_threshold"))
                ),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit strict vs adaptive line-domain stitch precision for selected DST outputs.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--selection", action="append", nargs=2, metavar=("NAME", "SELECTED_ROWS_CSV"), required=True)
    parser.add_argument("--output-root", action="append", nargs=2, metavar=("NAME", "OUTPUT_DIR"), required=True)
    parser.add_argument("--candidate-root", action="append", nargs=2, metavar=("CANDIDATE", "OUTPUT_DIR"), default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--line-radius-px", type=int, default=2)
    parser.add_argument("--adaptive-precision-radius-px", type=int, default=2)
    parser.add_argument("--strict-precision-threshold", type=float, default=0.75)
    parser.add_argument("--adaptive-precision-threshold", type=float, default=0.90)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["sample_id"]: row for row in read_csv(dataset_dir / "manifest.csv")}
    output_roots = {name: Path(path) for name, path in args.output_root}
    candidate_roots = {name: Path(path) for name, path in args.candidate_root}
    rows: list[dict[str, Any]] = []

    for selection_name, csv_path in args.selection:
        root = output_roots.get(selection_name)
        if root is None:
            raise ValueError(f"Missing --output-root for selection {selection_name!r}")
        for row in read_csv(Path(csv_path)):
            sample_id = row["sample_id"]
            sample = manifest[sample_id]
            dst_path = root / sample_id / "prediction.dst"
            candidate_name = row.get("chosen_candidate", "")
            if not dst_path.exists() and candidate_name in candidate_roots:
                dst_path = candidate_roots[candidate_name] / sample_id / "prediction.dst"
            if not dst_path.exists():
                raise FileNotFoundError(dst_path)
            metrics = coverage_metrics(
                dst_path,
                dataset_dir / sample["mask_path"],
                target_width_mm=args.target_width_mm,
                line_radius_px=args.line_radius_px,
                adaptive_precision_radius_px=args.adaptive_precision_radius_px,
            )
            enriched = {
                "selection": selection_name,
                "sample_id": sample_id,
                "source_name": row.get("source_name", sample.get("source_name", "")),
                "category": row.get("category", sample.get("category", "")),
                "chosen_candidate": row.get("chosen_candidate", ""),
                "strict_precision_threshold": args.strict_precision_threshold,
                "adaptive_precision_threshold": args.adaptive_precision_threshold,
                **metrics,
                "strict_precision_low": metrics["stitch_precision_ratio"] < args.strict_precision_threshold,
                "adaptive_precision_low": metrics["adaptive_stitch_precision_ratio"] < args.adaptive_precision_threshold,
            }
            rows.append(enriched)

    low_rows = sorted(
        [row for row in rows if row["adaptive_precision_low"]],
        key=lambda row: (safe_float(row.get("adaptive_stitch_precision_ratio")), row.get("selection", ""), row.get("sample_id", "")),
    )
    write_csv(rows, output_dir / "adaptive_precision_rows.csv")
    write_csv(low_rows, output_dir / "adaptive_precision_review_queue.csv")
    by_selection = summarize(rows, "selection")
    by_source = summarize(rows, "source_name")
    write_csv(by_selection, output_dir / "adaptive_precision_by_selection.csv")
    write_csv(by_source, output_dir / "adaptive_precision_by_source.csv")
    write_json(
        {
            "dataset_dir": args.dataset_dir,
            "target_width_mm": args.target_width_mm,
            "line_radius_px": args.line_radius_px,
            "adaptive_precision_radius_px": args.adaptive_precision_radius_px,
            "strict_precision_threshold": args.strict_precision_threshold,
            "adaptive_precision_threshold": args.adaptive_precision_threshold,
            "samples": len(rows),
            "review_samples": len(low_rows),
            "by_selection": by_selection,
            "by_source": by_source,
            "top_review_samples": low_rows[:10],
        },
        output_dir / "adaptive_precision_summary.json",
    )
    print(json.dumps({"samples": len(rows), "review_samples": len(low_rows), "by_selection": by_selection}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
