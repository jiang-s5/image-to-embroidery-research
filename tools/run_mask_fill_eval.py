from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_executability import analyze_file
from eval_stitch_coverage import coverage_metrics
from generate_mask_fill_dst import generate_mask_fill_dst
from score_unified import score_metrics


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
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "samples": len(rows),
        "hard_fail": sum(1 for row in rows if row.get("quality_level") == "hard_fail"),
    }
    for key in keys:
        values = [safe_float(row.get(key)) for row in rows]
        summary[f"mean_{key}"] = round(mean(values), 6) if values else 0.0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate and evaluate simple mask-fill DST candidates.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--use-eval-mask-dir", required=True)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--row-spacing-mm", type=float, default=1.2)
    parser.add_argument("--max-stitch-mm", type=float, default=3.2)
    parser.add_argument("--max-connect-mm", type=float, default=6.0)
    parser.add_argument("--min-connect-inside-fraction", type=float, default=0.95)
    parser.add_argument("--min-component-pixels", type=int, default=64)
    parser.add_argument("--min-run-mm", type=float, default=1.0)
    parser.add_argument("--use-mask-path-connectors", action="store_true")
    parser.add_argument("--max-mask-path-mm", type=float, default=24.0)
    parser.add_argument("--max-mask-path-expansions", type=int, default=8000)
    parser.add_argument("--add-outline", action="store_true")
    parser.add_argument("--outline-stride-px", type=int, default=2)
    parser.add_argument("--outline-min-area-px", type=int, default=64)
    parser.add_argument("--outline-external-only", action="store_true")
    parser.add_argument("--outline-inset-px", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    eval_root = Path(args.use_eval_mask_dir)
    eval_mask_lookup: dict[str, Path] = {}
    for row in read_csv(eval_root / "manifest_eval_masks.csv"):
        if row.get("eval_mask_path"):
            eval_mask_lookup[row["sample_id"]] = eval_root / row["eval_mask_path"]

    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for row in read_csv(dataset_dir / args.manifest):
        sample_id = row["sample_id"]
        eval_mask = eval_mask_lookup.get(sample_id)
        if eval_mask is None:
            eval_mask = dataset_dir / row["mask_path"]
        sample_out = output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        output_dst = sample_out / "prediction.dst"
        generator_report = generate_mask_fill_dst(
            dataset_dir / row["mask_path"],
            output_dst,
            connector_mask_path=eval_mask,
            target_width_mm=args.target_width_mm,
            row_spacing_mm=args.row_spacing_mm,
            max_stitch_mm=args.max_stitch_mm,
            max_connect_mm=args.max_connect_mm,
            min_connect_inside_fraction=args.min_connect_inside_fraction,
            min_component_pixels=args.min_component_pixels,
            min_run_mm=args.min_run_mm,
            use_mask_path_connectors=args.use_mask_path_connectors,
            max_mask_path_mm=args.max_mask_path_mm,
            max_mask_path_expansions=args.max_mask_path_expansions,
            add_outline=args.add_outline,
            outline_stride_px=args.outline_stride_px,
            outline_min_area_px=args.outline_min_area_px,
            outline_include_holes=not args.outline_external_only,
            outline_inset_px=args.outline_inset_px,
        )
        (sample_out / "generator_report.json").write_text(json.dumps(generator_report, ensure_ascii=False, indent=2), encoding="utf-8")
        pred = analyze_file(
            output_dst,
            max_stitch_mm=4.0,
            high_risk_jump_mm=8.0,
            mask_path=eval_mask,
            target_width_mm=args.target_width_mm,
        )
        (sample_out / "eval_executability.json").write_text(json.dumps({"pred": pred}, ensure_ascii=False, indent=2), encoding="utf-8")
        score = score_metrics(pred)
        coverage = coverage_metrics(
            output_dst,
            dataset_dir / row["mask_path"],
            target_width_mm=args.target_width_mm,
            line_radius_px=2,
        )
        (sample_out / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
        metric_rows.append(
            {
                "sample_id": sample_id,
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                "mode": ("mask_fill_edgewalk_outline" if args.add_outline and args.use_mask_path_connectors else "mask_fill_edgewalk" if args.use_mask_path_connectors else "mask_fill_serpentine"),
                "unified_loss": score["unified_loss"],
                "exec_score": score["exec_score"],
                "visual_risk": score["visual_risk"],
                "quality_level": score["quality_level"],
                "jump_count": pred.get("jump_count", ""),
                "trim_count": pred.get("trim_count", ""),
                "jump_path_mm": pred.get("jump_path_mm", ""),
                "off_mask_stitch_length_mm": pred.get("off_mask_stitch_length_mm", ""),
                "visible_connector_count": pred.get("visible_connector_count", ""),
                "visible_connector_length_mm": pred.get("visible_connector_length_mm", ""),
                "fill_rows": generator_report["fill_rows"],
                "safe_connects": generator_report["safe_connects"],
                "mask_path_connects": generator_report["mask_path_connects"],
                "rejected_connects": generator_report["rejected_connects"],
                "rejected_mask_paths": generator_report["rejected_mask_paths"],
                "outline_paths": generator_report["outline_paths"],
                "outline_points": generator_report["outline_points"],
            }
        )
        coverage_rows.append(
            {
                "sample_id": sample_id,
                "source_name": row.get("source_name", ""),
                "category": row.get("category", ""),
                **{key: coverage[key] for key in ("target_pixels", "stitch_pixels", "overlap_pixels", "off_target_pixels", "coverage_ratio", "stitch_precision_ratio")},
            }
        )

    write_csv(metric_rows, output_dir / "source_aware_hybrid_rows.csv")
    write_csv(coverage_rows, output_dir / "coverage_rows.csv")
    metrics_summary = summarize(
        metric_rows,
        [
            "unified_loss",
            "exec_score",
            "visual_risk",
            "jump_count",
            "trim_count",
            "off_mask_stitch_length_mm",
            "visible_connector_count",
            "visible_connector_length_mm",
        ],
    )
    coverage_summary = summarize(coverage_rows, ["coverage_ratio", "stitch_precision_ratio"])
    (output_dir / "source_aware_hybrid_summary.json").write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "coverage_summary.json").write_text(json.dumps(coverage_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics_summary, "coverage": coverage_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
