from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval_stitch_coverage import coverage_metrics
from score_unified import extract_pred, score_metrics


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


def load_eval_metrics(eval_json: Path) -> dict[str, Any]:
    if not eval_json.exists():
        return {}
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    pred = extract_pred(payload)
    score = score_metrics(pred)
    return {
        **pred,
        **{key: value for key, value in score.items() if key not in {"score_terms", "score_weights", "score_thresholds"}},
    }


def candidate_roots(args: argparse.Namespace, sample_id: str) -> list[Path]:
    roots: list[Path] = []
    if args.candidate_root:
        roots.extend(Path(path) for path in args.candidate_root)
    else:
        root = Path(args.results_root)
        roots.extend(path for path in root.iterdir() if path.is_dir() and (path / sample_id / "prediction.dst").exists())
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if not (root / sample_id / "prediction.dst").exists():
            continue
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return sorted(unique, key=lambda path: path.name.lower())


def balanced_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    coverage_deficit = max(0.0, args.target_coverage - safe_float(row.get("coverage_ratio")))
    adaptive_coverage_deficit = max(0.0, args.target_adaptive_coverage - safe_float(row.get("adaptive_coverage_ratio")))
    adaptive_precision_deficit = max(0.0, args.target_adaptive_precision - safe_float(row.get("adaptive_stitch_precision_ratio")))
    return round(
        safe_float(row.get("unified_loss"))
        + args.jump_weight * safe_float(row.get("jump_count"))
        + args.trim_weight * safe_float(row.get("trim_count"))
        + args.coverage_deficit_weight * coverage_deficit
        + args.adaptive_coverage_deficit_weight * adaptive_coverage_deficit
        + args.adaptive_precision_deficit_weight * adaptive_precision_deficit
        + args.off_mask_weight * safe_float(row.get("off_mask_stitch_length_mm"))
        + args.visible_weight * safe_float(row.get("visible_connector_count")),
        8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan existing M2 candidate DST outputs for selected benchmark samples.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--candidate-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--line-radius-px", type=int, default=2)
    parser.add_argument("--adaptive-precision-radius-px", type=int, default=2)
    parser.add_argument("--target-coverage", type=float, default=0.98)
    parser.add_argument("--target-adaptive-coverage", type=float, default=0.94)
    parser.add_argument("--target-adaptive-precision", type=float, default=0.90)
    parser.add_argument("--jump-weight", type=float, default=0.004)
    parser.add_argument("--trim-weight", type=float, default=0.002)
    parser.add_argument("--coverage-deficit-weight", type=float, default=0.08)
    parser.add_argument("--adaptive-coverage-deficit-weight", type=float, default=0.12)
    parser.add_argument("--adaptive-precision-deficit-weight", type=float, default=0.16)
    parser.add_argument("--off-mask-weight", type=float, default=0.01)
    parser.add_argument("--visible-weight", type=float, default=0.08)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["sample_id"]: row for row in read_csv(dataset_dir / "manifest.csv")}
    rows: list[dict[str, Any]] = []

    for sample_id in args.sample_id:
        if sample_id not in manifest:
            raise ValueError(f"Unknown sample_id {sample_id!r}")
        sample = manifest[sample_id]
        mask_path = dataset_dir / sample["mask_path"]
        for root in candidate_roots(args, sample_id):
            sample_root = root / sample_id
            dst_path = sample_root / "prediction.dst"
            exec_metrics = load_eval_metrics(sample_root / "eval_executability.json")
            coverage = coverage_metrics(
                dst_path,
                mask_path,
                target_width_mm=args.target_width_mm,
                line_radius_px=args.line_radius_px,
                adaptive_precision_radius_px=args.adaptive_precision_radius_px,
            )
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "source_name": sample.get("source_name", ""),
                "category": sample.get("category", ""),
                "candidate": root.name,
                "dst_path": str(dst_path),
                "target_mask_path": str(mask_path),
                "quality_level": exec_metrics.get("quality_level", ""),
                "unified_loss": exec_metrics.get("unified_loss", ""),
                "jump_count": exec_metrics.get("jump_count", ""),
                "trim_count": exec_metrics.get("trim_count", ""),
                "off_mask_stitch_length_mm": exec_metrics.get("off_mask_stitch_length_mm", ""),
                "visible_connector_count": exec_metrics.get("visible_connector_count", ""),
                "jump_path_mm": exec_metrics.get("jump_path_mm", ""),
                "visible_connector_length_mm": exec_metrics.get("visible_connector_length_mm", ""),
                **coverage,
            }
            row["balanced_score"] = balanced_score(row, args)
            rows.append(row)

    top_balanced = sorted(
        rows,
        key=lambda row: (
            row.get("quality_level") == "hard_fail",
            safe_float(row.get("balanced_score")),
            safe_float(row.get("jump_count")),
            safe_float(row.get("unified_loss")),
        ),
    )
    top_adaptive_precision = sorted(
        rows,
        key=lambda row: (
            row.get("quality_level") == "hard_fail",
            -safe_float(row.get("adaptive_stitch_precision_ratio")),
            -safe_float(row.get("adaptive_coverage_ratio")),
            safe_float(row.get("jump_count")),
        ),
    )
    top_low_jump = sorted(
        rows,
        key=lambda row: (
            row.get("quality_level") == "hard_fail",
            safe_float(row.get("jump_count")),
            safe_float(row.get("unified_loss")),
            -safe_float(row.get("adaptive_stitch_precision_ratio")),
        ),
    )

    write_csv(rows, output_dir / "candidate_rows.csv")
    write_csv(top_balanced[:50], output_dir / "top_balanced.csv")
    write_csv(top_adaptive_precision[:50], output_dir / "top_adaptive_precision.csv")
    write_csv(top_low_jump[:50], output_dir / "top_low_jump.csv")
    write_json(
        {
            "dataset_dir": args.dataset_dir,
            "sample_ids": args.sample_id,
            "candidate_count": len(rows),
            "top_balanced": top_balanced[:10],
            "top_adaptive_precision": top_adaptive_precision[:10],
            "top_low_jump": top_low_jump[:10],
            "args": vars(args),
        },
        output_dir / "candidate_scan_summary.json",
    )
    print(json.dumps({"candidate_count": len(rows), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
