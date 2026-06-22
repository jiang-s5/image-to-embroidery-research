from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

from train_m2_candidate_selector import build_candidate_rows, groups_by_sample, predict, safe_float, selectable_candidates


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return round(mean(safe_float(row.get(key)) for row in rows), 6)

    chosen_counts: dict[str, int] = {}
    for row in rows:
        chosen_counts[str(row["chosen_candidate"])] = chosen_counts.get(str(row["chosen_candidate"]), 0) + 1
    return {
        "samples": len(rows),
        "hard_fail": sum(1 for row in rows if row.get("quality_level") == "hard_fail"),
        "mean_predicted_score": avg("predicted_score"),
        "mean_oracle_score": avg("oracle_score"),
        "mean_unified_loss": avg("unified_loss"),
        "mean_jump_count": avg("jump_count"),
        "mean_trim_count": avg("trim_count"),
        "mean_off_mask_stitch_length_mm": avg("off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg("visible_connector_count"),
        "mean_coverage_ratio": avg("coverage_ratio"),
        "mean_stitch_precision_ratio": avg("stitch_precision_ratio"),
        "chosen_counts": chosen_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a trained M2 learned candidate selector.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument("--precision-weight", type=float, default=0.15)
    parser.add_argument("--hard-fail-penalty", type=float, default=0.04)
    parser.add_argument("--exclude-hard-fail", action="store_true")
    parser.add_argument("--enforce-coverage-floor", action="store_true")
    parser.add_argument("--coverage-floor-tolerance", type=float, default=0.0)
    parser.add_argument("--coverage-floor-mode", choices=["branch", "source"], default="branch")
    parser.add_argument("--coverage-floor-line-sources", default="QuickDraw,Rendered text")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [(name, Path(path)) for name, path in args.candidate]
    candidate_dirs = {name: path for name, path in candidates}
    rows = build_candidate_rows(
        Path(args.dataset_dir),
        candidates,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_weight,
        args.precision_weight,
        args.hard_fail_penalty,
    )
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}
    selected: list[dict[str, Any]] = []
    for sample_id, group in sorted(groups_by_sample(rows).items()):
        selectable = selectable_candidates(
            group,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
        )
        predictions = predict(model, selectable)
        ranked = sorted(zip(selectable, predictions), key=lambda item: item[1])
        chosen, predicted_score = ranked[0]
        sample_out = output_dir / sample_id
        sample_out.mkdir(parents=True, exist_ok=True)
        source_dir = candidate_dirs[str(chosen["candidate"])] / sample_id
        for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
            source = source_dir / filename
            if source.exists():
                shutil.copy2(source, sample_out / filename)
        selected.append(
            {
                "sample_id": sample_id,
                "source_name": chosen.get("source_name", ""),
                "category": chosen.get("category", ""),
                "chosen_candidate": chosen["candidate"],
                "predicted_score": round(float(predicted_score), 8),
                "oracle_score": chosen["oracle_score"],
                "quality_level": chosen["oracle_quality_level"],
                "unified_loss": chosen["unified_loss"],
                "jump_count": chosen["jump_count"],
                "trim_count": chosen["trim_count"],
                "off_mask_stitch_length_mm": chosen["off_mask_stitch_length_mm"],
                "visible_connector_count": chosen["visible_connector_count"],
                "coverage_ratio": chosen["coverage_ratio"],
                "stitch_precision_ratio": chosen["stitch_precision_ratio"],
            }
        )

    write_csv(selected, output_dir / "learned_selected_rows.csv")
    summary = summarize(selected)
    (output_dir / "learned_selected_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
