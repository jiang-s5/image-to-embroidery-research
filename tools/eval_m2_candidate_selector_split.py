from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from train_m2_candidate_selector import (
    build_candidate_rows,
    feature_names,
    fit_ridge,
    groups_by_sample,
    predict,
    safe_float,
    selectable_candidates,
    summarize_selected,
    write_csv,
)


def read_manifest_phases(dataset_dir: Path) -> dict[str, str]:
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        return {row["sample_id"]: row.get("phase", "") for row in csv.DictReader(handle)}


def select_with_model(
    model: dict[str, Any],
    rows: list[dict[str, Any]],
    exclude_hard_fail: bool = False,
    enforce_coverage_floor: bool = False,
    flat_min_coverage: float = 0.75,
    line_min_coverage: float = 0.45,
    coverage_floor_tolerance: float = 0.0,
    coverage_floor_mode: str = "branch",
    coverage_floor_line_sources: set[str] | None = None,
    style_aware_hard_gate: bool = False,
    style_aware_max_off_mask_mm: float = 0.05,
    style_aware_max_jump_count: float = 15.0,
    style_aware_max_trim_count: float = 3.0,
    style_aware_min_precision: float = 0.75,
    style_aware_min_coverage: float = 0.80,
    mask_fill_hard_gate: bool = False,
    mask_fill_max_off_mask_mm: float = 0.05,
    mask_fill_max_jump_count: float = 20.0,
    mask_fill_max_trim_count: float = 3.0,
    mask_fill_min_precision: float = 0.70,
    mask_fill_min_coverage: float = 0.80,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sample_id, group in sorted(groups_by_sample(rows).items()):
        selectable = selectable_candidates(
            group,
            exclude_hard_fail,
            enforce_coverage_floor,
            flat_min_coverage,
            line_min_coverage,
            coverage_floor_tolerance,
            coverage_floor_mode,
            coverage_floor_line_sources,
            style_aware_hard_gate,
            style_aware_max_off_mask_mm,
            style_aware_max_jump_count,
            style_aware_max_trim_count,
            style_aware_min_precision,
            style_aware_min_coverage,
            mask_fill_hard_gate,
            mask_fill_max_off_mask_mm,
            mask_fill_max_jump_count,
            mask_fill_max_trim_count,
            mask_fill_min_precision,
            mask_fill_min_coverage,
        )
        predictions = predict(model, selectable)
        ranked = sorted(zip(selectable, predictions), key=lambda item: item[1])
        chosen = dict(ranked[0][0])
        oracle_pool = selectable_candidates(
            group,
            exclude_hard_fail,
            enforce_coverage_floor,
            flat_min_coverage,
            line_min_coverage,
            coverage_floor_tolerance,
            coverage_floor_mode,
            coverage_floor_line_sources,
            style_aware_hard_gate,
            style_aware_max_off_mask_mm,
            style_aware_max_jump_count,
            style_aware_max_trim_count,
            style_aware_min_precision,
            style_aware_min_coverage,
            mask_fill_hard_gate,
            mask_fill_max_off_mask_mm,
            mask_fill_max_jump_count,
            mask_fill_max_trim_count,
            mask_fill_min_precision,
            mask_fill_min_coverage,
        )
        oracle = min(oracle_pool, key=lambda row: safe_float(row["oracle_score"]))
        chosen["predicted_score"] = round(ranked[0][1], 8)
        chosen["oracle_candidate"] = oracle["candidate"]
        chosen["learned_matches_oracle"] = 1 if chosen["candidate"] == oracle["candidate"] else 0
        selected.append(chosen)
    return selected


def fixed_candidate_summary(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    selected = []
    for _sample_id, group in groups_by_sample(rows).items():
        matches = [row for row in group if row["candidate"] == candidate]
        if matches:
            selected.append(dict(matches[0]))
    return summarize_selected(selected)


def oracle_summary(
    rows: list[dict[str, Any]],
    exclude_hard_fail: bool,
    enforce_coverage_floor: bool,
    flat_min_coverage: float,
    line_min_coverage: float,
    coverage_floor_tolerance: float,
    coverage_floor_mode: str,
    coverage_floor_line_sources: set[str],
    style_aware_hard_gate: bool,
    style_aware_max_off_mask_mm: float,
    style_aware_max_jump_count: float,
    style_aware_max_trim_count: float,
    style_aware_min_precision: float,
    style_aware_min_coverage: float,
    mask_fill_hard_gate: bool,
    mask_fill_max_off_mask_mm: float,
    mask_fill_max_jump_count: float,
    mask_fill_max_trim_count: float,
    mask_fill_min_precision: float,
    mask_fill_min_coverage: float,
) -> dict[str, Any]:
    selected = []
    for _sample_id, group in groups_by_sample(rows).items():
        pool = selectable_candidates(
            group,
            exclude_hard_fail,
            enforce_coverage_floor,
            flat_min_coverage,
            line_min_coverage,
            coverage_floor_tolerance,
            coverage_floor_mode,
            coverage_floor_line_sources,
            style_aware_hard_gate,
            style_aware_max_off_mask_mm,
            style_aware_max_jump_count,
            style_aware_max_trim_count,
            style_aware_min_precision,
            style_aware_min_coverage,
            mask_fill_hard_gate,
            mask_fill_max_off_mask_mm,
            mask_fill_max_jump_count,
            mask_fill_max_trim_count,
            mask_fill_min_precision,
            mask_fill_min_coverage,
        )
        best = min(pool, key=lambda row: safe_float(row["oracle_score"]))
        row = dict(best)
        row["learned_matches_oracle"] = 1
        selected.append(row)
    return summarize_selected(selected)


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "features"}
        flat.append(item)
    return flat


def main() -> int:
    parser = argparse.ArgumentParser(description="Train M2 candidate selector on one manifest phase and test on another.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--train-phase", default="core")
    parser.add_argument("--test-phase", default="full")
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
    parser.add_argument("--style-aware-hard-gate", action="store_true")
    parser.add_argument("--style-aware-max-off-mask-mm", type=float, default=0.05)
    parser.add_argument("--style-aware-max-jump-count", type=float, default=15.0)
    parser.add_argument("--style-aware-max-trim-count", type=float, default=3.0)
    parser.add_argument("--style-aware-min-precision", type=float, default=0.75)
    parser.add_argument("--style-aware-min-coverage", type=float, default=0.80)
    parser.add_argument("--mask-fill-hard-gate", action="store_true")
    parser.add_argument("--mask-fill-max-off-mask-mm", type=float, default=0.05)
    parser.add_argument("--mask-fill-max-jump-count", type=float, default=20.0)
    parser.add_argument("--mask-fill-max-trim-count", type=float, default=3.0)
    parser.add_argument("--mask-fill-min-precision", type=float, default=0.70)
    parser.add_argument("--mask-fill-min-coverage", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [(name, Path(path)) for name, path in args.candidate]
    rows = build_candidate_rows(
        dataset_dir,
        candidates,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_weight,
        args.precision_weight,
        args.hard_fail_penalty,
    )
    phases = read_manifest_phases(dataset_dir)
    train_rows = [row for row in rows if phases.get(str(row["sample_id"])) == args.train_phase]
    test_rows = [row for row in rows if phases.get(str(row["sample_id"])) == args.test_phase]
    names = feature_names(train_rows)
    model = fit_ridge(train_rows, names, args.alpha)
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}
    selected = select_with_model(
        model,
        test_rows,
        args.exclude_hard_fail,
        args.enforce_coverage_floor,
        args.flat_min_coverage,
        args.line_min_coverage,
        args.coverage_floor_tolerance,
        args.coverage_floor_mode,
        coverage_floor_line_sources,
        args.style_aware_hard_gate,
        args.style_aware_max_off_mask_mm,
        args.style_aware_max_jump_count,
        args.style_aware_max_trim_count,
        args.style_aware_min_precision,
        args.style_aware_min_coverage,
        args.mask_fill_hard_gate,
        args.mask_fill_max_off_mask_mm,
        args.mask_fill_max_jump_count,
        args.mask_fill_max_trim_count,
        args.mask_fill_min_precision,
        args.mask_fill_min_coverage,
    )

    write_csv(flatten_rows(train_rows), output_dir / "train_candidate_rows.csv")
    write_csv(flatten_rows(test_rows), output_dir / "test_candidate_rows.csv")
    write_csv(flatten_rows(selected), output_dir / "test_selected_rows.csv")
    (output_dir / "m2_candidate_selector_split_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    fixed = {name: fixed_candidate_summary(test_rows, name) for name, _path in candidates}
    summary = {
        "train_phase": args.train_phase,
        "test_phase": args.test_phase,
        "train_samples": len(groups_by_sample(train_rows)),
        "test_samples": len(groups_by_sample(test_rows)),
        "candidate_rows_train": len(train_rows),
        "candidate_rows_test": len(test_rows),
        "alpha": args.alpha,
        "exclude_hard_fail": args.exclude_hard_fail,
        "enforce_coverage_floor": args.enforce_coverage_floor,
        "coverage_floor_tolerance": args.coverage_floor_tolerance,
        "coverage_floor_mode": args.coverage_floor_mode,
        "coverage_floor_line_sources": sorted(coverage_floor_line_sources),
        "style_aware_hard_gate": args.style_aware_hard_gate,
        "style_aware_max_off_mask_mm": args.style_aware_max_off_mask_mm,
        "style_aware_max_jump_count": args.style_aware_max_jump_count,
        "style_aware_max_trim_count": args.style_aware_max_trim_count,
        "style_aware_min_precision": args.style_aware_min_precision,
        "style_aware_min_coverage": args.style_aware_min_coverage,
        "mask_fill_hard_gate": args.mask_fill_hard_gate,
        "mask_fill_max_off_mask_mm": args.mask_fill_max_off_mask_mm,
        "mask_fill_max_jump_count": args.mask_fill_max_jump_count,
        "mask_fill_max_trim_count": args.mask_fill_max_trim_count,
        "mask_fill_min_precision": args.mask_fill_min_precision,
        "mask_fill_min_coverage": args.mask_fill_min_coverage,
        "feature_count": len(names),
        "learned_selector": summarize_selected(selected),
        "oracle": oracle_summary(
            test_rows,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
            args.style_aware_hard_gate,
            args.style_aware_max_off_mask_mm,
            args.style_aware_max_jump_count,
            args.style_aware_max_trim_count,
            args.style_aware_min_precision,
            args.style_aware_min_coverage,
            args.mask_fill_hard_gate,
            args.mask_fill_max_off_mask_mm,
            args.mask_fill_max_jump_count,
            args.mask_fill_max_trim_count,
            args.mask_fill_min_precision,
            args.mask_fill_min_coverage,
        ),
        "fixed_candidates": fixed,
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
