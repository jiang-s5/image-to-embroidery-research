from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from statistics import mean
from typing import Any

from train_m2_candidate_selector import (
    build_candidate_rows,
    groups_by_sample,
    safe_float,
    selectable_candidates,
    write_csv,
)


def read_manifest_phases(dataset_dir: Path) -> dict[str, str]:
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        return {row["sample_id"]: row.get("phase", "") for row in csv.DictReader(handle)}


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def calibrated_score(row: dict[str, Any], config: dict[str, float]) -> tuple[float, dict[str, float]]:
    coverage = safe_float(row.get("coverage_ratio"))
    precision = safe_float(row.get("stitch_precision_ratio"))
    coverage_deficit = max(0.0, config["min_coverage"] - coverage)
    precision_deficit = max(0.0, config["min_precision"] - precision)
    jump_norm = safe_float(row.get("jump_count")) / max(1.0, config["jump_scale"])
    trim_norm = safe_float(row.get("trim_count")) / max(1.0, config["trim_scale"])
    off_mask = safe_float(row.get("off_mask_stitch_length_mm"))
    visible = safe_float(row.get("visible_connector_count"))
    hard_penalty = config["hard_fail_penalty"] if row.get("oracle_quality_level") == "hard_fail" else 0.0
    score = (
        config["unified_weight"] * safe_float(row.get("unified_loss"))
        + config["coverage_weight"] * coverage_deficit
        + config["precision_weight"] * precision_deficit
        + config["jump_weight"] * jump_norm
        + config["trim_weight"] * trim_norm
        + config["off_mask_weight"] * off_mask
        + config["visible_weight"] * visible
        + hard_penalty
    )
    return score, {
        "calibrated_score": round(score, 8),
        "coverage_deficit": round(coverage_deficit, 6),
        "precision_deficit": round(precision_deficit, 6),
        "jump_norm": round(jump_norm, 6),
        "trim_norm": round(trim_norm, 6),
        "off_mask_penalty": round(config["off_mask_weight"] * off_mask, 6),
        "visible_penalty": round(config["visible_weight"] * visible, 6),
        "hard_penalty": round(hard_penalty, 6),
    }


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "features"}


def select_rows(
    rows: list[dict[str, Any]],
    config: dict[str, float],
    flat_min_coverage: float,
    line_min_coverage: float,
    exclude_hard_fail: bool,
    enforce_coverage_floor: bool,
    coverage_floor_tolerance: float,
    coverage_floor_mode: str,
    coverage_floor_line_sources: set[str],
    style_aware_hard_gate: bool,
    mask_fill_hard_gate: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sample_id, group in sorted(groups_by_sample(rows).items()):
        candidates = selectable_candidates(
            group,
            exclude_hard_fail=exclude_hard_fail,
            enforce_coverage_floor=enforce_coverage_floor,
            flat_min_coverage=flat_min_coverage,
            line_min_coverage=line_min_coverage,
            coverage_floor_tolerance=coverage_floor_tolerance,
            coverage_floor_mode=coverage_floor_mode,
            coverage_floor_line_sources=coverage_floor_line_sources,
            style_aware_hard_gate=style_aware_hard_gate,
            style_aware_max_off_mask_mm=config["style_aware_max_off_mask_mm"],
            style_aware_max_jump_count=config["style_aware_max_jump_count"],
            style_aware_max_trim_count=config["style_aware_max_trim_count"],
            style_aware_min_precision=config["style_aware_min_precision"],
            style_aware_min_coverage=config["style_aware_min_coverage"],
            mask_fill_hard_gate=mask_fill_hard_gate,
            mask_fill_max_off_mask_mm=config["mask_fill_max_off_mask_mm"],
            mask_fill_max_jump_count=config["mask_fill_max_jump_count"],
            mask_fill_max_trim_count=config["mask_fill_max_trim_count"],
            mask_fill_min_precision=config["mask_fill_min_precision"],
            mask_fill_min_coverage=config["mask_fill_min_coverage"],
        )
        scored = []
        for row in candidates:
            score, terms = calibrated_score(row, config)
            scored.append((score, row, terms))
        score, chosen, terms = min(scored, key=lambda item: item[0])
        out = flatten_row(dict(chosen))
        out["chosen_candidate"] = out.pop("candidate")
        out["quality_level"] = out.get("oracle_quality_level", "")
        out["calibrated_score"] = round(score, 8)
        out.update(terms)
        selected.append(out)
    return selected


def summarize_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return round(mean(safe_float(row.get(key)) for row in rows), 6) if rows else 0.0

    chosen_counts: dict[str, int] = {}
    for row in rows:
        candidate = str(row.get("chosen_candidate", ""))
        chosen_counts[candidate] = chosen_counts.get(candidate, 0) + 1
    return {
        "samples": len(rows),
        "hard_fail": sum(1 for row in rows if row.get("quality_level") == "hard_fail"),
        "mean_calibrated_score": avg("calibrated_score"),
        "mean_unified_loss": avg("unified_loss"),
        "mean_jump_count": avg("jump_count"),
        "mean_trim_count": avg("trim_count"),
        "mean_off_mask_stitch_length_mm": avg("off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg("visible_connector_count"),
        "mean_coverage_ratio": avg("coverage_ratio"),
        "mean_stitch_precision_ratio": avg("stitch_precision_ratio"),
        "chosen_counts": chosen_counts,
    }


def selection_objective(summary: dict[str, Any], coverage_target: float) -> float:
    coverage_deficit = max(0.0, coverage_target - safe_float(summary.get("mean_coverage_ratio")))
    return (
        safe_float(summary.get("mean_unified_loss"))
        + 0.25 * coverage_deficit
        + 0.04 * safe_float(summary.get("mean_off_mask_stitch_length_mm"))
        + 0.01 * safe_float(summary.get("mean_visible_connector_count"))
    )


def config_grid(args: argparse.Namespace) -> list[dict[str, float]]:
    configs: list[dict[str, float]] = []
    for coverage_weight, precision_weight, jump_weight, trim_weight, off_mask_weight in itertools.product(
        parse_float_list(args.coverage_weights),
        parse_float_list(args.precision_weights),
        parse_float_list(args.jump_weights),
        parse_float_list(args.trim_weights),
        parse_float_list(args.off_mask_weights),
    ):
        configs.append(
            {
                "unified_weight": args.unified_weight,
                "coverage_weight": coverage_weight,
                "precision_weight": precision_weight,
                "jump_weight": jump_weight,
                "trim_weight": trim_weight,
                "off_mask_weight": off_mask_weight,
                "visible_weight": args.visible_weight,
                "hard_fail_penalty": args.hard_fail_penalty,
                "min_coverage": args.min_coverage,
                "min_precision": args.min_precision,
                "jump_scale": args.jump_scale,
                "trim_scale": args.trim_scale,
                "style_aware_max_off_mask_mm": args.style_aware_max_off_mask_mm,
                "style_aware_max_jump_count": args.style_aware_max_jump_count,
                "style_aware_max_trim_count": args.style_aware_max_trim_count,
                "style_aware_min_precision": args.style_aware_min_precision,
                "style_aware_min_coverage": args.style_aware_min_coverage,
                "mask_fill_max_off_mask_mm": args.mask_fill_max_off_mask_mm,
                "mask_fill_max_jump_count": args.mask_fill_max_jump_count,
                "mask_fill_max_trim_count": args.mask_fill_max_trim_count,
                "mask_fill_min_precision": args.mask_fill_min_precision,
                "mask_fill_min_coverage": args.mask_fill_min_coverage,
            }
        )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune a two-stage calibrated M2 candidate selector.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--config-json", default="", help="Apply an existing calibrated_config.json instead of tuning a grid.")
    parser.add_argument("--train-phase", default="core")
    parser.add_argument("--test-phase", default="full")
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--enforce-coverage-floor", action="store_true")
    parser.add_argument("--coverage-floor-tolerance", type=float, default=0.0)
    parser.add_argument("--coverage-floor-mode", choices=["branch", "source"], default="branch")
    parser.add_argument("--coverage-floor-line-sources", default="QuickDraw,Rendered text")
    parser.add_argument("--coverage-weight-for-oracle", type=float, default=0.40)
    parser.add_argument("--precision-weight-for-oracle", type=float, default=0.15)
    parser.add_argument("--hard-fail-penalty", type=float, default=0.04)
    parser.add_argument("--exclude-hard-fail", action="store_true")
    parser.add_argument("--style-aware-hard-gate", action="store_true")
    parser.add_argument("--mask-fill-hard-gate", action="store_true")
    parser.add_argument("--unified-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weights", default="0.20,0.35,0.50")
    parser.add_argument("--precision-weights", default="0.05,0.15,0.30")
    parser.add_argument("--jump-weights", default="0.00,0.02")
    parser.add_argument("--trim-weights", default="0.00,0.02")
    parser.add_argument("--off-mask-weights", default="0.02,0.05,0.10")
    parser.add_argument("--visible-weight", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--min-precision", type=float, default=0.72)
    parser.add_argument("--jump-scale", type=float, default=20.0)
    parser.add_argument("--trim-scale", type=float, default=8.0)
    parser.add_argument("--selection-coverage-target", type=float, default=0.82)
    parser.add_argument("--style-aware-max-off-mask-mm", type=float, default=0.50)
    parser.add_argument("--style-aware-max-jump-count", type=float, default=24.0)
    parser.add_argument("--style-aware-max-trim-count", type=float, default=5.0)
    parser.add_argument("--style-aware-min-precision", type=float, default=0.68)
    parser.add_argument("--style-aware-min-coverage", type=float, default=0.78)
    parser.add_argument("--mask-fill-max-off-mask-mm", type=float, default=0.50)
    parser.add_argument("--mask-fill-max-jump-count", type=float, default=24.0)
    parser.add_argument("--mask-fill-max-trim-count", type=float, default=5.0)
    parser.add_argument("--mask-fill-min-precision", type=float, default=0.68)
    parser.add_argument("--mask-fill-min-coverage", type=float, default=0.78)
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
        args.coverage_weight_for_oracle,
        args.precision_weight_for_oracle,
        args.hard_fail_penalty,
    )
    coverage_floor_line_sources = {item.strip() for item in args.coverage_floor_line_sources.split(",") if item.strip()}
    if args.config_json:
        config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        selected = select_rows(
            rows,
            config,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
            args.style_aware_hard_gate,
            args.mask_fill_hard_gate,
        )
        write_csv(selected, output_dir / "applied_selected_rows.csv")
        summary = {
            "mode": "apply",
            "samples": len(groups_by_sample(rows)),
            "candidate_rows": len(rows),
            "exclude_hard_fail": args.exclude_hard_fail,
            "enforce_coverage_floor": args.enforce_coverage_floor,
            "coverage_floor_tolerance": args.coverage_floor_tolerance,
            "coverage_floor_mode": args.coverage_floor_mode,
            "coverage_floor_line_sources": sorted(coverage_floor_line_sources),
            "style_aware_hard_gate": args.style_aware_hard_gate,
            "mask_fill_hard_gate": args.mask_fill_hard_gate,
            "config_json": args.config_json,
            "config": config,
            "applied": summarize_selected(selected),
        }
        (output_dir / "applied_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    phases = read_manifest_phases(dataset_dir)
    train_rows = [row for row in rows if phases.get(str(row["sample_id"])) == args.train_phase]
    test_rows = [row for row in rows if phases.get(str(row["sample_id"])) == args.test_phase]

    grid_rows: list[dict[str, Any]] = []
    best: tuple[float, int, dict[str, float], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None = None
    for config_id, config in enumerate(config_grid(args)):
        selected_train = select_rows(
            train_rows,
            config,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
            args.style_aware_hard_gate,
            args.mask_fill_hard_gate,
        )
        selected_test = select_rows(
            test_rows,
            config,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
            args.style_aware_hard_gate,
            args.mask_fill_hard_gate,
        )
        train_summary = summarize_selected(selected_train)
        test_summary = summarize_selected(selected_test)
        objective = selection_objective(train_summary, args.selection_coverage_target)
        row = {
            "config_id": config_id,
            "train_objective": round(objective, 8),
            **{f"config.{key}": value for key, value in config.items()},
            **{f"train.{key}": value for key, value in train_summary.items() if key != "chosen_counts"},
            **{f"test.{key}": value for key, value in test_summary.items() if key != "chosen_counts"},
        }
        grid_rows.append(row)
        if best is None or objective < best[0]:
            best = (objective, config_id, config, selected_train, selected_test, train_summary, test_summary)

    if best is None:
        raise RuntimeError("No calibrated selector configs were evaluated")

    objective, config_id, config, selected_train, selected_test, train_summary, test_summary = best
    write_csv(grid_rows, output_dir / "calibrated_grid_rows.csv")
    write_csv(selected_train, output_dir / "train_selected_rows.csv")
    write_csv(selected_test, output_dir / "test_selected_rows.csv")
    summary = {
        "train_phase": args.train_phase,
        "test_phase": args.test_phase,
        "train_samples": len(groups_by_sample(train_rows)),
        "test_samples": len(groups_by_sample(test_rows)),
        "candidate_rows_train": len(train_rows),
        "candidate_rows_test": len(test_rows),
        "configs_evaluated": len(grid_rows),
        "best_config_id": config_id,
        "best_train_objective": round(objective, 8),
        "exclude_hard_fail": args.exclude_hard_fail,
        "enforce_coverage_floor": args.enforce_coverage_floor,
        "coverage_floor_tolerance": args.coverage_floor_tolerance,
        "coverage_floor_mode": args.coverage_floor_mode,
        "coverage_floor_line_sources": sorted(coverage_floor_line_sources),
        "style_aware_hard_gate": args.style_aware_hard_gate,
        "mask_fill_hard_gate": args.mask_fill_hard_gate,
        "best_config": config,
        "train": train_summary,
        "test": test_summary,
    }
    (output_dir / "calibrated_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "calibrated_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
