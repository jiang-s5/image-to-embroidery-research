from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_m2_candidate_selector import build_candidate_rows, groups_by_sample, safe_float, write_csv
from tune_m2_calibrated_selector import config_grid, select_rows, summarize_selected, selection_objective


def config_row(config_id: int, config: dict[str, Any], objective: float, summary: dict[str, Any], sample_id: str) -> dict[str, Any]:
    return {
        "held_out_sample_id": sample_id,
        "config_id": config_id,
        "train_objective": round(objective, 8),
        **{f"config.{key}": value for key, value in config.items()},
        **{f"train.{key}": value for key, value in summary.items() if key not in {"chosen_counts", "coverage_profile_counts"}},
    }


def annotate_selected(row: dict[str, Any], sample_id: str, config_id: int, objective: float, config: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["held_out_sample_id"] = sample_id
    out["loo_best_config_id"] = config_id
    out["loo_train_objective"] = round(objective, 8)
    out["loo_coverage_profile"] = config.get("coverage_profile", "global")
    out["loo_coverage_weight"] = config.get("coverage_weight", "")
    out["loo_precision_weight"] = config.get("precision_weight", "")
    out["loo_jump_weight"] = config.get("jump_weight", "")
    out["loo_trim_weight"] = config.get("trim_weight", "")
    out["loo_off_mask_weight"] = config.get("off_mask_weight", "")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation for calibrated M2 candidate selector.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--flat-min-coverage", type=float, default=0.75)
    parser.add_argument("--line-min-coverage", type=float, default=0.45)
    parser.add_argument("--enforce-coverage-floor", action="store_true")
    parser.add_argument("--enforce-adaptive-coverage-floor", action="store_true")
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
    parser.add_argument("--coverage-profiles", default="global")
    parser.add_argument("--coverage-weights", default="0.20,0.35,0.50")
    parser.add_argument("--precision-weights", default="0.05,0.15,0.30")
    parser.add_argument("--jump-weights", default="0.00,0.02")
    parser.add_argument("--trim-weights", default="0.00,0.02")
    parser.add_argument("--off-mask-weights", default="0.02,0.05,0.10")
    parser.add_argument("--visible-weight", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--adaptive-flat-min-coverage", type=float, default=0.84)
    parser.add_argument("--adaptive-line-min-coverage", type=float, default=0.58)
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
    groups = groups_by_sample(rows)
    configs = config_grid(args)

    selected_rows: list[dict[str, Any]] = []
    best_config_rows: list[dict[str, Any]] = []
    all_config_rows: list[dict[str, Any]] = []

    for sample_id in sorted(groups):
        train_rows = [row for row in rows if str(row["sample_id"]) != sample_id]
        held_rows = groups[sample_id]
        best: tuple[float, int, dict[str, Any], dict[str, Any]] | None = None
        for config_id, config in enumerate(configs):
            selected_train = select_rows(
                train_rows,
                config,
                args.flat_min_coverage,
                args.line_min_coverage,
                args.exclude_hard_fail,
                args.enforce_coverage_floor,
                args.enforce_adaptive_coverage_floor,
                args.coverage_floor_tolerance,
                args.coverage_floor_mode,
                coverage_floor_line_sources,
                args.style_aware_hard_gate,
                args.mask_fill_hard_gate,
            )
            train_summary = summarize_selected(selected_train)
            objective = selection_objective(train_summary, args.selection_coverage_target)
            all_config_rows.append(config_row(config_id, config, objective, train_summary, sample_id))
            if best is None or objective < best[0]:
                best = (objective, config_id, config, train_summary)
        if best is None:
            raise RuntimeError(f"No configs evaluated for {sample_id}")
        objective, config_id, config, train_summary = best
        best_config_rows.append(config_row(config_id, config, objective, train_summary, sample_id))
        selected_held = select_rows(
            held_rows,
            config,
            args.flat_min_coverage,
            args.line_min_coverage,
            args.exclude_hard_fail,
            args.enforce_coverage_floor,
            args.enforce_adaptive_coverage_floor,
            args.coverage_floor_tolerance,
            args.coverage_floor_mode,
            coverage_floor_line_sources,
            args.style_aware_hard_gate,
            args.mask_fill_hard_gate,
        )
        selected_rows.extend(annotate_selected(row, sample_id, config_id, objective, config) for row in selected_held)

    summary = {
        "mode": "leave_one_out",
        "samples": len(groups),
        "candidate_rows": len(rows),
        "configs_evaluated_per_fold": len(configs),
        "exclude_hard_fail": args.exclude_hard_fail,
        "enforce_coverage_floor": args.enforce_coverage_floor,
        "enforce_adaptive_coverage_floor": args.enforce_adaptive_coverage_floor,
        "coverage_floor_tolerance": args.coverage_floor_tolerance,
        "coverage_floor_mode": args.coverage_floor_mode,
        "coverage_floor_line_sources": sorted(coverage_floor_line_sources),
        "style_aware_hard_gate": args.style_aware_hard_gate,
        "mask_fill_hard_gate": args.mask_fill_hard_gate,
        "coverage_profiles": args.coverage_profiles,
        "coverage_weights": args.coverage_weights,
        "precision_weights": args.precision_weights,
        "jump_weights": args.jump_weights,
        "trim_weights": args.trim_weights,
        "off_mask_weights": args.off_mask_weights,
        "leave_one_out": summarize_selected(selected_rows),
        "best_config_counts": {},
    }
    for row in best_config_rows:
        config_key = str(row["config_id"])
        summary["best_config_counts"][config_key] = summary["best_config_counts"].get(config_key, 0) + 1
    summary["mean_loo_train_objective"] = round(
        sum(safe_float(row.get("loo_train_objective")) for row in selected_rows) / max(1, len(selected_rows)),
        6,
    )

    write_csv(selected_rows, output_dir / "loo_selected_rows.csv")
    write_csv(best_config_rows, output_dir / "loo_best_config_rows.csv")
    write_csv(all_config_rows, output_dir / "loo_all_config_rows.csv")
    (output_dir / "loo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
