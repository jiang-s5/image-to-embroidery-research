from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from eval_stitch_coverage import coverage_metrics
from score_unified import extract_pred, score_metrics


METRIC_KEYS = (
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
    "adaptive_coverage_ratio",
    "adaptive_stitch_precision_ratio",
    "overfill_outside_adaptive_ratio",
)


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


def avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(safe_float(row.get(key)) for row in rows), 8)


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
                "profile_switches": sum(1 for row in items if str(row.get("profile_switched", "")).lower() == "true"),
                "hard_fail": sum(1 for row in items if str(row.get("quality_level", "")).lower() == "hard_fail"),
                "mean_unified_loss": avg(items, "unified_loss"),
                "mean_jump_count": avg(items, "jump_count"),
                "mean_trim_count": avg(items, "trim_count"),
                "mean_off_mask_stitch_length_mm": avg(items, "off_mask_stitch_length_mm"),
                "mean_visible_connector_count": avg(items, "visible_connector_count"),
                "mean_coverage_ratio": avg(items, "coverage_ratio"),
                "mean_stitch_precision_ratio": avg(items, "stitch_precision_ratio"),
                "mean_adaptive_coverage_ratio": avg(items, "adaptive_coverage_ratio"),
                "mean_adaptive_stitch_precision_ratio": avg(items, "adaptive_stitch_precision_ratio"),
                "mean_overfill_outside_adaptive_ratio": avg(items, "overfill_outside_adaptive_ratio"),
            }
        )
    return output


def copy_outputs(source_root: Path, sample_id: str, output_root: Path) -> None:
    sample_out = output_root / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json", "coverage_report.json"):
        src = source_root / sample_id / filename
        if src.exists():
            shutil.copy2(src, sample_out / filename)


def metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(safe_float(candidate.get(key)) - safe_float(baseline.get(key)), 8)


def load_shape_candidate_rows(dataset_dir: Path, candidate_root: Path, target_width_mm: float) -> dict[str, dict[str, Any]]:
    manifest = {row["sample_id"]: row for row in read_csv(dataset_dir / "manifest.csv")}
    rows: dict[str, dict[str, Any]] = {}
    for sample_id, sample in manifest.items():
        dst_path = candidate_root / sample_id / "prediction.dst"
        eval_path = candidate_root / sample_id / "eval_executability.json"
        if not dst_path.exists() or not eval_path.exists():
            continue
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pred = extract_pred(payload)
        score = score_metrics(pred)
        adaptive = coverage_metrics(
            dst_path,
            dataset_dir / sample["mask_path"],
            target_width_mm=target_width_mm,
            line_radius_px=2,
            adaptive_precision_radius_px=2,
        )
        rows[sample_id] = {
            "sample_id": sample_id,
            "source_name": sample.get("source_name", ""),
            "category": sample.get("category", ""),
            "mode": candidate_root.name,
            "quality_level": score.get("quality_level", ""),
            "unified_loss": score.get("unified_loss", ""),
            "exec_score": score.get("exec_score", ""),
            "visual_risk": score.get("visual_risk", ""),
            "jump_count": pred.get("jump_count", ""),
            "trim_count": pred.get("trim_count", ""),
            "jump_path_mm": pred.get("jump_path_mm", ""),
            "off_mask_stitch_length_mm": pred.get("off_mask_stitch_length_mm", ""),
            "visible_connector_count": pred.get("visible_connector_count", ""),
            "visible_connector_length_mm": pred.get("visible_connector_length_mm", ""),
            "candidate_root": str(candidate_root),
            "candidate_name": candidate_root.name,
            **adaptive,
        }
    return rows


def shape_profile_is_worth_it(candidate: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if str(candidate.get("quality_level", "")).lower() == "hard_fail":
        return False, "candidate_hard_fail"
    off_mask_reduction = -metric_delta(candidate, baseline, "off_mask_stitch_length_mm")
    adaptive_precision_gain = metric_delta(candidate, baseline, "adaptive_stitch_precision_ratio")
    strict_precision_gain = metric_delta(candidate, baseline, "stitch_precision_ratio")
    if off_mask_reduction < args.min_off_mask_reduction:
        return False, "off_mask_reduction_too_small"
    if adaptive_precision_gain < args.min_adaptive_precision_gain:
        return False, "adaptive_precision_gain_too_small"
    if strict_precision_gain < args.min_strict_precision_gain:
        return False, "strict_precision_gain_too_small"
    if safe_float(candidate.get("coverage_ratio")) < args.min_coverage:
        return False, "coverage_too_low"
    if safe_float(candidate.get("adaptive_coverage_ratio")) < args.min_adaptive_coverage:
        return False, "adaptive_coverage_too_low"
    if metric_delta(candidate, baseline, "visible_connector_count") > args.max_visible_increase:
        return False, "visible_connector_increase"
    if metric_delta(candidate, baseline, "unified_loss") > args.max_loss_increase:
        return False, "loss_increase_too_large"
    if metric_delta(candidate, baseline, "jump_count") > args.max_jump_increase:
        return False, "jump_increase_too_large"
    if metric_delta(candidate, baseline, "trim_count") > args.max_trim_increase:
        return False, "trim_increase_too_large"
    return True, "shape_offmask_precision_gain"


def shape_profile_score(candidate: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace) -> float:
    off_mask_reduction = -metric_delta(candidate, baseline, "off_mask_stitch_length_mm")
    adaptive_precision_gain = metric_delta(candidate, baseline, "adaptive_stitch_precision_ratio")
    strict_precision_gain = metric_delta(candidate, baseline, "stitch_precision_ratio")
    coverage_deficit = max(0.0, args.target_coverage - safe_float(candidate.get("coverage_ratio")))
    adaptive_coverage_deficit = max(0.0, args.target_adaptive_coverage - safe_float(candidate.get("adaptive_coverage_ratio")))
    return (
        safe_float(candidate.get("unified_loss"))
        + args.jump_weight * safe_float(candidate.get("jump_count"))
        + args.trim_weight * safe_float(candidate.get("trim_count"))
        + args.coverage_deficit_weight * coverage_deficit
        + args.adaptive_coverage_deficit_weight * adaptive_coverage_deficit
        - args.off_mask_reduction_weight * off_mask_reduction
        - args.adaptive_precision_gain_weight * adaptive_precision_gain
        - args.strict_precision_gain_weight * strict_precision_gain
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an Openclipart/shape-aware fill profile on top of an existing M2 profile system.")
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--shape-candidate", action="append", required=True, metavar="OUTPUT_DIR")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shape-source-name", default="Openclipart")
    parser.add_argument("--target-width-mm", type=float, default=90.0)
    parser.add_argument("--min-off-mask-reduction", type=float, default=1.0)
    parser.add_argument("--min-adaptive-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-strict-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-coverage", type=float, default=0.96)
    parser.add_argument("--min-adaptive-coverage", type=float, default=0.90)
    parser.add_argument("--max-loss-increase", type=float, default=0.05)
    parser.add_argument("--max-jump-increase", type=float, default=6.0)
    parser.add_argument("--max-trim-increase", type=float, default=2.0)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--target-coverage", type=float, default=0.98)
    parser.add_argument("--target-adaptive-coverage", type=float, default=0.94)
    parser.add_argument("--jump-weight", type=float, default=0.004)
    parser.add_argument("--trim-weight", type=float, default=0.002)
    parser.add_argument("--coverage-deficit-weight", type=float, default=0.08)
    parser.add_argument("--adaptive-coverage-deficit-weight", type=float, default=0.10)
    parser.add_argument("--off-mask-reduction-weight", type=float, default=0.03)
    parser.add_argument("--adaptive-precision-gain-weight", type=float, default=0.06)
    parser.add_argument("--strict-precision-gain-weight", type=float, default=0.04)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    base_root = Path(args.base_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_csv(Path(args.base_selection))
    candidate_sets: list[tuple[Path, dict[str, dict[str, Any]]]] = []
    for item in args.shape_candidate:
        root = Path(item)
        candidate_sets.append((root, load_shape_candidate_rows(dataset_dir, root, args.target_width_mm)))

    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for base in base_rows:
        sample_id = base["sample_id"]
        profile = str(base.get("active_profile", "safe_low_cost") or "safe_low_cost")
        reason = str(base.get("profile_reason", "base_profile"))
        chosen: dict[str, Any] = dict(base)
        source_root = base_root
        candidate_pool: list[tuple[float, Path, dict[str, Any], str]] = []

        if base.get("source_name") == args.shape_source_name:
            rejection_reasons: list[str] = []
            for candidate_root, candidate_rows in candidate_sets:
                candidate = candidate_rows.get(sample_id)
                if candidate is None:
                    continue
                allowed, gate_reason = shape_profile_is_worth_it(candidate, base, args)
                if allowed:
                    candidate_pool.append((shape_profile_score(candidate, base, args), candidate_root, candidate, gate_reason))
                else:
                    rejection_reasons.append(f"{candidate_root.name}:{gate_reason}")
            if candidate_pool:
                _score, candidate_root, candidate, gate_reason = sorted(candidate_pool, key=lambda item: item[0])[0]
                profile = "shape_fill_precision"
                reason = gate_reason
                chosen = dict(base)
                chosen.update(candidate)
                chosen["chosen_candidate"] = candidate_root.name
                source_root = candidate_root
            else:
                reason = f"{reason}|shape_rejected:{';'.join(rejection_reasons) or 'no_candidate'}"

        copy_outputs(source_root, sample_id, output_dir)
        chosen["active_profile"] = profile
        chosen["profile_reason"] = reason
        chosen["profile_switched"] = profile != str(base.get("active_profile", "safe_low_cost") or "safe_low_cost")
        for key in METRIC_KEYS:
            chosen[f"delta_vs_base_{key}"] = metric_delta(chosen, base, key)
        selected.append(chosen)
        decisions.append(
            {
                "sample_id": sample_id,
                "source_name": base.get("source_name", ""),
                "category": base.get("category", ""),
                "base_profile": base.get("active_profile", ""),
                "active_profile": profile,
                "profile_reason": reason,
                "profile_switched": chosen["profile_switched"],
                "base_candidate": base.get("chosen_candidate", ""),
                "chosen_candidate": chosen.get("chosen_candidate", ""),
                "off_mask_reduction": -metric_delta(chosen, base, "off_mask_stitch_length_mm"),
                "adaptive_precision_gain": metric_delta(chosen, base, "adaptive_stitch_precision_ratio"),
                "strict_precision_gain": metric_delta(chosen, base, "stitch_precision_ratio"),
                "adaptive_coverage_delta": metric_delta(chosen, base, "adaptive_coverage_ratio"),
                "coverage_delta": metric_delta(chosen, base, "coverage_ratio"),
                "loss_delta": metric_delta(chosen, base, "unified_loss"),
                "jump_delta": metric_delta(chosen, base, "jump_count"),
                "trim_delta": metric_delta(chosen, base, "trim_count"),
                "visible_delta": metric_delta(chosen, base, "visible_connector_count"),
            }
        )

    switched = [row for row in decisions if str(row.get("profile_switched", "")).lower() == "true"]
    by_profile = summarize(selected, "active_profile")
    by_source = summarize(selected, "source_name")
    by_category = summarize(selected, "category")
    summary = {
        "model_id": "m2_68_shape_aware_profile",
        "samples": len(selected),
        "base_profile_system": "m2_67_line_art_profile",
        "shape_candidate_roots": [root.name for root, _rows in candidate_sets],
        "profile_switches": len(switched),
        "hard_fail": sum(1 for row in selected if str(row.get("quality_level", "")).lower() == "hard_fail"),
        "mean_unified_loss": avg(selected, "unified_loss"),
        "mean_jump_count": avg(selected, "jump_count"),
        "mean_trim_count": avg(selected, "trim_count"),
        "mean_off_mask_stitch_length_mm": avg(selected, "off_mask_stitch_length_mm"),
        "mean_visible_connector_count": avg(selected, "visible_connector_count"),
        "mean_coverage_ratio": avg(selected, "coverage_ratio"),
        "mean_stitch_precision_ratio": avg(selected, "stitch_precision_ratio"),
        "mean_adaptive_coverage_ratio": avg(selected, "adaptive_coverage_ratio"),
        "mean_adaptive_stitch_precision_ratio": avg(selected, "adaptive_stitch_precision_ratio"),
        "mean_overfill_outside_adaptive_ratio": avg(selected, "overfill_outside_adaptive_ratio"),
        "by_profile": by_profile,
        "by_source": by_source,
        "switched_samples": switched,
        "args": vars(args),
    }

    write_csv(selected, output_dir / "shape_profile_selected_rows.csv")
    write_csv(decisions, output_dir / "shape_profile_decision_rows.csv")
    write_csv(switched, output_dir / "shape_profile_switched_rows.csv")
    write_csv(by_profile, output_dir / "shape_profile_by_profile.csv")
    write_csv(by_source, output_dir / "shape_profile_by_source.csv")
    write_csv(by_category, output_dir / "shape_profile_by_category.csv")
    write_json(summary, output_dir / "shape_profile_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
