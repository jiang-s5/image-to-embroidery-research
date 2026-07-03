from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


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


def short_candidate_name(candidate: str, prefix: str) -> str:
    if candidate.startswith(prefix):
        return candidate[len(prefix) :]
    return candidate


def copy_outputs(source_dir: Path, sample_id: str, sample_out: Path) -> None:
    sample_out.mkdir(parents=True, exist_ok=True)
    for filename in ("prediction.dst", "eval_executability.json", "generator_report.json"):
        source = source_dir / sample_id / filename
        if source.exists():
            shutil.copy2(source, sample_out / filename)


def copy_baseline_outputs(
    baseline_output: Path,
    candidate_roots: dict[str, Path],
    chosen_candidate: str,
    sample_id: str,
    sample_out: Path,
) -> None:
    copy_outputs(baseline_output, sample_id, sample_out)
    if (sample_out / "prediction.dst").exists():
        return
    fallback = candidate_roots.get(chosen_candidate)
    if fallback is not None:
        copy_outputs(fallback, sample_id, sample_out)


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
                "changed_samples": sum(1 for row in items if str(row.get("changed_by_text_rescue", "")).lower() == "true"),
                "hard_fail": sum(1 for row in items if row.get("quality_level") == "hard_fail"),
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


def rescue_score(candidate: dict[str, str], current: dict[str, Any], args: argparse.Namespace) -> float:
    precision_gain = safe_float(candidate.get("adaptive_stitch_precision_ratio")) - safe_float(
        current.get("adaptive_stitch_precision_ratio")
    )
    adaptive_coverage_deficit = max(0.0, args.target_adaptive_coverage - safe_float(candidate.get("adaptive_coverage_ratio")))
    coverage_deficit = max(0.0, args.target_coverage - safe_float(candidate.get("coverage_ratio")))
    return (
        safe_float(candidate.get("unified_loss"))
        + args.jump_weight * safe_float(candidate.get("jump_count"))
        + args.trim_weight * safe_float(candidate.get("trim_count"))
        + args.adaptive_coverage_deficit_weight * adaptive_coverage_deficit
        + args.coverage_deficit_weight * coverage_deficit
        - args.precision_gain_weight * precision_gain
    )


def passes_text_rescue_gate(candidate: dict[str, str], current: dict[str, Any], args: argparse.Namespace) -> bool:
    precision_gain = safe_float(candidate.get("adaptive_stitch_precision_ratio")) - safe_float(
        current.get("adaptive_stitch_precision_ratio")
    )
    if safe_float(candidate.get("adaptive_stitch_precision_ratio")) + 1e-9 < args.min_adaptive_precision:
        return False
    if precision_gain + 1e-9 < args.min_adaptive_precision_gain:
        return False
    if safe_float(candidate.get("adaptive_coverage_ratio")) + 1e-9 < args.min_adaptive_coverage:
        return False
    if safe_float(candidate.get("coverage_ratio")) + 1e-9 < args.min_coverage:
        return False
    if safe_float(candidate.get("visible_connector_count")) > safe_float(current.get("visible_connector_count")) + args.max_visible_increase + 1e-9:
        return False
    if safe_float(candidate.get("off_mask_stitch_length_mm")) > safe_float(current.get("off_mask_stitch_length_mm")) + args.max_off_mask_increase + 1e-9:
        return False
    if safe_float(candidate.get("unified_loss")) > safe_float(current.get("unified_loss")) + args.max_loss_slack + 1e-9:
        return False
    if safe_float(candidate.get("jump_count")) > safe_float(current.get("jump_count")) + args.max_jump_increase + 1e-9:
        return False
    if safe_float(candidate.get("trim_count")) > safe_float(current.get("trim_count")) + args.max_trim_increase + 1e-9:
        return False
    return True


def enrich_with_adaptive(base_row: dict[str, str], adaptive_row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base_row)
    for key in (
        "adaptive_coverage_ratio",
        "adaptive_stitch_precision_ratio",
        "overfill_outside_adaptive_ratio",
        "adaptive_precision_radius_px",
        "line_radius_px",
    ):
        out[key] = adaptive_row.get(key, "")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a conservative text-only adaptive-precision rescue on top of M2.62.")
    parser.add_argument("--baseline-selection", required=True)
    parser.add_argument("--baseline-output-dir", required=True)
    parser.add_argument("--baseline-adaptive-rows", required=True)
    parser.add_argument("--candidate-adaptive-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-results-root", default="results")
    parser.add_argument("--baseline-candidate-root", action="append", nargs=2, metavar=("NAME", "DIR"), default=[])
    parser.add_argument("--candidate-prefix", default="public_benchmark_v1_ext33_")
    parser.add_argument("--text-source-name", default="Rendered text")
    parser.add_argument("--min-adaptive-precision", type=float, default=0.95)
    parser.add_argument("--min-adaptive-precision-gain", type=float, default=0.02)
    parser.add_argument("--min-adaptive-coverage", type=float, default=0.88)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--max-loss-slack", type=float, default=0.04)
    parser.add_argument("--max-jump-increase", type=float, default=4.0)
    parser.add_argument("--max-trim-increase", type=float, default=2.0)
    parser.add_argument("--max-off-mask-increase", type=float, default=0.25)
    parser.add_argument("--max-visible-increase", type=float, default=0.0)
    parser.add_argument("--target-adaptive-coverage", type=float, default=0.92)
    parser.add_argument("--target-coverage", type=float, default=0.98)
    parser.add_argument("--jump-weight", type=float, default=0.004)
    parser.add_argument("--trim-weight", type=float, default=0.002)
    parser.add_argument("--adaptive-coverage-deficit-weight", type=float, default=0.15)
    parser.add_argument("--coverage-deficit-weight", type=float, default=0.05)
    parser.add_argument("--precision-gain-weight", type=float, default=0.04)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(args.candidate_results_root)
    baseline_candidate_roots = {name: Path(path) for name, path in args.baseline_candidate_root}
    baseline_output = Path(args.baseline_output_dir)
    baseline_rows = read_csv(Path(args.baseline_selection))
    adaptive_rows = {
        row["sample_id"]: row
        for row in read_csv(Path(args.baseline_adaptive_rows))
        if row.get("selection") in ("m2_62", "baseline", "")
    }
    candidate_rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(Path(args.candidate_adaptive_rows)):
        candidate_rows_by_sample[row["sample_id"]].append(row)

    selected: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for base_row in baseline_rows:
        sample_id = base_row["sample_id"]
        current = enrich_with_adaptive(base_row, adaptive_rows[sample_id])
        current["base_candidate"] = base_row.get("chosen_candidate", "")
        current["changed_by_text_rescue"] = False
        current["text_rescue_mode"] = "non_text_source"
        current["text_rescue_candidate_dir"] = ""
        current["text_rescue_score"] = ""
        current["text_rescue_precision_gain"] = ""
        current["text_rescue_loss_delta"] = ""
        current["text_rescue_jump_delta"] = ""
        current["text_rescue_adaptive_coverage_delta"] = ""

        if base_row.get("source_name") != args.text_source_name:
            copy_baseline_outputs(
                baseline_output,
                baseline_candidate_roots,
                base_row.get("chosen_candidate", ""),
                sample_id,
                output_dir / sample_id,
            )
            selected.append(current)
            continue

        rescue_pool: list[tuple[float, dict[str, str]]] = []
        for candidate in candidate_rows_by_sample.get(sample_id, []):
            if not passes_text_rescue_gate(candidate, current, args):
                rejected.append(
                    {
                        "sample_id": sample_id,
                        "source_name": base_row.get("source_name", ""),
                        "category": base_row.get("category", ""),
                        "baseline_candidate": base_row.get("chosen_candidate", ""),
                        "candidate_dir": candidate.get("candidate", ""),
                        "candidate": short_candidate_name(candidate.get("candidate", ""), args.candidate_prefix),
                        "adaptive_precision_gain": round(
                            safe_float(candidate.get("adaptive_stitch_precision_ratio"))
                            - safe_float(current.get("adaptive_stitch_precision_ratio")),
                            8,
                        ),
                        "loss_delta": round(safe_float(candidate.get("unified_loss")) - safe_float(current.get("unified_loss")), 8),
                        "jump_delta": round(safe_float(candidate.get("jump_count")) - safe_float(current.get("jump_count")), 8),
                        "adaptive_coverage_delta": round(
                            safe_float(candidate.get("adaptive_coverage_ratio")) - safe_float(current.get("adaptive_coverage_ratio")),
                            8,
                        ),
                        "candidate_adaptive_precision": candidate.get("adaptive_stitch_precision_ratio", ""),
                        "candidate_adaptive_coverage": candidate.get("adaptive_coverage_ratio", ""),
                    }
                )
                continue
            rescue_pool.append((rescue_score(candidate, current, args), candidate))

        if not rescue_pool:
            current["text_rescue_mode"] = "no_safe_text_rescue"
            copy_baseline_outputs(
                baseline_output,
                baseline_candidate_roots,
                base_row.get("chosen_candidate", ""),
                sample_id,
                output_dir / sample_id,
            )
            selected.append(current)
            continue

        score, rescue = sorted(rescue_pool, key=lambda item: item[0])[0]
        chosen_candidate_dir = rescue["candidate"]
        chosen_short = short_candidate_name(chosen_candidate_dir, args.candidate_prefix)
        current.update(
            {
                "chosen_candidate": chosen_short,
                "changed_by_text_rescue": chosen_short != base_row.get("chosen_candidate", ""),
                "text_rescue_mode": "text_adaptive_precision_rescue",
                "text_rescue_candidate_dir": chosen_candidate_dir,
                "text_rescue_score": round(score, 8),
                "text_rescue_precision_gain": round(
                    safe_float(rescue.get("adaptive_stitch_precision_ratio")) - safe_float(current.get("adaptive_stitch_precision_ratio")),
                    8,
                ),
                "text_rescue_loss_delta": round(safe_float(rescue.get("unified_loss")) - safe_float(current.get("unified_loss")), 8),
                "text_rescue_jump_delta": round(safe_float(rescue.get("jump_count")) - safe_float(current.get("jump_count")), 8),
                "text_rescue_adaptive_coverage_delta": round(
                    safe_float(rescue.get("adaptive_coverage_ratio")) - safe_float(current.get("adaptive_coverage_ratio")),
                    8,
                ),
                "unified_loss": rescue.get("unified_loss", ""),
                "jump_count": rescue.get("jump_count", ""),
                "trim_count": rescue.get("trim_count", ""),
                "off_mask_stitch_length_mm": rescue.get("off_mask_stitch_length_mm", ""),
                "visible_connector_count": rescue.get("visible_connector_count", ""),
                "coverage_ratio": rescue.get("coverage_ratio", ""),
                "stitch_precision_ratio": rescue.get("stitch_precision_ratio", ""),
                "adaptive_coverage_ratio": rescue.get("adaptive_coverage_ratio", ""),
                "adaptive_stitch_precision_ratio": rescue.get("adaptive_stitch_precision_ratio", ""),
                "overfill_outside_adaptive_ratio": rescue.get("overfill_outside_adaptive_ratio", ""),
            }
        )
        copy_outputs(candidate_root / chosen_candidate_dir, sample_id, output_dir / sample_id)
        selected.append(current)
        if current["changed_by_text_rescue"]:
            changed.append(current)

    write_csv(selected, output_dir / "text_rescue_selected_rows.csv")
    write_csv(changed, output_dir / "text_rescue_changed_rows.csv")
    write_csv(rejected, output_dir / "text_rescue_rejected_rows.csv")
    by_source = summarize(selected, "source_name")
    by_category = summarize(selected, "category")
    write_csv(by_source, output_dir / "text_rescue_by_source.csv")
    write_csv(by_category, output_dir / "text_rescue_by_category.csv")
    summary = {
        "samples": len(selected),
        "text_source_name": args.text_source_name,
        "text_samples": sum(1 for row in selected if row.get("source_name") == args.text_source_name),
        "changed_samples": len(changed),
        "hard_fail": sum(1 for row in selected if row.get("quality_level") == "hard_fail"),
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
        "by_source": by_source,
        "changed_rows": [
            {
                "sample_id": row["sample_id"],
                "category": row.get("category", ""),
                "base_candidate": row.get("base_candidate", ""),
                "chosen_candidate": row.get("chosen_candidate", ""),
                "precision_gain": row.get("text_rescue_precision_gain", ""),
                "loss_delta": row.get("text_rescue_loss_delta", ""),
                "jump_delta": row.get("text_rescue_jump_delta", ""),
                "adaptive_coverage_delta": row.get("text_rescue_adaptive_coverage_delta", ""),
            }
            for row in changed
        ],
        "args": vars(args),
    }
    write_json(summary, output_dir / "text_rescue_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
