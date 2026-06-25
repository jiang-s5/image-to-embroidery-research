from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = [
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "coverage_ratio",
    "stitch_precision_ratio",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


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


def parse_name_path(items: list[list[str]]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for name, path in items:
        parsed.append((name, Path(path)))
    names = [name for name, _ in parsed]
    if len(names) != len(set(names)):
        raise ValueError("Selection names must be unique.")
    return parsed


def risk_flags(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    quality = str(row.get("quality_level", "")).strip().lower()
    if quality in {"warning", "fail", "hard_fail"}:
        flags.append(f"quality_{quality}")
    if safe_float(row.get("unified_loss")) > args.max_loss:
        flags.append("high_loss")
    if safe_float(row.get("jump_count")) > args.max_jump:
        flags.append("high_jump")
    if safe_float(row.get("trim_count")) > args.max_trim:
        flags.append("high_trim")
    if safe_float(row.get("off_mask_stitch_length_mm")) > args.max_off_mask:
        flags.append("off_mask")
    if safe_float(row.get("visible_connector_count")) > args.max_visible:
        flags.append("visible_connector")
    if safe_float(row.get("coverage_ratio")) < args.min_coverage:
        flags.append("low_coverage")
    if safe_float(row.get("stitch_precision_ratio")) < args.min_precision:
        flags.append("low_precision")
    return flags


def risk_score(row: dict[str, Any], flags: list[str], args: argparse.Namespace) -> float:
    loss = safe_float(row.get("unified_loss"))
    jump = safe_float(row.get("jump_count"))
    trim = safe_float(row.get("trim_count"))
    off_mask = safe_float(row.get("off_mask_stitch_length_mm"))
    visible = safe_float(row.get("visible_connector_count"))
    coverage = safe_float(row.get("coverage_ratio"))
    precision = safe_float(row.get("stitch_precision_ratio"))
    coverage_deficit = max(0.0, args.min_coverage - coverage)
    precision_deficit = max(0.0, args.min_precision - precision)
    return round(
        loss
        + 0.006 * jump
        + 0.004 * trim
        + 0.030 * off_mask
        + 0.015 * visible
        + 0.500 * coverage_deficit
        + 0.250 * precision_deficit
        + 0.050 * len(flags),
        8,
    )


def annotate_rows(selection_name: str, rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        enriched: dict[str, Any] = {"selection": selection_name}
        enriched.update(row)
        flags = risk_flags(enriched, args)
        enriched["risk_flags"] = "|".join(flags)
        enriched["risk_flag_count"] = len(flags)
        enriched["risk_score"] = risk_score(enriched, flags, args)
        enriched["needs_review"] = bool(flags)
        annotated.append(enriched)
    return annotated


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(item, "")) for item in group_keys)
        groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        worst = max(group, key=lambda row: safe_float(row.get("risk_score")))
        summary: dict[str, Any] = {field: value for field, value in zip(group_keys, key)}
        summary.update(
            {
                "samples": len(group),
                "review_samples": sum(1 for row in group if str(row.get("needs_review", "")).lower() == "true"),
                "review_rate": round(sum(1 for row in group if str(row.get("needs_review", "")).lower() == "true") / len(group), 6),
                "mean_risk_score": round(mean(safe_float(row.get("risk_score")) for row in group), 8),
                "worst_risk_score": safe_float(worst.get("risk_score")),
                "worst_sample_id": worst.get("sample_id", ""),
                "worst_flags": worst.get("risk_flags", ""),
                "mean_unified_loss": round(mean(safe_float(row.get("unified_loss")) for row in group), 8),
                "mean_jump_count": round(mean(safe_float(row.get("jump_count")) for row in group), 8),
                "mean_trim_count": round(mean(safe_float(row.get("trim_count")) for row in group), 8),
                "mean_off_mask_stitch_length_mm": round(mean(safe_float(row.get("off_mask_stitch_length_mm")) for row in group), 8),
                "mean_visible_connector_count": round(mean(safe_float(row.get("visible_connector_count")) for row in group), 8),
                "mean_coverage_ratio": round(mean(safe_float(row.get("coverage_ratio")) for row in group), 8),
                "mean_stitch_precision_ratio": round(mean(safe_float(row.get("stitch_precision_ratio")) for row in group), 8),
            }
        )
        summaries.append(summary)
    return summaries


def flag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        flags = [item for item in str(row.get("risk_flags", "")).split("|") if item]
        for flag in flags:
            counts[flag] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit sample-level and source-level risks in selected M2 policy rows.")
    parser.add_argument("--selection", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-loss", type=float, default=0.10)
    parser.add_argument("--max-jump", type=float, default=10.0)
    parser.add_argument("--max-trim", type=float, default=3.0)
    parser.add_argument("--max-off-mask", type=float, default=1.0)
    parser.add_argument("--max-visible", type=float, default=0.0)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-precision", type=float, default=0.75)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = parse_name_path(args.selection)

    all_rows: list[dict[str, Any]] = []
    source_files: dict[str, str] = {}
    for name, path in selections:
        source_files[name] = str(path)
        all_rows.extend(annotate_rows(name, read_csv(path), args))

    review_rows = sorted(
        [row for row in all_rows if row["needs_review"]],
        key=lambda row: (-safe_float(row.get("risk_score")), row.get("selection", ""), row.get("sample_id", "")),
    )
    all_rows = sorted(all_rows, key=lambda row: (row.get("selection", ""), -safe_float(row.get("risk_score"))))
    by_selection = summarize(all_rows, ["selection"])
    by_source = summarize(all_rows, ["selection", "source_name"])
    by_category = summarize(all_rows, ["selection", "source_name", "category"])

    write_csv(all_rows, output_dir / "sample_risk_rows.csv")
    write_csv(review_rows, output_dir / "review_queue.csv")
    write_csv(by_selection, output_dir / "risk_by_selection.csv")
    write_csv(by_source, output_dir / "risk_by_source.csv")
    write_csv(by_category, output_dir / "risk_by_category.csv")
    write_json(
        {
            "selections": source_files,
            "thresholds": {
                "max_loss": args.max_loss,
                "max_jump": args.max_jump,
                "max_trim": args.max_trim,
                "max_off_mask": args.max_off_mask,
                "max_visible": args.max_visible,
                "min_coverage": args.min_coverage,
                "min_precision": args.min_precision,
            },
            "samples": len(all_rows),
            "review_samples": len(review_rows),
            "flag_counts": flag_counts(all_rows),
            "by_selection": by_selection,
            "top_review_samples": review_rows[:10],
        },
        output_dir / "risk_audit_summary.json",
    )
    print(json.dumps({"samples": len(all_rows), "review_samples": len(review_rows), "flag_counts": flag_counts(all_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

