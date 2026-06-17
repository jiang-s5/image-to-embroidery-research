from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PARETO_KEYS = [
    "unified_loss",
    "jump_count",
    "trim_count",
    "off_mask_stitch_length_mm",
    "visible_connector_count",
    "jump_path_mm",
]


def safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    return groups


def summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"method": name, "samples": len(rows)}
    for key in PARETO_KEYS + ["exec_score", "visual_risk", "stitch_count", "stitch_path_mm"]:
        values = [safe_float(row.get(key)) for row in rows if str(row.get(key, "")) != ""]
        summary[key] = round(mean(values), 8) if values else 0.0
    summary["hard_fail"] = sum(1 for row in rows if row.get("quality_level") == "hard_fail")
    summary["warning"] = sum(1 for row in rows if row.get("quality_level") == "warning")
    summary["pass"] = sum(1 for row in rows if row.get("quality_level") == "pass")
    summary["excellent"] = sum(1 for row in rows if row.get("quality_level") == "excellent")
    return summary


def dominates(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> bool:
    a_values = [safe_float(a.get(key)) for key in keys]
    b_values = [safe_float(b.get(key)) for key in keys]
    return all(av <= bv for av, bv in zip(a_values, b_values)) and any(av < bv for av, bv in zip(a_values, b_values))


def pareto_front(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    front = []
    for candidate in rows:
        if not any(dominates(other, candidate, keys) for other in rows if other is not candidate):
            front.append(candidate)
    return sorted(front, key=lambda row: safe_float(row.get("unified_loss")))


def load_sweep_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def planner_args_for_method(config: dict[str, Any], method: str) -> dict[str, Any]:
    for preset in config.get("presets", []):
        if isinstance(preset, dict) and preset.get("name") == method:
            args = preset.get("args", {})
            return args if isinstance(args, dict) else {}
    return {}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary_rows: list[dict[str, Any]], front: list[dict[str, Any]], recommended: dict[str, Any], path: Path) -> None:
    lines = [
        "# B1 Clean Selection",
        "",
        "Lower is better. The recommendation is selected from the Pareto front using the lowest mean unified loss.",
        "",
        "## Recommendation",
        "",
        f"- recommended method: `{recommended.get('method', '')}`",
        f"- mean unified loss: `{safe_float(recommended.get('unified_loss')):.6f}`",
        f"- mean visual risk: `{safe_float(recommended.get('visual_risk')):.6f}`",
        "",
        "## Pareto Front",
        "",
        "| Method | Samples | Unified Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Jump Path mm | Hard Fail | Warning |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in front:
        lines.append(
            "| {method} | {samples} | {loss:.6f} | {jump:.3f} | {trim:.3f} | {off:.3f} | {visible:.3f} | {jump_path:.3f} | {hard_fail} | {warning} |".format(
                method=row.get("method", ""),
                samples=int(row.get("samples", 0)),
                loss=safe_float(row.get("unified_loss")),
                jump=safe_float(row.get("jump_count")),
                trim=safe_float(row.get("trim_count")),
                off=safe_float(row.get("off_mask_stitch_length_mm")),
                visible=safe_float(row.get("visible_connector_count")),
                jump_path=safe_float(row.get("jump_path_mm")),
                hard_fail=int(row.get("hard_fail", 0)),
                warning=int(row.get("warning", 0)),
            )
        )
    lines += [
        "",
        "## All Methods",
        "",
        "| Method | Unified Loss | Exec Score | Visual Risk | Hard Fail | Warning | Pass | Excellent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: safe_float(item.get("unified_loss"))):
        lines.append(
            "| {method} | {loss:.6f} | {exec_score:.6f} | {visual:.6f} | {hard_fail} | {warning} | {passed} | {excellent} |".format(
                method=row.get("method", ""),
                loss=safe_float(row.get("unified_loss")),
                exec_score=safe_float(row.get("exec_score")),
                visual=safe_float(row.get("visual_risk")),
                hard_fail=int(row.get("hard_fail", 0)),
                warning=int(row.get("warning", 0)),
                passed=int(row.get("pass", 0)),
                excellent=int(row.get("excellent", 0)),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select B1_clean from scored sweep results.")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--group-key", default="method")
    parser.add_argument("--sweep-config", default="")
    parser.add_argument("--output-dir", default="results/pareto/b1_clean")
    args = parser.parse_args()

    rows = read_csv(Path(args.scores_csv))
    summaries = [summarize_group(name, group) for name, group in group_rows(rows, args.group_key).items()]
    front = pareto_front(summaries, PARETO_KEYS)
    recommended = front[0] if front else (sorted(summaries, key=lambda row: safe_float(row.get("unified_loss")))[0] if summaries else {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summaries, output_dir / "method_summary.csv")
    write_csv(front, output_dir / "pareto_front.csv")
    write_markdown(summaries, front, recommended, output_dir / "b1_clean_selection.md")

    sweep_config = load_sweep_config(Path(args.sweep_config)) if args.sweep_config else {}
    b1_clean = {
        "name": "b1_clean",
        "source_method": recommended.get("method", ""),
        "selection_metric": "unified_loss",
        "mean_unified_loss": safe_float(recommended.get("unified_loss")),
        "planner_args": planner_args_for_method(sweep_config, str(recommended.get("method", ""))),
    }
    (output_dir / "b1_clean.json").write_text(json.dumps(b1_clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recommended": recommended, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
