from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from score_unified import score_metrics


DEFAULT_BASE_ARGS = {
    "--geometry-planner": True,
    "--model-path-order": True,
    "--use-continuity-planner": True,
    "--serpentine-fill": True,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} must be JSON-compatible YAML for this dependency-free sweep tool: {exc}") from exc


def append_args(command: list[str], args: dict[str, Any]) -> None:
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                command.append(key)
        elif value is not None and str(value) != "":
            command.extend([key, str(value)])


def run_command(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def discover_pair_cases(pair_root: Path, target_root: Path | None) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for pair_dir in sorted(path for path in pair_root.iterdir() if path.is_dir() and path.name.startswith("pair_")):
        input_path = pair_dir / "preprocess_v2" / "selected_design.png"
        mask_path = pair_dir / "preprocess_v2" / "selected_mask.png"
        if not input_path.exists() or not mask_path.exists():
            input_path = pair_dir / "preprocess_v2" / "thread_design.png"
            mask_path = pair_dir / "preprocess_v2" / "thread_mask.png"
        target_path = ""
        if target_root:
            candidate = target_root / pair_dir.name / "target_normalized.dst"
            if candidate.exists():
                target_path = str(candidate)
        if input_path.exists() and mask_path.exists():
            cases.append(
                {
                    "sample_id": pair_dir.name,
                    "input": str(input_path),
                    "mask": str(mask_path),
                    "target": target_path,
                }
            )
    return cases


def read_cases_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = []
    for row in rows:
        cases.append(
            {
                "sample_id": str(row.get("sample_id") or row.get("pair") or row.get("pair_id") or Path(str(row.get("input", ""))).stem),
                "input": str(row.get("input", "")),
                "mask": str(row.get("mask", "")),
                "target": str(row.get("target", "")),
            }
        )
    return cases


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


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    lines = [
        "# B1 Sweep Report",
        "",
        "Lower unified loss is better. The paired cases are evaluation cases only; this script does not add them to training.",
        "",
        "| Method | Samples | Mean Unified Loss | Mean Exec | Mean Visual Risk | Mean Jumps | Mean Trims | Mean Off-Mask mm | Mean Visible Connectors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, group in sorted(by_method.items(), key=lambda item: sum(float(row["unified_loss"]) for row in item[1]) / max(1, len(item[1]))):
        n = len(group)
        avg = lambda key: sum(float(row.get(key, 0.0)) for row in group) / max(1, n)
        lines.append(
            "| {method} | {n} | {loss:.6f} | {exec_score:.6f} | {visual:.6f} | {jump:.3f} | {trim:.3f} | {off:.3f} | {visible:.3f} |".format(
                method=method,
                n=n,
                loss=avg("unified_loss"),
                exec_score=avg("exec_score"),
                visual=avg("visual_risk"),
                jump=avg("jump_count"),
                trim=avg("trim_count"),
                off=avg("off_mask_stitch_length_mm"),
                visible=avg("visible_connector_count"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B1/B1_clean planner parameter sweep on image/mask cases.")
    parser.add_argument("--cases-csv", default="")
    parser.add_argument("--pair-root", default="")
    parser.add_argument("--target-root", default="")
    parser.add_argument("--config", default="configs/sweep_b1.yaml")
    parser.add_argument("--output-dir", default="results/b1_sweep")
    parser.add_argument("--checkpoint", default="checkpoints/best_model13_multiformat_all_vector_continuity.pt")
    parser.add_argument("--planner-config", default="configs/relation_planner.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    config = read_config(Path(args.config))
    presets = [preset for preset in config.get("presets", []) if isinstance(preset, dict)]
    if not presets:
        raise SystemExit(f"No presets found in {args.config}")

    if args.cases_csv:
        cases = read_cases_csv(Path(args.cases_csv))
    elif args.pair_root:
        target_root = Path(args.target_root) if args.target_root else None
        cases = discover_pair_cases(Path(args.pair_root), target_root)
    else:
        raise SystemExit("Provide --cases-csv or --pair-root.")
    cases = cases[: args.limit] if args.limit else cases
    if not cases:
        raise SystemExit("No cases found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    thresholds = config.get("thresholds", {}) if isinstance(config.get("thresholds", {}), dict) else {}
    weights = config.get("weights", {}) if isinstance(config.get("weights", {}), dict) else {}

    for case in cases:
        for preset in presets:
            name = str(preset["name"])
            case_out = output_dir / name / str(case["sample_id"])
            eval_json = case_out / "executability_eval.json"
            if not args.reuse or not eval_json.exists():
                infer_command = [
                    sys.executable,
                    "infer_model3_portrait_hybrid.py",
                    case["input"],
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(case_out),
                    "--planner-config",
                    args.planner_config,
                    "--external-foreground-mask",
                    case["mask"],
                ]
                if args.cpu:
                    infer_command.append("--cpu")
                append_args(infer_command, DEFAULT_BASE_ARGS)
                append_args(infer_command, preset.get("args", {}) if isinstance(preset.get("args", {}), dict) else {})
                run_command(infer_command, root)

                pred = case_out / "embroidery_output.dst"
                eval_command = [
                    sys.executable,
                    "tools/eval_executability.py",
                    "--pred",
                    str(pred),
                    "--mask",
                    str(case_out / "hybrid_export_mask.png"),
                    "--report",
                    str(eval_json),
                    "--target-width-mm",
                    "90",
                    "--max-stitch-mm",
                    "4",
                    "--high-risk-jump-mm",
                    "8",
                ]
                if case.get("target"):
                    eval_command.extend(["--gt", case["target"]])
                run_command(eval_command, root)
            payload = read_json(eval_json)
            pred_metrics = payload.get("pred", {})
            if not isinstance(pred_metrics, dict):
                pred_metrics = {}
            score = score_metrics(
                pred_metrics,
                thresholds={k: float(v) for k, v in thresholds.items()},
                weights={k: float(v) for k, v in weights.items()},
            )
            row = {
                "sample_id": case["sample_id"],
                "method": name,
                "eval_json": str(eval_json),
                "input": case["input"],
                "mask": case["mask"],
                "target": case.get("target", ""),
                **{key: pred_metrics.get(key, "") for key in sorted(set(score["score_thresholds"]) | {"round_trip_parse_success", "stitch_count", "stitch_path_mm"})},
                "unified_loss": score["unified_loss"],
                "exec_score": score["exec_score"],
                "visual_risk": score["visual_risk"],
                "quality_level": score["quality_level"],
            }
            rows.append(row)

    write_csv(rows, output_dir / "sweep_results.csv")
    write_summary(rows, output_dir / "sweep_report.md")
    print(json.dumps({"cases": len(cases), "presets": len(presets), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
