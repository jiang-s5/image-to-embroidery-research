from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean


METRIC_KEYS = [
    "round_trip_parse_success",
    "stitch_count",
    "jump_count",
    "trim_count",
    "illegal_long_stitch_count",
    "high_risk_jump_count",
    "max_stitch_mm",
    "max_jump_mm",
    "stitch_path_mm",
    "jump_path_mm",
    "lock_tack_like_count",
    "unsupported_command_count",
    "off_mask_stitch_length_mm",
    "off_mask_stitch_count",
    "visible_connector_count",
    "visible_connector_length_mm",
]


PRESETS: dict[str, dict[str, object]] = {
    "a0_like": {
        "description": "Original continuity planner defaults.",
        "relation": False,
        "args": {},
    },
    "a1_relation": {
        "description": "A0 plus relation-aware transition cost, without conservative fixed parameters.",
        "relation": True,
        "args": {},
    },
    "conservative_v1": {
        "description": "Current A2 fixed-prior parameters from anti-overfit controls.",
        "relation": True,
        "args": {
            "--row-step-px": 2,
            "--point-step-px": 2,
            "--min-component-px": 24,
            "--max-components": 74,
            "--long-jump-weight": 0.2025,
            "--two-opt-passes": 1,
            "--connect-near-mm": 0.9,
            "--continuity-order-weight": 0.7675,
            "--continuity-connect-threshold": 0.2665,
            "--max-stitch-mm": 3.82,
        },
    },
    "conservative_v2_more_connect": {
        "description": "Slightly more near-connect stitching with the same short-stitch guard.",
        "relation": True,
        "args": {
            "--row-step-px": 2,
            "--point-step-px": 2,
            "--min-component-px": 22,
            "--max-components": 80,
            "--long-jump-weight": 0.3,
            "--two-opt-passes": 1,
            "--connect-near-mm": 1.2,
            "--continuity-order-weight": 0.8,
            "--continuity-connect-threshold": 0.255,
            "--max-stitch-mm": 3.75,
        },
    },
    "conservative_v3_less_connect": {
        "description": "Less aggressive connector use to reduce possible visible stitch artifacts.",
        "relation": True,
        "args": {
            "--row-step-px": 2,
            "--point-step-px": 2,
            "--min-component-px": 26,
            "--max-components": 68,
            "--long-jump-weight": 0.16,
            "--two-opt-passes": 1,
            "--connect-near-mm": 0.6,
            "--continuity-order-weight": 0.74,
            "--continuity-connect-threshold": 0.275,
            "--max-stitch-mm": 3.9,
        },
    },
}


def run_command(command: list[str], cwd: Path) -> None:
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_has_required_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        pred = read_json(path).get("pred", {})
    except json.JSONDecodeError:
        return False
    if not isinstance(pred, dict):
        return False
    return all(key in pred for key in METRIC_KEYS)


def safe_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def input_files(input_dir: Path, limit: int) -> list[Path]:
    files = sorted(path for path in input_dir.rglob("*.png") if path.is_file())
    return files[:limit] if limit else files


def split_for_index(index: int, train_count: int) -> str:
    return "train" if index <= train_count else "val"


def append_planner_args(command: list[str], args: dict[str, object]) -> None:
    for key, value in args.items():
        command.append(key)
        command.append(str(value).lower() if isinstance(value, bool) else str(value))


def score_row(row: dict[str, object], stitch_path_reference: float | None = None) -> float:
    if safe_float(row.get("round_trip_parse_success")) < 0.5:
        return 1_000_000.0
    stitch_path = safe_float(row.get("stitch_path_mm"))
    reference = stitch_path_reference if stitch_path_reference and stitch_path_reference > 0 else stitch_path
    stitch_path_overage = max(0.0, stitch_path - reference)
    return (
        safe_float(row.get("jump_path_mm"))
        + 10.0 * safe_float(row.get("jump_count"))
        + 18.0 * safe_float(row.get("trim_count"))
        + 120.0 * safe_float(row.get("illegal_long_stitch_count"))
        + 60.0 * safe_float(row.get("high_risk_jump_count"))
        + 4.0 * safe_float(row.get("unsupported_command_count"))
        + 0.25 * stitch_path_overage
        + 1.0 * safe_float(row.get("off_mask_stitch_length_mm"))
        + 20.0 * safe_float(row.get("visible_connector_count"))
    )


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_preset_split: dict[tuple[str, str], list[dict[str, object]]] = {}
    baseline_by_sample = {
        str(row["sample_id"]): row
        for row in rows
        if row.get("preset") == "a0_like"
    }
    scored_rows = []
    for row in rows:
        base = baseline_by_sample.get(str(row["sample_id"]))
        reference = safe_float(base.get("stitch_path_mm")) if base else None
        row = dict(row)
        row["score"] = round(score_row(row, reference), 6)
        scored_rows.append(row)
        by_preset_split.setdefault((str(row["preset"]), str(row["split"])), []).append(row)

    summary: dict[str, object] = {"presets": {}, "rank_train": [], "rank_val": []}
    by_preset: dict[str, dict[str, object]] = {}
    for (preset, split), preset_rows in by_preset_split.items():
        metrics = {"samples": len(preset_rows)}
        for key in METRIC_KEYS + ["score"]:
            values = [safe_float(row.get(key)) for row in preset_rows]
            metrics[key] = round(mean(values), 6) if values else 0.0
        by_preset.setdefault(preset, {})[split] = metrics
    summary["presets"] = by_preset
    for split in ("train", "val"):
        ranking = []
        for preset, splits in by_preset.items():
            split_metrics = splits.get(split)
            if isinstance(split_metrics, dict):
                ranking.append(
                    {
                        "preset": preset,
                        "score": split_metrics.get("score", 0.0),
                        "jump_count": split_metrics.get("jump_count", 0.0),
                        "trim_count": split_metrics.get("trim_count", 0.0),
                        "jump_path_mm": split_metrics.get("jump_path_mm", 0.0),
                        "stitch_path_mm": split_metrics.get("stitch_path_mm", 0.0),
                        "visible_connector_count": split_metrics.get("visible_connector_count", 0.0),
                        "off_mask_stitch_length_mm": split_metrics.get("off_mask_stitch_length_mm", 0.0),
                    }
                )
        summary[f"rank_{split}"] = sorted(ranking, key=lambda item: safe_float(item["score"]))
    return summary


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Fixed Planner Parameter Sweep",
        "",
        "Lower score is better. The score penalizes jump path, jump count, trim count, illegal long stitches, high-risk jumps, unsupported commands, and stitch-path overage relative to `a0_like`.",
        "",
    ]
    for split_key, title in (("rank_train", "Train Ranking"), ("rank_val", "Validation Ranking")):
        lines += [
            f"## {title}",
            "",
            "| Rank | Preset | Score | Jump Count | Trim Count | Jump Path mm | Stitch Path mm | Visible Connectors | Off-Mask Stitch mm |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for index, row in enumerate(summary.get(split_key, []), start=1):  # type: ignore[arg-type]
            lines.append(
                "| {rank} | {preset} | {score:.3f} | {jump:.3f} | {trim:.3f} | {jump_path:.3f} | {stitch_path:.3f} | {visible:.3f} | {off_mask:.3f} |".format(
                    rank=index,
                    preset=row["preset"],
                    score=float(row["score"]),
                    jump=float(row["jump_count"]),
                    trim=float(row["trim_count"]),
                    jump_path=float(row["jump_path_mm"]),
                    stitch_path=float(row["stitch_path_mm"]),
                    visible=float(row.get("visible_connector_count", 0.0)),
                    off_mask=float(row.get("off_mask_stitch_length_mm", 0.0)),
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep fixed planner parameter presets with a train/validation split.")
    parser.add_argument("--input-dir", default="datasets/mini_demo/inputs")
    parser.add_argument("--output-dir", default="outputs/fixed_planner_param_sweep")
    parser.add_argument("--checkpoint", default="checkpoints/best_model13_multiformat_all_vector_continuity.pt")
    parser.add_argument("--planner-config", default="configs/relation_planner.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--preset", action="append", default=[], choices=sorted(PRESETS))
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_presets = args.preset or list(PRESETS)
    rows: list[dict[str, object]] = []

    for index, image_path in enumerate(input_files(Path(args.input_dir), args.limit), start=1):
        sample_id = image_path.stem
        split = split_for_index(index, args.train_count)
        preprocess_dir = output_dir / "preprocess" / sample_id
        input_path = preprocess_dir / "input_resized.png"
        if not args.reuse or not input_path.exists():
            run_command(
                [
                    sys.executable,
                    "tools/preprocess_real_image.py",
                    str(image_path),
                    "--output-dir",
                    str(preprocess_dir),
                    "--colors",
                    "10",
                ],
                root,
            )

        for preset_name in selected_presets:
            preset = PRESETS[preset_name]
            run_dir = output_dir / "runs" / sample_id / preset_name
            eval_path = run_dir / "executability_eval.json"
            needs_infer = not args.reuse or not (run_dir / "summary.json").exists() or not (run_dir / "embroidery_output.dst").exists()
            if needs_infer:
                command = [
                    sys.executable,
                    "infer_model3_portrait_hybrid.py",
                    str(input_path),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(run_dir),
                    "--geometry-planner",
                    "--model-path-order",
                    "--use-continuity-planner",
                    "--serpentine-fill",
                ]
                if bool(preset.get("relation", False)):
                    command += ["--planner-config", args.planner_config]
                if args.cpu:
                    command.append("--cpu")
                append_planner_args(command, preset["args"])  # type: ignore[arg-type]
                run_command(command, root)
            if not args.reuse or not eval_has_required_metrics(eval_path):
                run_command(
                    [
                        sys.executable,
                        "tools/eval_executability.py",
                        "--pred",
                        str(run_dir / "embroidery_output.dst"),
                        "--mask",
                        str(run_dir / "hybrid_export_mask.png"),
                        "--report",
                        str(eval_path),
                    ],
                    root,
                )

            metrics = read_json(eval_path)["pred"]  # type: ignore[index]
            row = {
                "sample_id": sample_id,
                "split": split,
                "preset": preset_name,
                "description": preset["description"],
            }
            for key in METRIC_KEYS:
                row[key] = metrics.get(key, "")
            rows.append(row)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["sample_id", "preset"])
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "comparison.md")
    print(json.dumps({"metrics": str(metrics_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
