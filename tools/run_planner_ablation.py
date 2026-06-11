from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean


BASE_METHODS = {
    "A0_continuity": {
        "description": "geometry planner + vector-continuity planner",
        "extra": [],
        "retrieval_mode": "",
    },
    "A1_relation": {
        "description": "A0 + relation-aware transition cost",
        "extra": ["--planner-config"],
        "retrieval_mode": "",
    },
    "A2_retrieval_relation": {
        "description": "A1 + retrieval planner prior",
        "extra": ["--planner-config", "--retrieval-index"],
        "retrieval_mode": "base",
    },
    "A2_fixed_params": {
        "description": "A1 + fixed conservative planner parameters, without retrieval",
        "extra": ["--planner-config", "--fixed-planner-params"],
        "retrieval_mode": "",
    },
}


CONTROL_METHODS = {
    "A2_leave_one_out": {
        "description": "A2 with the current sample removed from the retrieval index",
        "extra": ["--planner-config", "--retrieval-index"],
        "retrieval_mode": "leave_one_out",
    },
    "A2_external_index": {
        "description": "A2 with an external train/dataset retrieval index",
        "extra": ["--planner-config", "--retrieval-index"],
        "retrieval_mode": "external",
    },
    "A2_random_prior": {
        "description": "A2 with random retrieval priors from the base index",
        "extra": ["--planner-config", "--retrieval-index"],
        "retrieval_mode": "random",
    },
    "A2_shuffled_prior": {
        "description": "A2 with nearest-neighbor features but shuffled planner priors",
        "extra": ["--planner-config", "--retrieval-index"],
        "retrieval_mode": "shuffled",
    },
}


METRIC_KEYS = [
    "round_trip_parse_success",
    "stitch_count",
    "jump_count",
    "trim_count",
    "color_change_count",
    "unsupported_command_count",
    "illegal_long_stitch_count",
    "high_risk_jump_count",
    "max_stitch_mm",
    "max_jump_mm",
    "stitch_path_mm",
    "jump_path_mm",
    "lock_tack_like_count",
]


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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def input_files(input_dir: Path, limit: int) -> list[Path]:
    files = sorted(path for path in input_dir.rglob("*.png") if path.is_file())
    return files[:limit] if limit else files


def safe_float(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def item_matches_sample(item: dict[str, object], sample_id: str) -> bool:
    source = str(item.get("source", ""))
    item_id = str(item.get("id", ""))
    if sample_id and sample_id in source:
        return True
    if sample_id and sample_id in item_id:
        return True
    try:
        return Path(source).stem == sample_id
    except OSError:
        return False


def items_from_index(payload: dict[str, object], sample_id: str = "") -> list[dict[str, object]]:
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    typed_items = [item for item in items if isinstance(item, dict)]
    if not sample_id:
        return typed_items
    return [item for item in typed_items if not item_matches_sample(item, sample_id)]


def index_with_items(payload: dict[str, object], items: list[dict[str, object]]) -> dict[str, object]:
    result = deepcopy(payload)
    result["items"] = items
    return result


def shuffled_prior_index(payload: dict[str, object], sample_id: str, rng: random.Random) -> dict[str, object]:
    items = deepcopy(items_from_index(payload, sample_id))
    priors = [item.get("planner_prior") for item in items]
    if len(priors) > 1:
        rng.shuffle(priors)
        for item, prior in zip(items, priors):
            item["planner_prior"] = prior
    return index_with_items(payload, items)


def random_prior_index(payload: dict[str, object], sample_id: str, rng: random.Random, top_k: int) -> dict[str, object]:
    items = deepcopy(items_from_index(payload, sample_id))
    if not items:
        return index_with_items(payload, [])
    count = min(max(1, top_k), len(items))
    selected = rng.sample(items, count)
    return index_with_items(payload, selected)


def retrieval_index_for_method(
    method: str,
    mode: str,
    sample_id: str,
    base_index_path: Path,
    external_index_path: Path | None,
    temp_dir: Path,
    base_payload: dict[str, object],
    rng: random.Random,
    random_top_k: int,
) -> Path:
    if mode == "base":
        return base_index_path
    if mode == "external":
        if external_index_path is None:
            raise SystemExit("--external-retrieval-index is required for A2_external_index.")
        return external_index_path
    if mode == "leave_one_out":
        payload = index_with_items(base_payload, deepcopy(items_from_index(base_payload, sample_id)))
    elif mode == "random":
        payload = random_prior_index(base_payload, sample_id, rng, random_top_k)
    elif mode == "shuffled":
        payload = shuffled_prior_index(base_payload, sample_id, rng)
    else:
        return base_index_path
    path = temp_dir / sample_id / f"{method}.json"
    write_json(path, payload)
    return path


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary: dict[str, object] = {"methods": {}, "deltas_vs_A0": {}}
    for method, method_rows in by_method.items():
        metrics = {}
        for key in METRIC_KEYS:
            values = [safe_float(row.get(key)) for row in method_rows]
            metrics[key] = round(mean(values), 6) if values else 0.0
        metrics["samples"] = len(method_rows)
        summary["methods"][method] = metrics

    baseline = by_method.get("A0_continuity", [])
    baseline_by_sample = {str(row["sample_id"]): row for row in baseline}
    for method, method_rows in by_method.items():
        if method == "A0_continuity":
            continue
        delta_rows = []
        for row in method_rows:
            base = baseline_by_sample.get(str(row["sample_id"]))
            if not base:
                continue
            delta_rows.append(
                {
                    key: safe_float(row.get(key)) - safe_float(base.get(key))
                    for key in METRIC_KEYS
                    if key != "round_trip_parse_success"
                }
            )
        delta_summary = {}
        for key in delta_rows[0].keys() if delta_rows else []:
            values = [item[key] for item in delta_rows]
            delta_summary[key] = round(mean(values), 6)
        summary["deltas_vs_A0"][method] = delta_summary
    return summary


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    methods = summary["methods"]  # type: ignore[index]
    deltas = summary["deltas_vs_A0"]  # type: ignore[index]
    lines = [
        "# Planner Ablation Report",
        "",
        "Lower is better for jump, trim, illegal stitch, and unsupported command metrics.",
        "",
        "| Method | Samples | Parse Success | Stitch Count | Stitch Path mm | Jump Count | Jump Path mm | Max Jump mm | Trim Count | Illegal Long Stitch | Unsupported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in methods.items():  # type: ignore[union-attr]
        lines.append(
            "| {method} | {samples} | {parse:.3f} | {stitch:.3f} | {stitch_path:.3f} | {jump:.3f} | {jump_path:.3f} | {max_jump:.3f} | {trim:.3f} | {illegal:.3f} | {unsupported:.3f} |".format(
                method=method,
                samples=int(metrics["samples"]),
                parse=float(metrics["round_trip_parse_success"]),
                stitch=float(metrics["stitch_count"]),
                stitch_path=float(metrics["stitch_path_mm"]),
                jump=float(metrics["jump_count"]),
                jump_path=float(metrics["jump_path_mm"]),
                max_jump=float(metrics["max_jump_mm"]),
                trim=float(metrics["trim_count"]),
                illegal=float(metrics["illegal_long_stitch_count"]),
                unsupported=float(metrics["unsupported_command_count"]),
            )
        )
    lines += ["", "## Deltas vs A0", ""]
    for method, metrics in deltas.items():  # type: ignore[union-attr]
        if not metrics:
            continue
        lines.append(f"### {method}")
        lines.append("")
        lines.append(f"- jump_count: {metrics.get('jump_count', 0):.3f}")
        lines.append(f"- jump_path_mm: {metrics.get('jump_path_mm', 0):.3f}")
        lines.append(f"- max_jump_mm: {metrics.get('max_jump_mm', 0):.3f}")
        lines.append(f"- trim_count: {metrics.get('trim_count', 0):.3f}")
        lines.append(f"- stitch_count: {metrics.get('stitch_count', 0):.3f}")
        lines.append(f"- stitch_path_mm: {metrics.get('stitch_path_mm', 0):.3f}")
        lines.append(f"- illegal_long_stitch_count: {metrics.get('illegal_long_stitch_count', 0):.3f}")
        lines.append(f"- unsupported_command_count: {metrics.get('unsupported_command_count', 0):.3f}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A0/A1/A2 planner ablation on a folder of demo inputs.")
    parser.add_argument("--input-dir", default="datasets/mini_demo/inputs")
    parser.add_argument("--output-dir", default="outputs/planner_ablation_mini_demo")
    parser.add_argument("--checkpoint", default="checkpoints/best_model13_multiformat_all_vector_continuity.pt")
    parser.add_argument("--retrieval-index", default="datasets/mini_demo/retrieval_planner_index.json")
    parser.add_argument("--external-retrieval-index", default="")
    parser.add_argument("--planner-config", default="configs/relation_planner.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-validation-controls", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260611)
    parser.add_argument("--random-top-k", type=int, default=5)
    parser.add_argument("--use-canonical", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = dict(BASE_METHODS)
    if args.include_validation_controls:
        methods.update(CONTROL_METHODS)
    base_index_path = Path(args.retrieval_index)
    external_index_path = Path(args.external_retrieval_index) if args.external_retrieval_index else None
    base_payload = read_json(base_index_path) if base_index_path.exists() else {"items": []}
    temp_index_dir = output_dir / "_control_indices"
    rng = random.Random(args.random_seed)
    rows: list[dict[str, object]] = []

    for index, image_path in enumerate(input_files(Path(args.input_dir), args.limit), start=1):
        sample_id = image_path.stem
        print(json.dumps({"sample": index, "id": sample_id}, ensure_ascii=False), flush=True)
        preprocess_dir = output_dir / "preprocess" / sample_id
        canonical_path = preprocess_dir / ("canonical_input.png" if args.use_canonical else "input_resized.png")
        if not args.reuse or not canonical_path.exists():
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

        for method, config in methods.items():
            run_dir = output_dir / "runs" / sample_id / method
            eval_path = run_dir / "executability_eval.json"
            if not args.reuse or not eval_path.exists():
                retrieval_index_path: Path | None = None
                retrieval_mode = str(config.get("retrieval_mode", ""))
                command = [
                    sys.executable,
                    "infer_model3_portrait_hybrid.py",
                    str(canonical_path),
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    str(run_dir),
                    "--geometry-planner",
                    "--model-path-order",
                    "--use-continuity-planner",
                    "--serpentine-fill",
                ]
                if args.cpu:
                    command.append("--cpu")
                if "--planner-config" in config["extra"]:
                    command += ["--planner-config", args.planner_config]
                if "--fixed-planner-params" in config["extra"]:
                    command += [
                        "--row-step-px",
                        "2",
                        "--point-step-px",
                        "2",
                        "--min-component-px",
                        "24",
                        "--max-components",
                        "74",
                        "--long-jump-weight",
                        "0.2025",
                        "--two-opt-passes",
                        "1",
                        "--connect-near-mm",
                        "0.9",
                        "--continuity-order-weight",
                        "0.7675",
                        "--continuity-connect-threshold",
                        "0.2665",
                        "--max-stitch-mm",
                        "3.82",
                    ]
                if "--retrieval-index" in config["extra"]:
                    retrieval_index_path = retrieval_index_for_method(
                        method=method,
                        mode=retrieval_mode,
                        sample_id=sample_id,
                        base_index_path=base_index_path,
                        external_index_path=external_index_path,
                        temp_dir=temp_index_dir,
                        base_payload=base_payload,
                        rng=rng,
                        random_top_k=args.random_top_k,
                    )
                    command += ["--retrieval-index", str(retrieval_index_path)]
                run_command(command, root)
                run_command(
                    [
                        sys.executable,
                        "tools/eval_executability.py",
                        "--pred",
                        str(run_dir / "embroidery_output.dst"),
                        "--report",
                        str(eval_path),
                    ],
                    root,
                )

            infer_summary = read_json(run_dir / "summary.json")
            eval_summary = read_json(eval_path)["pred"]  # type: ignore[index]
            row = {
                "sample_id": sample_id,
                "method": method,
                "description": config["description"],
                "input_used": str(canonical_path),
                "hybrid_pixels": infer_summary.get("hybrid_pixels", 0),
                "components": infer_summary.get("dst_summary", {}).get("components", 0) if isinstance(infer_summary.get("dst_summary"), dict) else 0,
                "retrieval_mode": config.get("retrieval_mode", ""),
            }
            for key in METRIC_KEYS:
                row[key] = eval_summary.get(key, "")
            rows.append(row)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["sample_id", "method"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "comparison.md")
    print(json.dumps({"metrics": str(metrics_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
