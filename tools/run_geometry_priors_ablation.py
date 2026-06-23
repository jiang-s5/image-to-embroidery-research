from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from score_unified import score_metrics


VARIANTS = {
    "B0_m2_1_globalrepair20": {
        "description": "M2.1 globalrepair20 baseline, no geometry priors.",
        "geometry": False,
        "dt_weight": 0.0,
        "sobel_weight": 0.0,
        "canny_weight": 0.0,
        "repair_veto": [],
    },
    "B1_edt_repair_veto": {
        "description": "M2.1 plus EDT safety-margin veto only for safe-connect repair.",
        "geometry": True,
        "dt_weight": 0.0,
        "sobel_weight": 0.0,
        "canny_weight": 0.0,
        "repair_veto": ["dt"],
    },
    "B2_canny_repair_veto": {
        "description": "M2.1 plus Canny visual-edge veto only for safe-connect repair.",
        "geometry": True,
        "dt_weight": 0.0,
        "sobel_weight": 0.0,
        "canny_weight": 0.0,
        "repair_veto": ["canny"],
    },
    "B3_edt_canny_repair_veto": {
        "description": "M2.1 plus EDT and Canny vetoes only for safe-connect repair.",
        "geometry": True,
        "dt_weight": 0.0,
        "sobel_weight": 0.0,
        "canny_weight": 0.0,
        "repair_veto": ["dt", "canny"],
    },
    "B4_soft_geometry_rerank": {
        "description": "M2.1 plus weak EDT/Sobel/Canny Graph-TSP cost terms, without repair veto.",
        "geometry": True,
        "dt_weight": 0.03,
        "sobel_weight": 0.02,
        "canny_weight": 0.03,
        "repair_veto": [],
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


def benchmark_samples(input_dir: Path, limit: int) -> list[dict[str, Any]]:
    manifest = input_dir / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        samples: list[dict[str, Any]] = []
        for row in rows:
            image_field = row.get("image_png_path") or row.get("image_path")
            if not image_field:
                continue
            image_path = input_dir / str(image_field)
            if not image_path.exists():
                continue
            mask_field = row.get("mask_path") or ""
            mask_path = input_dir / str(mask_field) if mask_field else None
            samples.append(
                {
                    "sample_id": row.get("sample_id") or image_path.stem,
                    "image_path": image_path,
                    "mask_path": mask_path if mask_path and mask_path.exists() else None,
                    "source_name": row.get("source_name") or row.get("source") or "",
                    "category": row.get("category") or "",
                }
            )
        return samples[:limit] if limit else samples
    suffixes = {".png", ".jpg", ".jpeg"}
    files = sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)
    samples = [{"sample_id": path.stem, "image_path": path, "mask_path": None, "source_name": "", "category": ""} for path in files]
    return samples[:limit] if limit else samples


def safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def flatten_row(sample_id: str, variant: str, summary: dict[str, Any], eval_report: dict[str, Any]) -> dict[str, Any]:
    pred = eval_report.get("pred", {})
    if not isinstance(pred, dict):
        pred = {}
    score = score_metrics(pred)
    dst_summary = summary.get("dst_summary", {})
    graph_stats = dst_summary.get("graph_tsp_stats", {}) if isinstance(dst_summary, dict) else {}
    return {
        "sample_id": sample_id,
        "variant": variant,
        "unified_loss": score["unified_loss"],
        "exec_score": score["exec_score"],
        "visual_risk": score["visual_risk"],
        "quality_level": score["quality_level"],
        "jump_count": pred.get("jump_count", ""),
        "trim_count": pred.get("trim_count", ""),
        "jump_path_mm": pred.get("jump_path_mm", ""),
        "off_mask_stitch_length_mm": pred.get("off_mask_stitch_length_mm", ""),
        "visible_connector_count": pred.get("visible_connector_count", ""),
        "visible_connector_length_mm": pred.get("visible_connector_length_mm", ""),
        "stitch_count": pred.get("stitch_count", ""),
        "safe_connect_repairs": dst_summary.get("safe_connect_repairs", "") if isinstance(dst_summary, dict) else "",
        "geometry_edges_sampled": graph_stats.get("geometry_edges_sampled", ""),
        "geometry_visible_risk_edges": graph_stats.get("geometry_visible_risk_edges", ""),
        "geometry_mean_dt_q05_px": graph_stats.get("geometry_mean_dt_q05_px", ""),
        "geometry_mean_sobel_cross": graph_stats.get("geometry_mean_sobel_cross", ""),
        "geometry_mean_canny_cross": graph_stats.get("geometry_mean_canny_cross", ""),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summary: dict[str, Any] = {"variants": {}}
    for variant, group in by_variant.items():
        metrics: dict[str, Any] = {"samples": len(group)}
        for key in (
            "unified_loss",
            "exec_score",
            "visual_risk",
            "jump_count",
            "trim_count",
            "off_mask_stitch_length_mm",
            "visible_connector_count",
            "visible_connector_length_mm",
            "geometry_visible_risk_edges",
        ):
            values = [safe_float(row.get(key)) for row in group if row.get(key) != ""]
            metrics[f"mean_{key}"] = round(mean(values), 6) if values else 0.0
        metrics["hard_fail"] = sum(1 for row in group if row.get("quality_level") == "hard_fail")
        summary["variants"][variant] = metrics
    return summary


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B0-B3 geometry-prior planner ablations.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--m2-edge-policy", default="checkpoints/m2_edge_policy_m2_1_globalrepair20.json")
    parser.add_argument("--output-dir", default="results/geometry_priors_ablation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-components", type=int, default=40)
    parser.add_argument("--safe-connect-repair-max-mm", type=float, default=20.0)
    parser.add_argument("--m2-edge-policy-top-k", type=int, default=4)
    parser.add_argument("--m2-jump-aware-weight", type=float, default=0.20)
    parser.add_argument("--m2-visible-weight", type=float, default=0.30)
    parser.add_argument("--m2-offmask-weight", type=float, default=0.25)
    args = parser.parse_args()

    repo = Path.cwd()
    output_dir = Path(args.output_dir)
    samples = benchmark_samples(Path(args.input_dir), args.limit)
    if not samples:
        raise SystemExit(f"No image files found under {args.input_dir}")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        image_path = Path(sample["image_path"])
        sample_id = str(sample["sample_id"])
        for variant, spec in VARIANTS.items():
            run_dir = output_dir / variant / sample_id
            run_dir.mkdir(parents=True, exist_ok=True)
            infer_cmd = [
                args.python,
                "infer_model3_portrait_hybrid.py",
                str(image_path),
                "--checkpoint",
                args.checkpoint,
                "--output-dir",
                str(run_dir),
                "--output-prefix",
                "prediction",
                "--geometry-planner",
                "--model-path-order",
                "--graph-tsp-planner",
                "--m2-edge-policy",
                args.m2_edge_policy,
                "--m2-edge-policy-top-k",
                str(args.m2_edge_policy_top_k),
                "--m2-hard-safe-filter",
                "--m2-jump-aware-weight",
                str(args.m2_jump_aware_weight),
                "--m2-visible-weight",
                str(args.m2_visible_weight),
                "--m2-offmask-weight",
                str(args.m2_offmask_weight),
                "--safe-connect-repair",
                "--safe-connect-repair-max-mm",
                str(args.safe_connect_repair_max_mm),
                "--safe-connect-repair-global-mask",
                "--max-components",
                str(args.max_components),
            ]
            if sample.get("mask_path"):
                infer_cmd.extend(["--external-foreground-mask", str(sample["mask_path"])])
            if args.cpu:
                infer_cmd.append("--cpu")
            if bool(spec["geometry"]):
                infer_cmd.extend(
                    [
                        "--geometry-priors",
                        "--m2-dt-weight",
                        str(spec["dt_weight"]),
                        "--m2-sobel-weight",
                        str(spec["sobel_weight"]),
                        "--m2-canny-weight",
                        str(spec["canny_weight"]),
                    ]
                )
                repair_veto = set(spec.get("repair_veto", []))
                if "dt" in repair_veto:
                    infer_cmd.append("--gp-repair-veto-dt")
                if "sobel" in repair_veto:
                    infer_cmd.append("--gp-repair-veto-sobel")
                if "canny" in repair_veto:
                    infer_cmd.append("--gp-repair-veto-canny")

            summary_path = run_dir / "summary.json"
            mask_path = run_dir / "hybrid_export_mask.png"
            eval_path = run_dir / "eval_executability.json"
            if not summary_path.exists():
                run_command(infer_cmd, repo)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            dst_path = Path(summary["dst_summary"]["dst"])
            eval_cmd = [
                args.python,
                "tools/eval_executability.py",
                "--pred",
                str(dst_path),
                "--mask",
                str(mask_path),
                "--report",
                str(eval_path),
            ]
            if not eval_path.exists():
                run_command(eval_cmd, repo)
            eval_report = json.loads(eval_path.read_text(encoding="utf-8"))
            rows.append(flatten_row(sample_id, variant, summary, eval_report))

    write_csv(rows, output_dir / "geometry_priors_ablation_rows.csv")
    summary = summarize_rows(rows)
    (output_dir / "geometry_priors_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
