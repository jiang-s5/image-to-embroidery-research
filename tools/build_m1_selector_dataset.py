from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HARD_ORDER_KEYS = (
    "parse_fail",
    "visible_connector_count",
    "off_mask_stitch_length_mm",
    "jump_count",
    "trim_count",
    "unified_loss",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_config(path: Path) -> dict[str, Any]:
    payload = safe_read_json(path)
    if not payload:
        raise SystemExit(f"{path} must be JSON-compatible YAML for this dependency-free M1 tool.")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
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


def flatten_numeric(prefix: str, payload: dict[str, Any], out: dict[str, Any], keys: tuple[str, ...] | None = None) -> None:
    items = payload.items() if keys is None else ((key, payload.get(key)) for key in keys)
    for key, value in items:
        if isinstance(value, bool):
            out[f"{prefix}.{key}"] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[f"{prefix}.{key}"] = float(value)


def make_hard_sort_values(row: dict[str, Any]) -> dict[str, float]:
    parse_success = safe_float(row.get("round_trip_parse_success"))
    return {
        "parse_fail": 0.0 if parse_success >= 0.5 else 1.0,
        "visible_connector_count": safe_float(row.get("visible_connector_count")),
        "off_mask_stitch_length_mm": safe_float(row.get("off_mask_stitch_length_mm")),
        "jump_count": safe_float(row.get("jump_count")),
        "trim_count": safe_float(row.get("trim_count")),
        "unified_loss": safe_float(row.get("unified_loss")),
    }


def hard_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    values = make_hard_sort_values(row)
    return tuple(values[key] for key in HARD_ORDER_KEYS)


def hard_score(row: dict[str, Any]) -> float:
    values = make_hard_sort_values(row)
    # This continuous proxy preserves the "visual first" hard policy while
    # giving the selector a trainable scalar target.
    return (
        10.0 * values["parse_fail"]
        + values["visible_connector_count"] / 3.0
        + values["off_mask_stitch_length_mm"] / 8.0
        + values["jump_count"] / 25.0
        + values["trim_count"] / 20.0
        + values["unified_loss"]
    )


def preset_args_by_method(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    presets = config.get("presets", [])
    out: dict[str, dict[str, Any]] = {}
    if isinstance(presets, list):
        for item in presets:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            args = item.get("args", {})
            if name and isinstance(args, dict):
                out[name] = args
    return out


def resolve_summary_path(outputs_root: Path, method: str, sample_id: str) -> Path:
    return outputs_root / method / sample_id / "summary.json"


def load_sample_preprocess(pair_root: Path, sample_id: str) -> dict[str, Any]:
    return safe_read_json(pair_root / sample_id / "preprocess_v2" / "summary.json")


def add_ratio_features(prefix: str, counts: dict[str, Any], out: dict[str, Any]) -> None:
    total = sum(max(0.0, safe_float(value)) for value in counts.values())
    if total <= 0:
        return
    for key, value in counts.items():
        out[f"{prefix}.{key}_ratio"] = max(0.0, safe_float(value)) / total


def add_sample_preset_interactions(features: dict[str, Any]) -> None:
    sample_keys = (
        "pre.selected_foreground_ratio",
        "pre.region_foreground_ratio",
        "pre.thread_foreground_ratio",
        "pre.region_thread_area_ratio",
        "pred.mask_mean",
        "pred.density_mean",
        "pred.stitch_type.1_ratio",
        "pred.stitch_type.2_ratio",
        "pred.stitch_type.3_ratio",
    )
    preset_keys = (
        "preset.graph_tsp_planner",
        "preset.mask_safe_connectors",
        "preset.connect_near_mm",
        "preset.continuity_connect_max_mm",
        "preset.continuity_connect_threshold",
        "preset.max_stitch_mm",
        "preset.row_step_px",
        "preset.point_step_px",
        "preset.min_active_px",
        "preset.max_components",
    )
    for sample_key in sample_keys:
        sample_value = features.get(sample_key)
        if not isinstance(sample_value, (int, float)):
            continue
        for preset_key in preset_keys:
            preset_value = features.get(preset_key)
            if not isinstance(preset_value, (int, float)):
                continue
            clean_sample = sample_key.replace(".", "_")
            clean_preset = preset_key.replace(".", "_")
            features[f"interaction.{clean_sample}__x__{clean_preset}"] = float(sample_value) * float(preset_value)

    selected = safe_float(features.get("pre.selected_foreground_ratio"))
    density = safe_float(features.get("pred.density_mean"))
    mask = safe_float(features.get("pred.mask_mean"))
    satin = safe_float(features.get("pred.stitch_type.2_ratio"))
    fill = safe_float(features.get("pred.stitch_type.3_ratio"))
    running = safe_float(features.get("pred.stitch_type.1_ratio"))
    features["derived.coverage_density"] = selected * density
    features["derived.mask_density"] = mask * density
    features["derived.running_to_satin_ratio"] = running / max(1e-6, satin)
    features["derived.fill_or_satin_ratio"] = fill + satin


def build_candidate_features(
    sample_id: str,
    method: str,
    row: dict[str, Any],
    preset_args: dict[str, Any],
    preprocess: dict[str, Any],
    summary: dict[str, Any],
    include_post_export_features: bool,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "method": method,
        "sample_id": sample_id,
    }

    source_size = preprocess.get("source_size")
    if isinstance(source_size, list) and len(source_size) >= 2:
        width = safe_float(source_size[0])
        height = safe_float(source_size[1])
        features["pre.source_width"] = width
        features["pre.source_height"] = height
        features["pre.source_aspect"] = width / max(1.0, height)
    flatten_numeric(
        "pre",
        preprocess,
        features,
        keys=(
            "size",
            "colors",
            "color_distance",
            "min_component_px",
            "max_components",
            "initial_foreground_ratio",
            "refined_foreground_ratio",
            "thread_foreground_ratio",
            "region_foreground_ratio",
            "selected_foreground_ratio",
            "region_thread_area_ratio",
        ),
    )
    selected_source = str(preprocess.get("selected_mask_source", "unknown"))
    features[f"pre.selected_source={selected_source}"] = 1.0

    flatten_numeric(
        "pred",
        summary,
        features,
        keys=("foreground_pixels", "hybrid_pixels", "mask_mean", "density_mean"),
    )
    counts = summary.get("stitch_type_counts")
    if isinstance(counts, dict):
        add_ratio_features("pred.stitch_type", counts, features)

    for key, value in preset_args.items():
        clean_key = key.lstrip("-").replace("-", "_")
        if isinstance(value, bool):
            features[f"preset.{clean_key}"] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            features[f"preset.{clean_key}"] = float(value)

    add_sample_preset_interactions(features)

    if include_post_export_features:
        dst_summary = summary.get("dst_summary", {})
        if isinstance(dst_summary, dict):
            flatten_numeric(
                "post.dst",
                dst_summary,
                features,
                keys=(
                    "stitches",
                    "jumps",
                    "jump_distance_mm",
                    "max_jump_mm",
                    "stitch_path_mm",
                    "color_layers",
                    "color_changes",
                    "trims",
                    "components",
                    "continuity_connectors",
                    "outline_stitches",
                ),
            )
            type_counts = dst_summary.get("type_component_counts")
            if isinstance(type_counts, dict):
                add_ratio_features("post.type_component", type_counts, features)
            graph_stats = dst_summary.get("graph_tsp_stats")
            if isinstance(graph_stats, dict):
                flatten_numeric("post.graph", graph_stats, features)

    return features


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build M1 planner selector training rows from a planner sweep.")
    parser.add_argument("--sweep-results", required=True)
    parser.add_argument("--sweep-config", default="configs/sweep_b1.yaml")
    parser.add_argument("--outputs-root", default="")
    parser.add_argument("--pair-root", default="")
    parser.add_argument("--output-dir", default="results/m1_selector/latest")
    parser.add_argument(
        "--include-post-export-features",
        action="store_true",
        help="Include outcome-derived DST/graph metrics. Off by default to avoid target leakage.",
    )
    args = parser.parse_args()

    sweep_path = Path(args.sweep_results)
    rows = read_csv(sweep_path)
    if not rows:
        raise SystemExit(f"No rows found in {sweep_path}")

    config = read_config(Path(args.sweep_config))
    preset_args = preset_args_by_method(config)
    outputs_root = Path(args.outputs_root) if args.outputs_root else sweep_path.parent
    pair_root = Path(args.pair_root) if args.pair_root else None
    output_dir = Path(args.output_dir)

    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row.get("sample_id", "")), []).append(row)

    hard_best_by_sample: dict[str, str] = {}
    mean_best_by_sample: dict[str, str] = {}
    rank_by_sample_method: dict[tuple[str, str], dict[str, int]] = {}
    label_rows: list[dict[str, Any]] = []
    for sample_id, group in by_sample.items():
        hard_sorted = sorted(group, key=hard_sort_key)
        mean_sorted = sorted(group, key=lambda item: safe_float(item.get("unified_loss")))
        hard_best_by_sample[sample_id] = str(hard_sorted[0].get("method", ""))
        mean_best_by_sample[sample_id] = str(mean_sorted[0].get("method", ""))
        for index, item in enumerate(hard_sorted):
            rank_by_sample_method.setdefault((sample_id, str(item.get("method", ""))), {})["hard_rank"] = index
        for index, item in enumerate(mean_sorted):
            rank_by_sample_method.setdefault((sample_id, str(item.get("method", ""))), {})["mean_rank"] = index
        label_rows.append(
            {
                "sample_id": sample_id,
                "best_hard_method": hard_best_by_sample[sample_id],
                "best_mean_method": mean_best_by_sample[sample_id],
                "candidate_count": len(group),
            }
        )

    selector_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        method = str(row.get("method", ""))
        preprocess = load_sample_preprocess(pair_root, sample_id) if pair_root else {}
        summary = safe_read_json(resolve_summary_path(outputs_root, method, sample_id))
        features = build_candidate_features(
            sample_id=sample_id,
            method=method,
            row=row,
            preset_args=preset_args.get(method, {}),
            preprocess=preprocess,
            summary=summary,
            include_post_export_features=args.include_post_export_features,
        )
        ranks = rank_by_sample_method.get((sample_id, method), {})
        metrics = {
            "unified_loss": safe_float(row.get("unified_loss")),
            "exec_score": safe_float(row.get("exec_score")),
            "visual_risk": safe_float(row.get("visual_risk")),
            "jump_count": safe_float(row.get("jump_count")),
            "trim_count": safe_float(row.get("trim_count")),
            "jump_path_mm": safe_float(row.get("jump_path_mm")),
            "off_mask_stitch_length_mm": safe_float(row.get("off_mask_stitch_length_mm")),
            "visible_connector_count": safe_float(row.get("visible_connector_count")),
            "visible_connector_length_mm": safe_float(row.get("visible_connector_length_mm")),
            "stitch_count": safe_float(row.get("stitch_count")),
            "stitch_path_mm": safe_float(row.get("stitch_path_mm")),
            "round_trip_parse_success": safe_float(row.get("round_trip_parse_success")),
            "hard_score": hard_score(row),
            "hard_rank": int(ranks.get("hard_rank", 999)),
            "mean_rank": int(ranks.get("mean_rank", 999)),
        }
        record = {
            "sample_id": sample_id,
            "method": method,
            "features": features,
            "metrics": metrics,
            "labels": {
                "is_best_hard": method == hard_best_by_sample.get(sample_id),
                "is_best_mean": method == mean_best_by_sample.get(sample_id),
            },
            "paths": {
                "eval_json": row.get("eval_json", ""),
                "input": row.get("input", ""),
                "mask": row.get("mask", ""),
                "target": row.get("target", ""),
                "summary_json": str(resolve_summary_path(outputs_root, method, sample_id)),
            },
        }
        selector_rows.append(record)
        flat_row: dict[str, Any] = {
            "sample_id": sample_id,
            "method": method,
            **metrics,
            "is_best_hard": record["labels"]["is_best_hard"],
            "is_best_mean": record["labels"]["is_best_mean"],
        }
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float, bool, str)):
                flat_row[f"feature.{key}"] = value
        flat_rows.append(flat_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(selector_rows, output_dir / "m1_selector_samples.jsonl")
    write_csv(flat_rows, output_dir / "m1_selector_candidates.csv")
    write_csv(label_rows, output_dir / "m1_selector_sample_labels.csv")

    methods = sorted({row["method"] for row in selector_rows})
    samples = sorted(by_sample)
    summary = {
        "version": "m1_selector_dataset_v1",
        "sweep_results": str(sweep_path),
        "sweep_config": args.sweep_config,
        "outputs_root": str(outputs_root),
        "pair_root": str(pair_root) if pair_root else "",
        "include_post_export_features": bool(args.include_post_export_features),
        "samples": len(samples),
        "methods": methods,
        "rows": len(selector_rows),
        "feature_mode": "pre_export_plus_preset" if not args.include_post_export_features else "includes_post_export_leaky_features",
        "hard_order_keys": HARD_ORDER_KEYS,
        "label_rows": label_rows,
    }
    (output_dir / "m1_selector_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
