from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_THRESHOLDS = {
    "jump_count": 25.0,
    "trim_count": 20.0,
    "jump_path_mm": 2500.0,
    "off_mask_stitch_length_mm": 8.0,
    "visible_connector_count": 3.0,
    "visible_connector_length_mm": 20.0,
}

DEFAULT_WEIGHTS = {
    "parse_fail": 0.30,
    "jump_count": 0.20,
    "trim_count": 0.12,
    "jump_path_mm": 0.08,
    "off_mask_stitch_length_mm": 0.18,
    "visible_connector_count": 0.08,
    "visible_connector_length_mm": 0.04,
}

FALLBACK_WEIGHTS = {
    "parse_fail": 0.35,
    "jump_count": 0.25,
    "trim_count": 0.15,
    "off_mask_stitch_length_mm": 0.15,
    "visible_connector_count": 0.10,
}

HARD_FAIL_THRESHOLDS = {
    "jump_count": 25.0,
    "trim_count": 20.0,
    "off_mask_stitch_length_mm": 8.0,
    "visible_connector_count": 3.0,
}

WARNING_THRESHOLDS = {
    "jump_count": 15.0,
    "trim_count": 10.0,
    "off_mask_stitch_length_mm": 4.0,
    "visible_connector_count": 1.0,
}

EXCELLENT_THRESHOLDS = {
    "jump_count": 10.0,
    "trim_count": 6.0,
    "off_mask_stitch_length_mm": 2.0,
    "visible_connector_count": 0.5,
}

DIAGNOSTIC_METRICS = {
    "safe_connect_repairs",
    "graph_tsp_stats.geometry_edges_sampled",
    "graph_tsp_stats.geometry_visible_risk_edges",
    "graph_tsp_stats.geometry_mean_dt_q05_px",
    "graph_tsp_stats.geometry_mean_sobel_cross",
    "graph_tsp_stats.geometry_mean_canny_cross",
    "graph_tsp_stats.geometry_dt_penalty_sum",
    "graph_tsp_stats.geometry_sobel_penalty_sum",
    "graph_tsp_stats.geometry_canny_penalty_sum",
}


def safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def clipped(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return min(1.0, max(0.0, value / threshold))


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tiny YAML subset fallback: key: value pairs only. This keeps the
        # scoring tool dependency-free for lightweight research packaging.
        config: dict[str, Any] = {}
        section: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                section = line[:-1].strip()
                config[section] = {}
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    parsed: Any = float(value)
                except ValueError:
                    parsed = value.strip("\"'")
                if section and raw_line.startswith(" "):
                    config.setdefault(section, {})[key] = parsed
                else:
                    config[key] = parsed
        return config


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda item: display_path(item).lower())


def extract_pred(payload: dict[str, Any]) -> dict[str, Any]:
    pred = payload.get("pred", payload)
    return pred if isinstance(pred, dict) else {}


def nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part, "")
    return current


def score_metrics(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    full_metric_available = "jump_path_mm" in metrics and "visible_connector_length_mm" in metrics
    base_weights = DEFAULT_WEIGHTS if full_metric_available else FALLBACK_WEIGHTS
    weights = {**base_weights, **(weights or {})}
    parse_success = safe_float(metrics.get("round_trip_parse_success", metrics.get("parse_success", 0.0))) >= 0.5
    parse_fail = 0.0 if parse_success else 1.0

    terms: dict[str, float] = {"parse_fail": parse_fail}
    for key in (
        "jump_count",
        "trim_count",
        "jump_path_mm",
        "off_mask_stitch_length_mm",
        "visible_connector_count",
        "visible_connector_length_mm",
    ):
        if key in weights:
            terms[key] = clipped(safe_float(metrics.get(key)), thresholds.get(key, 1.0))

    unified = sum(weights.get(key, 0.0) * value for key, value in terms.items())
    return {
        "unified_loss": round(unified, 8),
        "exec_score": round(
            weights.get("parse_fail", 0.0) * terms.get("parse_fail", 0.0)
            + weights.get("jump_count", 0.0) * terms.get("jump_count", 0.0)
            + weights.get("trim_count", 0.0) * terms.get("trim_count", 0.0)
            + weights.get("jump_path_mm", 0.0) * terms.get("jump_path_mm", 0.0),
            8,
        ),
        "visual_risk": round(
            weights.get("off_mask_stitch_length_mm", 0.0) * terms.get("off_mask_stitch_length_mm", 0.0)
            + weights.get("visible_connector_count", 0.0) * terms.get("visible_connector_count", 0.0)
            + weights.get("visible_connector_length_mm", 0.0) * terms.get("visible_connector_length_mm", 0.0),
            8,
        ),
        "score_terms": terms,
        "score_weights": weights,
        "score_thresholds": thresholds,
        "quality_level": quality_level(metrics, parse_success=parse_success),
    }


def quality_level(metrics: dict[str, Any], parse_success: bool | None = None) -> str:
    parse_ok = safe_float(metrics.get("round_trip_parse_success", metrics.get("parse_success", 0.0))) >= 0.5
    if parse_success is not None:
        parse_ok = parse_success
    if not parse_ok:
        return "hard_fail"
    if any(safe_float(metrics.get(key)) > threshold for key, threshold in HARD_FAIL_THRESHOLDS.items()):
        return "hard_fail"
    if any(safe_float(metrics.get(key)) > threshold for key, threshold in WARNING_THRESHOLDS.items()):
        return "warning"
    if all(safe_float(metrics.get(key)) <= threshold for key, threshold in EXCELLENT_THRESHOLDS.items()):
        return "excellent"
    return "pass"


def row_from_eval(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pred = extract_pred(payload)
    thresholds = config.get("thresholds", {}) if isinstance(config.get("thresholds", {}), dict) else {}
    weights = config.get("weights", {}) if isinstance(config.get("weights", {}), dict) else {}
    score = score_metrics(pred, thresholds={k: safe_float(v) for k, v in thresholds.items()}, weights={k: safe_float(v) for k, v in weights.items()})
    row = {
        "eval_json": display_path(path),
        "sample_id": path.parent.name,
        "method": path.parent.parent.name,
        **{key: pred.get(key, "") for key in sorted(set(DEFAULT_THRESHOLDS) | set(HARD_FAIL_THRESHOLDS) | {"round_trip_parse_success", "stitch_count", "stitch_path_mm", "max_jump_mm"})},
        **{key: nested_get(pred, key) for key in sorted(DIAGNOSTIC_METRICS)},
        **{key: value for key, value in score.items() if key not in {"score_terms", "score_weights", "score_thresholds"}},
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method", "")), []).append(row)
    methods: dict[str, dict[str, Any]] = {}
    for method, group in by_method.items():
        methods[method] = {
            "samples": len(group),
            "mean_unified_loss": round(mean(safe_float(row.get("unified_loss")) for row in group), 8),
            "mean_exec_score": round(mean(safe_float(row.get("exec_score")) for row in group), 8),
            "mean_visual_risk": round(mean(safe_float(row.get("visual_risk")) for row in group), 8),
            "hard_fail": sum(1 for row in group if row.get("quality_level") == "hard_fail"),
            "warning": sum(1 for row in group if row.get("quality_level") == "warning"),
            "pass": sum(1 for row in group if row.get("quality_level") == "pass"),
            "excellent": sum(1 for row in group if row.get("quality_level") == "excellent"),
        }
    return {"methods": dict(sorted(methods.items(), key=lambda item: safe_float(item[1]["mean_unified_loss"])))}


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


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Unified Executability Score",
        "",
        "Lower `mean_unified_loss` is better. The score combines command executability and visual-risk metrics.",
        "",
        "| Method | Samples | Mean Unified Loss | Exec Score | Visual Risk | Hard Fail | Warning | Pass | Excellent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in summary.get("methods", {}).items():
        lines.append(
            "| {method} | {samples} | {loss:.6f} | {exec_score:.6f} | {visual:.6f} | {hard_fail} | {warning} | {passed} | {excellent} |".format(
                method=method,
                samples=int(metrics["samples"]),
                loss=float(metrics["mean_unified_loss"]),
                exec_score=float(metrics["mean_exec_score"]),
                visual=float(metrics["mean_visual_risk"]),
                hard_fail=int(metrics["hard_fail"]),
                warning=int(metrics["warning"]),
                passed=int(metrics["pass"]),
                excellent=int(metrics["excellent"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute unified executability + visual-risk score for eval JSON files.")
    parser.add_argument("--eval-json", action="append", default=[], help="One eval JSON file. May be repeated.")
    parser.add_argument("--eval-glob", action="append", default=[], help="Glob pattern for eval JSON files. May be repeated.")
    parser.add_argument("--config", default="", help="Optional JSON/YAML config with weights and thresholds.")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    config = load_config(Path(args.config)) if args.config else {}
    paths = [Path(item) for item in args.eval_json]
    for pattern in args.eval_glob:
        paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
    paths = unique_existing_paths(paths)
    rows = [row_from_eval(path, config) for path in paths]
    summary = summarize(rows)

    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        write_markdown(summary, Path(args.output_md))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
