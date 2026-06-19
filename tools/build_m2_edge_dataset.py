from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return payload


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def distance(a: list[float], b: list[float]) -> float:
    return float(math.hypot(safe_float(a[0]) - safe_float(b[0]), safe_float(a[1]) - safe_float(b[1])))


def bbox_features(node: dict[str, Any]) -> dict[str, float]:
    bbox = node.get("bbox_px", [0.0, 0.0, 0.0, 0.0])
    if not isinstance(bbox, list) or len(bbox) < 4:
        bbox = [0.0, 0.0, 0.0, 0.0]
    width = max(0.0, safe_float(bbox[2]) - safe_float(bbox[0]))
    height = max(0.0, safe_float(bbox[3]) - safe_float(bbox[1]))
    return {
        "candidate_bbox_width_px": width,
        "candidate_bbox_height_px": height,
        "candidate_bbox_area_px": width * height,
        "candidate_bbox_aspect": width / max(1e-6, height),
    }


def endpoint_after_visit(node: dict[str, Any], reverse: bool) -> list[float]:
    # If reverse is false, the polyline is traversed start -> end.
    # If reverse is true, it is traversed end -> start.
    key = "start_mm" if reverse else "end_mm"
    point = node.get(key, [0.0, 0.0])
    return point if isinstance(point, list) and len(point) >= 2 else [0.0, 0.0]


def candidate_features(
    task: dict[str, Any],
    node: dict[str, Any],
    previous_node: dict[str, Any],
    current_mm: list[float],
    step_index: int,
    remaining_count: int,
    selected_edge: dict[str, Any],
) -> dict[str, float]:
    start_mm = node.get("start_mm", [0.0, 0.0])
    end_mm = node.get("end_mm", [0.0, 0.0])
    if not isinstance(start_mm, list) or len(start_mm) < 2:
        start_mm = [0.0, 0.0]
    if not isinstance(end_mm, list) or len(end_mm) < 2:
        end_mm = [0.0, 0.0]
    dist_start = distance(current_mm, start_mm)
    dist_end = distance(current_mm, end_mm)
    min_dist = min(dist_start, dist_end)
    best_target = end_mm if dist_end < dist_start else start_mm
    dx = safe_float(best_target[0]) - safe_float(current_mm[0])
    dy = safe_float(best_target[1]) - safe_float(current_mm[1])
    prev_length = safe_float(previous_node.get("length_mm"))
    prev_points = safe_float(previous_node.get("point_count"))
    candidate_length = safe_float(node.get("length_mm"))
    candidate_points = safe_float(node.get("point_count"))
    task_type = int(safe_float(task.get("type_id")))
    features = {
        "distance_to_start_mm": dist_start,
        "distance_to_end_mm": dist_end,
        "min_endpoint_distance_mm": min_dist,
        "endpoint_distance_gap_mm": abs(dist_start - dist_end),
        "choose_reverse_by_distance": 1.0 if dist_end < dist_start else 0.0,
        "dx_best_endpoint_mm": dx,
        "dy_best_endpoint_mm": dy,
        "abs_dx_best_endpoint_mm": abs(dx),
        "abs_dy_best_endpoint_mm": abs(dy),
        "candidate_length_mm": candidate_length,
        "candidate_point_count": candidate_points,
        "previous_length_mm": prev_length,
        "previous_point_count": prev_points,
        "length_ratio_to_previous": candidate_length / max(1e-6, prev_length),
        "point_ratio_to_previous": candidate_points / max(1e-6, prev_points),
        "remaining_count": float(remaining_count),
        "step_index": float(step_index),
        "task_area_px": safe_float(task.get("area_px")),
        "task_order_score": safe_float(task.get("order_score")),
        "task_type_running": 1.0 if task_type == 1 else 0.0,
        "task_type_satin": 1.0 if task_type == 2 else 0.0,
        "task_type_fill": 1.0 if task_type == 3 else 0.0,
        "selected_edge_distance_mm": safe_float(selected_edge.get("distance_mm")),
        "selected_edge_cost": safe_float(selected_edge.get("cost")),
    }
    features.update(bbox_features(node))
    return features


def sample_id_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("m2_trace_"):
        return parent[len("m2_trace_") :]
    if parent.startswith("trace_"):
        return parent[len("trace_") :]
    return parent


def rows_from_trace(path: Path, include_start_step: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = read_json(path)
    sample_id = sample_id_from_path(path)
    rows: list[dict[str, Any]] = []
    trace_summary = {
        "sample_id": sample_id,
        "trace_path": str(path),
        "tasks": 0,
        "nodes": 0,
        "decisions": 0,
        "candidate_rows": 0,
        "positive_rows": 0,
        "skipped_single_node_tasks": 0,
    }
    for task in trace.get("tasks", []):
        if not isinstance(task, dict):
            continue
        graph = task.get("graph", {})
        if not isinstance(graph, dict):
            continue
        nodes_raw = graph.get("nodes", [])
        edges_raw = graph.get("selected_edges", [])
        if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
            continue
        nodes = {int(safe_float(node.get("node_id"))): node for node in nodes_raw if isinstance(node, dict)}
        trace_summary["tasks"] += 1
        trace_summary["nodes"] += len(nodes)
        if len(nodes) <= 1:
            trace_summary["skipped_single_node_tasks"] += 1
            continue
        remaining = set(nodes)
        previous_node_id = -1
        previous_node: dict[str, Any] | None = None
        current_mm: list[float] | None = None
        for edge in edges_raw:
            if not isinstance(edge, dict):
                continue
            to_node = int(safe_float(edge.get("to_node"), -9999.0))
            from_node = int(safe_float(edge.get("from_node"), -9999.0))
            if to_node not in nodes or to_node not in remaining:
                continue
            if from_node == -1:
                if include_start_step:
                    current_mm = nodes[to_node].get("start_mm", [0.0, 0.0])
                    previous_node = nodes[to_node]
                remaining.remove(to_node)
                previous_node_id = to_node
                previous_node = nodes[to_node]
                current_mm = endpoint_after_visit(nodes[to_node], bool(edge.get("reverse", False)))
                continue
            if previous_node is None or current_mm is None or previous_node_id not in nodes:
                previous_node = nodes.get(from_node)
                current_mm = endpoint_after_visit(previous_node, False) if previous_node else [0.0, 0.0]
            if not remaining:
                break
            decision_id = f"{sample_id}:task{task.get('task_index')}:step{edge.get('step')}"
            trace_summary["decisions"] += 1
            for candidate_id in sorted(remaining):
                node = nodes[candidate_id]
                label = 1 if candidate_id == to_node else 0
                rows.append(
                    {
                        "sample_id": sample_id,
                        "trace_path": str(path),
                        "task_index": int(safe_float(task.get("task_index"))),
                        "decision_id": decision_id,
                        "step": int(safe_float(edge.get("step"))),
                        "from_node": from_node,
                        "candidate_node": candidate_id,
                        "selected_node": to_node,
                        "label": label,
                        "features": candidate_features(
                            task,
                            node,
                            previous_node or nodes[from_node],
                            current_mm,
                            int(safe_float(edge.get("step"))),
                            len(remaining),
                            edge,
                        ),
                    }
                )
                trace_summary["candidate_rows"] += 1
                trace_summary["positive_rows"] += label
            remaining.remove(to_node)
            previous_node_id = to_node
            previous_node = nodes[to_node]
            current_mm = endpoint_after_visit(nodes[to_node], bool(edge.get("reverse", False)))
    return rows, trace_summary


def downsample_negatives(rows: list[dict[str, Any]], max_negatives: int) -> list[dict[str, Any]]:
    if max_negatives <= 0:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)
    kept: list[dict[str, Any]] = []
    for group in grouped.values():
        positives = [row for row in group if int(row.get("label", 0)) == 1]
        negatives = [row for row in group if int(row.get("label", 0)) == 0]
        negatives.sort(
            key=lambda row: (
                safe_float(row.get("features", {}).get("min_endpoint_distance_mm")) if isinstance(row.get("features", {}), dict) else 0.0,
                int(row.get("candidate_node", 0)),
            )
        )
        kept.extend(positives)
        kept.extend(negatives[:max_negatives])
    kept.sort(
        key=lambda row: (
            str(row.get("sample_id", "")),
            str(row.get("decision_id", "")),
            -int(row.get("label", 0)),
            int(row.get("candidate_node", 0)),
        )
    )
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an M2 edge-decision dataset from graph_tsp_trace.json files.")
    parser.add_argument("--trace-glob", action="append", required=True, help="Glob for graph_tsp_trace.json files. May be repeated.")
    parser.add_argument("--output-dir", default="results/m2_edge_policy/latest_pair_20260619")
    parser.add_argument("--include-start-step", action="store_true")
    parser.add_argument("--max-negatives-per-decision", type=int, default=24)
    args = parser.parse_args()

    trace_paths: list[Path] = []
    for pattern in args.trace_glob:
        trace_paths.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    trace_paths = sorted(set(path for path in trace_paths if path.exists()))
    if not trace_paths:
        raise SystemExit("No trace files found.")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in trace_paths:
        rows, summary = rows_from_trace(path, args.include_start_step)
        all_rows.extend(rows)
        summaries.append(summary)
    raw_candidate_rows = len(all_rows)
    if not all_rows:
        raise SystemExit("No M2 candidate rows were created. Need traces with multi-node graph tasks.")
    all_rows = downsample_negatives(all_rows, args.max_negatives_per_decision)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "m2_edge_candidates.jsonl"
    summary_path = output_dir / "m2_edge_dataset_summary.json"
    write_jsonl(all_rows, rows_path)
    write_csv(summaries, output_dir / "m2_edge_trace_summary.csv")
    payload = {
        "version": "m2_edge_dataset_v1",
        "trace_count": len(trace_paths),
        "raw_candidate_rows": raw_candidate_rows,
        "candidate_rows": len(all_rows),
        "positive_rows": sum(int(row["label"]) for row in all_rows),
        "decision_count": len({row["decision_id"] for row in all_rows}),
        "sample_count": len({row["sample_id"] for row in all_rows}),
        "max_negatives_per_decision": args.max_negatives_per_decision,
        "trace_summaries": summaries,
        "output_jsonl": str(rows_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
