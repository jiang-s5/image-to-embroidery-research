from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from train_m1_planner_selector import predict, read_jsonl, row_metric, safe_float


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a trained M1 planner selector to candidate rows.")
    parser.add_argument("--model", default="checkpoints/m1_planner_selector.json")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--output-csv", default="results/m1_selector/latest_pair_20260618/m1_selector_applied.csv")
    args = parser.parse_args()

    model_path = Path(args.model)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    rows = read_jsonl(Path(args.candidates_jsonl))
    shrinkage = safe_float(model.get("shrinkage"))
    scores = predict(model, rows, shrinkage)

    by_sample: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_sample[str(row.get("sample_id", ""))].append((row, float(score)))

    selected_rows: list[dict[str, Any]] = []
    for sample_id, group in sorted(by_sample.items()):
        chosen, score = min(group, key=lambda item: item[1])
        selected_rows.append(
            {
                "sample_id": sample_id,
                "selected_method": chosen.get("method", ""),
                "m1_score": round(score, 8),
                "unified_loss_if_evaluated": row_metric(chosen, "unified_loss"),
                "hard_score_if_evaluated": row_metric(chosen, "hard_score"),
                "jump_count_if_evaluated": row_metric(chosen, "jump_count"),
                "trim_count_if_evaluated": row_metric(chosen, "trim_count"),
                "visible_connector_count_if_evaluated": row_metric(chosen, "visible_connector_count"),
                "off_mask_stitch_length_mm_if_evaluated": row_metric(chosen, "off_mask_stitch_length_mm"),
            }
        )

    write_csv(selected_rows, Path(args.output_csv))
    print(json.dumps({"model": str(model_path), "samples": len(selected_rows), "output_csv": args.output_csv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
