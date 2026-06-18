from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_m1_direct_selector import matrix, predict_model, safe_float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


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


def load_runtime_model(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights", {})
    norm = payload.get("normalization", {})
    return {
        "payload": payload,
        "feature_names": [str(item) for item in payload.get("feature_names", [])],
        "methods": [str(item) for item in payload.get("methods", [])],
        "preset_args": payload.get("preset_args", {}),
        "runtime_model": {
            "params": {
                "w1": np.asarray(weights.get("w1"), dtype=np.float64),
                "b1": np.asarray(weights.get("b1"), dtype=np.float64),
                "w2": np.asarray(weights.get("w2"), dtype=np.float64),
                "b2": np.asarray(weights.get("b2"), dtype=np.float64),
            },
            "mean": np.asarray(norm.get("mean"), dtype=np.float64),
            "std": np.asarray(norm.get("std"), dtype=np.float64),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply strict M1 direct selector to sample-level image/geometry features.")
    parser.add_argument("--model", default="checkpoints/m1_direct_selector.json")
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--output-csv", default="results/m1_direct_selector/latest_pair_20260618/m1_direct_applied.csv")
    parser.add_argument("--output-json", default="results/m1_direct_selector/latest_pair_20260618/m1_direct_selected_configs.json")
    args = parser.parse_args()

    model = load_runtime_model(Path(args.model))
    samples = read_jsonl(Path(args.samples_jsonl))
    predictions = predict_model(model["runtime_model"], samples, model["feature_names"], model["methods"])
    rows: list[dict[str, Any]] = []
    configs: dict[str, Any] = {}
    for pred in predictions:
        method = str(pred["predicted_method"])
        sample_id = str(pred["sample_id"])
        config_args = model["preset_args"].get(method, {})
        rows.append(
            {
                "sample_id": sample_id,
                "predicted_method": method,
                "confidence": round(safe_float(pred["confidence"]), 8),
                "config_args_json": json.dumps(config_args, ensure_ascii=False, sort_keys=True),
            }
        )
        configs[sample_id] = {
            "predicted_method": method,
            "confidence": safe_float(pred["confidence"]),
            "config_args": config_args,
            "probabilities": pred.get("probabilities", {}),
        }

    write_csv(rows, Path(args.output_csv))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(samples), "output_csv": args.output_csv, "output_json": args.output_json}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
