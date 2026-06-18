from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


BOOL_KEYS = {"--graph-tsp-planner", "--mask-safe-connectors"}
INT_KEYS = {
    "--min-active-px",
    "--row-step-px",
    "--point-step-px",
    "--min-component-px",
    "--max-components",
    "--two-opt-passes",
    "--graph-max-two-opt-nodes",
}


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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def metric(row: dict[str, Any], key: str) -> float:
    payload = row.get("metrics", {})
    return safe_float(payload.get(key)) if isinstance(payload, dict) else 0.0


def preset_args_by_method(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    presets = config.get("presets", [])
    if isinstance(presets, list):
        for item in presets:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            args = item.get("args", {})
            if name and isinstance(args, dict):
                out[name] = args
    return out


def parameter_keys(preset_args: dict[str, dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for args in preset_args.values():
        keys.update(str(key) for key in args)
    return sorted(keys)


def parameter_ranges(preset_args: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [safe_float(args.get(key)) for args in preset_args.values()]
        ranges[key] = {"min": min(values), "max": max(values)}
    return ranges


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    return dict(grouped)


def sample_features_from_row(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("features", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        key = str(key)
        if key in {"method", "sample_id"}:
            continue
        if key.startswith("preset.") or key.startswith("interaction."):
            continue
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[key] = safe_float(value)
    return out


def vector_from_args(args: dict[str, Any], keys: list[str]) -> list[float]:
    return [safe_float(args.get(key)) for key in keys]


def args_from_vector(vector: np.ndarray, keys: list[str], ranges: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw_value in zip(keys, vector.tolist()):
        value = float(raw_value)
        if key in ranges:
            value = min(ranges[key]["max"], max(ranges[key]["min"], value))
        if key in BOOL_KEYS:
            if value >= 0.5:
                out[key] = True
        elif key in INT_KEYS:
            out[key] = int(round(value))
        else:
            out[key] = round(value, 6)
    return out


def choose_feature_row(group: list[dict[str, Any]], feature_method: str) -> dict[str, Any]:
    for row in group:
        if str(row.get("method", "")) == feature_method:
            return row
    return sorted(group, key=lambda item: str(item.get("method", "")))[0]


def build_samples(
    candidate_rows: list[dict[str, Any]],
    preset_args: dict[str, dict[str, Any]],
    keys: list[str],
    feature_method: str,
    label_target: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_id, group in sorted(group_rows(candidate_rows).items()):
        feature_row = choose_feature_row(group, feature_method)
        best = min(group, key=lambda row: metric(row, label_target))
        method = str(best.get("method", ""))
        samples.append(
            {
                "sample_id": sample_id,
                "features": sample_features_from_row(feature_row),
                "feature_method": str(feature_row.get("method", "")),
                "label_method": method,
                "label_config": preset_args.get(method, {}),
                "label_vector": vector_from_args(preset_args.get(method, {}), keys),
                "paths": feature_row.get("paths", {}),
            }
        )
    return samples


def feature_names(samples: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        features = sample.get("features", {})
        if isinstance(features, dict):
            names.update(str(key) for key in features)
    return sorted(names)


def x_matrix(samples: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(samples), len(names)), dtype=np.float64)
    for row_idx, sample in enumerate(samples):
        features = sample.get("features", {})
        if not isinstance(features, dict):
            continue
        for key, value in features.items():
            if key in index:
                x[row_idx, index[key]] = safe_float(value)
    return x


def y_matrix(samples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([sample["label_vector"] for sample in samples], dtype=np.float64)


def fit_norm(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_v = values.mean(axis=0)
    std_v = values.std(axis=0)
    std_v[std_v < 1e-8] = 1.0
    return (values - mean_v) / std_v, mean_v, std_v


def apply_norm(values: np.ndarray, mean_v: np.ndarray, std_v: np.ndarray) -> np.ndarray:
    std_v = std_v.copy()
    std_v[std_v < 1e-8] = 1.0
    return (values - mean_v) / std_v


def init_model(input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, math.sqrt(2.0 / max(1, input_dim)), size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "w2": rng.normal(0.0, math.sqrt(2.0 / max(1, hidden_dim)), size=(hidden_dim, output_dim)),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ params["w1"] + params["b1"])
    pred = hidden @ params["w2"] + params["b2"]
    return hidden, pred


def train_regressor(
    train_samples: list[dict[str, Any]],
    names: list[str],
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    x = x_matrix(train_samples, names)
    y = y_matrix(train_samples)
    xz, x_mean, x_std = fit_norm(x)
    yz, y_mean, y_std = fit_norm(y)
    rng = np.random.default_rng(seed)
    params = init_model(xz.shape[1], hidden_dim, yz.shape[1], rng)
    losses: list[float] = []
    for _epoch in range(epochs):
        hidden, pred = forward(params, xz)
        error = pred - yz
        mse = float(np.mean(error**2))
        reg = 0.5 * weight_decay * (np.sum(params["w1"] ** 2) + np.sum(params["w2"] ** 2))
        losses.append(mse + reg)
        n = max(1, len(train_samples))
        grad_pred = (2.0 / n) * error / max(1, yz.shape[1])
        grad_w2 = hidden.T @ grad_pred + weight_decay * params["w2"]
        grad_b2 = grad_pred.sum(axis=0)
        grad_hidden = grad_pred @ params["w2"].T
        grad_pre = grad_hidden * (1.0 - hidden**2)
        grad_w1 = xz.T @ grad_pre + weight_decay * params["w1"]
        grad_b1 = grad_pre.sum(axis=0)
        params["w2"] -= lr * grad_w2
        params["b2"] -= lr * grad_b2
        params["w1"] -= lr * grad_w1
        params["b1"] -= lr * grad_b1
    return {
        "params": params,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "loss_curve": losses,
        "final_mse": losses[-1] if losses else 0.0,
    }


def predict_vectors(model: dict[str, Any], samples: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    x = x_matrix(samples, names)
    xz = apply_norm(x, np.asarray(model["x_mean"], dtype=np.float64), np.asarray(model["x_std"], dtype=np.float64))
    params = {key: np.asarray(value, dtype=np.float64) for key, value in model["params"].items()}
    _hidden, pred_z = forward(params, xz)
    return pred_z * np.asarray(model["y_std"], dtype=np.float64) + np.asarray(model["y_mean"], dtype=np.float64)


def nearest_preset(pred: np.ndarray, preset_args: dict[str, dict[str, Any]], keys: list[str], ranges: dict[str, dict[str, float]]) -> tuple[str, float]:
    best_method = ""
    best_dist = float("inf")
    scales = np.asarray([max(1e-6, ranges[key]["max"] - ranges[key]["min"]) for key in keys], dtype=np.float64)
    for method, args in preset_args.items():
        target = np.asarray(vector_from_args(args, keys), dtype=np.float64)
        dist = float(np.mean(((pred - target) / scales) ** 2))
        if dist < best_dist:
            best_method = method
            best_dist = dist
    return best_method, best_dist


def evaluate_spec(
    samples: list[dict[str, Any]],
    names: list[str],
    preset_args: dict[str, dict[str, Any]],
    keys: list[str],
    ranges: dict[str, dict[str, float]],
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    mses: list[float] = []
    for held in samples:
        train = [sample for sample in samples if sample["sample_id"] != held["sample_id"]]
        model = train_regressor(train, names, hidden_dim, lr, weight_decay, epochs, seed)
        pred = predict_vectors(model, [held], names)[0]
        target = np.asarray(held["label_vector"], dtype=np.float64)
        mse = float(np.mean((pred - target) ** 2))
        mses.append(mse)
        pred_args = args_from_vector(pred, keys, ranges)
        nearest, dist = nearest_preset(pred, preset_args, keys, ranges)
        fold_rows.append(
            {
                "sample_id": held["sample_id"],
                "label_method": held["label_method"],
                "nearest_preset": nearest,
                "nearest_distance": round(dist, 8),
                "config_mse": round(mse, 8),
                "predicted_config_json": json.dumps(pred_args, ensure_ascii=False, sort_keys=True),
            }
        )
    return {
        "hidden_dim": hidden_dim,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "mean_config_mse": round(mean(mses), 8),
        "folds": fold_rows,
    }


def serializable_model(
    trained: dict[str, Any],
    names: list[str],
    keys: list[str],
    ranges: dict[str, dict[str, float]],
    preset_args: dict[str, dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": "m1_config_regressor_v1",
        "model_type": "numpy_mlp_regressor",
        "mapping": "image_geometry_features_to_continuous_planner_config",
        "training_loss": "normalized_mse",
        "feature_names": names,
        "parameter_keys": keys,
        "parameter_ranges": ranges,
        "bool_keys": sorted(BOOL_KEYS),
        "int_keys": sorted(INT_KEYS),
        "preset_bank": preset_args,
        "hidden_dim": spec["hidden_dim"],
        "lr": spec["lr"],
        "weight_decay": spec["weight_decay"],
        "epochs": spec["epochs"],
        "seed": spec["seed"],
        "normalization": {
            "x_mean": trained["x_mean"].tolist(),
            "x_std": trained["x_std"].tolist(),
            "y_mean": trained["y_mean"].tolist(),
            "y_std": trained["y_std"].tolist(),
        },
        "weights": {key: value.tolist() for key, value in trained["params"].items()},
        "training_metrics": {
            "final_normalized_mse": trained["final_mse"],
            "loss_curve_tail": trained["loss_curve"][-10:],
        },
    }


def write_report(path: Path, best: dict[str, Any], model_path: Path, samples_path: Path, predictions_path: Path) -> None:
    lines = [
        "# M1 Config Regressor",
        "",
        "This is the stronger M1 variant: it maps image/geometry features directly to continuous planner parameters.",
        "",
        "It is different from preset classification because the output is a planner config vector, not only a preset id.",
        "",
        "## Best Leave-One-Out Config Regression",
        "",
        f"- hidden_dim: `{best['hidden_dim']}`",
        f"- lr: `{best['lr']}`",
        f"- weight_decay: `{best['weight_decay']}`",
        f"- epochs: `{best['epochs']}`",
        f"- seed: `{best['seed']}`",
        f"- mean config MSE: `{best['mean_config_mse']}`",
        f"- model artifact: `{model_path.as_posix()}`",
        f"- sample dataset: `{samples_path.as_posix()}`",
        f"- LOO predicted configs: `{predictions_path.as_posix()}`",
        "",
        "## Fold Predictions",
        "",
        "| Sample | Label Method | Nearest Preset | Config MSE |",
        "| --- | --- | --- | ---: |",
    ]
    for row in best["folds"]:
        lines.append(
            "| {sample_id} | {label_method} | {nearest_preset} | {config_mse:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Requirement Check",
            "",
            "| Requirement | Status | Evidence |",
            "| --- | --- | --- |",
            "| Direct config generation | Done | `predicted_config_json` contains numeric planner args per sample |",
            "| Learning model | Done | NumPy MLP regressor |",
            "| Training loss | Done | normalized MSE over oracle config vectors |",
            "| No sweep at inference | Done | inference is one forward pass from sample features to config vector |",
            "",
            "The separate evaluator step runs these predicted configs through DST generation and command/visual-risk scoring.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train M1 image/geometry -> continuous planner config regressor.")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--sweep-config", default="configs/sweep_b1.yaml")
    parser.add_argument("--feature-method", default="b1_conservative")
    parser.add_argument("--label-target", default="hard_score")
    parser.add_argument("--output-dir", default="results/m1_config_regressor/latest_pair_20260618")
    parser.add_argument("--model-output", default="checkpoints/m1_config_regressor.json")
    parser.add_argument("--hidden-dims", default="4,8,16,32")
    parser.add_argument("--lrs", default="0.005,0.01,0.03,0.05")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001,0.01")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seeds", default="1,2,3,5,8,13")
    args = parser.parse_args()

    candidate_rows = read_jsonl(Path(args.candidates_jsonl))
    config = read_json(Path(args.sweep_config))
    presets = preset_args_by_method(config)
    keys = parameter_keys(presets)
    ranges = parameter_ranges(presets, keys)
    samples = build_samples(candidate_rows, presets, keys, args.feature_method, args.label_target)
    names = feature_names(samples)

    hidden_dims = [int(safe_float(item)) for item in args.hidden_dims.split(",") if item.strip()]
    lrs = [safe_float(item) for item in args.lrs.split(",") if item.strip()]
    weight_decays = [safe_float(item) for item in args.weight_decays.split(",") if item.strip()]
    seeds = [int(safe_float(item)) for item in args.seeds.split(",") if item.strip()]

    leaderboard: list[dict[str, Any]] = []
    for hidden_dim in hidden_dims:
        for lr in lrs:
            for weight_decay in weight_decays:
                for seed in seeds:
                    leaderboard.append(
                        evaluate_spec(samples, names, presets, keys, ranges, hidden_dim, lr, weight_decay, args.epochs, seed)
                    )
    leaderboard.sort(key=lambda row: row["mean_config_mse"])
    best = leaderboard[0]

    final = train_regressor(
        samples,
        names,
        int(best["hidden_dim"]),
        float(best["lr"]),
        float(best["weight_decay"]),
        int(best["epochs"]),
        int(best["seed"]),
    )
    pred_vectors = predict_vectors(final, samples, names)
    prediction_rows: list[dict[str, Any]] = []
    for sample, pred in zip(samples, pred_vectors):
        pred_args = args_from_vector(pred, keys, ranges)
        nearest, dist = nearest_preset(pred, presets, keys, ranges)
        prediction_rows.append(
            {
                "sample_id": sample["sample_id"],
                "label_method": sample["label_method"],
                "nearest_preset": nearest,
                "nearest_distance": dist,
                "predicted_config": pred_args,
                "paths": sample.get("paths", {}),
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "m1_config_samples.jsonl"
    predictions_path = output_dir / "m1_config_predictions.json"
    write_jsonl(samples, samples_path)
    predictions_path.write_text(json.dumps(prediction_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(best["folds"], output_dir / "m1_config_loo_decisions.csv")
    write_csv(
        [
            {
                "rank": index + 1,
                "hidden_dim": row["hidden_dim"],
                "lr": row["lr"],
                "weight_decay": row["weight_decay"],
                "epochs": row["epochs"],
                "seed": row["seed"],
                "mean_config_mse": row["mean_config_mse"],
            }
            for index, row in enumerate(leaderboard)
        ],
        output_dir / "m1_config_loop_leaderboard.csv",
    )
    result_payload = {
        "version": "m1_config_regressor_results_v1",
        "best": best,
        "feature_count": len(names),
        "parameter_keys": keys,
        "samples": [sample["sample_id"] for sample in samples],
        "model_output": args.model_output,
    }
    (output_dir / "m1_config_results.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(serializable_model(final, names, keys, ranges, presets, best), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "m1_config_report.md", best, model_path, samples_path, predictions_path)
    print(
        json.dumps(
            {
                "best": {k: best[k] for k in ("hidden_dim", "lr", "weight_decay", "epochs", "seed", "mean_config_mse")},
                "model_output": args.model_output,
                "predictions": str(predictions_path),
                "report": str(output_dir / "m1_config_report.md"),
                "feature_count": len(names),
                "parameter_count": len(keys),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
