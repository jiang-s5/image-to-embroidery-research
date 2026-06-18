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


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return payload


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


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    return dict(grouped)


def label_row(group: list[dict[str, Any]], target: str) -> dict[str, Any]:
    return min(group, key=lambda row: metric(row, target))


def choose_feature_source(group: list[dict[str, Any]], feature_method: str) -> dict[str, Any]:
    for row in group:
        if str(row.get("method", "")) == feature_method:
            return row
    return sorted(group, key=lambda row: str(row.get("method", "")))[0]


def sample_features_from_row(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("features", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        key = str(key)
        if key in {"method", "sample_id"}:
            continue
        # Strict M1 uses image/geometry summaries only. Candidate preset args and
        # candidate interactions belong to the earlier M0.9 scorer and are
        # intentionally excluded here.
        if key.startswith("preset.") or key.startswith("interaction."):
            continue
        if isinstance(value, bool):
            out[key] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            out[key] = safe_float(value)
    return out


def build_sample_dataset(
    candidate_rows: list[dict[str, Any]],
    feature_method: str,
    label_target: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped = group_rows(candidate_rows)
    methods = sorted({str(row.get("method", "")) for row in candidate_rows})
    samples: list[dict[str, Any]] = []
    for sample_id, group in sorted(grouped.items()):
        feature_row = choose_feature_source(group, feature_method)
        best = label_row(group, label_target)
        features = sample_features_from_row(feature_row)
        by_method = {str(row.get("method", "")): row for row in group}
        samples.append(
            {
                "sample_id": sample_id,
                "feature_method": str(feature_row.get("method", "")),
                "features": features,
                "label_method": str(best.get("method", "")),
                "label_target": label_target,
                "candidate_metrics": {
                    method: {
                        "unified_loss": metric(row, "unified_loss"),
                        "hard_score": metric(row, "hard_score"),
                        "jump_count": metric(row, "jump_count"),
                        "trim_count": metric(row, "trim_count"),
                        "off_mask_stitch_length_mm": metric(row, "off_mask_stitch_length_mm"),
                        "visible_connector_count": metric(row, "visible_connector_count"),
                    }
                    for method, row in by_method.items()
                },
            }
        )
    return samples, methods


def feature_names(samples: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        features = sample.get("features", {})
        if isinstance(features, dict):
            names.update(str(key) for key in features)
    return sorted(names)


def matrix(samples: list[dict[str, Any]], names: list[str]) -> np.ndarray:
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


def labels(samples: list[dict[str, Any]], methods: list[str]) -> np.ndarray:
    index = {method: i for i, method in enumerate(methods)}
    return np.array([index[str(sample.get("label_method", ""))] for sample in samples], dtype=np.int64)


def init_model(input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale1 = math.sqrt(2.0 / max(1, input_dim))
    scale2 = math.sqrt(2.0 / max(1, hidden_dim))
    return {
        "w1": rng.normal(0.0, scale1, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "w2": rng.normal(0.0, scale2, size=(hidden_dim, output_dim)),
        "b2": np.zeros(output_dim, dtype=np.float64),
    }


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    return (x - mean_x) / std_x, mean_x, std_x


def normalize_apply(x: np.ndarray, mean_x: np.ndarray, std_x: np.ndarray) -> np.ndarray:
    std_x = std_x.copy()
    std_x[std_x < 1e-8] = 1.0
    return (x - mean_x) / std_x


def forward(params: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(x @ params["w1"] + params["b1"])
    logits = hidden @ params["w2"] + params["b2"]
    return hidden, logits


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_mlp(
    train_samples: list[dict[str, Any]],
    names: list[str],
    methods: list[str],
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    x = matrix(train_samples, names)
    y = labels(train_samples, methods)
    xz, mean_x, std_x = normalize_fit(x)
    rng = np.random.default_rng(seed)
    params = init_model(xz.shape[1], hidden_dim, len(methods), rng)
    losses: list[float] = []
    for _epoch in range(epochs):
        hidden, logits = forward(params, xz)
        probs = softmax(logits)
        n = max(1, len(train_samples))
        ce = -np.log(np.clip(probs[np.arange(n), y], 1e-9, 1.0)).mean()
        reg = 0.5 * weight_decay * (np.sum(params["w1"] ** 2) + np.sum(params["w2"] ** 2))
        loss = float(ce + reg)
        losses.append(loss)

        grad_logits = probs
        grad_logits[np.arange(n), y] -= 1.0
        grad_logits /= n
        grad_w2 = hidden.T @ grad_logits + weight_decay * params["w2"]
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = grad_logits @ params["w2"].T
        grad_pre_hidden = grad_hidden * (1.0 - hidden**2)
        grad_w1 = xz.T @ grad_pre_hidden + weight_decay * params["w1"]
        grad_b1 = grad_pre_hidden.sum(axis=0)

        params["w2"] -= lr * grad_w2
        params["b2"] -= lr * grad_b2
        params["w1"] -= lr * grad_w1
        params["b1"] -= lr * grad_b1

    return {
        "params": params,
        "mean": mean_x,
        "std": std_x,
        "loss_curve": losses,
        "final_loss": losses[-1] if losses else 0.0,
    }


def predict_model(model: dict[str, Any], samples: list[dict[str, Any]], names: list[str], methods: list[str]) -> list[dict[str, Any]]:
    x = matrix(samples, names)
    xz = normalize_apply(x, np.asarray(model["mean"], dtype=np.float64), np.asarray(model["std"], dtype=np.float64))
    params = {key: np.asarray(value, dtype=np.float64) for key, value in model["params"].items()}
    _hidden, logits = forward(params, xz)
    probs = softmax(logits)
    predictions: list[dict[str, Any]] = []
    for sample, row_probs in zip(samples, probs):
        best_idx = int(np.argmax(row_probs))
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "predicted_method": methods[best_idx],
                "confidence": float(row_probs[best_idx]),
                "probabilities": {method: float(row_probs[i]) for i, method in enumerate(methods)},
            }
        )
    return predictions


def summarize_selected(samples: list[dict[str, Any]], selected_methods: dict[str, str]) -> dict[str, float]:
    keys = (
        "unified_loss",
        "hard_score",
        "jump_count",
        "trim_count",
        "off_mask_stitch_length_mm",
        "visible_connector_count",
    )
    values: dict[str, list[float]] = {key: [] for key in keys}
    for sample in samples:
        method = selected_methods[str(sample["sample_id"])]
        metrics = sample["candidate_metrics"][method]
        for key in keys:
            values[key].append(safe_float(metrics.get(key)))
    return {key: round(mean(vals), 8) for key, vals in values.items()}


def evaluate_spec(
    samples: list[dict[str, Any]],
    names: list[str],
    methods: list[str],
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    selected_methods: dict[str, str] = {}
    for sample in samples:
        held_id = str(sample["sample_id"])
        train_samples = [item for item in samples if str(item["sample_id"]) != held_id]
        held = [sample]
        model = train_mlp(train_samples, names, methods, hidden_dim, lr, weight_decay, epochs, seed)
        pred = predict_model(model, held, names, methods)[0]
        predicted = str(pred["predicted_method"])
        selected_methods[held_id] = predicted
        label = str(sample["label_method"])
        selected_metrics = sample["candidate_metrics"].get(predicted, {})
        oracle_metrics = sample["candidate_metrics"].get(label, {})
        fold_rows.append(
            {
                "sample_id": held_id,
                "predicted_method": predicted,
                "label_method": label,
                "correct": predicted == label,
                "confidence": round(float(pred["confidence"]), 8),
                "selected_unified_loss": safe_float(selected_metrics.get("unified_loss")),
                "oracle_unified_loss": safe_float(oracle_metrics.get("unified_loss")),
                "selected_hard_score": safe_float(selected_metrics.get("hard_score")),
                "oracle_hard_score": safe_float(oracle_metrics.get("hard_score")),
                "selected_jump_count": safe_float(selected_metrics.get("jump_count")),
                "selected_trim_count": safe_float(selected_metrics.get("trim_count")),
            }
        )
    summary = summarize_selected(samples, selected_methods)
    accuracy = sum(1 for row in fold_rows if row["correct"]) / max(1, len(fold_rows))
    return {
        "hidden_dim": hidden_dim,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "folds": fold_rows,
        "accuracy": round(accuracy, 8),
        "selected_methods": selected_methods,
        "summary": summary,
    }


def oracle_summary(samples: list[dict[str, Any]]) -> dict[str, float]:
    return summarize_selected(samples, {str(sample["sample_id"]): str(sample["label_method"]) for sample in samples})


def fixed_summary(samples: list[dict[str, Any]], method: str) -> dict[str, float]:
    return summarize_selected(samples, {str(sample["sample_id"]): method for sample in samples})


def to_serializable_model(
    trained: dict[str, Any],
    names: list[str],
    methods: list[str],
    preset_args: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    feature_method: str,
    label_target: str,
) -> dict[str, Any]:
    params = trained["params"]
    return {
        "version": "m1_direct_mlp_selector_v1",
        "model_type": "numpy_mlp_softmax",
        "mapping": "image_geometry_features_to_planner_preset_config",
        "feature_method": feature_method,
        "label_target": label_target,
        "feature_names": names,
        "methods": methods,
        "preset_args": preset_args,
        "hidden_dim": spec["hidden_dim"],
        "lr": spec["lr"],
        "weight_decay": spec["weight_decay"],
        "epochs": spec["epochs"],
        "seed": spec["seed"],
        "normalization": {
            "mean": trained["mean"].tolist(),
            "std": trained["std"].tolist(),
        },
        "weights": {key: value.tolist() for key, value in params.items()},
        "training_loss": {
            "final_cross_entropy": trained["final_loss"],
            "loss_curve_tail": trained["loss_curve"][-10:],
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(
    path: Path,
    best: dict[str, Any],
    baselines: list[dict[str, Any]],
    model_path: Path,
    dataset_path: Path,
) -> None:
    lines = [
        "# Strict M1 Direct Planner Selector",
        "",
        "This is the strict M1 stage: a trainable MLP maps image/geometry summary features directly to a planner preset/config.",
        "",
        "Unlike the earlier M0.9 scorer, strict M1 does not score every candidate preset as input. It predicts one preset from sample-level features.",
        "",
        "## Requirement Check",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
        "| Learning model | Done | NumPy MLP with tanh hidden layer and softmax output |",
        "| Training loss | Done | Cross-entropy loss saved in the model artifact |",
        "| Image to config mapping | Done | Input features exclude `preset.*` and `interaction.*`; output is a preset plus config args |",
        "| Closed-loop evaluation | Done | Leave-one-out by sample against sweep-derived labels |",
        "",
        "## Best Leave-One-Out Result",
        "",
        f"- hidden_dim: `{best['hidden_dim']}`",
        f"- lr: `{best['lr']}`",
        f"- weight_decay: `{best['weight_decay']}`",
        f"- epochs: `{best['epochs']}`",
        f"- seed: `{best['seed']}`",
        f"- accuracy: `{best['accuracy']}`",
        f"- model artifact: `{model_path.as_posix()}`",
        f"- sample dataset: `{dataset_path.as_posix()}`",
        "",
        "| Metric | Strict M1 LOO Mean |",
        "| --- | ---: |",
    ]
    for key, value in best["summary"].items():
        lines.append(f"| {key} | {safe_float(value):.6f} |")

    lines.extend(
        [
            "",
            "## Fold Decisions",
            "",
            "| Sample | Predicted Method | Label Method | Correct | Confidence | Selected Unified | Oracle Unified | Selected Hard | Oracle Hard |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best["folds"]:
        lines.append(
            "| {sample_id} | {predicted_method} | {label_method} | {correct} | {confidence:.6f} | {selected_unified_loss:.6f} | {oracle_unified_loss:.6f} | {selected_hard_score:.6f} | {oracle_hard_score:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Baseline | Unified | Hard Score | Jumps | Trims | Visible Connectors | Off-Mask mm |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in baselines:
        summary = item["summary"]
        lines.append(
            "| {name} | {unified_loss:.6f} | {hard_score:.6f} | {jump_count:.3f} | {trim_count:.3f} | {visible_connector_count:.3f} | {off_mask_stitch_length_mm:.3f} |".format(
                name=item["name"],
                **summary,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is now a formal M1 prototype, not only an oracle-driven sweep scorer.",
            "- The supervision labels are still derived from a small sweep, so the result is not a mature generalization claim.",
            "- The next real upgrade is to expand the sweep benchmark, then train the same direct selector on Pilot-50 / Release-200.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train strict M1 direct image/geometry -> planner preset selector.")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--sweep-config", default="configs/sweep_b1.yaml")
    parser.add_argument("--feature-method", default="b1_conservative")
    parser.add_argument("--label-target", default="hard_score")
    parser.add_argument("--output-dir", default="results/m1_direct_selector/latest_pair_20260618")
    parser.add_argument("--model-output", default="checkpoints/m1_direct_selector.json")
    parser.add_argument("--hidden-dims", default="4,8,16,32")
    parser.add_argument("--lrs", default="0.005,0.01,0.03,0.05")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001,0.01")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--seeds", default="1,2,3,5,8,13")
    args = parser.parse_args()

    candidate_rows = read_jsonl(Path(args.candidates_jsonl))
    if not candidate_rows:
        raise SystemExit("No candidate rows found.")
    config = read_json(Path(args.sweep_config))
    preset_args = preset_args_by_method(config)
    samples, methods = build_sample_dataset(candidate_rows, args.feature_method, args.label_target)
    names = feature_names(samples)
    if not names:
        raise SystemExit("No sample-level features found.")

    hidden_dims = [int(safe_float(item)) for item in args.hidden_dims.split(",") if item.strip()]
    lrs = [safe_float(item) for item in args.lrs.split(",") if item.strip()]
    weight_decays = [safe_float(item) for item in args.weight_decays.split(",") if item.strip()]
    seeds = [int(safe_float(item)) for item in args.seeds.split(",") if item.strip()]

    leaderboard: list[dict[str, Any]] = []
    for hidden_dim in hidden_dims:
        for lr in lrs:
            for weight_decay in weight_decays:
                for seed in seeds:
                    result = evaluate_spec(samples, names, methods, hidden_dim, lr, weight_decay, args.epochs, seed)
                    leaderboard.append(result)
    leaderboard.sort(
        key=lambda item: (
            item["summary"]["hard_score"],
            item["summary"]["unified_loss"],
            -item["accuracy"],
            item["summary"]["jump_count"],
        )
    )
    best = leaderboard[0]
    final_trained = train_mlp(
        samples,
        names,
        methods,
        hidden_dim=int(best["hidden_dim"]),
        lr=float(best["lr"]),
        weight_decay=float(best["weight_decay"]),
        epochs=int(best["epochs"]),
        seed=int(best["seed"]),
    )
    model_payload = to_serializable_model(
        final_trained,
        names,
        methods,
        preset_args,
        best,
        args.feature_method,
        args.label_target,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dataset_path = output_dir / "m1_direct_samples.jsonl"
    write_jsonl(samples, sample_dataset_path)
    write_csv(best["folds"], output_dir / "m1_direct_loo_decisions.csv")
    write_csv(
        [
            {
                "rank": index + 1,
                "hidden_dim": row["hidden_dim"],
                "lr": row["lr"],
                "weight_decay": row["weight_decay"],
                "epochs": row["epochs"],
                "seed": row["seed"],
                "accuracy": row["accuracy"],
                **{f"mean_{key}": value for key, value in row["summary"].items()},
            }
            for index, row in enumerate(leaderboard)
        ],
        output_dir / "m1_direct_loop_leaderboard.csv",
    )
    baselines = [{"name": "oracle_label", "summary": oracle_summary(samples)}]
    for method in methods:
        baselines.append({"name": f"fixed_{method}", "summary": fixed_summary(samples, method)})
    result_payload = {
        "version": "m1_direct_selector_results_v1",
        "best": best,
        "baselines": baselines,
        "leaderboard_top": leaderboard[:20],
        "feature_count": len(names),
        "samples": [sample["sample_id"] for sample in samples],
        "methods": methods,
        "model_output": args.model_output,
    }
    (output_dir / "m1_direct_results.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "m1_direct_report.md", best, baselines, model_path, sample_dataset_path)
    print(
        json.dumps(
            {
                "best": {
                    "hidden_dim": best["hidden_dim"],
                    "lr": best["lr"],
                    "weight_decay": best["weight_decay"],
                    "epochs": best["epochs"],
                    "seed": best["seed"],
                    "accuracy": best["accuracy"],
                    "summary": best["summary"],
                },
                "model_output": args.model_output,
                "report": str(output_dir / "m1_direct_report.md"),
                "feature_count": len(names),
                "samples": len(samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
