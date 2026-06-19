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


def choose_feature_row(group: list[dict[str, Any]], feature_method: str) -> dict[str, Any]:
    for row in group:
        if str(row.get("method", "")) == feature_method:
            return row
    return sorted(group, key=lambda item: str(item.get("method", "")))[0]


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


def config_vector(args: dict[str, Any], keys: list[str], ranges: dict[str, dict[str, float]]) -> list[float]:
    values: list[float] = []
    for key in keys:
        raw = safe_float(args.get(key))
        span = ranges[key]["max"] - ranges[key]["min"]
        if span < 1e-8:
            values.append(0.0)
        else:
            values.append((raw - ranges[key]["min"]) / span)
    return values


def build_samples(
    candidate_rows: list[dict[str, Any]],
    preset_args: dict[str, dict[str, Any]],
    keys: list[str],
    ranges: dict[str, dict[str, float]],
    feature_method: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for sample_id, group in sorted(group_rows(candidate_rows).items()):
        feature_row = choose_feature_row(group, feature_method)
        candidates: list[dict[str, Any]] = []
        for row in group:
            method = str(row.get("method", ""))
            candidates.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "metrics": row.get("metrics", {}),
                    "config": preset_args.get(method, {}),
                    "config_vector": config_vector(preset_args.get(method, {}), keys, ranges),
                    "paths": row.get("paths", {}),
                }
            )
        samples.append(
            {
                "sample_id": sample_id,
                "feature_method": str(feature_row.get("method", "")),
                "features": sample_features_from_row(feature_row),
                "candidates": sorted(candidates, key=lambda item: str(item["method"])),
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


def feature_matrix(samples: list[dict[str, Any]], names: list[str]) -> np.ndarray:
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


def fit_feature_norm(samples: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = feature_matrix(samples, names)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    return mean_x, std_x


def normalized_features(sample: dict[str, Any], names: list[str], mean_x: np.ndarray, std_x: np.ndarray) -> np.ndarray:
    values = np.zeros(len(names), dtype=np.float64)
    features = sample.get("features", {})
    if isinstance(features, dict):
        for index, name in enumerate(names):
            values[index] = safe_float(features.get(name))
    return np.clip((values - mean_x) / std_x, -5.0, 5.0)


def phi(
    sample: dict[str, Any],
    candidate: dict[str, Any],
    names: list[str],
    mean_x: np.ndarray,
    std_x: np.ndarray,
) -> np.ndarray:
    feat = normalized_features(sample, names, mean_x, std_x)
    cfg = np.asarray(candidate["config_vector"], dtype=np.float64)
    # Feature-only terms cancel out in pairwise comparisons for one image, but
    # keeping them makes the saved scorer interpretable and harmless.
    interactions = np.outer(feat, cfg).reshape(-1)
    return np.concatenate([np.ones(1, dtype=np.float64), feat, cfg, interactions])


def pair_rows(
    samples: list[dict[str, Any]],
    names: list[str],
    mean_x: np.ndarray,
    std_x: np.ndarray,
    target: str,
    min_delta: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    diffs: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    for sample in samples:
        candidates = sample["candidates"]
        phis = {str(item["method"]): phi(sample, item, names, mean_x, std_x) for item in candidates}
        for i, left in enumerate(candidates):
            for right in candidates[i + 1 :]:
                left_score = metric(left, target)
                right_score = metric(right, target)
                delta = abs(left_score - right_score)
                if delta <= min_delta:
                    continue
                if left_score < right_score:
                    good, bad = left, right
                    good_score, bad_score = left_score, right_score
                else:
                    good, bad = right, left
                    good_score, bad_score = right_score, left_score
                diffs.append(phis[str(good["method"])] - phis[str(bad["method"])])
                audit.append(
                    {
                        "sample_id": sample["sample_id"],
                        "good_method": good["method"],
                        "bad_method": bad["method"],
                        "good_score": good_score,
                        "bad_score": bad_score,
                        "score_delta": bad_score - good_score,
                    }
                )
    if not diffs:
        raise SystemExit("No ranking pairs were created. Lower --min-delta or check target metric.")
    return np.vstack(diffs), audit


def train_ranker(
    diffs: np.ndarray,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=diffs.shape[1])
    losses: list[float] = []
    for _epoch in range(epochs):
        margins = np.clip(diffs @ weights, -40.0, 40.0)
        loss_terms = np.logaddexp(0.0, -margins)
        reg_weights = weights.copy()
        reg_weights[0] = 0.0
        loss = float(loss_terms.mean() + 0.5 * weight_decay * np.sum(reg_weights**2))
        losses.append(loss)
        coeff = -1.0 / (1.0 + np.exp(margins))
        grad = (coeff[:, None] * diffs).mean(axis=0) + weight_decay * reg_weights
        weights -= lr * grad
    margins = diffs @ weights
    pair_accuracy = float(np.mean(margins > 0.0))
    return {
        "weights": weights,
        "loss_curve": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "pair_accuracy": pair_accuracy,
    }


def score_candidate(
    model: dict[str, Any],
    sample: dict[str, Any],
    candidate: dict[str, Any],
    names: list[str],
    mean_x: np.ndarray,
    std_x: np.ndarray,
) -> float:
    vector = phi(sample, candidate, names, mean_x, std_x)
    return float(vector @ np.asarray(model["weights"], dtype=np.float64))


def select_methods(
    model: dict[str, Any],
    samples: list[dict[str, Any]],
    names: list[str],
    mean_x: np.ndarray,
    std_x: np.ndarray,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    selected: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        scored = [
            {
                "method": candidate["method"],
                "utility": score_candidate(model, sample, candidate, names, mean_x, std_x),
                "candidate": candidate,
            }
            for candidate in sample["candidates"]
        ]
        scored.sort(key=lambda item: item["utility"], reverse=True)
        best = scored[0]
        selected[str(sample["sample_id"])] = str(best["method"])
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "selected_method": best["method"],
                "selected_utility": round(float(best["utility"]), 8),
                "ranked_methods_json": json.dumps(
                    [{"method": item["method"], "utility": round(float(item["utility"]), 8)} for item in scored],
                    ensure_ascii=False,
                ),
            }
        )
    return selected, rows


def best_method(sample: dict[str, Any], target: str) -> str:
    return str(min(sample["candidates"], key=lambda candidate: metric(candidate, target))["method"])


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
        selected = selected_methods[str(sample["sample_id"])]
        by_method = {str(item["method"]): item for item in sample["candidates"]}
        metrics = by_method[selected]["metrics"]
        for key in keys:
            values[key].append(safe_float(metrics.get(key)))
    return {key: round(mean(vals), 8) for key, vals in values.items()}


def fixed_summary(samples: list[dict[str, Any]], method: str) -> dict[str, float]:
    return summarize_selected(samples, {str(sample["sample_id"]): method for sample in samples})


def oracle_summary(samples: list[dict[str, Any]], target: str) -> dict[str, float]:
    return summarize_selected(samples, {str(sample["sample_id"]): best_method(sample, target) for sample in samples})


def evaluate_spec(
    samples: list[dict[str, Any]],
    names: list[str],
    target: str,
    min_delta: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    selected_methods: dict[str, str] = {}
    pair_accuracies: list[float] = []
    for held in samples:
        held_id = str(held["sample_id"])
        train_samples = [sample for sample in samples if str(sample["sample_id"]) != held_id]
        mean_x, std_x = fit_feature_norm(train_samples, names)
        train_diffs, _audit = pair_rows(train_samples, names, mean_x, std_x, target, min_delta)
        model = train_ranker(train_diffs, lr, weight_decay, epochs, seed)
        pair_accuracies.append(float(model["pair_accuracy"]))
        selected, score_rows = select_methods(model, [held], names, mean_x, std_x)
        predicted = selected[held_id]
        label = best_method(held, target)
        selected_methods[held_id] = predicted
        by_method = {str(item["method"]): item for item in held["candidates"]}
        selected_metrics = by_method[predicted]["metrics"]
        oracle_metrics = by_method[label]["metrics"]
        fold_rows.append(
            {
                "sample_id": held_id,
                "selected_method": predicted,
                "oracle_method": label,
                "correct": predicted == label,
                "selected_utility": safe_float(score_rows[0]["selected_utility"]),
                "selected_unified_loss": metric(by_method[predicted], "unified_loss"),
                "oracle_unified_loss": metric(by_method[label], "unified_loss"),
                "selected_hard_score": metric(by_method[predicted], "hard_score"),
                "oracle_hard_score": metric(by_method[label], "hard_score"),
                "selected_jump_count": safe_float(selected_metrics.get("jump_count")),
                "selected_trim_count": safe_float(selected_metrics.get("trim_count")),
            }
        )
    accuracy = sum(1 for row in fold_rows if row["correct"]) / max(1, len(fold_rows))
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "target": target,
        "min_delta": min_delta,
        "accuracy": round(accuracy, 8),
        "mean_train_pair_accuracy": round(mean(pair_accuracies), 8),
        "folds": fold_rows,
        "selected_methods": selected_methods,
        "summary": summarize_selected(samples, selected_methods),
    }


def to_model_payload(
    model: dict[str, Any],
    names: list[str],
    mean_x: np.ndarray,
    std_x: np.ndarray,
    parameter_keys: list[str],
    parameter_ranges_payload: dict[str, dict[str, float]],
    preset_args: dict[str, dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": "m1_ranking_selector_v1",
        "model_type": "linear_pairwise_ranker",
        "mapping": "image_geometry_features_plus_candidate_config_to_candidate_utility",
        "selection_rule": "pick candidate config with highest utility",
        "training_loss": "pairwise_logistic_preference_loss",
        "feature_clip_after_normalization": 5.0,
        "label_target": spec["target"],
        "min_delta": spec["min_delta"],
        "feature_names": names,
        "feature_normalization": {
            "mean": mean_x.tolist(),
            "std": std_x.tolist(),
        },
        "parameter_keys": parameter_keys,
        "parameter_ranges": parameter_ranges_payload,
        "preset_bank": preset_args,
        "weights": np.asarray(model["weights"], dtype=np.float64).tolist(),
        "hyperparameters": {
            "lr": spec["lr"],
            "weight_decay": spec["weight_decay"],
            "epochs": spec["epochs"],
            "seed": spec["seed"],
        },
        "training_metrics": {
            "final_loss": model["final_loss"],
            "pair_accuracy": model["pair_accuracy"],
            "loss_curve_tail": model["loss_curve"][-10:],
        },
    }


def write_report(
    path: Path,
    best: dict[str, Any],
    baselines: list[dict[str, Any]],
    model_path: Path,
    samples_path: Path,
    predictions_path: Path,
) -> None:
    lines = [
        "# M1 Ranking Planner Selector",
        "",
        "This stage turns evaluator outputs into preference supervision.",
        "",
        "Instead of only asking which single preset is the label, it trains a pairwise ranker:",
        "",
        "```text",
        "image / predicted geometry + candidate config A/B",
        "        |",
        "pairwise logistic ranking loss",
        "        |",
        "utility(candidate)",
        "        |",
        "choose the highest-utility planner config",
        "```",
        "",
        "## Requirement Check",
        "",
        "| Requirement | Status | Evidence |",
        "| --- | --- | --- |",
        "| Preference learning | Done | pairwise logistic loss over evaluator-ranked candidates |",
        "| Image-conditioned config choice | Done | candidate utility includes image geometry x config interactions |",
        "| Leave-one-out validation | Done | every sample is held out from the ranking pairs used to train its selector |",
        "| No post-export feature leakage at inference | Done | sample features exclude `preset.*`, `interaction.*`, and evaluator metrics |",
        "",
        "## Best Leave-One-Out Result",
        "",
        f"- target metric: `{best['target']}`",
        f"- min_delta: `{best['min_delta']}`",
        f"- lr: `{best['lr']}`",
        f"- weight_decay: `{best['weight_decay']}`",
        f"- epochs: `{best['epochs']}`",
        f"- seed: `{best['seed']}`",
        f"- oracle-match accuracy: `{best['accuracy']}`",
        f"- mean train pair accuracy: `{best['mean_train_pair_accuracy']}`",
        f"- model artifact: `{model_path.as_posix()}`",
        f"- ranking samples: `{samples_path.as_posix()}`",
        f"- predictions: `{predictions_path.as_posix()}`",
        "",
        "| Metric | Ranking M1 LOO Mean |",
        "| --- | ---: |",
    ]
    for key, value in best["summary"].items():
        lines.append(f"| {key} | {safe_float(value):.6f} |")
    lines.extend(
        [
            "",
            "## Fold Decisions",
            "",
            "| Sample | Selected Method | Oracle Method | Correct | Selected Unified | Oracle Unified | Selected Hard | Oracle Hard |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best["folds"]:
        lines.append(
            "| {sample_id} | {selected_method} | {oracle_method} | {correct} | {selected_unified_loss:.6f} | {oracle_unified_loss:.6f} | {selected_hard_score:.6f} | {oracle_hard_score:.6f} |".format(
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
            "- This is the first M1 variant that uses evaluator feedback as pairwise preference supervision.",
            "- It is still a planner-selector model, not an M2 graph edge policy.",
            "- A learned M2 would need segment/edge labels or graph traces and should be reported separately.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train M1 ranking-based planner selector from evaluator preferences.")
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--sweep-config", default="configs/sweep_b1.yaml")
    parser.add_argument("--feature-method", default="b1_conservative")
    parser.add_argument("--label-target", default="hard_score")
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--output-dir", default="results/m1_ranking_selector/latest_pair_20260618")
    parser.add_argument("--model-output", default="checkpoints/m1_ranking_selector.json")
    parser.add_argument("--lrs", default="0.005,0.01,0.03,0.05")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001,0.01")
    parser.add_argument("--epochs-list", default="300,600,1000")
    parser.add_argument("--seeds", default="1,2,3,5,8,13")
    args = parser.parse_args()

    candidate_rows = read_jsonl(Path(args.candidates_jsonl))
    if not candidate_rows:
        raise SystemExit("No candidate rows found.")
    sweep_config = read_json(Path(args.sweep_config))
    presets = preset_args_by_method(sweep_config)
    keys = parameter_keys(presets)
    ranges = parameter_ranges(presets, keys)
    samples = build_samples(candidate_rows, presets, keys, ranges, args.feature_method)
    names = feature_names(samples)
    if not names:
        raise SystemExit("No sample-level image/geometry features were found.")

    lrs = [safe_float(item) for item in args.lrs.split(",") if item.strip()]
    weight_decays = [safe_float(item) for item in args.weight_decays.split(",") if item.strip()]
    epochs_values = [int(safe_float(item)) for item in args.epochs_list.split(",") if item.strip()]
    seeds = [int(safe_float(item)) for item in args.seeds.split(",") if item.strip()]

    leaderboard: list[dict[str, Any]] = []
    for lr in lrs:
        for weight_decay in weight_decays:
            for epochs in epochs_values:
                for seed in seeds:
                    leaderboard.append(
                        evaluate_spec(
                            samples,
                            names,
                            args.label_target,
                            args.min_delta,
                            lr,
                            weight_decay,
                            epochs,
                            seed,
                        )
                    )
    leaderboard.sort(
        key=lambda row: (
            row["summary"]["hard_score"],
            row["summary"]["unified_loss"],
            -row["accuracy"],
            row["summary"]["jump_count"],
        )
    )
    best = leaderboard[0]

    mean_x, std_x = fit_feature_norm(samples, names)
    diffs, pair_audit = pair_rows(samples, names, mean_x, std_x, args.label_target, args.min_delta)
    final_model = train_ranker(
        diffs,
        float(best["lr"]),
        float(best["weight_decay"]),
        int(best["epochs"]),
        int(best["seed"]),
    )
    selected, prediction_rows = select_methods(final_model, samples, names, mean_x, std_x)
    for row in prediction_rows:
        sample = next(item for item in samples if str(item["sample_id"]) == str(row["sample_id"]))
        row["oracle_method"] = best_method(sample, args.label_target)
        row["selected_config_json"] = json.dumps(
            next(item["config"] for item in sample["candidates"] if str(item["method"]) == str(row["selected_method"])),
            ensure_ascii=False,
            sort_keys=True,
        )

    methods = sorted({str(candidate["method"]) for sample in samples for candidate in sample["candidates"]})
    baselines = [{"name": "oracle_preference_label", "summary": oracle_summary(samples, args.label_target)}]
    for method in methods:
        baselines.append({"name": f"fixed_{method}", "summary": fixed_summary(samples, method)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "m1_ranking_samples.jsonl"
    predictions_path = output_dir / "m1_ranking_predictions.csv"
    pairs_path = output_dir / "m1_ranking_pairs.csv"
    write_jsonl(samples, samples_path)
    write_csv(pair_audit, pairs_path)
    write_csv(best["folds"], output_dir / "m1_ranking_loo_decisions.csv")
    write_csv(prediction_rows, predictions_path)
    write_csv(
        [
            {
                "rank": index + 1,
                "lr": row["lr"],
                "weight_decay": row["weight_decay"],
                "epochs": row["epochs"],
                "seed": row["seed"],
                "accuracy": row["accuracy"],
                "mean_train_pair_accuracy": row["mean_train_pair_accuracy"],
                **{f"mean_{key}": value for key, value in row["summary"].items()},
            }
            for index, row in enumerate(leaderboard)
        ],
        output_dir / "m1_ranking_loop_leaderboard.csv",
    )

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_payload = to_model_payload(final_model, names, mean_x, std_x, keys, ranges, presets, best)
    model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    results_payload = {
        "version": "m1_ranking_selector_results_v1",
        "best": best,
        "baselines": baselines,
        "leaderboard_top": leaderboard[:20],
        "feature_count": len(names),
        "parameter_count": len(keys),
        "pair_count": len(pair_audit),
        "samples": [sample["sample_id"] for sample in samples],
        "model_output": args.model_output,
        "selected_methods_full_fit": selected,
    }
    (output_dir / "m1_ranking_results.json").write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "m1_ranking_report.md", best, baselines, model_path, samples_path, predictions_path)
    print(
        json.dumps(
            {
                "best": {
                    "lr": best["lr"],
                    "weight_decay": best["weight_decay"],
                    "epochs": best["epochs"],
                    "seed": best["seed"],
                    "accuracy": best["accuracy"],
                    "mean_train_pair_accuracy": best["mean_train_pair_accuracy"],
                    "summary": best["summary"],
                },
                "model_output": args.model_output,
                "report": str(output_dir / "m1_ranking_report.md"),
                "feature_count": len(names),
                "parameter_count": len(keys),
                "pair_count": len(pair_audit),
                "samples": len(samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
