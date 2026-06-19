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


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        features = row.get("features", {})
        if isinstance(features, dict):
            names.update(str(key) for key in features)
    return sorted(names)


def matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(rows), len(names)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        features = row.get("features", {})
        if not isinstance(features, dict):
            continue
        for key, value in features.items():
            if key in index:
                x[row_idx, index[key]] = safe_float(value)
    return x


def fit_norm(rows: list[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = matrix(rows, names)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    return mean_x, std_x


def transform(rows: list[dict[str, Any]], names: list[str], mean_x: np.ndarray, std_x: np.ndarray) -> np.ndarray:
    return np.clip((matrix(rows, names) - mean_x) / std_x, -5.0, 5.0)


def group_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("decision_id", ""))].append(row)
    return dict(grouped)


def build_pair_diffs(rows: list[dict[str, Any]], names: list[str], mean_x: np.ndarray, std_x: np.ndarray) -> tuple[np.ndarray, int]:
    groups = group_by_decision(rows)
    diffs: list[np.ndarray] = []
    skipped = 0
    for group in groups.values():
        positives = [row for row in group if int(row.get("label", 0)) == 1]
        negatives = [row for row in group if int(row.get("label", 0)) == 0]
        if len(positives) != 1 or not negatives:
            skipped += 1
            continue
        pos_x = transform(positives, names, mean_x, std_x)[0]
        neg_x = transform(negatives, names, mean_x, std_x)
        for row in neg_x:
            diffs.append(pos_x - row)
    if not diffs:
        raise SystemExit("No pairwise M2 training differences were created.")
    return np.vstack(diffs), skipped


def train_ranker(diffs: np.ndarray, lr: float, weight_decay: float, epochs: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=diffs.shape[1])
    losses: list[float] = []
    for _epoch in range(epochs):
        margins = np.clip(diffs @ weights, -40.0, 40.0)
        reg = weights.copy()
        loss = float(np.logaddexp(0.0, -margins).mean() + 0.5 * weight_decay * np.sum(reg**2))
        losses.append(loss)
        coeff = -1.0 / (1.0 + np.exp(margins))
        grad = (coeff[:, None] * diffs).mean(axis=0) + weight_decay * reg
        weights -= lr * grad
    pair_accuracy = float(np.mean((diffs @ weights) > 0.0))
    return {
        "weights": weights,
        "final_loss": losses[-1] if losses else 0.0,
        "pair_accuracy": pair_accuracy,
        "loss_curve": losses,
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    names: list[str],
    mean_x: np.ndarray,
    std_x: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    groups = group_by_decision(rows)
    decisions: list[dict[str, Any]] = []
    correct = 0
    top3 = 0
    ranks: list[int] = []
    for decision_id, group in sorted(groups.items()):
        positives = [row for row in group if int(row.get("label", 0)) == 1]
        if len(positives) != 1:
            continue
        x = transform(group, names, mean_x, std_x)
        utilities = x @ weights
        ranked = sorted(zip(group, utilities.tolist()), key=lambda item: item[1], reverse=True)
        selected = ranked[0][0]
        positive_node = int(positives[0].get("candidate_node"))
        rank = next((idx + 1 for idx, item in enumerate(ranked) if int(item[0].get("candidate_node")) == positive_node), len(ranked))
        is_correct = int(selected.get("candidate_node")) == positive_node
        correct += 1 if is_correct else 0
        top3 += 1 if rank <= 3 else 0
        ranks.append(rank)
        decisions.append(
            {
                "sample_id": selected.get("sample_id", ""),
                "decision_id": decision_id,
                "selected_node": int(selected.get("candidate_node")),
                "oracle_node": positive_node,
                "correct": bool(is_correct),
                "oracle_rank": rank,
                "candidate_count": len(group),
                "selected_utility": round(float(ranked[0][1]), 8),
            }
        )
    n = max(1, len(decisions))
    return (
        {
            "decision_count": float(len(decisions)),
            "accuracy": correct / n,
            "top3_accuracy": top3 / n,
            "mean_oracle_rank": mean(ranks) if ranks else 0.0,
        },
        decisions,
    )


def evaluate_spec(
    rows: list[dict[str, Any]],
    names: list[str],
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    sample_ids = sorted({str(row.get("sample_id", "")) for row in rows})
    fold_rows: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    pair_accuracies: list[float] = []
    for held_id in sample_ids:
        train_rows = [row for row in rows if str(row.get("sample_id", "")) != held_id]
        held_rows = [row for row in rows if str(row.get("sample_id", "")) == held_id]
        mean_x, std_x = fit_norm(train_rows, names)
        diffs, skipped = build_pair_diffs(train_rows, names, mean_x, std_x)
        model = train_ranker(diffs, lr, weight_decay, epochs, seed)
        pair_accuracies.append(float(model["pair_accuracy"]))
        metrics, decisions = evaluate_rows(held_rows, names, mean_x, std_x, np.asarray(model["weights"], dtype=np.float64))
        all_decisions.extend(decisions)
        fold_rows.append(
            {
                "held_sample": held_id,
                "decision_count": int(metrics["decision_count"]),
                "accuracy": round(metrics["accuracy"], 8),
                "top3_accuracy": round(metrics["top3_accuracy"], 8),
                "mean_oracle_rank": round(metrics["mean_oracle_rank"], 8),
                "train_pair_accuracy": round(float(model["pair_accuracy"]), 8),
                "skipped_train_decisions": skipped,
            }
        )
    total_decisions = sum(int(row["decision_count"]) for row in fold_rows)
    weighted_accuracy = sum(float(row["accuracy"]) * int(row["decision_count"]) for row in fold_rows) / max(1, total_decisions)
    weighted_top3 = sum(float(row["top3_accuracy"]) * int(row["decision_count"]) for row in fold_rows) / max(1, total_decisions)
    weighted_rank = sum(float(row["mean_oracle_rank"]) * int(row["decision_count"]) for row in fold_rows) / max(1, total_decisions)
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "seed": seed,
        "folds": fold_rows,
        "decisions": all_decisions,
        "summary": {
            "decision_count": total_decisions,
            "accuracy": round(weighted_accuracy, 8),
            "top3_accuracy": round(weighted_top3, 8),
            "mean_oracle_rank": round(weighted_rank, 8),
            "mean_train_pair_accuracy": round(mean(pair_accuracies), 8),
        },
    }


def train_full_model(
    rows: list[dict[str, Any]],
    names: list[str],
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    mean_x, std_x = fit_norm(rows, names)
    diffs, skipped = build_pair_diffs(rows, names, mean_x, std_x)
    model = train_ranker(diffs, lr, weight_decay, epochs, seed)
    return {
        "weights": model["weights"],
        "mean_x": mean_x,
        "std_x": std_x,
        "pair_accuracy": model["pair_accuracy"],
        "final_loss": model["final_loss"],
        "loss_curve_tail": model["loss_curve"][-10:],
        "skipped_decisions": skipped,
    }


def write_report(path: Path, best: dict[str, Any], model_path: Path, dataset_path: Path) -> None:
    lines = [
        "# M2 Edge Policy Prototype",
        "",
        "This is the first learned M2 stage. It trains an edge-level ranker from `graph_tsp_trace.json` decisions.",
        "",
        "The target is not a final DST replacement yet. The target is local routing imitation:",
        "",
        "```text",
        "current graph node + remaining candidate node",
        "        |",
        "M2 edge utility",
        "        |",
        "choose next node",
        "```",
        "",
        "## Best Leave-One-Sample-Out Result",
        "",
        f"- lr: `{best['lr']}`",
        f"- weight_decay: `{best['weight_decay']}`",
        f"- epochs: `{best['epochs']}`",
        f"- seed: `{best['seed']}`",
        f"- model artifact: `{model_path.as_posix()}`",
        f"- dataset: `{dataset_path.as_posix()}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in best["summary"].items():
        lines.append(f"| {key} | {safe_float(value):.6f} |")
    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            "| Held Sample | Decisions | Accuracy | Top-3 Accuracy | Mean Oracle Rank | Train Pair Acc |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best["folds"]:
        lines.append(
            "| {held_sample} | {decision_count} | {accuracy:.6f} | {top3_accuracy:.6f} | {mean_oracle_rank:.3f} | {train_pair_accuracy:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Stage Boundary",
            "",
            "- Done: graph trace -> edge dataset -> learned edge utility -> leave-one-sample-out validation.",
            "- Not done: replacing Graph-TSP inside final DST generation.",
            "- Next: integrate this utility as an optional reranker in the graph planner and compare generated DST metrics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an M2 edge-level ranking policy from graph trace decisions.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", default="results/m2_edge_policy/latest_pair_20260619")
    parser.add_argument("--model-output", default="checkpoints/m2_edge_policy.json")
    parser.add_argument("--lrs", default="0.001,0.003,0.005,0.01,0.03")
    parser.add_argument("--weight-decays", default="0,0.0001,0.001,0.01")
    parser.add_argument("--epochs-list", default="200,500,1000")
    parser.add_argument("--seeds", default="1,2,3,5,8")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.dataset_jsonl))
    if not rows:
        raise SystemExit("No M2 rows found.")
    names = feature_names(rows)
    if not names:
        raise SystemExit("No M2 features found.")

    lrs = [safe_float(item) for item in args.lrs.split(",") if item.strip()]
    weight_decays = [safe_float(item) for item in args.weight_decays.split(",") if item.strip()]
    epochs_values = [int(safe_float(item)) for item in args.epochs_list.split(",") if item.strip()]
    seeds = [int(safe_float(item)) for item in args.seeds.split(",") if item.strip()]

    leaderboard: list[dict[str, Any]] = []
    for lr in lrs:
        for weight_decay in weight_decays:
            for epochs in epochs_values:
                for seed in seeds:
                    leaderboard.append(evaluate_spec(rows, names, lr, weight_decay, epochs, seed))
    leaderboard.sort(
        key=lambda row: (
            -row["summary"]["accuracy"],
            -row["summary"]["top3_accuracy"],
            row["summary"]["mean_oracle_rank"],
            -row["summary"]["mean_train_pair_accuracy"],
        )
    )
    best = leaderboard[0]
    full = train_full_model(rows, names, float(best["lr"]), float(best["weight_decay"]), int(best["epochs"]), int(best["seed"]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(best["folds"], output_dir / "m2_edge_policy_loo_folds.csv")
    write_csv(best["decisions"], output_dir / "m2_edge_policy_loo_decisions.csv")
    write_csv(
        [
            {
                "rank": index + 1,
                "lr": row["lr"],
                "weight_decay": row["weight_decay"],
                "epochs": row["epochs"],
                "seed": row["seed"],
                **{f"mean_{key}": value for key, value in row["summary"].items()},
            }
            for index, row in enumerate(leaderboard)
        ],
        output_dir / "m2_edge_policy_leaderboard.csv",
    )
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "version": "m2_edge_policy_v1",
        "model_type": "linear_pairwise_edge_ranker",
        "mapping": "edge_features_to_next_node_utility",
        "training_loss": "pairwise_logistic_loss",
        "feature_names": names,
        "feature_clip_after_normalization": 5.0,
        "normalization": {
            "mean": full["mean_x"].tolist(),
            "std": full["std_x"].tolist(),
        },
        "weights": np.asarray(full["weights"], dtype=np.float64).tolist(),
        "hyperparameters": {
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
            "epochs": best["epochs"],
            "seed": best["seed"],
        },
        "training_metrics": {
            "pair_accuracy": full["pair_accuracy"],
            "final_loss": full["final_loss"],
            "loss_curve_tail": full["loss_curve_tail"],
            "skipped_decisions": full["skipped_decisions"],
        },
    }
    model_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    results_payload = {
        "version": "m2_edge_policy_results_v1",
        "best": {key: value for key, value in best.items() if key != "decisions"},
        "leaderboard_top": [{key: value for key, value in row.items() if key != "decisions"} for row in leaderboard[:20]],
        "feature_count": len(names),
        "row_count": len(rows),
        "sample_count": len({str(row.get("sample_id", "")) for row in rows}),
        "model_output": str(model_path),
    }
    (output_dir / "m2_edge_policy_results.json").write_text(json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "m2_edge_policy_report.md", best, model_path, Path(args.dataset_jsonl))
    print(
        json.dumps(
            {
                "best": {
                    "lr": best["lr"],
                    "weight_decay": best["weight_decay"],
                    "epochs": best["epochs"],
                    "seed": best["seed"],
                    "summary": best["summary"],
                },
                "model_output": str(model_path),
                "report": str(output_dir / "m2_edge_policy_report.md"),
                "feature_count": len(names),
                "row_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
