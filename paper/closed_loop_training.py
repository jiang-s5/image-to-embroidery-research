"""
Closed-Loop Optimization System for M1 + M2
-------------------------------------------
This module implements the real optimization loop:

image → M1/M2 → DST → evaluation → learning signal → update

This is the core step from "system paper" → "learning system".
"""

import json
from pathlib import Path


# -----------------------------
# Unified Objective Function
# -----------------------------

def compute_score(metrics: dict, lam: float = 1.0) -> float:
    """
    Unified objective used for both M1 and M2 optimization.
    """
    exec_score = (
        metrics.get("jump_count", 0)
        + metrics.get("trim_count", 0)
        + metrics.get("illegal_stitch_count", 0)
    )

    visual_risk = (
        metrics.get("off_mask_stitch_length_mm", 0.0)
        + metrics.get("visible_connector_count", 0)
    )

    return exec_score + lam * visual_risk


# -----------------------------
# M1 Update Signal (Config Level)
# -----------------------------

def update_m1(dataset, model, lr=1e-3):
    """
    Dataset: (image_features, best_config)
    Model: M1-config regressor
    """
    for sample in dataset:
        x = sample["features"]
        y = sample["best_config"]

        pred = model.forward(x)
        loss = ((pred - y) ** 2).mean()

        model.backward(loss, lr=lr)


# -----------------------------
# M2 Update Signal (Edge Level)
# -----------------------------

def update_m2(graph_batch, policy_model, lr=1e-4):
    """
    graph_batch: list of (graph, optimal_edges)
    """
    for graph, labels in graph_batch:
        logits = policy_model.forward(graph)

        loss = 0
        for e, label in zip(logits, labels):
            loss += -label * e  # simplified supervised loss

        policy_model.backward(loss, lr=lr)


# -----------------------------
# Closed Loop Training Step
# -----------------------------

def training_step(data, m1_model, m2_model, evaluator):
    """
    One full loop:
    1. predict config / edges
    2. generate DST
    3. evaluate
    4. update models
    """

    m1_outputs = []
    m2_outputs = []

    for sample in data:
        img = sample["image"]
        feat = sample["features"]

        # M1 prediction
        config = m1_model.forward(feat)

        # M2 prediction (placeholder graph)
        graph = sample["graph"]
        edges = m2_model.forward(graph)

        # DST generation (external pipeline)
        dst = sample["generator"](config, edges)

        # evaluation
        metrics = evaluator(dst)
        score = compute_score(metrics)

        m1_outputs.append((feat, config, score))
        m2_outputs.append((graph, edges, score))

    # update models
    update_m1(m1_outputs, m1_model)
    update_m2(m2_outputs, m2_model)

    return m1_outputs, m2_outputs


# -----------------------------
# Convergence Check
# -----------------------------

def check_convergence(history, eps=1e-3):
    if len(history) < 5:
        return False
    return abs(history[-1] - history[-2]) < eps
