# Weakly Supervised Constrained Graph Learning for Machine Embroidery Path Planning

## (M1–M2 Unified Method Paper Draft)

---

# Abstract

We propose a unified framework for machine embroidery path planning under weak supervision, consisting of two key components: (1) M1-config regressor, which learns continuous planner parameters from image/geometry representations; and (2) M2 edge-level relation planner, which models embroidery generation as a structured graph decision problem. 

Unlike prior heuristic or retrieval-based approaches, our method introduces a constrained optimization formulation that jointly minimizes execution cost and visual artifacts such as off-mask stitching and visible connector structures.

We evaluate the system using a full DST/PES execution pipeline and propose a dual-objective evaluation metric combining executability and visual risk.

---

# 1. Introduction

Machine embroidery generation requires converting visual designs into executable stitch commands (DST/PES format). This process is constrained by:

- machine execution limits (jump, trim, stitch length)
- spatial constraints (fabric mask, regions)
- visual quality constraints (connector visibility, path continuity)

Existing approaches rely on:

- rule-based planners (heuristics)
- parameter tuning (fixed strategies)
- retrieval-based priors (image-level similarity)

However, these methods fail to generalize to complex geometry and often produce visually invalid stitching artifacts.

We address this by proposing a two-stage learning system:

- M1: learns global planner configuration
- M2: learns edge-level routing decisions

---

# 2. Problem Formulation

Given an input image I, we aim to generate a valid embroidery program P such that:

P = {s1, s2, ..., sn}

subject to constraints:

- executable stitching constraints
- mask adherence constraints
- minimal visual artifact constraints

We define objective function:

minimize:

J(P) = ExecScore(P) + λ * VisualRisk(P)

---

# 3. System Overview

Our system consists of three modules:

## 3.1 Geometry Encoder
Extracts structured representation:

- mask
- boundary
- centerline
- density map
- stitch direction field

## 3.2 M1 Config Regressor
Learns mapping:

I → θ

where θ includes:

- max_stitch_mm
- connect_near_mm
- trim_threshold
- graph_connectivity_bias
- component constraints

## 3.3 M2 Edge-Level Planner
Models embroidery as graph G = (V, E)

At each step t:

s_t = (v_i, v_j, context)

a_t ∈ {connect, jump, trim, color_change, stop}

---

# 4. M1: Continuous Parameter Learning

## 4.1 Oracle Construction
We construct training data via sweep-based oracle:

θ* = argminθ J(DST(θ))

Dataset:

(I, θ*)

## 4.2 Model
We use regression model:

θ̂ = f_φ(I)

where f_φ is MLP/CNN-based encoder.

## 4.3 Loss Function

L_M1 = ||θ̂ - θ*||²

Optionally augmented with ranking loss:

L_rank = max(0, J(θ̂) - J(θ*))

---

# 5. M2: Edge-Level Relation Planner

## 5.1 Graph Construction
We construct graph:

G = (V, E)

Nodes represent stitchable segments.
Edges represent candidate transitions.

## 5.2 Edge Representation
Each edge e_ij includes:

- spatial distance
- direction alignment
- mask compliance
- stitch feasibility
- local geometry context

## 5.3 Policy Learning
We learn policy:

π(e_ij, a_t | context)

where actions include:

- connect
- jump
- trim
- color_change

## 5.4 Loss Function

L_M2 = L_action + λ1 * L_exec + λ2 * L_visual

---

# 6. Evaluation Metrics

We propose dual evaluation system:

## 6.1 Executability Metrics

- jump_count
- trim_count
- illegal_long_stitch
- parse_success_rate

## 6.2 Visual Risk Metrics

- off_mask_stitch_length_mm
- off_mask_stitch_count
- visible_connector_count
- visible_connector_length_mm

## 6.3 Unified Score

J(P) = ExecScore + λ * VisualRisk

---

# 7. Experimental Design

We evaluate:

- B1 heuristic planner
- M1-config regressor
- M2 edge-level planner

Ablation settings:

- retrieval vs fixed vs learned
- mask-aware vs mask-unaware
- graph vs non-graph routing

---

# 8. Key Findings (Current Stage)

1. Fixed conservative planner parameters outperform retrieval-based priors.
2. Visual artifacts (connector leakage) correlate weakly with jump reduction.
3. Continuous parameter regression (M1) improves stability over discrete selection.
4. Edge-level modeling (M2) provides finer control over stitching legality.

---

# 9. Limitations

- current RL-based optimization is not included due to instability
- dataset size is limited (mini-demo scale)
- full real-world digitizer dataset is unavailable

---

# 10. Future Work

- reinforcement learning stitching policy (M3)
- differentiable planner optimization
- large-scale embroidery benchmark (Pilot-50 / 200 / 500)
- human-in-the-loop preference refinement

---

# 11. Conclusion

We present a structured learning framework for machine embroidery generation that moves from heuristic planning to continuous parameter learning (M1) and edge-level routing (M2). The system introduces a unified execution-aware and visual-risk-aware optimization objective, forming a foundation for future learning-based embroidery generation systems.

---
