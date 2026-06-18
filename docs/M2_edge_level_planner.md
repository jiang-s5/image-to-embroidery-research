# M2 Edge-Level Relation Planner (Design Spec)

## 1. Motivation
M1-config regressor learns global planner parameters but does not explicitly model local stitching decisions. M2 aims to move from global parameter prediction to **edge-level structured routing**, where each stitch transition is a learned decision.

---

## 2. Problem Definition
We reformulate embroidery generation as a graph decision problem:

- Nodes: stitchable points / segment endpoints
- Edges: possible transitions between nodes
- Goal: construct valid path with minimal execution + visual risk cost

Formally:

G = (V, E)

At each step t:

s_t = (v_i, v_j, context)

a_t ∈ {connect, jump, trim, color_change, stop}

---

## 3. Model Architecture

### 3.1 Input Representation
- Image features (CNN/ViT)
- Geometry features (mask, boundary, centerline)
- Segment graph embedding

### 3.2 Edge Encoder
Each candidate edge e_ij is encoded as:

- spatial distance
- direction vector
- mask compliance score
- continuity score
- stitch type compatibility

---

### 3.3 Relation Network
A GNN / Transformer over graph edges:

h_e = f(edge_features, node_context, global_context)

Outputs:

P(a_t | e_ij, context)

---

## 4. Decision Policy
At each step:

1. sample candidate edges
2. compute edge score
3. select action

Policy:

π(e_ij, a_t) → probability of transition

---

## 5. Training Strategy

### 5.1 Supervised Pretraining (from M1 oracle)
Use M1-config + sweep-generated DST traces:

- reconstruct optimal transitions
- label edges: connect/jump/trim

Loss:

L_supervised = cross_entropy(edge_action)

---

### 5.2 Consistency Regularization
Ensure path validity:

- no illegal long stitches
- mask compliance
- continuity preservation

L_reg = violation penalties

---

## 6. Integration with Executability System
M2 is evaluated using:

- ExecScore (jump, trim, illegal stitches)
- VisualRiskScore (off-mask, visible connectors)

Final objective:

L_total = L_supervised + λ * L_reg + μ * VisualRisk

---

## 7. Key Innovation

M2 replaces:

- global planner parameters (M1)

with:

- edge-level learned routing policy

This shifts the problem from:

> parameter optimization → structured decision making

---

## 8. Expected Benefits

- finer control of stitching paths
- reduces hidden artifacts (visible connectors)
- improves generalization across complex geometries
- enables future RL fine-tuning

---

## 9. Relationship to Existing System

| Module | Role |
|--------|------|
| B1 | heuristic baseline |
| M1 | global parameter regression |
| M2 | edge-level routing policy |

---

## 10. Future Extensions

- M3: reinforcement learning refinement
- differentiable routing objective
- hierarchical stitching policy

---
