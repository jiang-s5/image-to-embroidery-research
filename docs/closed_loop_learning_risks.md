# Closed-Loop Learning Risks

This note defines the current boundary between the implemented M1 planner selectors and the future M2 learned graph policy.

## Current Status

Implemented:

- M0.9 candidate scorer: image/geometry + candidate config -> score.
- Strict M1 direct selector: image/geometry -> preset config.
- M1 continuous config regressor: image/geometry -> numeric planner config -> DST evaluation.
- M1 ranking selector: image/geometry + candidate config -> utility from pairwise evaluator preferences.
- Deterministic Graph-TSP planner: segment/polyline graph -> route order through handcrafted edge costs.

Not implemented yet:

- learned M2 edge policy;
- GNN route model;
- reinforcement-learning policy update;
- joint M1/M2 closed-loop co-training.

## Why The Distinction Matters

The current loop can generate, export, evaluate, and use the evaluator to train M1 variants. That is a real learning loop for planner selection.

It is not yet a full structured policy learner because the graph route itself is still chosen by deterministic costs, not by a learned edge model.

## Credit Assignment

When a generated DST is bad, the error may come from several layers:

- geometry prediction: mask, centerline, endpoint, stitch type;
- M1 config prediction: thresholds, graph penalties, component limits;
- graph extraction: fragmented or missing nodes;
- deterministic planner: wrong edge costs or TSP ordering;
- exporter: command conversion, jump/trim thresholds, max stitch constraints.

M2 should only be trained after graph traces are reliable enough to tell which edge choices were good or bad.

## Reward Hacking Risks

Lower jump and trim counts can be misleading. A planner may reduce jumps by adding visible stitch connectors across blank fabric.

Every planner-learning report should include:

- `jump_count`;
- `trim_count`;
- `stitch_count`;
- `stitch_path_mm`;
- `max_stitch_mm`;
- `off_mask_stitch_length_mm`;
- `visible_connector_count`;
- render-back preview review.

Do not promote a model only because `jump_count` or `trim_count` decreases.

## Stable M1 Promotion Rule

Promote an M1 variant only if it improves on leave-one-out or holdout samples without increasing visual risk:

```text
selected config
  improves hard_score or unified_loss
  and does not inflate visible_connector_count
  and does not inflate off_mask_stitch_length_mm
  and keeps render-back previews visually clean
```

With the current 4-sample benchmark:

- M1 continuous config regressor is the best practical M1 result.
- M1 ranking selector is implemented, but should be treated as a diagnostic because it underperforms on leave-one-out hard score.

## Future M2 Entry Criteria

Before implementing learned M2, collect graph traces with:

- node geometry;
- edge costs;
- selected route;
- off-mask edge risk;
- visible connector risk;
- jump/trim decision;
- final evaluator metrics.

Then train an edge policy with constraints:

```text
edge features
  -> edge utility
  -> constrained route construction
  -> DST export
  -> evaluator
```

Minimum constraints:

- mask-safe edge filter;
- maximum visible connector budget;
- maximum off-mask edge length;
- route connectivity check;
- no duplicate segment visits unless explicitly allowed.

This makes M2 a constrained graph policy rather than an unconstrained image model.
