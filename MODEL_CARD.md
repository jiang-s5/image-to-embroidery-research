# Model Card

## Research Objective

Image-to-embroidery stitch planning with machine-executable DST/PES output.

The model does not directly emit raw DST commands. It predicts geometry and planning maps that are converted into embroidery paths by post-processing and a path planner.

## Model Lineage

| Model | Main Change | Status |
| --- | --- | --- |
| model1 | Basic mask/density/direction supervision | early baseline |
| model2 | 11-channel rich geometry labels | baseline geometry |
| model3 | Added entry/exit endpoint heatmaps | useful cartoon behavior |
| model5/model6 | Added path-order supervision | path-order probe |
| model7/model8 | Geometry/planner separation experiments | architecture probe |
| model9 | Hard-standard geometry/path experiments | intermediate |
| model10 | Added vector-continuity supervision | continuity breakthrough |
| model13 | Multiformat vector-continuity training | current promoted checkpoint |
| model14B | Gentle geometry-continuity continuation | local experiment, not promoted |

## Current Promoted Checkpoint

```text
model13_multiformat_all_vector_continuity_20260514/
best_model13_multiformat_all_vector_continuity.pt
```

Summary metrics from the local run:

```text
epochs: 24
best_epoch: 24
best_score: 3.7382206823
mask_iou: 0.9533605923
boundary_f1: 0.6999207076
centerline_f1: 0.4802198375
stitch_type_miou: 0.1895635666
endpoint_mae: 0.1178026602
path_order_mae: 0.2845349568
segment_order_mae: 0.1954096120
stitch_trace_f1: 0.5709184910
near_connect_f1: 0.4392108341
continuity_mae: 0.1324329144
```

## Current Planner Selector

The current best planning package is M2.14:

```text
models/model_m2_14_current_best/
configs/best_current_model_m2_14_outline_selector.json
```

M2.14 keeps model13 as the geometry checkpoint and improves the post-processing planner selection layer. It adds a gated inset-outline candidate to the M2.13 hard-safe selector.

Summary results:

| Evaluation | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Coverage |
| --- | ---: | ---: | ---: | ---: |
| Public ext33 LOO | 0 | 0.096228 | 8.848485 | 0.855224 |
| Core-to-full holdout | 0 | 0.111060 | 10.1875 | 0.826159 |
| Incoming review holdout | 0 | 0.090709 | 7.5000 | 0.856321 |

The edge-walk fill branch remains the main fill improvement. The inset-outline branch is a gated candidate: direct outlines were unsafe, while inset outlines can help selected public/holdout samples without regressing incoming review.

## Later Experiment: M2.15 Satin-Column Probe

M2.15 tested a first satin-column candidate, but it was not promoted. It improved public leave-one-out metrics slightly, but core-to-full and incoming-review evaluations were materially unchanged.

The result suggests that true satin behavior needs paired inner/outer rails and stitch-style classification rather than simple centroid-normal columns.

## Later Experiment: M2.16 Satin-Rail Probe

M2.16 tested continuous outer/inner rail zigzag stitching, but it was not promoted. It tied M2.15 on public LOO and did not improve core-to-full or incoming-review results.

The result suggests that paired rails should be derived from distance-transform level sets instead of centroid-normal projections.

## Later Experiment: M2.17 Distance-Transform Satin-Rail Probe

M2.17 implemented distance-transform rail pairing, deriving outer and inner satin-like rails from mask-internal distance-to-boundary level sets. It was not promoted. Public leave-one-out improved only marginally, core-to-full was unchanged, and the selector did not directly choose the DT-satin candidate.

This result narrows the next direction: distance-transform rails are useful geometry primitives, but satin cannot be an additive after-pass on top of fill rows. The next upgrade should classify stitch style first, then generate fill/running/satin routes separately.

## Later Experiment: M2.18 Component-Level Style-Aware Routing

M2.18 implemented the first component-level style gate: thin or line-like connected components are routed through skeleton running stitches, while broader components continue through the edge-walk fill branch. It also added explicit candidate-type features to the M2 selector.

This was not promoted. Standalone style-aware fill substantially reduced jump and trim counts compared with previous fill candidates, but the learned selector over-selected it on incoming review, increasing unified loss and off-mask length. The result supports stitch-style classification as the next research direction, but the style gate must become hard-safe before replacing M2.14.

## Later Experiment: M2.19-M2.21 Hard-Safe Candidate Gates

M2.19-M2.21 added optional hard gates for style-aware and mask-fill candidate families. These gates reject candidates based on off-mask stitch length, jump count, trim count, stitch precision, and coverage.

They were not promoted. Strict gates removed off-mask risk but over-pruned high-coverage candidates, while relaxed gates still underperformed M2.14. The selector code keeps these gates as research controls; the next step is a calibrated two-stage selector rather than fixed thresholds alone.

## Later Experiment: M2.22-M2.25 Calibrated Selector

M2.22-M2.25 added an interpretable calibrated selector that ranks candidates by unified loss plus coverage, precision, jump, trim, off-mask, and visible-connector terms. M2.22 slightly improved incoming review loss over M2.14, but public full coverage dropped too far. M2.25 added a hard coverage floor and recovered coverage, but regressed unified loss, jump, trim, and incoming review quality.

These were not promoted. The next selector should use adaptive coverage targets rather than a fixed coverage floor.

## Later Experiment: M2.26-M2.29 Adaptive Coverage Selector

M2.26-M2.29 tested source-aware adaptive coverage targets. M2.26 lowered public-full loss from M2.14's `0.111060` to `0.095392`, with lower jump and trim counts, but coverage dropped from `0.826159` to `0.734277`. M2.28-M2.29 added adaptive coverage floors and recovered coverage to `0.844804`, but loss regressed to `0.122887`.

These were not promoted. The result supports adaptive coverage as a useful direction, but the next version should use a smooth coverage-risk curve rather than a hard floor.

## Later Experiment: model14B

```text
model14B_gentle_continue_geometry_20260514/
best_model14B_gentle_continue_geometry.pt
```

Model14B was completed but not promoted. Its own summary says model13 remains the main best checkpoint. It is still useful as a local experiment for continuity/order behavior.

## Output Channels

The current model family predicts geometry/planner channels such as:

- mask;
- density;
- axis x/y and axis confidence;
- boundary;
- centerline;
- stitch type;
- entry and exit endpoint heatmaps;
- path/segment order;
- stitch trace;
- near-connect map;
- jump endpoint map.

## Evaluation Standard

Evaluation must include three layers:

1. Dense prediction:
   - mask IoU;
   - boundary F1;
   - centerline F1;
   - stitch type mIoU;
   - endpoint MAE;
   - path-order MAE.

2. Machine-executable DST metrics:
   - illegal long stitch count;
   - high-risk jump count;
   - maximum stitch length;
   - maximum jump length;
   - trim count;
   - color changes;
   - continuity score.

3. Visual embroidery realism:
   - off-mask stitch length;
   - visible connector count;
   - closed and continuous outlines;
   - realistic fill/tatami direction;
   - satin-like border behavior;
   - fewer broken fragments;
   - recognizable cartoon/person features.

## Planner-Only Ablation Path

The promoted checkpoint remains `model13`. Recent planner work is evaluated as a post-processing ablation, not as a new model checkpoint:

- A0: geometry planner + vector-continuity planner;
- A1: relation-aware transition cost from `configs/relation_planner.yaml`;
- A2: A1 plus retrieval planner priors from `datasets/mini_demo/retrieval_planner_index.json` or a larger case bank;
- A2-fixed: A1 plus the conservative planner parameters observed in A2, without retrieval;
- A2-controls: leave-one-out, external-index, random-prior, and shuffled-prior variants for overfit checks;
- A3: future beam-search ordering over top-k candidate transitions.

Use `tools/run_planner_ablation.py` and `tools/eval_executability.py` to compare these variants before deciding whether to train a relation head. Pass a `--mask` to `tools/eval_executability.py` so lower jump/trim counts are checked against `off_mask_stitch_length_mm` and `visible_connector_count`. If `A2-fixed`, random-prior, or shuffled-prior performs like `A2`, the improvement should be attributed to conservative planner parameters rather than retrieval-specific generalization.

The current mini-demo control result supports this conservative interpretation: the fixed-prior planner reproduces the command-level jump/trim improvements seen in retrieval variants. This is useful, but it means retrieval should remain a hypothesis until it beats fixed/random/shuffled controls on a held-out set.

Use `tools/sweep_fixed_planner_params.py` to tune fixed planner presets with a train/validation-style split. A preset should not be promoted only because it reduces jumps on the same examples used to choose the parameters; it should also avoid visible connector growth on validation samples.

## Known Limitation

The model has improved continuity and jump safety, but it still does not fully reproduce a professional digitizer's satin/fill style. Lower jump/trim counts can also hide visible connector artifacts, so mask-based visual metrics and render-back visual checks from `tools/make_planner_visual_report.py` should accompany command-level metrics. The next research step is stronger segment-level graph planning, edge-level relation learning, and stitch-style generation.
