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
   - closed and continuous outlines;
   - realistic fill/tatami direction;
   - satin-like border behavior;
   - fewer broken fragments;
   - recognizable cartoon/person features.

## Known Limitation

The model has improved continuity and jump safety, but it still does not fully reproduce a professional digitizer's satin/fill style. The next research step is stronger segment-level graph planning and stitch-style generation.
