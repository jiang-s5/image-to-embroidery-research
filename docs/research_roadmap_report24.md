# Research Roadmap From Report 24

This note records the actionable direction extracted from `deep-research-report (24).md`.

## Main Judgment

The project should not be positioned as a generic image-to-DST demo. The stronger research framing is:

> Weakly supervised, relation-aware planning for machine-executable embroidery.

The repository already has DST-derived labels, multi-task geometry prediction, vector-continuity supervision, command-level executability metrics, and planner ablations. The next research risk is not model size; it is weak evidence. The key gaps are:

- external benchmark coverage is still too small;
- retrieval controls must remain strict;
- current retrieval is image-level parameter tuning, not edge-level route decision;
- relation/planner logic is still heuristic, not a learned edge-level planner;
- visual tradeoff metrics are needed to detect visible stitch connectors.

## Immediate Engineering Actions

1. Keep A0/A1/A2/A2-fixed/LOO/random/shuffled/external controls as the default planner benchmark.
2. Treat fixed conservative planner parameters as the current proven improvement.
3. Treat image-level retrieval as unproven until it beats fixed/random/shuffled controls on held-out data.
4. Add visual tradeoff metrics to every executability report:
   - `off_mask_stitch_length_mm`
   - `off_mask_stitch_count`
   - `visible_connector_count`
   - `visible_connector_length_mm`
5. Use train/validation-style parameter sweeps for fixed planner presets, rather than tuning on a single mini-demo table.

## Method Direction

The strongest next method step is edge-level retrieval plus a learned relation head:

```text
Input image
  -> canonicalization
  -> geometry/stitch-type prediction
  -> segment graph
  -> candidate transition edges
  -> edge-level retrieval memory
  -> learned relation head
  -> hard-constrained planner
  -> DST/PES export
  -> executability + visual tradeoff evaluation
```

The edge memory item should describe a transition, not a whole image:

```text
(segment_i exit, segment_j entry, geometry, same_color, stitch_pair, local_context, actual_decision)
```

Possible labels:

- `connect`
- `jump`
- `trim`
- `color_change`
- `not_next`
- `expected_jump_mm`
- pairwise path order

## Benchmark Direction

Mini demo remains a smoke test only. Main claims should eventually use:

- Pilot-50: protocol check;
- Release-200: first publishable benchmark;
- Full-500: stable extended benchmark.

Each sample should include metadata:

- `image_id`
- `source_type`
- `license`
- `attribution`
- `split`
- `difficulty`
- `category`
- `num_dominant_colors`
- `has_text`
- `thin_structure_score`
- `hole_count`
- `edge_density`
- `background_complexity`
- `canonicalized_path`
- `releaseable`
- `retrieval_index_eligible`

## Evaluation Direction

Command-level metrics remain necessary:

- parse success;
- jump count/path;
- trim count;
- illegal long stitches;
- high-risk jumps;
- max stitch/jump length;
- lock-tack-like count.

They are not sufficient. Every main table should also include visual/tradeoff checks:

- off-mask stitch length;
- visible connector count;
- render-back contact sheet;
- failure examples;
- human A/B preference for a held-out subset.

## Paper Claim Discipline

Current evidence supports:

> Conservative continuity-aware planner parameters improve command-level executability.

Current evidence does not yet support:

> Image-level retrieval generalizes as an adaptive planner.

The latter requires retrieval to beat fixed/random/shuffled controls on an external holdout.
