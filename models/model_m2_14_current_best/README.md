# Model M2.14 Current Best

M2.14 is the current best candidate-augmented embroidery planner selector.

It extends M2.13 with one gated outline candidate:

`mask_fill_edgewalk_outline_ext_in3_rows16_p40`

This candidate uses edge-walk fill plus an inset external outline. The outline is shifted inward by 3 px to reduce off-mask stitch risk.

## Main Files

- `m2_candidate_selector_model.json` - learned M2 candidate selector weights.
- `best_current_model_m2_14_outline_selector.json` - model and experiment configuration.
- `M2_14_OUTLINE_CURRENT_BEST_RESULTS.md` - public benchmark, holdout, and incoming review results.

## Why This Is Current Best

Compared with M2.13:

- public LOO unified loss improved from `0.096830` to `0.096228`;
- public LOO coverage improved from `0.843746` to `0.855224`;
- core-to-full unified loss improved from `0.111771` to `0.111060`;
- incoming review stayed at the strong M2.13 result: `0.090709` unified loss and `0` hard_fail.

## Important Boundary

The outline branch is not a full satin-border digitizer. Direct contour outlines increased off-mask risk; M2.14 keeps outline as a hard-safe candidate only.

The next step is true stitch-style planning: fill/tatami, satin border, running detail, and underlay layers.
