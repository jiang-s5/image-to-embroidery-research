# Model M2.13 Current Best

M2.13 is the current best candidate-augmented embroidery planner selector.

It extends M2.12 with one new fill candidate:

`mask_fill_edgewalk_rows16_p40`

This candidate connects fill rows through foreground-only edge-walk paths instead of using only straight row connectors or row-level jumps.

## Main Files

- `m2_candidate_selector_model.json` - learned M2 candidate selector weights.
- `best_current_model_m2_13_edgewalk_selector.json` - model and experiment configuration.
- `M2_13_EDGEWALK_CURRENT_BEST_RESULTS.md` - public benchmark, holdout, and incoming review results.

## Why This Is Current Best

Compared with M2.12:

- public LOO unified loss improved from `0.098180` to `0.096830`;
- public LOO coverage improved from `0.780257` to `0.843746`;
- incoming review unified loss improved from `0.123580` to `0.090709`;
- incoming review coverage improved from `0.720063` to `0.856321`;
- hard_fail remained `0`.

## Important Boundary

M2.13 is still not a commercial digitizer. It improves dense fill coverage by safely using edge-walk fill when appropriate, but it does not yet model satin borders, underlay, color-layer order, or professional fill direction.

The next step is segment-aware stitch-style planning.
