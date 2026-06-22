# Model M2.30 Current Best

M2.30 is the current best candidate-augmented embroidery planner selector.

It promotes a new high-coverage candidate:

`mask_fill_edgewalk_nearestrow_rows16_p40`

This candidate keeps the existing mask-path edge-walk fill, but changes the fill-row ordering inside each component from fixed scanline order to nearest-endpoint order. The goal is to preserve coverage while reducing unnecessary jumps and trims.

## Main Files

- `best_current_model_m2_30_nearestrow_selector.json` - model and experiment configuration.
- `M2_30_NEARESTROW_CURRENT_BEST_RESULTS.md` - public benchmark and incoming review results.

## Why This Is Current Best

Compared with M2.14 on the public core-to-full holdout:

- unified loss improves from `0.111060` to `0.074071`;
- jump count improves from `10.1875` to `6.9375`;
- trim count improves from `3.7500` to `2.0625`;
- coverage improves from `0.826159` to `0.927512`;
- hard_fail remains `0`.

On incoming review:

- unified loss improves from `0.090709` to `0.084419`;
- jump count improves from `7.5` to `6.75`;
- coverage stays effectively tied: `0.856321` to `0.856344`;
- hard_fail remains `0`.

## Important Boundary

M2.30 improves path continuity and coverage through better row ordering. It is still not a full professional digitizer with learned satin, tatami, underlay, pull compensation, and color-layer sequencing.
