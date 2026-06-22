# Model M2.34 Current Best

M2.34 is the current best candidate-augmented embroidery planner selector.

It promotes two high-coverage fill-inset candidates:

```text
mask_fill_edgewalk_nearestrow_inset1_rows16_p40
mask_fill_edgewalk_nearestrow_inset2_rows16_p40
```

These candidates keep nearest-endpoint row ordering from M2.30, but erode the fill-row mask by 1px or 2px before generating horizontal fill rows. Connector validation still uses the evaluation mask.

## Main Files

- `best_current_model_m2_34_fill_inset_selector.json` - model and experiment configuration.
- `calibrated_config.json` - calibrated selector settings selected on public core split.
- `M2_34_FILL_INSET_CURRENT_BEST_RESULTS.md` - public benchmark and incoming review results.

## Why This Is Current Best

On the public ext33 leave-one-out validation:

- hard_fail remains `0`;
- unified loss improves from M2.30 `0.068519` to `0.051174`;
- jump count improves from `6.636364` to `5.636364`;
- trim count improves from `1.848485` to `0.424242`;
- off-mask stitch length improves from `0.105806mm` to `0.082527mm`;
- coverage improves from `0.911108` to `0.968881`.

Compared with M2.30 on the public core-to-full holdout:

- unified loss improves from `0.074071` to `0.051748`;
- jump count improves from `6.9375` to `5.7500`;
- trim count improves from `2.0625` to `0.4375`;
- off-mask stitch length improves from `0.187256mm` to `0.061937mm`;
- coverage improves from `0.927512` to `0.951650`;
- hard_fail remains `0`.

On incoming review:

- unified loss improves from `0.084419` to `0.078290`;
- trim count improves from `4.7500` to `1.2500`;
- coverage improves from `0.856344` to `0.995549`;
- off-mask remains `0`;
- hard_fail remains `0`;
- jump count increases from `6.7500` to `8.5000`, so incoming jump behavior remains a watch item.

## Important Boundary

M2.34 improves executable DST path quality through better high-coverage fill geometry. It is still not a full professional digitizer with learned satin, tatami, underlay, pull compensation, color-layer sequencing, or realistic 3D thread rendering.
