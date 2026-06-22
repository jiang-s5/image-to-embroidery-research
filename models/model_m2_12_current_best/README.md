# Model M2.12 Current Best

M2.12 is the current safest candidate-augmented embroidery planner selector.

It extends M2.10 with one experimental dense fill candidate:

`mask_fill_rows16_c6`

The selector still uses hard-safe filtering, so hard_fail outputs are excluded whenever an executable candidate exists.

## Main Files

- `m2_candidate_selector_model.json` - learned M2 candidate selector weights.
- `best_current_model_m2_12_maskfill_selector.json` - model and experiment configuration.
- `M2_12_MASKFILL_CURRENT_BEST_RESULTS.md` - public benchmark, holdout, and incoming review results.

## Why This Is Current Best

Compared with M2.10 on public benchmark v1 ext33 leave-one-out:

- unified loss improved from `0.104648` to `0.098180`;
- jump count improved from `9.121212` to `8.727273`;
- hard_fail remained `0`.

On core-to-full and incoming review holdouts, M2.12 does not regress from M2.10.

## Important Boundary

M2.12 is not a commercial digitizer and does not yet produce fully professional satin/fill embroidery.

The dense mask-fill candidate reaches near-complete coverage, but usually creates too many row-level jumps/trims. It is included as a diagnostic candidate and selected only when safe.

The next research step is segment-level fill/satin planning with border underlay, color-layer ordering, and component-level entry/exit continuity.
