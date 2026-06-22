# M2.13 Edge-Walk Fill Current Best Results

Date: 2026-06-22

M2.13 extends M2.12 with a mask-internal edge-walk fill candidate:

`mask_fill_edgewalk_rows16_p40`

The key change is that row-to-row fill transitions no longer rely only on straight connectors. When a straight connector would leave the foreground mask, the generator searches for a short foreground-only path and stitches that path as an internal connector. This is closer to real digitizing behavior, where transitions are hidden inside filled regions instead of turning every row transition into a jump.

## Public Benchmark v1 ext33 Leave-One-Out

| Model | Candidates | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| M2.12 mask-fill selector | 12 | 0 | 0.098180 | 8.727273 | 4.333333 | 0.780257 |
| M2.13 edge-walk selector | 13 | 0 | 0.096830 | 8.939394 | 3.454545 | 0.843746 |

M2.13 improves public LOO unified loss and coverage. Jump count is slightly higher than M2.12, but trim count is lower and coverage is much better.

## Core-to-Full Holdout

Train split: core, 17 samples.

Test split: full, 16 samples.

| Model | Hard Fail | Oracle Match | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Off-Mask mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.12 mask-fill selector | 0 | 13/16 | 0.111759 | 10.2500 | 4.5000 | 0.746555 | 0.000000 |
| M2.13 edge-walk selector | 0 | 12/16 | 0.111771 | 10.3125 | 3.6875 | 0.826159 | 0.187262 |

The core-to-full result is a tradeoff: executable loss is essentially unchanged, hard_fail stays at zero, coverage improves sharply, trim count drops, and off-mask stitch length remains small but nonzero.

## Incoming Review Holdout

Dataset: `datasets/incoming_review_eval_v1`

Status: evaluation-only, not used for training.

| Model | Samples | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| M2.12 mask-fill selector | 4 | 0 | 0.123580 | 11.5000 | 4.7500 | 0.720063 |
| M2.13 edge-walk selector | 4 | 0 | 0.090709 | 7.5000 | 4.7500 | 0.856321 |

Chosen candidates on incoming review:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 2 |
| mask_fill_edgewalk_rows16_p40 | 2 |

This is the strongest evidence for promotion: the edge-walk candidate is selected on two incoming review samples and improves unified loss, jump count, and coverage without introducing hard_fail or visible connectors.

## Edge-Walk Candidate Diagnostic

Standalone fill candidates on public ext33:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|
| naive rows16_c6 | 31/33 | 0.309077 | 148.060606 | 21.787879 | 0.999656 |
| edgewalk rows16_p24 | 26/33 | 0.307985 | 115.757576 | 21.787879 | 0.999714 |
| edgewalk rows16_p40 | 20/33 | 0.266298 | 90.696970 | 16.424242 | 0.999722 |
| edgewalk rows20_p24 | 26/33 | 0.295382 | 97.636364 | 18.212121 | 0.997461 |

Edge-walk does not make dense fill universally safe, but it is a real improvement over naive row fill. It turns part of the dense-fill branch from a pure negative diagnostic into a usable candidate for some images.

## Selector Choice Distribution

Public LOO:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 12 |
| mask_fill_edgewalk_rows16_p40 | 5 |
| mask_fill_rows16_c6 | 1 |
| skeleton_c20_m4 | 5 |
| skeleton_c30_m4 | 4 |
| skeleton_c30_m8 | 2 |
| skeleton_c30_m12 | 2 |
| skeleton_c30_m20 | 2 |

Core-to-full test:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 6 |
| mask_fill_edgewalk_rows16_p40 | 3 |
| skeleton_c20_m4 | 1 |
| skeleton_c30_m4 | 1 |
| skeleton_c30_m8 | 2 |
| skeleton_c30_m12 | 1 |
| skeleton_c30_m20 | 2 |

## Interpretation

M2.13 is the current recommended best.

Compared with M2.12, it is the first version that improves not only command-level safety but also coverage/fill behavior. It still does not reproduce a professional digitizer's full satin/fill/underlay stack, but it is moving in the right direction: dense fill is now an optional, gated candidate that the selector can safely use when it helps.

## Remaining Gap

M2.13 still has three important limitations:

- edge-walk fill can reduce precision because internal connector paths may over-stitch parts of the foreground;
- it does not classify regions into satin border, tatami fill, running detail, and underlay;
- it is still deterministic candidate generation plus learned selection, not a fully learned stitch-style generator.

The next step should be segment-aware stitch-style planning:

1. classify components/regions into fill, satin border, and running detail;
2. generate border/outline layers separately from fill layers;
3. use component-level entry/exit planning for dense fill;
4. add a render-back realism score for fill direction, border thickness, and feature preservation.
