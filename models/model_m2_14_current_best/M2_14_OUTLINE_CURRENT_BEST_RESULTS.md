# M2.14 Inset-Outline Gated Current Best Results

Date: 2026-06-22

M2.14 extends M2.13 by adding one more candidate:

`mask_fill_edgewalk_outline_ext_in3_rows16_p40`

This candidate combines edge-walk fill with an inset external outline. The outline is shifted inward by 3 px before contour stitching. This avoids the main failure of direct boundary outline, which placed stitches too close to the mask edge and increased off-mask/visible-connector risk.

## Public Benchmark v1 ext33 Leave-One-Out

| Model | Candidates | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| M2.13 edge-walk selector | 13 | 0 | 0.096830 | 8.939394 | 3.454545 | 0.843746 |
| M2.14 outline selector | 14 | 0 | 0.096228 | 8.848485 | 3.484848 | 0.855224 |

M2.14 gives a small public LOO gain. Coverage improves, jump count slightly improves, and hard_fail remains zero.

## Core-to-Full Holdout

Train split: core, 17 samples.

Test split: full, 16 samples.

| Model | Hard Fail | Oracle Match | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Off-Mask mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.13 edge-walk selector | 0 | 12/16 | 0.111771 | 10.3125 | 3.6875 | 0.826159 | 0.187262 |
| M2.14 outline selector | 0 | 13/16 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.187262 |

The holdout improvement is small but positive. Coverage is unchanged from M2.13 on this split, while unified loss and jump count improve slightly.

## Incoming Review Holdout

Dataset: `datasets/incoming_review_eval_v1`

Status: evaluation-only, not used for training.

| Model | Samples | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| M2.13 edge-walk selector | 4 | 0 | 0.090709 | 7.5000 | 4.7500 | 0.856321 |
| M2.14 outline selector | 4 | 0 | 0.090709 | 7.5000 | 4.7500 | 0.856321 |

Chosen candidates on incoming review:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 2 |
| mask_fill_edgewalk_rows16_p40 | 2 |

M2.14 does not regress incoming review. The selector does not choose the outline candidate on incoming samples, which is the desired hard-safe behavior.

## Outline Diagnostic

Standalone outline candidates on public ext33:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Off-Mask mm | Mean Visible Connectors | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| edgewalk rows16_p40 | 20/33 | 0.266298 | 90.696970 | 1.313558 | 0.000000 | 0.999722 |
| direct outline all contours | 22/33 | 0.370494 | 97.242424 | 9.413852 | 1.757576 | 1.000000 |
| direct outline external only | 22/33 | 0.357667 | 95.606061 | 6.527079 | 1.090909 | 0.999926 |
| inset outline external in2 | 20/33 | 0.294022 | 96.151515 | 1.647724 | 0.060606 | 0.999812 |
| inset outline external in3 | 20/33 | 0.290617 | 97.303030 | 1.524758 | 0.030303 | 0.999743 |

The diagnostic result is important: raw boundary outlines are unsafe. Inset outline is safer, but still not strong enough to be used as a universal production branch. It is useful as a gated candidate only.

## Interpretation

M2.14 is the current recommended best, but the improvement over M2.13 is modest.

The main research value of this step is not that outline solved professional satin borders. It showed:

- direct contour outlines create off-mask risk;
- inward-shifted outlines are safer;
- outline should remain a candidate selected by hard-safe ranking, not a mandatory post-process.

## Next Step

The next meaningful upgrade should move from geometric outline tracing to true stitch-style planning:

1. classify regions into fill, satin border, running detail, and underlay;
2. generate satin-like borders as paired rails or short perpendicular columns rather than one contour polyline;
3. preserve face/cartoon details with separate detail layers;
4. add a render-back realism score for border thickness and fill direction.
