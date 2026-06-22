# M2.17 Distance-Transform Satin-Rail Exploratory Result

Date: 2026-06-22

M2.17 tested a distance-transform rail candidate:

`mask_fill_edgewalk_dtsatin_o3_i9_s7`

Unlike M2.15 and M2.16, which derived satin-like pairs from centroid-normal projections, this candidate derives rail pairs from distance-transform level sets inside the mask:

```text
outer rail: distance-to-boundary >= 3 px
inner rail: distance-to-boundary >= 9 px
pairing: nearest inner-rail point for each sampled outer-rail point
sequence: outer_0 -> inner_0 -> outer_1 -> inner_1 -> ...
```

The goal was to make satin-like border strokes follow a stable mask-internal safety distance instead of projecting every boundary point toward the component centroid.

## Public Benchmark v1 ext33 Leave-One-Out

| Model | Candidates | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 14 | 0 | 0.096228 | 8.848485 | 3.484848 | 0.855224 | 0.837897 |
| M2.17 DT-satin selector | 15 | 0 | 0.095790 | 8.939394 | 3.212121 | 0.856572 | 0.837471 |

M2.17 gives a very small leave-one-out unified-loss improvement, but this is not strong evidence of a better satin candidate. The selector did not choose the DT-satin candidate in leave-one-out. The small change comes from the extra candidate altering the fitted ridge selector, not from direct DT-satin usage.

Chosen candidates in M2.17 leave-one-out:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 9 |
| mask_fill_edgewalk_rows16_p40 | 7 |
| skeleton_c20_m4 | 6 |
| skeleton_c30_m8 | 3 |
| skeleton_c30_m4 | 3 |
| skeleton_c30_m20 | 2 |
| skeleton_c30_m12 | 2 |
| mask_fill_edgewalk_outline_ext_in3_rows16_p40 | 1 |
| mask_fill_edgewalk_dtsatin_o3_i9_s7 | 0 |

## Core-to-Full Holdout

| Model | Hard Fail | Oracle Match | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Off-Mask mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 0 | 13/16 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.187262 |
| M2.17 DT-satin selector | 0 | 12/16 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.187262 |

Core-to-full is materially unchanged from M2.14, while oracle match is slightly worse. This is not enough to promote M2.17.

## Incoming Review Holdout

The DT-satin candidate was also generated on the incoming review set and then evaluated through the M2.17 selector.

| Model | Samples | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 4 | 0 | 0.090709 | 7.5000 | 4.7500 | 0.856321 |
| M2.17 DT-satin selector | 4 | 0 | 0.090709 | 7.5000 | 4.7500 | 0.856321 |

Chosen candidates on incoming review:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 2 |
| mask_fill_edgewalk_rows16_p40 | 2 |
| mask_fill_edgewalk_dtsatin_o3_i9_s7 | 0 |

The selector safely avoids DT-satin on incoming review. There is no regression, but also no improvement.

## Standalone DT-Satin Candidates

Standalone public ext33 candidates:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| DT-satin o3 i9 s7 pair18 | 20/33 | 0.282915 | 94.636364 | 17.121212 | 1.329273 | 0.999726 | 0.719269 |
| DT-satin o3 i8 s14 pair14 max90 | 20/33 | 0.281441 | 94.424242 | 17.090909 | 1.329273 | 0.999726 | 0.719283 |

Standalone incoming review:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|
| DT-satin o3 i9 s7 pair18 | 1/4 | 0.183752 | 33.0000 | 6.5000 | 0.997118 |

## Decision

M2.17 is **not promoted**.

M2.14 remains the current best. M2.17 is useful because it validates the next technical bottleneck: simply appending satin-like rails after fill stitching does not produce professional digitizing behavior. It increases command count and does not get selected by the hard-safe planner.

## Research Interpretation

Distance-transform rails are more geometrically meaningful than centroid-normal rails, but the current implementation still treats satin as an additive after-pass on top of fill rows. That is the wrong abstraction for professional-looking embroidery.

The next step should not be another rail variant on top of the same fill branch. It should separate stitch styles before route generation:

1. classify regions or components into fill, satin border, running detail, and underlay;
2. generate satin rails only for components where border stitching is the primary style;
3. replace local fill rows near the border instead of appending border rails after fill;
4. add a render-back border-thickness/edge-coverage metric so real satin quality is measured directly;
5. keep DT rails as a geometry primitive, not as a universal candidate.

