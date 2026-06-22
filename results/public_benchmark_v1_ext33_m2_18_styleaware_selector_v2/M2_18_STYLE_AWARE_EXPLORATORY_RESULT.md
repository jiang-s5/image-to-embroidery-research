# M2.18 Component-Level Style-Aware Routing Exploratory Result

Date: 2026-06-22

M2.18 tested the first component-level stitch-style routing branch:

`mask_fill_edgewalk_styleaware_r008_d45`

This is a direct follow-up to M2.17. Instead of appending satin-like rails after fill stitching, M2.18 first classifies each connected mask component as either:

- `running`: thin or line-like component, traced through the skeleton;
- `fill`: larger component, stitched with the existing mask-fill edge-walk branch.

The running/fill decision uses component-level geometry:

```text
skeleton_ratio = skeleton_pixels / mask_pixels
max_distance_px = max distance-transform value inside the component

running if:
  skeleton_ratio >= 0.08
  or max_distance_px <= 4.5
```

The candidate selector also received explicit candidate-type features such as `candidate_is_mask_fill`, `candidate_is_style_aware`, `candidate_is_outline`, `candidate_is_satin`, and related coverage/line-score interactions.

## Standalone Public Benchmark v1 ext33

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| edgewalk fill rows16 p40 | 20/33 | 0.266298 | 90.696970 | 16.424242 | 1.313558 | 0.999722 | 0.720092 |
| DT-satin o3 i9 s7 | 20/33 | 0.282915 | 94.636364 | 17.121212 | 1.329273 | 0.999726 | 0.719269 |
| style-aware r0.08 d4.5 | 19/33 | 0.236194 | 49.878788 | 9.484848 | 0.987197 | 0.983542 | 0.764977 |

This is the strongest result in the M2.15-M2.18 fill-family experiments. Component-level style routing almost halves jump count compared with dense edge-walk fill while keeping high coverage.

## Public Benchmark v1 ext33 Leave-One-Out

| Model | Candidates | Feature Count | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 14 | 19 | 0 | 0.096228 | 8.848485 | 3.484848 | 0.855224 |
| M2.18 style-aware selector v2 | 15 | 28 | 0 | 0.095790 | 8.939394 | 3.212121 | 0.856572 |

Chosen candidates in M2.18 leave-one-out:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 11 |
| mask_fill_edgewalk_styleaware_r008_d45 | 6 |
| skeleton_c20_m4 | 4 |
| skeleton_c30_m8 | 3 |
| skeleton_c30_m4 | 3 |
| skeleton_c30_m12 | 2 |
| skeleton_c30_m20 | 2 |
| mask_fill_edgewalk_outline_ext_in3_rows16_p40 | 1 |
| mask_fill_edgewalk_rows16_p40 | 1 |

M2.18 now actually uses the style-aware candidate in leave-one-out. The improvement is still small and comes with a slight jump increase, but the branch is no longer purely decorative.

## Core-to-Full Holdout

| Model | Hard Fail | Oracle Match | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Off-Mask mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 0 | 13/16 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.187262 |
| M2.18 style-aware selector v2 | 0 | 12/16 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.187262 |

Core-to-full is materially unchanged from M2.14. Oracle match is slightly worse.

## Incoming Review Holdout

| Model | Samples | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Off-Mask mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 4 | 0 | 0.090709 | 7.5000 | 4.7500 | 0.856321 | 0.000000 |
| M2.18 style-aware selector v2 | 4 | 0 | 0.094245 | 8.5000 | 3.5000 | 0.927795 | 0.123875 |

M2.18 increases coverage and reduces trim count on incoming review, but unified loss, jump count, and off-mask length regress. It selected the style-aware branch on 2/4 incoming samples, which is too aggressive for promotion.

Chosen candidates on incoming review:

| Candidate | Count |
|---|---:|
| mask_fill_edgewalk_styleaware_r008_d45 | 2 |
| auto_evalrepair_r10_c20 | 1 |
| mask_fill_edgewalk_outline_ext_in3_rows16_p40 | 1 |

## Decision

M2.18 is **not promoted**.

M2.14 remains the current best.

## Research Interpretation

M2.18 is more important than M2.15-M2.17 conceptually. It confirms the core hypothesis that stitch style must be decided before route generation, not appended afterward.

However, the current component classifier is still a hand-tuned geometry rule. It can over-assign the style-aware fill branch to incoming samples where a safer auto/outline branch is better. The next step is not another rail variant. The next step is a safer style gate:

1. keep component-level style routing as a candidate;
2. train or tune a hard-safe selector specifically for style-aware candidates;
3. add off-mask and incoming-review penalties to the style-aware selection gate;
4. separate style classification from candidate score regression;
5. eventually replace the hand rule with learned stitch-style labels from DST-derived supervision.
