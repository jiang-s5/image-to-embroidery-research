# M2.73 Stitch-Type Texture Profile

Date: 2026-07-03

## Summary

M2.73 is the first production profile that safely introduces satin-like stitch texture into the selected public benchmark outputs.

It is a conservative upgrade on top of M2.72:

```text
M2.72 line skeleton fidelity
  -> safe texture candidates
  -> guarded stitch-type texture promotion
```

The goal is not to reduce jump count or unified loss at all costs. The goal is to move one step closer to professional digitizing behavior, where different regions are not all rendered as the same generic row-fill or running-stitch style.

## Why Not Promote Old Satin Candidates Directly

Earlier satin, satin-rail, dt-satin, and style-aware candidates had the right visual idea but were too unsafe globally.

Representative full-candidate results:

| Candidate | Hard Fail | Mean Loss | Mean Jump | Mean Trim | Mean Off-Mask |
|---|---:|---:|---:|---:|---:|
| old satin w8/s7 | 20 / 33 | 0.282392 | 94.88 | 17.21 | 1.3436 |
| old satin-rail w8/s7 | 20 / 33 | 0.282264 | 94.88 | 17.21 | 1.3484 |
| old dt-satin o3/i9/s7 | 20 / 33 | 0.282915 | 94.64 | 17.12 | 1.3293 |
| old style-aware | 19 / 33 | 0.236194 | 49.88 | 9.48 | 0.9872 |

That is why M2.73 does not enable these as a global branch.

## New Candidate Strategy

M2.73 starts from the currently safe fill baseline:

```text
eval mask as fill + connector mask
safe_dt adaptive fill inset
nearest endpoint row order
mask-path connectors
quantized segment validation
```

Then it tests three light texture candidates:

| Candidate | Result |
|---|---|
| outline_safeadt | rejected, 16 hard_fail |
| satinrail_safeadt_light | 0 hard_fail, candidate pool only |
| dtsatin_safeadt_light | 0 hard_fail, candidate pool only |

The light satin candidates are still worse as global outputs, so they are only allowed through a strict per-sample gate.

## Promotion Gate

A texture candidate may replace M2.72 only if:

```text
hard_fail does not appear
off-mask does not increase
visible connector does not increase
jump count does not increase
trim count does not increase
unified loss increases by at most 0.002
coverage does not decrease
strict precision drops by at most 0.01
texture score improves by at least 0.5
satin-like segment count is at least 60
```

QuickDraw and rendered text are excluded from this profile. QuickDraw already has the line-skeleton profile, and text needs its own satin/text-legibility planner rather than generic shape texture.

## Switched Samples

M2.73 switches 2 samples:

| Sample | Source | Category | Candidate | Loss Delta | Jump Delta | Trim Delta | Coverage Delta | Precision Delta | Texture Delta |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ocp_005 | Openclipart | public_domain_logo | satinrail_safeadt_light | +0.00000289 | 0 | 0 | 0 | -0.009194 | +0.970 |
| omj_005 | OpenMoji | sun | dtsatin_safeadt_light | 0 | 0 | 0 | +0.000790 | -0.004865 | +1.276 |

These are small but meaningful promotions because they add satin-like structure without increasing command-level risk.

## M2.72 vs M2.73

| Metric | M2.72 | M2.73 | Delta |
|---|---:|---:|---:|
| hard_fail | 0 | 0 | 0 |
| unified loss | 0.05506661 | 0.05506670 | +0.00000009 |
| jump count | 5.69696970 | 5.69696970 | 0 |
| trim count | 0.84848485 | 0.84848485 | 0 |
| off-mask length | 0.12120606 | 0.12120606 | 0 |
| visible connector | 0 | 0 | 0 |
| coverage | 0.97970561 | 0.97972955 | +0.00002394 |
| strict precision | 0.83696715 | 0.83654112 | -0.00042603 |
| adaptive coverage | 0.92858318 | 0.92859709 | +0.00001391 |
| adaptive precision | 0.96633306 | 0.96587603 | -0.00045703 |
| low adaptive precision samples | 0 | 0 | 0 |

Component-risk audit is effectively unchanged:

| Metric | M2.72 | M2.73 |
|---|---:|---:|
| mean risk score | 0.10425597 | 0.10425605 |
| review samples | 16 | 16 |
| worst sample | txt_005 | txt_005 |
| worst risk score | 0.28471031 | 0.28471031 |

## Interpretation

M2.73 is not a broad quality jump. It is a guarded representation upgrade:

```text
single generic stitch look
  -> selected satin-like texture where it is safe
```

This matters because professional embroidery software does not output one uniform stitch style. It mixes running, satin, fill/tatami, underlay, and color-layer sequencing.

M2.73 proves that the current system can safely admit stitch-type-specific candidates, but only in a narrow subset of images. This is useful progress, and it also exposes the next bottleneck: generic satin candidates are still too unstable to use broadly.

## Artifacts

Config:

```text
configs/best_current_model_m2_73_stitch_type_texture_profile.json
```

Selection:

```text
results/public_benchmark_v1_ext33_m2_73_stitch_type_texture_profile/
```

Candidate outputs:

```text
results/public_benchmark_v1_ext33_m2_73_texture_satinrail_safeadt_light/
results/public_benchmark_v1_ext33_m2_73_texture_dtsatin_safeadt_light/
results/public_benchmark_v1_ext33_m2_73_texture_outline_safeadt/
```

Audits:

```text
results/public_benchmark_v1_ext33_m2_73_stitch_type_texture_profile_component_audit/
results/public_benchmark_v1_ext33_m2_73_stitch_type_texture_profile_adaptive_audit/
```

## Next Direction

The next step should be a real stitch-type-aware planner, not just candidate gating:

```text
thin centerlines -> running stitch
wide contours / text strokes -> satin stitch
large regions -> fill / tatami rows
safe borders -> satin rail or dt-satin
```

The selector needs richer features:

- stroke width distribution,
- component compactness,
- contour length,
- skeleton-to-area ratio,
- region hole count,
- local edge confidence,
- expected satin pair rejection rate,
- expected precision loss.

That is the route from safe M2.73 texture gating toward a professional digitizing engine.
