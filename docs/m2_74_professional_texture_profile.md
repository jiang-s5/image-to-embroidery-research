# M2.74 Professional Texture Profile

Date: 2026-07-03

## Summary

M2.74 is a texture-quality step on top of M2.73. It accepts a limited amount of extra routing cost when the output gains satin-like texture, coverage, and precision without adding hard failures, visible connectors, or off-mask stitch length.

This profile exists because professional embroidery software is not optimized only for the fewest jumps. A digitized design often uses extra layers for satin borders, dense fill, underlay, and texture. If the selector only minimizes jump count and unified loss, it rejects many outputs that look more like real embroidery.

The new positioning is:

| Profile | Purpose |
|---|---|
| M2.73 | strict no-regression texture gate |
| M2.74 | professional texture profile with controlled jump/loss tradeoff |

## What Changed

M2.74 keeps the M2.72 base and M2.73 texture candidates, then adds two lighter satin-rail probes:

| Candidate | Result |
|---|---|
| satinrail_w4_s18_i4_cap40 | tested, candidate pool |
| satinrail_w5_s16_i5_cap45 | tested, candidate pool |

The promoted switches still all come from `dtsatin_safeadt_light`. The new probes are useful as negative evidence: making satin-rail lighter reduces global risk but does not yet produce more safe per-sample wins under the current gate.

## Promotion Gate

M2.74 relaxes the M2.73 no-regression rule in a controlled way:

```text
hard_fail must remain absent
off-mask stitch length must not increase
visible connector count must not increase
jump increase <= 8
trim increase <= 3
unified loss increase <= 0.030
coverage may drop by at most 0.015
strict precision may drop by at most 0.040
texture score gain >= 0.35
satin-like segment count >= 35
```

QuickDraw and rendered text remain excluded. QuickDraw needs line-domain running stitches, and text needs a dedicated satin/text-legibility planner.

## Switched Samples

M2.74 switches 4 samples:

| Sample | Source | Category | Candidate | Loss Delta | Jump Delta | Trim Delta | Coverage Delta | Precision Delta | Texture Delta |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ocp_005 | Openclipart | public_domain_logo | dtsatin_safeadt_light | +0.00000289 | 0 | 0 | 0 | -0.017393 | +1.000 |
| omj_001 | OpenMoji | heart | dtsatin_safeadt_light | +0.01842164 | +8 | -8 | +0.044481 | +0.015922 | +0.806 |
| omj_003 | OpenMoji | butterfly | dtsatin_safeadt_light | +0.02556037 | +3 | 0 | -0.001258 | +0.033672 | +1.142 |
| omj_005 | OpenMoji | sun | dtsatin_safeadt_light | 0 | 0 | 0 | +0.000790 | -0.004865 | +1.276 |

The most important new wins are `omj_001` and `omj_003`: they add satin-like texture while improving strict precision and keeping off-mask and visible-connector metrics at zero.

## M2.73 vs M2.74

| Metric | M2.73 | M2.74 | Delta |
|---|---:|---:|---:|
| texture switches | 2 | 4 | +2 |
| hard_fail | 0 | 0 | 0 |
| unified loss | 0.05506670 | 0.05639949 | +0.00133279 |
| jump count | 5.69696970 | 6.03030303 | +0.33333333 |
| trim count | 0.84848485 | 0.60606061 | -0.24242424 |
| off-mask length | 0.12120606 | 0.12120606 | 0 |
| visible connector | 0 | 0 | 0 |
| coverage | 0.97972955 | 0.98103933 | +0.00130978 |
| strict precision | 0.83654112 | 0.83779552 | +0.00125440 |
| adaptive coverage | 0.92859709 | 0.92973376 | +0.00113667 |
| adaptive precision | 0.96587603 | 0.96704606 | +0.00117003 |
| low adaptive precision samples | 0 | 0 | 0 |
| texture score | 0.14866667 | 0.20860606 | +0.05993939 |
| dt-satin segments | 3.48484848 | 11.90909091 | +8.42424243 |

Component-risk audit:

| Metric | M2.73 | M2.74 | Delta |
|---|---:|---:|---:|
| mean risk score | 0.10425605 | 0.10740702 | +0.00315097 |
| review samples | 16 | 16 | 0 |
| worst sample | txt_005 | txt_005 | unchanged |
| worst risk score | 0.28471031 | 0.28471031 | 0 |

## Interpretation

M2.74 is not the lowest-loss profile. It is the current best professional-texture profile.

The tradeoff is acceptable for the professional-digitizing research direction because it:

- doubles satin-like texture promotion from 2 to 4 samples;
- improves coverage, strict precision, adaptive coverage, and adaptive precision;
- keeps `0` hard_fail and `0` visible connectors;
- does not increase off-mask stitch length;
- reduces trim count;
- exposes the true next bottleneck: stitch-type selection needs a learned planner rather than a fixed candidate gate.

The cost is also explicit:

- mean unified loss increases by `0.00133279`;
- mean jump count increases by `0.33333333`;
- mean component risk score increases by `0.00315097`.

## Artifacts

Config:

```text
configs/best_current_model_m2_74_professional_texture_profile.json
```

Selection:

```text
results/public_benchmark_v1_ext33_m2_74_professional_texture_profile/
```

Candidate outputs:

```text
results/public_benchmark_v1_ext33_m2_73_texture_satinrail_safeadt_light/
results/public_benchmark_v1_ext33_m2_73_texture_dtsatin_safeadt_light/
results/public_benchmark_v1_ext33_m2_74_texture_satinrail_w4_s18_i4_cap40/
results/public_benchmark_v1_ext33_m2_74_texture_satinrail_w5_s16_i5_cap45/
```

Audits:

```text
results/public_benchmark_v1_ext33_m2_74_professional_texture_profile_component_audit/
results/public_benchmark_v1_ext33_m2_74_professional_texture_profile_adaptive_audit/
```

## Next Direction

M2.74 confirms that a professional-quality objective cannot be only `minimize jump/loss`.

The next model should learn stitch-type planning directly:

```text
component geometry
  -> running / satin / fill probability
  -> expected texture gain
  -> expected jump/loss risk
  -> safe stitch-type promotion
```

The training target should include both executable safety and texture value. This is the route from a safe candidate selector toward a real digitizing engine with running, satin, fill/tatami, underlay, pull compensation, and render-back thread quality.
