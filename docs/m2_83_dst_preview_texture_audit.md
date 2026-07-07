# M2.83 DST Preview Texture Audit

Date: 2026-07-07

## Summary

M2.83 adds the missing evaluation layer that became obvious after M2.82.

M2.82 safely expanded the candidate pool from texture-only candidates to a broader multi-family set. Command-level metrics stayed safe, but that still did not prove the output looked more like professional embroidery.

M2.83 reads the generated DST files directly and computes a preview-texture proxy score from the actual stitch trajectory.

This is a diagnostic evaluator, not a new generator.

## Tool

New:

```text
tools/audit_dst_preview_texture.py
```

Run:

```powershell
python tools\audit_dst_preview_texture.py --render-previews
```

Default comparison:

```text
m2_81 = results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile
m2_82 = results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit/dst_preview_texture_rows.csv
results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit/dst_preview_texture_delta_rows.csv
results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit/dst_preview_texture_summary.json
```

The command also creates local preview PNGs in:

```text
results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit/previews
```

The PNG previews are intentionally treated as local review artifacts. The reproducible numeric audit is stored in CSV/JSON.

## What It Measures

M2.83 combines existing safety/coverage metrics with DST-derived stitch-trajectory metrics:

| Metric | Meaning |
|---|---|
| `angle_entropy` | normalized orientation entropy of stitch segments |
| `dominant_angle_mass` | amount of coherent dominant stitch direction |
| `stitch_density_per_100mm2` | stitch density normalized by pattern area |
| `rhythm_score` | regularity of stitch-length rhythm |
| `generator_texture_score` | existing family-texture proxy from fill/satin/outline features |
| `coverage_score` | coverage and precision proxy |
| `safety_score` | command-risk penalty from visible/off-mask/illegal stitches |
| `professional_preview_score` | combined preview-texture proxy |

The scoring is deliberately conservative. It does not claim to replace a professional embroidery preview engine or human digitizer review.

## Result

| Profile | Preview score | Texture score | Angle entropy | Dominant angle | Density / 100mm2 | Rhythm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.81 | 0.73007454 | 0.20860606 | 0.49069049 | 0.61630507 | 17.25413234 | 0.60764702 | 0.98103933 | 0.83779552 |
| M2.82 | 0.72976094 | 0.19769697 | 0.49145153 | 0.61826383 | 17.35027878 | 0.60626542 | 0.98105379 | 0.83837373 |

Delta:

| Metric | M2.82 - M2.81 |
|---|---:|
| preview score | -0.00031360 |
| generator texture score | -0.01090909 |
| coverage | +0.00001445 |
| precision | +0.00057821 |
| visible connectors | 0 |
| off-mask length | 0 |
| improved preview samples | 0 |
| worse preview samples | 2 |

The two negative-preview samples are:

| Sample | Preview delta | Texture delta | Precision delta |
|---|---:|---:|---:|
| `ocp_005` | -0.00246777 | -0.32000000 | +0.017393 |
| `omj_003` | -0.00788097 | -0.04000000 | +0.001688 |

## Interpretation

M2.83 changes the conclusion about M2.82.

M2.82 is command-safe, but it is not visually stronger under the DST-derived preview-texture proxy. It raises coverage and precision slightly, but reduces the texture-family score and does not improve the combined preview score on any sample.

So the safest research conclusion is:

```text
M2.82 proves multi-family candidates can be audited safely,
but future promotion must be preview-texture gated.
```

## Next Optimization

The next model/policy should add:

```text
M2.84 = M2.82 candidate policy + M2.83 preview-texture gate
```

Promotion rule:

```text
candidate may replace baseline only if
  command safety gates pass
  and professional_preview_score does not decrease
  and texture_family_score does not regress beyond a small tolerance
```

That is the most direct response to the professional-quality problem: command-level safety is not enough; the selected DST must also improve or preserve stitch-preview texture.
