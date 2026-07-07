# M2.91 Professional Candidate Generation Audit

Date: 2026-07-07

## Summary

M2.91 is the first step after M2.90 that tries to generate richer professional stitch-family candidates directly, instead of only selecting from existing generator roots.

M2.90 already showed that the system can trade a bounded amount of execution cost for better DST-derived professional texture. The limitation was that M2.90 could only choose a better candidate when that candidate already existed.

M2.91 tests this next question:

```text
Can we add new satin / DT-satin style candidate roots and let the preview-texture selector decide which ones are safe enough to use?
```

The answer is mixed but useful:

- direct outline-underlay stacking is unsafe and creates off-mask / visible-connector failures;
- safe DT-satin generation is viable;
- the new generated roots are selected by the M2.91 sweep and by the professional-texture objective;
- the strongest texture profile improves professional preview score substantially, but increases jumps and unified loss.

This makes M2.91 a candidate-generation audit, not a default production profile.

## Why This Matters

The user's main qualitative complaint was:

```text
the output still looks like one simple line stitch, not a dimensional professional embroidery design
```

M2.91 directly targets that gap by adding denser stitch-family candidates, especially DT-satin and satin-rail-like outputs. It moves the system from:

```text
professional stitch-family selection
```

toward:

```text
professional stitch-family generation
```

That is the right direction for real digitizing quality, but the new candidates must remain gated by command safety and preview-risk metrics.

## Generated Candidate Roots

M2.91 generated six new candidate families.

### Negative Underlay Profiles

The direct layered-underlay profiles were intentionally tested and rejected.

| Profile | Hard Fail | Mean Loss | Mean Jumps | Mean Off-Mask mm | Mean Visible |
|---|---:|---:|---:|---:|---:|
| layered DT-satin underlay | 11 | 0.233557 | 13.696970 | 6.401085 | 1.363636 |
| layered satin-rail underlay | 10 | 0.232518 | 13.606061 | 6.468291 | 1.363636 |
| layered style-aware underlay | 8 | 0.225278 | 12.727273 | 6.337309 | 1.363636 |

Interpretation:

```text
Do not hard-stack outline underlay as a generated layer.
```

It produces a more dimensional stitch family, but the connector and mask behavior is not safe. Underlay should later be reintroduced as a component-aware, inside-mask, segment-level operation, not a raw global layer.

### Safer Generated Profiles

| Profile | Hard Fail | Mean Loss | Mean Jumps | Mean Trims | Mean Off-Mask mm | Mean Visible | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| safe DT-satin dense | 0 | 0.123009 | 13.030303 | 1.878788 | 0.143667 | 0.0 | 0.997599 | 0.816691 |
| safe satin-rail dense | 5 | 0.137370 | 15.030303 | 2.181818 | 0.213239 | 0.0 | 0.997777 | 0.811736 |
| safe DT-satin balanced | 0 | 0.119693 | 12.878788 | 1.818182 | 0.068055 | 0.0 | 0.997618 | 0.816886 |

Interpretation:

```text
safe DT-satin is the most promising M2.91 generation family.
```

The dense and balanced DT-satin roots keep `0` hard_fail and `0` visible connectors, so they are suitable for gated candidate selection. The satin-rail dense root can still be useful per-sample, but it is not safe as a standalone profile because it has nonzero hard_fail.

## Strict Candidate Sweep

The M2.91 sweep expands candidate discovery to include the new generated roots.

```text
results/public_benchmark_v1_ext33_m2_91_professional_candidate_sweep/
```

| Metric | Value |
|---|---:|
| candidate roots | 54 |
| candidate rows | 1782 |
| gate-pass rows | 37 |
| selected samples | 33 |
| preview-sweep switches | 8 |
| hard_fail | 0 |
| visible connectors | 0 |
| mean unified loss | 0.05614614 |
| mean jumps | 6.06060606 |
| mean professional preview score | 0.73754405 |
| mean generator texture score | 0.33957576 |

This strict sweep is close to the M2.89 safety baseline, but it proves the generated M2.91 candidates are not dead-on-arrival. At least one generated DT-satin candidate is selected under tight safety gates.

## Professional Texture Objective

The professional-texture objective is then rerun on the M2.91 candidate table:

```text
results/public_benchmark_v1_ext33_m2_91_professional_texture_objective/
```

| Metric | M2.89 strict | M2.91 professional texture | Delta |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| switches | 0 | 25 | +25 |
| hard_fail | 0 | 0 | 0 |
| mean unified loss | 0.05706876 | 0.10187967 | +0.04481091 |
| mean jumps | 6.15151515 | 11.15151515 | +5.00000000 |
| mean trims | 0.60606061 | 1.45454545 | +0.84848484 |
| mean off-mask length mm | 0.10619091 | 0.01501515 | -0.09117576 |
| mean visible connectors | 0.0 | 0.0 | 0 |
| mean coverage | 0.98167906 | 0.99249652 | +0.01081746 |
| mean precision | 0.83918282 | 0.84118715 | +0.00200433 |
| mean professional preview score | 0.73544111 | 0.81147022 | +0.07602911 |
| mean generator texture score | 0.22612121 | 1.77945455 | +1.55333334 |
| mean family texture score | 0.25195960 | 0.77317172 | +0.52121212 |

This is the strongest professional-texture signal so far. It improves professional preview and texture much more than M2.90 while preserving `0` hard_fail and `0` visible connectors.

The tradeoff is also clear: M2.91 increases command cost and jump count more than M2.90.

## Source-Level Behavior

M2.91 is strongest on clean shape/icon sources:

| Source | Samples | Switches | Mean Preview | Mean Texture | Mean Jumps |
|---|---:|---:|---:|---:|---:|
| OpenMoji | 10 | 9 | 0.88750921 | 2.08000000 | 11.6 |
| Openclipart | 10 | 10 | 0.89246727 | 2.59700000 | 12.2 |
| QuickDraw | 8 | 1 | 0.61485554 | 0.12225000 | 8.5 |
| Rendered text | 5 | 5 | 0.81198162 | 2.19480000 | 12.4 |

This matches the intended use:

- clean icons and text benefit from satin / DT-satin structure;
- QuickDraw line art mostly stays with the safer line-oriented baseline.

## Current Position

Use the profiles this way:

| Profile | Use |
|---|---|
| M2.89 strict | conservative safety baseline |
| M2.90 hard texture | stronger professional-texture selector over existing roots |
| M2.91 professional texture | strongest texture experiment with newly generated candidate roots |

M2.91 should be described as:

```text
an experimental professional candidate-generation audit with hard safety gates
```

not as:

```text
a final commercial embroidery digitizer
```

## What Changed From M2.90

M2.90:

```text
select better candidates from existing roots
```

M2.91:

```text
generate new satin / DT-satin candidate roots, then select under safety and preview-texture gates
```

That is a real research step, because it tests whether richer stitch-family candidates can be introduced without breaking hard safety.

## Next Step

M2.92 should not simply make the texture gate more aggressive. The next improvement should reduce execution cost while keeping the M2.91 texture gain:

- add jump-aware reranking after professional-texture scoring;
- make DT-satin generation component-aware so it does not over-segment small shapes;
- reintroduce underlay only as inside-mask segment-level underlay, not as a raw global layer;
- add per-source profile caps so QuickDraw line art does not get forced into dense fill;
- generate A/B preview reports for the switched samples.

The next target is:

```text
keep most of M2.91's professional texture gain while bringing jumps and unified loss closer to M2.90 balanced.
```
