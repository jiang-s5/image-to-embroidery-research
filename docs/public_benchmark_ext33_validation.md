# Public Benchmark ext33 Validation Notes

Date: 2026-06-22

This note reconciles the original 7-sample `public_benchmark_v1` pilot with the later
33-sample external benchmark used by the M2.30-M2.40 experiments.

## Benchmark Status

`datasets/public_benchmark_v1` is the lightweight smoke benchmark:

| Source | Samples |
|---|---:|
| OpenMoji | 5 |
| Rendered text | 2 |
| Total | 7 |

`datasets/public_benchmark_v1_ext33` is the current external validation benchmark:

| Source | Samples |
|---|---:|
| Openclipart | 10 |
| OpenMoji | 10 |
| QuickDraw | 8 |
| Rendered text | 5 |
| Total | 33 |

The original benchmark plan also listed Oxford-IIIT Pet samples for real-photo
validation. Those are not yet included in `public_benchmark_v1_ext33`, so the
current benchmark is still strongest for flat icons, cartoons, line drawings, and
text. It is not yet a full real-photo validation set.

## Baseline Shift

The external benchmark shows that M2.1 should now be treated as a weak baseline,
not as the current promoted system. M2.1 was useful on the earlier paired
held-out comparison, but it does not generalize well to the 33-sample public
benchmark.

| System | Samples | Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.1 B0 globalrepair20 | 33 | 0.395948 | 43.333 | 8.394 | 1170.1083 | 257.061 | 32 |
| M2.34 fill-inset selector LOO | 33 | 0.051174 | 5.636 | 0.424 | 0.0825 | 0.000 | 0 |

This is the strongest current evidence that the project has moved beyond the
original M2.1 baseline. M2.34 is the current balanced best system.

## M2.34 Per-Domain Result

M2.34 remains hard-fail free across all four external domains:

| Source | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Openclipart | 10 | 0.043547 | 4.900 | 0.300 | 0.0495 | 0.9618 | 0.9000 | 0 |
| OpenMoji | 10 | 0.056662 | 5.800 | 0.600 | 0.2228 | 0.9969 | 0.8994 | 0 |
| QuickDraw | 8 | 0.068351 | 7.750 | 0.625 | 0.0000 | 0.9615 | 0.6554 | 0 |
| Rendered text | 5 | 0.027968 | 3.400 | 0.000 | 0.0000 | 0.9387 | 0.7628 | 0 |

QuickDraw is the weakest domain by precision, which is expected because line
drawings stress running-stitch and skeleton structure rather than filled regions.

## Geometry-Prior Ablation

The 7-sample repair-veto ablation supports a negative conclusion for strong
EDT/Canny repair vetoes:

| Variant | Samples | Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 M2.1 globalrepair20 | 7 | 0.301113 | 40.857 | 7.000 | 113.1525 | 27.000 | 6 |
| B1 EDT repair veto | 7 | 0.343299 | 118.143 | 7.286 | 116.6724 | 27.143 | 6 |
| B2 Canny repair veto | 7 | 0.355069 | 178.571 | 7.429 | 117.2839 | 27.143 | 6 |
| B3 EDT + Canny repair veto | 7 | 0.357378 | 190.286 | 7.714 | 116.4761 | 27.143 | 7 |

The conclusion is not that geometry priors are useless. The conclusion is that
hard edge/distance vetoes over-constrain the planner and increase jump count
without reducing hard-fail risk. Future geometry priors should be used as soft
diagnostics, weak reranking features, or component-level design features rather
than as hard repair vetoes.

## Learned Selector Status

M2.39 is the first learned listwise selector that beats M2.34 on public unified
loss, but it loses too much coverage:

| System | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 fill-inset selector | 0.051174 | 5.636 | 0.424 | 0.0825 | 0.9689 | 0.8197 | 0 |
| M2.39 listwise a0.001 t0.02 | 0.050832 | 5.576 | 0.394 | 0.0975 | 0.9384 | 0.8234 | 0 |
| M2.40 coverage-aware teacher cov0.50 | 0.052026 | 5.697 | 0.424 | 0.0975 | 0.9608 | 0.8237 | 0 |

M2.39/M2.40 are useful research branches, but neither replaces M2.34 as the
current balanced best.

## Current Decision

For reporting and GitHub documentation:

1. Use M2.1 as the historical baseline, not the final promoted system.
2. Use M2.34 as the current best balanced system.
3. Record M2.39/M2.40 as learned-selector experiments that expose a loss versus
   coverage tradeoff.
4. Treat the 7-sample geometry-prior experiment as a negative result showing
   that hard EDT/Canny vetoes are too conservative.

## Remaining Gaps

The external validation loop is much stronger than the original 7-sample pilot,
but it is not finished:

- `public_benchmark_v1_ext33` is ignored by the default dataset git rules; the
  benchmark should be published as an artifact or documented with exact rebuild
  instructions.
- Oxford-IIIT Pet or another real-photo-with-mask subset is still missing from
  the 33-sample benchmark.
- Per-sample qualitative previews should be added for the strongest and weakest
  cases before making broad visual-quality claims.
- The system improves executable DST path quality, but it is still not a
  professional digitizer with underlay, pull compensation, true satin/tatami
  modeling, semantic color sequencing, or 3D thread rendering.
