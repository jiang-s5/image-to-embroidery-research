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

## Core20 Geometry-Prior Follow-Up

A 20-sample follow-up was run on the first 20 samples of `public_benchmark_v1_ext33`
(10 Openclipart + 10 OpenMoji). This directly addresses the original requirement to
move beyond the 7-sample pilot and produce aggregate, per-domain, and per-sample
results.

Result files:

- `results/public_benchmark_v1_ext33_repair_veto_core20/geometry_priors_ablation_summary.json`
- `results/public_benchmark_v1_ext33_repair_veto_core20/geometry_priors_ablation_rows.csv`
- `results/public_benchmark_v1_ext33_repair_veto_core20/geometry_priors_domain_summary.csv`
- `results/public_benchmark_v1_ext33_repair_veto_core20/geometry_priors_per_sample_summary.csv`

Aggregate result:

| Variant | Samples | Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 M2.1 globalrepair20 | 20 | 0.349218 | 55.550 | 11.200 | 161.1440 | 32.650 | 19 |
| B1 EDT repair veto | 20 | 0.356347 | 121.500 | 11.000 | 165.4469 | 32.800 | 19 |
| B2 Canny repair veto | 20 | 0.361037 | 147.400 | 11.250 | 165.8244 | 32.800 | 20 |
| B3 EDT + Canny repair veto | 20 | 0.359774 | 166.300 | 11.550 | 164.8603 | 32.800 | 20 |
| B4 soft geometry rerank | 20 | 0.360913 | 50.150 | 10.250 | 166.0707 | 32.750 | 19 |

Per-domain result:

| Variant | Source | Samples | Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Hard Fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | Openclipart | 10 | 0.344470 | 40.600 | 8.700 | 305.1066 | 64.800 | 10 |
| B1 EDT | Openclipart | 10 | 0.342749 | 42.100 | 8.600 | 306.3096 | 64.800 | 10 |
| B2 Canny | Openclipart | 10 | 0.345570 | 44.600 | 8.600 | 305.4684 | 64.800 | 10 |
| B3 EDT+Canny | Openclipart | 10 | 0.343865 | 45.300 | 8.800 | 305.2641 | 64.800 | 10 |
| B4 soft | Openclipart | 10 | 0.353421 | 39.700 | 8.400 | 307.3256 | 64.800 | 10 |
| B0 | OpenMoji | 10 | 0.353966 | 70.500 | 13.700 | 17.1814 | 0.500 | 9 |
| B1 EDT | OpenMoji | 10 | 0.369945 | 200.900 | 13.400 | 24.5842 | 0.800 | 9 |
| B2 Canny | OpenMoji | 10 | 0.376505 | 250.200 | 13.900 | 26.1804 | 0.800 | 10 |
| B3 EDT+Canny | OpenMoji | 10 | 0.375682 | 287.300 | 14.300 | 24.4565 | 0.800 | 10 |
| B4 soft | OpenMoji | 10 | 0.368406 | 60.600 | 12.100 | 24.8158 | 0.700 | 9 |

This larger follow-up confirms the 7-sample pilot conclusion. Hard geometry repair
vetoes do not reduce hard-fail rate and substantially increase jump count,
especially on OpenMoji. B1 is marginally lower than B0 on Openclipart loss, but it
keeps all 10 Openclipart samples as hard-fail and worsens OpenMoji strongly.

B4 tests the softer alternative: weak EDT/Sobel/Canny cost terms without repair
veto. It lowers average jump and trim count compared with B0, but its unified loss
and off-mask/visual risk are worse. Therefore the recommended conclusion is still
to keep EDT/Canny as diagnostics or very weak optional features, not as the main
routing or repair mechanism.

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


## M2.41 Safe Adaptive Fill-Inset Selector

M2.41 adds a `safe_dt` adaptive fill-inset candidate to the M2.34 calibrated selector. The candidate is not promoted as a standalone planner, but it improves the selector when used as an optional candidate.

| System | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.34 core selector LOO | 33 | 0.051174 | 5.636 | 0.424 | 0.0825 | 0.9680 | 0.8197 | 0 |
| M2.41 core + safe_dt LOO | 33 | 0.049160 | 5.485 | 0.424 | 0.0500 | 0.9784 | 0.7992 | 0 |
| M2.34 incoming LOO | 4 | 0.078290 | 8.500 | 1.250 | 0.0000 | 0.9955 | 0.7811 | 0 |
| M2.41 incoming core + safe_dt LOO | 4 | 0.074710 | 8.250 | 1.000 | 0.0000 | 0.9965 | 0.7535 | 0 |

This is the first post-M2.34 change that improves both public and incoming unified loss while remaining hard-fail free. The tradeoff is lower stitch precision, so M2.41 is best described as the current unified-objective best, while M2.34 remains the precision-safer baseline. See `docs/m2_41_safe_adaptive_inset_selector.md`.


## M2.42 Precision-Aware Safe Adaptive Selector

M2.42 retunes the M2.41 selector with a stronger precision term. It keeps the same unified loss as M2.41 while slightly improving public stitch precision.

| System | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.41 core + safe_dt LOO | 33 | 0.049160 | 5.485 | 0.424 | 0.0500 | 0.9784 | 0.7992 | 0 |
| M2.42 precision-aware safe_dt LOO | 33 | 0.049160 | 5.485 | 0.424 | 0.0500 | 0.9787 | 0.8015 | 0 |

M2.42 is therefore the recommended current selector config, while M2.34 remains the precision-safer baseline. See `docs/m2_42_precision_safeadt_selector.md`.


## M2.43 Pareto-Aware Selector Audit

M2.43 adds held-out config summary and Pareto-front reporting to the LOO selector evaluation. It does not introduce a new generator. Its purpose is to make the loss/coverage/precision tradeoff explicit instead of relying on one selected LOO result.

New outputs from `tools/eval_m2_calibrated_selector_loo.py`:

- `loo_config_selected_rows.csv`
- `loo_config_summary_rows.csv`
- `loo_config_pareto_rows.csv`

Public ext33 result:

| Selector | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 dynamic LOO | 33 | 0.049160 | 5.485 | 0.424 | 0.0500 | 0.9787 | 0.8015 | 0 |
| M2.43 Pareto fixed config | 33 | 0.049596 | 5.394 | 0.697 | 0.0300 | 0.9657 | 0.8069 | 0 |

Incoming review result:

| Selector | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 incoming | 4 | 0.074710 | 8.250 | 1.000 | 0.0000 | 0.9965 | 0.7535 | 0 |
| M2.43 incoming Pareto | 4 | 0.074710 | 8.250 | 1.000 | 0.0000 | 0.9965 | 0.7535 | 0 |

Decision: M2.42 remains the default current selector. M2.43 is a reporting and optional public-balanced configuration step: it improves public off-mask and precision at the cost of slightly higher loss, lower coverage, and more trims. See `docs/m2_43_pareto_safeadt_selector.md`.


## M2.44 Pareto-Teacher Learned Selector

M2.44 feeds the M2.43 Pareto insight back into the learned listwise selector. The listwise teacher now supports coverage, precision, off-mask, visible-connector, jump, and trim penalties.

| Selector | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 calibrated default | 33 | 0.049160 | 5.485 | 0.424 | 0.0500 | 0.9787 | 0.8015 | 0 |
| M2.44 Pareto teacher learned | 33 | 0.050318 | 5.485 | 0.455 | 0.0935 | 0.9566 | 0.8168 | 0 |

Incoming review:

| Selector | Samples | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 incoming | 4 | 0.074710 | 8.250 | 1.000 | 0.0000 | 0.9965 | 0.7535 | 0 |
| M2.44 incoming applied | 4 | 0.078290 | 8.500 | 1.250 | 0.0000 | 0.9955 | 0.7811 | 0 |

M2.44 proves that multi-objective listwise supervision can move the learned selector toward higher precision, but it does not replace M2.42 because unified loss, jump/trim, and coverage are worse. See `docs/m2_44_pareto_teacher_selector.md`.

## Current Decision

For reporting and GitHub documentation:

1. Use M2.1 as the historical baseline, not the final promoted system.
2. Use M2.42 as the current recommended selector config; keep M2.43 as the Pareto audit / optional public-balanced setting, M2.44 as a learned precision-safe research branch, M2.34 as the precision-safer baseline, and M2.41 as the first safe_dt candidate proof.
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
