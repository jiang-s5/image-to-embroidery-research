# M2.43 Pareto-Aware Safe Adaptive Selector Audit

Date: 2026-06-25

M2.43 is a selector-audit upgrade, not a new stitch generator. It keeps the M2.42 safe_dt candidate pool and adds held-out Pareto reporting so the loss/coverage/precision tradeoff can be inspected directly.

## What Changed

`tools/eval_m2_calibrated_selector_loo.py` now writes three additional files:

- `loo_config_selected_rows.csv`: each config's selected held-out rows across folds.
- `loo_config_summary_rows.csv`: held-out summary metrics for every config.
- `loo_config_pareto_rows.csv`: non-dominated configs over loss, jumps, trims, off-mask, visible connectors, coverage, and precision.

This closes a weakness in M2.42: the old report showed the best LOO selector result, but did not show whether nearby configs gave a better precision/off-mask tradeoff.

## Public ext33 Result

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 dynamic LOO | 33 | 0 | 0.049160 | 5.484848 | 0.424242 | 0.049952 | 0.978737 | 0.801506 |
| M2.43 Pareto fixed config | 33 | 0 | 0.049596 | 5.393939 | 0.696970 | 0.030030 | 0.965657 | 0.806913 |

The public Pareto fixed config is not the lowest-loss setting. Its value is a more balanced tradeoff:

- lower off-mask length: `0.049952 -> 0.030030`
- slightly higher precision: `0.801506 -> 0.806913`
- slightly lower jump count: `5.484848 -> 5.393939`
- lower coverage and higher trim count
- small loss increase: `0.049160 -> 0.049596`

## Incoming Review Result

| Selector | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 incoming | 4 | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.43 incoming Pareto | 4 | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.996541 | 0.753519 |

Incoming review does not improve beyond M2.42. This is important because it prevents overclaiming: M2.43 is a stronger audit and optional public-balanced setting, not a global replacement.

## Decision

Keep M2.42 as the default current selector configuration.

Use M2.43 for:

1. Pareto-front reporting in experiments.
2. Selecting a precision/off-mask safer public-benchmark config.
3. Documenting the tradeoff between unified loss and stitch precision.

M2.34 remains the precision-safer historical baseline. M2.42 remains the recommended default. M2.43 adds the reporting infrastructure needed to choose the next selector using evidence instead of manual inspection.

## Artifacts

- `configs/best_current_model_m2_43_pareto_safeadt_selector.json`
- `results/public_benchmark_v1_ext33_m2_43_pareto_safeadt_plus_loo/`
- `results/incoming_review_eval_v1_m2_43_pareto_safeadt_plus_loo/`
