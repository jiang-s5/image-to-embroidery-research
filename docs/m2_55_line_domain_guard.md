# M2.55 Line-Domain Guard

Date: 2026-06-25

M2.55 follows the M2.54 risk audit.

M2.54 showed that the dominant residual weakness is not hard failure or visible connectors. The weak domain is line-like input, especially QuickDraw and rendered text, where stitch precision remains low and some samples still have excessive jump counts.

M2.55 adds a bounded line-domain guard on top of the M2.53/M2.51 low-jump policy.

This is a selector/reranking layer, not a new stitch generator.

## Tool

```text
tools/select_m2_line_domain_guard.py
```

The tool:

1. builds the same candidate rows used by the M2.51/M2.53 benefit-gate selector;
2. applies the robust low-jump profile as the baseline choice;
3. detects line-domain sources (`QuickDraw`, `Rendered text`);
4. applies an extra precision/jump-aware line score only for those sources;
5. caps line-domain replacements at `jump_count <= 10`;
6. writes selected rows, changed rows, source summaries, baseline deltas, and copied DST outputs.

## Guard Objective

The line-domain score is:

```text
loss
+ 0.004 * jump
+ 0.002 * trim
+ 0.010 * off_mask_mm
+ 0.005 * visible_connector
+ 0.55  * max(0, 0.88 - coverage)
+ 0.22  * max(0, 0.75 - precision)
```

Hard guard constraints:

```text
coverage >= 0.60
precision >= 0.45
jump_count <= 10
off_mask_mm <= 1.0
visible_connector_count <= 0
trim_count <= 6
loss_slack_vs_baseline <= 0.03
```

The jump cap is important. Without it, the guard improved precision on `qd_001` but increased jump from 10 to 11, adding a new high-jump risk flag.

## Inputs

M2.55 uses the same candidate families as M2.51/M2.53:

- auto eval-repair;
- auto skeleton hybrid;
- mask-fill edgewalk;
- nearest-row;
- nearest-row inset 1/2;
- adaptive nearest-row;
- safe-ADT nearest-row;
- skeleton connect-30 variants.

## Public Ext33 Result

Baseline is M2.51/M2.53 low-jump selection on `datasets/public_benchmark_v1_ext33`.

| Metric | M2.51 low_jump | M2.55 line guard | Change |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| hard_fail | 0 | 0 | 0 |
| mean unified loss | 0.048809 | 0.048797 | -0.000012 |
| mean jump | 4.909091 | 4.909091 | 0.000000 |
| mean trim | 0.606061 | 0.606061 | 0.000000 |
| mean off-mask mm | 0.199076 | 0.199076 | 0.000000 |
| mean visible connector | 0.000000 | 0.000000 | 0.000000 |
| mean coverage | 0.979476 | 0.979200 | -0.000276 |
| mean precision | 0.793264 | 0.793370 | +0.000106 |
| review samples | 15 | 15 | 0 |

Line-domain subset (`QuickDraw` + `Rendered text`, 13 samples):

| Metric | Baseline line subset | M2.55 line subset | Change |
|---|---:|---:|---:|
| mean unified loss | 0.048646 | 0.048616 | -0.000030 |
| mean jump | 5.538462 | 5.538462 | 0.000000 |
| mean trim | 0.307692 | 0.307692 | 0.000000 |
| mean off-mask mm | 0.036262 | 0.036262 | 0.000000 |
| mean coverage | 0.980583 | 0.979882 | -0.000701 |
| mean precision | 0.651302 | 0.651571 | +0.000269 |

Only one sample changes:

| Sample | Source | Category | Baseline | M2.55 | Effect |
|---|---|---|---|---|---|
| `qd_006` | QuickDraw | bird | `skeleton_c30_m8` | `mask_fill_edgewalk_nearestrow_inset2_rows16_p40` | loss improves by 0.000388, precision improves by 0.003497, jump unchanged |

## Incoming Review Regression

On `datasets/incoming_review_eval_v1`, M2.55 changes no samples because the incoming paired review set does not contain `QuickDraw` or `Rendered text` sources.

| Metric | M2.51 low_jump | M2.55 line guard |
|---|---:|---:|
| samples | 4 | 4 |
| hard_fail | 0 | 0 |
| mean unified loss | 0.073831 | 0.073831 |
| mean jump | 7.250000 | 7.250000 |
| mean coverage | 0.920700 | 0.920700 |
| mean precision | 0.788361 | 0.788361 |

## Interpretation

M2.55 is safe but only weakly improves the current system.

The main finding is that the remaining QuickDraw failures are not primarily selector failures. For example, `qd_007` already selects the best candidate available in the current candidate pool:

```text
candidate = mask_fill_edgewalk_nearestrow_inset2_rows16_p40
loss = 0.128886
jump = 14
precision = 0.661752
```

All other current candidates for `qd_007` have worse loss and/or more jumps. Therefore, reranking cannot bring `qd_007` below the jump threshold. This points to a generator-side limitation.

## Research Conclusion

M2.55 should not be presented as a major quality jump.

Its value is diagnostic:

```text
bounded reranking can make very small safe fixes,
but line-domain generalization now requires new line/running-stitch candidate generation.
```

The next real optimization should be M2.56:

```text
line-domain candidate generator
  -> skeleton stroke tracing
  -> path simplification
  -> closed-loop stroke ordering
  -> line-specific evaluator
  -> selector integration
```

## Artifacts

- `tools/select_m2_line_domain_guard.py`
- `configs/best_current_model_m2_55_line_domain_guard.json`
- `results/public_benchmark_v1_ext33_m2_55_line_domain_guard_low_jump/`
- `results/public_benchmark_v1_ext33_m2_55_line_domain_guard_risk_audit/`
- `results/incoming_review_eval_v1_m2_55_line_domain_guard_low_jump/`
