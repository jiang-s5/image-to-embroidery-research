# M2.60 Mask-Path Rescue

Date: 2026-06-25

M2.60 follows M2.59 anchored quality rescue.

M2.59 improved `qd_007` by adding a grouped-topology candidate, but the remaining line-domain issue was still fragmented skeleton components and high jumps. M2.60 tests whether rejected component connections can be converted into safe in-mask stitch paths instead of jumps.

## Generator Change

New tool:

```text
tools/run_line_domain_maskpath_merger_candidate.py
```

It extends the M2.59 grouped topology candidate:

```text
grouped component ordering
  -> direct safe connector
  -> if unsafe, search mask-internal path
  -> if route is valid, stitch along the mask path
  -> otherwise keep JUMP
```

The route is constrained by:

- target mask passability;
- maximum route length in mm;
- maximum route length relative to the straight-line distance;
- no mask dilation by default, because dilation created off-mask risk in the probe.

## Variant Probe

Six mask-path variants were tested.

Artifacts:

```text
results/m2_60_maskpath_merger_probe/variant_summary.csv
results/m2_60_maskpath_merger_probe/probe_summary.json
```

Best hard-fail-free standalone variant by mean loss was:

```text
mp_d0_r0_c20_i095_route32_f26_min20
```

However, this standalone variant is not promoted because it can lose too much coverage on some samples. The selector-level default uses the safer min12 variant:

```text
mp_d0_r0_c20_i095_route32_f26_min12
```

## Final Selector

Final output:

```text
results/public_benchmark_v1_ext33_m2_60_maskpath_rescue_selector/
```

Risk audit:

```text
results/public_benchmark_v1_ext33_m2_60_maskpath_rescue_risk_audit/
```

The selector is anchored to M2.59:

```text
--anchor-baseline-selection results/public_benchmark_v1_ext33_m2_59_anchored_quality_rescue_selector/line_guard_selected_rows.csv
```

and accepts mask-path candidates only when:

- jump decreases by at least 1;
- loss does not increase;
- coverage drops by at most 0.05;
- precision drops by at most 0.03;
- visible and off-mask do not increase.

## Result

Compared with M2.59:

| Metric | M2.59 | M2.60 mask-path rescue | Change |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| hard fail | 0 | 0 | 0 |
| mean loss | 0.048581 | 0.048007 | -0.000574 |
| mean jump | 4.909091 | 4.818182 | -0.090909 |
| mean trim | 0.575758 | 0.606061 | +0.030303 |
| off-mask mm | 0.199076 | 0.199076 | 0.000000 |
| visible connector | 0.000000 | 0.000000 | 0.000000 |
| mean coverage | 0.980840 | 0.980904 | +0.000064 |
| mean precision | 0.793433 | 0.793610 | +0.000177 |
| review samples | 15 | 15 | 0 |
| mean risk score | 0.127404 | 0.124630 | -0.002775 |
| worst risk score | 0.381297 | 0.381297 | 0.000000 |

Two samples changed:

| Sample | Baseline candidate | M2.60 candidate | Loss | Jump | Coverage | Precision | Off-mask |
|---|---|---|---:|---:|---:|---:|---:|
| `qd_004` | `skeleton_c30_m20` | `maskpath_c20_i095_r32` | -0.002190 | -1 | +0.041031 | +0.029201 | 0 |
| `qd_006` | `mask_fill_edgewalk_nearestrow_inset2_rows16_p40` | `maskpath_c20_i095_r32` | -0.016764 | -2 | -0.038933 | -0.023335 | 0 |

For `qd_006`, the selected mask-path candidate made:

```text
routed_connects = 3
rejected_component_connects = 2
route_length_mm_total = 59.8095
```

This confirms that the improvement comes from actual mask-internal path routing, not just selector noise.

## Combo Probe

A more aggressive combo selector included both min12 and min20 mask-path variants:

```text
results/public_benchmark_v1_ext33_m2_60_maskpath_combo_rescue_selector/
```

It lowered mean loss and jump further, but risk audit was worse because it selected a lower-coverage `qd_004` candidate:

```text
coverage = 0.890471
```

Therefore M2.60 promotes the safer min12 selector, not the combo selector.

## Decision

M2.60 is a real improvement over M2.59:

- lower mean loss;
- lower mean jump;
- lower mean risk score;
- no added hard failures;
- no added visible connector risk;
- no added off-mask risk.

Recommended current research default:

```text
M2.53/M2.51 low-jump robust policy
+ M2.55 bounded line-domain guard
+ M2.58 coverage-aware rescue gate
+ M2.59 anchored quality rescue
+ M2.60 mask-path rescue
```

## Remaining Limitation

M2.60 improves `qd_004` and `qd_006`, but it still does not reduce `qd_007` jump count. The route search reports no successful routed connector for `qd_007` under the safe route constraints.

The next stage should be:

```text
M2.61 explicit segment graph merge
```

Instead of connecting current component to next component greedily, M2.61 should build a segment graph first, then solve endpoint pairing globally under:

```text
jump + route length + off-mask + visible + coverage retention
```

## Artifacts

- `tools/run_line_domain_maskpath_merger_candidate.py`
- `configs/best_current_model_m2_60_maskpath_rescue.json`
- `results/m2_60_maskpath_merger_probe/`
- `results/public_benchmark_v1_ext33_m2_60_maskpath_rescue_selector/`
- `results/public_benchmark_v1_ext33_m2_60_maskpath_rescue_risk_audit/`

