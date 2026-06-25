# M2.59 Anchored Quality Rescue

Date: 2026-06-25

M2.59 follows M2.58.

M2.58 added a coverage-drop safety gate and blocked the M2.57 failure mode where a lower-jump topology candidate lost too much coverage. M2.59 tests whether a safer grouped-topology candidate can still improve the hard QuickDraw sample `qd_007`.

## Generator Change

New candidate generator:

```text
tools/run_line_domain_grouped_topology_candidate.py
```

The generator is based on the M2.57 open-trail topology candidate, but changes how the next skeleton component is selected.

Instead of choosing only the nearest component, it ranks component starts by:

```text
safe connector first
+ distance
+ inside-mask deficit
+ small component penalty
```

This is a first stroke-grouping probe: a farther component with a safe in-mask connector can beat a nearer component that would immediately require a jump.

## Selector Change

M2.59 also extends:

```text
tools/select_m2_line_domain_guard.py
```

with:

```text
--anchor-baseline-selection
```

This freezes the current baseline selection before adding new candidates. New candidates can only replace the baseline through an explicit rescue gate.

This matters because adding a new candidate can perturb the learned base ranking and change unrelated samples. Anchoring prevents that.

M2.59 uses:

```text
--line-relative-rescue
--line-rescue-min-jump-gain 0.0
--line-rescue-max-coverage-drop 0.05
--anchor-baseline-selection results/public_benchmark_v1_ext33_m2_58_coverage_aware_relative_rescue/line_guard_selected_rows.csv
```

So a candidate may replace the baseline when it:

- does not increase jump;
- does not increase visible/off-mask risk;
- improves or matches loss;
- does not drop coverage by more than `0.05`;
- preserves precision within the existing tolerance.

## Variant Probe

Ten grouped-topology variants were tested.

Artifacts:

```text
results/m2_59_grouped_topology_probe/variant_summary.csv
results/m2_59_grouped_topology_probe/probe_summary.json
```

Best hard-fail-free grouped variant by mean loss:

```text
g_d0_c20_i095_s4_min12_p100
```

However, the raw grouped candidates are not strong enough as standalone replacements:

| Metric | M2.58 default | Best grouped candidate |
|---|---:|---:|
| mean loss | 0.048797 | 0.071406 |
| mean jump | 4.909091 | 7.393939 |
| mean coverage | 0.979200 | 0.914694 |
| mean precision | 0.793370 | 0.847218 |
| QuickDraw loss | lower | 0.129376 |
| QuickDraw jump | lower | 14.250000 |
| text coverage | higher | 0.617972 |

The generator improves precision but still loses too much coverage on rendered text and keeps QuickDraw jump high.

## Anchored Quality Rescue Result

Selector output:

```text
results/public_benchmark_v1_ext33_m2_59_anchored_quality_rescue_selector/
```

Risk audit:

```text
results/public_benchmark_v1_ext33_m2_59_anchored_quality_rescue_risk_audit/
```

Compared with M2.58:

| Metric | M2.58 | M2.59 anchored quality rescue | Change |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| hard fail | 0 | 0 | 0 |
| mean loss | 0.048797 | 0.048581 | -0.000216 |
| mean jump | 4.909091 | 4.909091 | 0.000000 |
| mean trim | 0.606061 | 0.575758 | -0.030303 |
| off-mask mm | 0.199076 | 0.199076 | 0.000000 |
| visible connector | 0.000000 | 0.000000 | 0.000000 |
| mean coverage | 0.979200 | 0.980840 | +0.001640 |
| mean precision | 0.793370 | 0.793433 | +0.000063 |
| review samples | 15 | 15 | 0 |
| mean risk score | 0.127757 | 0.127404 | -0.000353 |
| worst risk score | 0.392948 | 0.381297 | -0.011651 |

Only one sample changed:

| Sample | Baseline candidate | M2.59 candidate | Loss | Jump | Trim | Coverage | Precision |
|---|---|---|---:|---:|---:|---:|---:|
| `qd_007` | `mask_fill_edgewalk_nearestrow_inset2_rows16_p40` | `grouped_topology_c20_i095` | -0.007136 | 0 | -1 | +0.054124 | +0.002058 |

The replacement keeps jump at `14`, reduces trim from `2` to `1`, keeps off-mask and visible connector at `0`, and improves coverage from `0.933862` to `0.987986`.

## Decision

M2.59 is a small positive update.

It should be treated as:

```text
candidate-generation probe + anchored selector safety upgrade
```

not as a solved line-domain planner.

Recommended current research default:

```text
M2.53/M2.51 low-jump robust policy
+ M2.55 bounded line-domain guard
+ M2.58 coverage-aware rescue gate
+ M2.59 anchored quality rescue
```

## Remaining Limitation

`qd_007` still has high jump count (`14`), so the main failure is not solved.

The next stage should be:

```text
M2.60 line-domain segment merger
```

The next generator should not only reorder components. It should explicitly merge fragmented skeleton segments into higher-level strokes, then plan stroke endpoints under a joint cost:

```text
jump + off-mask + visible connector + coverage retention
```

## Artifacts

- `tools/run_line_domain_grouped_topology_candidate.py`
- updated `tools/select_m2_line_domain_guard.py`
- `configs/best_current_model_m2_59_anchored_quality_rescue.json`
- `results/m2_59_grouped_topology_probe/`
- `results/public_benchmark_v1_ext33_m2_59_anchored_quality_rescue_selector/`
- `results/public_benchmark_v1_ext33_m2_59_anchored_quality_rescue_risk_audit/`
