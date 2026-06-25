# M2.57 Topology Open-Trail Probe

Date: 2026-06-25

M2.57 follows the M2.56 line-domain stroke candidate probe.

M2.56 showed that pure skeleton tracing improves line-like precision but creates too many jumps because each disconnected skeleton component is traced as an isolated closed walk.

M2.57 tests a topology-aware open-trail traversal:

```text
component start endpoint
  -> trace side branches
  -> continue along the main endpoint-to-endpoint path
  -> exit at another endpoint
```

The goal is to avoid returning every component to its starting point and to make inter-component ordering easier.

## Tool

```text
tools/run_line_domain_topology_candidate.py
```

It generates selector-compatible candidate outputs:

- `source_aware_hybrid_rows.csv`
- `coverage_rows.csv`
- per-sample DST / evaluation reports

It reuses the M2.56 candidate interface but changes the skeleton walk from closed DFS to endpoint-to-endpoint open traversal.

M2.57 also extends:

```text
tools/select_m2_line_domain_guard.py
```

with an optional relative rescue mode:

```text
--line-relative-rescue
```

This allows a line-domain candidate to replace the baseline only when it improves loss and jump relative to the baseline without increasing visible/off-mask risk.

## Variant Sweep

Sixteen open-trail variants were tested.

Summary artifacts:

```text
results/m2_57_topology_open_trail_probe/variant_summary.csv
results/m2_57_topology_open_trail_probe/probe_summary.json
```

## Main Result

The best hard-fail-free variant by QuickDraw loss was:

```text
d0_c30_i070_min100
```

Compared with M2.55:

| Metric | M2.55 baseline | Best M2.57 topology variant |
|---|---:|---:|
| mean loss | 0.048797 | 0.057846 |
| mean jump | 4.909091 | 5.000000 |
| mean coverage | 0.979200 | 0.755302 |
| mean precision | 0.793370 | 0.819447 |
| QuickDraw loss | baseline lower | 0.099104 |
| QuickDraw jump | baseline lower | 7.375000 |
| QuickDraw coverage | baseline higher | 0.541325 |
| QuickDraw precision | baseline lower | 0.687358 |
| `qd_007` loss | 0.128886 | 0.114131 |
| `qd_007` jump | 14 | 13 |
| `qd_007` coverage | 0.933862 | 0.723805 |

So M2.57 successfully finds a topology candidate that improves `qd_007` execution metrics, but it does so by losing too much coverage.

## Relative Rescue Test

M2.57 relative rescue result:

```text
results/public_benchmark_v1_ext33_m2_57_topology_relative_rescue/
```

Risk audit:

```text
results/public_benchmark_v1_ext33_m2_57_topology_relative_rescue_risk_audit/
```

Compared with M2.55:

| Metric | M2.55 | M2.57 relative rescue | Change |
|---|---:|---:|---:|
| mean loss | 0.048797 | 0.048350 | -0.000447 |
| mean jump | 4.909091 | 4.878788 | -0.030303 |
| mean trim | 0.606061 | 0.575758 | -0.030303 |
| mean off-mask mm | 0.199076 | 0.199076 | 0.000000 |
| visible connector | 0.000000 | 0.000000 | 0.000000 |
| mean coverage | 0.979200 | 0.972835 | -0.006365 |
| mean precision | 0.793370 | 0.793240 | -0.000130 |
| review samples | 15 | 15 | 0 |
| mean risk score | 0.127757 | 0.131225 | worse |
| worst risk score | 0.392948 | 0.507368 | worse |

The actual non-zero change is `qd_007`:

| Sample | Loss | Jump | Trim | Coverage | Precision |
|---|---:|---:|---:|---:|---:|
| M2.55 `qd_007` | 0.128886 | 14 | 2 | 0.933862 | 0.661752 |
| M2.57 `qd_007` | 0.114131 | 13 | 1 | 0.723805 | 0.657442 |

M2.57 improves execution but worsens coverage-driven risk. Therefore it is not promoted as the default.

## Decision

Current default remains:

```text
M2.53/M2.51 low-jump robust policy
+ M2.55 bounded line-domain guard
```

M2.57 is kept as a research probe.

It proves that topology-aware open trails are useful, but the next version needs coverage-aware stroke grouping instead of simply filtering components.

## Next Stage

M2.58 should focus on:

- grouping skeleton fragments into stroke-level components;
- endpoint pairing under a coverage-aware cost;
- preserving low-jump execution improvements only when coverage risk does not worsen;
- separate handling for thick text glyphs, where pure centerline coverage is too low.

## Artifacts

- `tools/run_line_domain_topology_candidate.py`
- updated `tools/select_m2_line_domain_guard.py`
- `configs/best_current_model_m2_57_topology_open_trail_probe.json`
- `results/m2_57_topology_open_trail_probe/`
- `results/public_benchmark_v1_ext33_m2_57_topology_relative_rescue/`
- `results/public_benchmark_v1_ext33_m2_57_topology_relative_rescue_risk_audit/`

