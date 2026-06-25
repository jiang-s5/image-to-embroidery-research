# M2.58 Coverage-Aware Relative Rescue

Date: 2026-06-25

M2.58 follows the M2.57 topology open-trail probe.

M2.57 found a topology candidate that improved the `qd_007` execution numbers, but only by dropping coverage too much:

| Metric | M2.55 baseline | M2.57 relative rescue |
|---|---:|---:|
| `qd_007` loss | 0.128886 | 0.114131 |
| `qd_007` jump | 14 | 13 |
| `qd_007` trim | 2 | 1 |
| `qd_007` coverage | 0.933862 | 0.723805 |
| `qd_007` precision | 0.661752 | 0.657442 |

This is a reward-hacking failure mode: the selector appears to improve execution metrics, but it loses too much of the intended line structure.

## Change

M2.58 adds a relative coverage-drop gate to:

```text
tools/select_m2_line_domain_guard.py
```

New argument:

```text
--line-rescue-max-coverage-drop
```

Default:

```text
0.05
```

A relative rescue candidate is now allowed only when:

```text
candidate_coverage >= baseline_coverage - max_coverage_drop
```

along with the existing loss, jump, precision, visible, and off-mask checks.

## Public Ext33 Result

Run:

```text
results/public_benchmark_v1_ext33_m2_58_coverage_aware_relative_rescue/
```

Risk audit:

```text
results/public_benchmark_v1_ext33_m2_58_coverage_aware_risk_audit/
```

Compared with M2.55 and M2.57:

| Metric | M2.55 | M2.57 rescue | M2.58 coverage-aware |
|---|---:|---:|---:|
| samples | 33 | 33 | 33 |
| hard fail | 0 | 0 | 0 |
| mean loss | 0.048797 | 0.048350 | 0.048797 |
| mean jump | 4.909091 | 4.878788 | 4.909091 |
| mean trim | 0.606061 | 0.575758 | 0.606061 |
| off-mask mm | 0.199076 | 0.199076 | 0.199076 |
| visible connector | 0.000000 | 0.000000 | 0.000000 |
| mean coverage | 0.979200 | 0.972835 | 0.979200 |
| mean precision | 0.793370 | 0.793240 | 0.793370 |
| review samples | 15 | 15 | 15 |
| mean risk score | 0.127757 | 0.131225 | 0.127757 |
| worst risk score | 0.392948 | 0.507368 | 0.392948 |
| worst sample | `qd_007` | `qd_007` | `qd_007` |

M2.58 rejects the M2.57 `qd_007` rescue because the candidate coverage delta is approximately:

```text
0.723805 - 0.933862 = -0.210057
```

which is below the allowed `-0.05` drop.

## Interpretation

M2.58 is not a large quality jump. It is a safety improvement.

It prevents topology candidates from being accepted when they reduce jump/loss by deleting too much stitch coverage. This makes the selector more robust against reward hacking.

The current recommended default is therefore:

```text
M2.53/M2.51 low-jump robust policy
+ M2.55 bounded line-domain guard
+ M2.58 coverage-aware relative rescue gate
```

On the current public benchmark this produces the same metric surface as M2.55, but it safely blocks the M2.57 coverage-collapse case.

## Remaining Limitation

M2.58 does not solve `qd_007`; it only avoids selecting a worse-looking structural tradeoff.

The next generator-side step remains:

```text
M2.59 coverage-aware stroke grouping
```

Required next work:

- group skeleton fragments into stroke-level components;
- optimize endpoint pairing with both jump and coverage costs;
- preserve line coverage before accepting lower-jump trails;
- handle thick text glyphs separately from thin QuickDraw strokes.

## Artifacts

- `tools/select_m2_line_domain_guard.py`
- `configs/best_current_model_m2_58_coverage_aware_relative_rescue.json`
- `results/public_benchmark_v1_ext33_m2_58_coverage_aware_relative_rescue/`
- `results/public_benchmark_v1_ext33_m2_58_coverage_aware_risk_audit/`

