# M2.61 Segment-Graph Probe and Component-Aware Jump Audit

Date: 2026-06-25

## Summary

M2.61 tested an explicit segment-graph/euler routing candidate for line-domain samples and added a component-aware jump audit. The routing candidate is **not promoted** because it did not improve the current M2.60 selector. The audit change is useful: it separates unavoidable jumps between disconnected visual components from excessive routing jumps.

Current default remains:

```text
M2.53/M2.51 robust policy
+ M2.55 line-domain guard
+ M2.58 coverage-aware rescue
+ M2.59 anchored quality rescue
+ M2.60 mask-path rescue
```

M2.61 adds an optional evaluation lens:

```text
component-aware jump audit
```

## Why This Was Needed

The previous risk audit marked `qd_007` as the worst sample mainly because of high jump count. Visual inspection showed that `qd_007` is a moon-face line drawing with separated strokes: one outer outline and several detached internal features. For this case, some jumps are structurally unavoidable unless the planner creates visible connector stitches through the black background.

So the correct question is not:

```text
How many jumps does the file have?
```

but:

```text
How many jumps exceed the number expected from disconnected visual components?
```

## Segment-Graph Candidate Probe

New candidate tool:

```text
tools/run_line_domain_segment_graph_merge_candidate.py
```

It tested:

- global component option selection
- lookahead endpoint scoring
- mask-route connection reuse
- euler walk mode for closed-loop components

Seven variants were summarized in:

```text
results/m2_61_segment_graph_component_jump_probe/variant_summary.csv
results/m2_61_segment_graph_component_jump_probe/probe_summary.json
```

Best segment-graph standalone variant:

| Variant | Mean Loss | Mean Jump | Mean Coverage | Mean Precision | Hard Fail |
|---|---:|---:|---:|---:|---:|
| `sg_euler_d0_l000_c20_i095_r32_f26_min12` | 0.059866 | 6.121212 | 0.922186 | 0.839258 | 0 |

This is still worse than M2.60:

| Selector | Mean Loss | Mean Jump | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|
| M2.60 | 0.048007 | 4.818182 | 0.980904 | 0.793610 |
| Best M2.61 segment-graph standalone | 0.059866 | 6.121212 | 0.922186 | 0.839258 |

For `qd_007`, the best conservative euler candidate stayed equal to M2.60:

| Sample | M2.60 Jump | M2.61 Best Jump | M2.60 Coverage | M2.61 Best Coverage |
|---|---:|---:|---:|---:|
| `qd_007` | 14 | 14 | 0.987986 | 0.987986 |

Higher lookahead made `qd_007` worse, increasing jump to 18. Therefore, this candidate is retained as a negative ablation, not as the new default.

## Component-Aware Jump Audit

Updated tool:

```text
tools/audit_m2_policy_risk.py
```

New optional parameters:

```text
--dataset-dir datasets/public_benchmark_v1_ext33
--component-aware-jump
--component-min-pixels 12
--component-jump-multiplier 2.0
--component-jump-bias 2.0
```

The audit computes:

```text
component_jump_allowance = component_jump_bias + component_jump_multiplier * mask_component_count
jump_excess_count = max(0, jump_count - component_jump_allowance)
```

When enabled, high-jump risk uses `jump_excess_count` instead of raw `jump_count`.

## Results

Component-aware audit output:

```text
results/public_benchmark_v1_ext33_m2_61_component_aware_jump_audit/
```

Risk comparison for M2.60:

| Audit | Mean Risk | Worst Risk | Worst Sample | Worst Flags |
|---|---:|---:|---|---|
| Original M2.60 audit | 0.12462978 | 0.38129687 | `qd_007` | `high_loss|high_jump|low_precision` |
| Component-aware audit | 0.09893281 | 0.25929687 | `qd_007` | `high_loss|low_precision` |

For `qd_007`:

| Metric | Value |
|---|---:|
| mask components | 5 |
| raw jump count | 14 |
| component jump allowance | 12 |
| jump excess count | 2 |

This means `qd_007` still needs review, but not because raw jump count alone is excessive. The remaining bottleneck is low precision and high loss.

## Research Interpretation

M2.61 confirms two points:

1. Naive segment-graph lookahead is not enough to improve line-domain routing. It can even worsen jump behavior when it changes a good local order.
2. Jump count should be interpreted relative to visual component structure. For disconnected line drawings, some jumps are necessary and should not be treated the same as avoidable planner fragmentation.

## Next Direction

The next useful model-side step is not more greedy endpoint ordering. It should target precision and repeated line tracing:

```text
M2.62 stroke precision rescue
```

Recommended components:

- detect duplicate/retrace-heavy skeleton walks
- prefer euler traversal only for truly eulerian components
- add stitch-pixel precision term into the line-domain selector
- add per-component coverage/precision diagnostics
- keep component-aware jump audit as the review metric

