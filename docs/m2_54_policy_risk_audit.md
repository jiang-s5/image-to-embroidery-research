# M2.54 Policy Risk Audit

Date: 2026-06-25

M2.54 continues from M2.53.

M2.53 selected a cross-evaluation robust policy, but it still reported aggregate behavior. M2.54 adds a sample-level and source-level risk audit so that a good mean score cannot hide bad individual cases.

This is an evaluation/diagnostic layer, not a new planner model.

## Tool

```text
tools/audit_m2_policy_risk.py
```

The tool reads selected policy rows and writes:

- `sample_risk_rows.csv`
- `review_queue.csv`
- `risk_by_selection.csv`
- `risk_by_source.csv`
- `risk_by_category.csv`
- `risk_audit_summary.json`

## Risk Thresholds

This run uses:

```text
max_loss = 0.10
max_jump = 10
max_trim = 3
max_off_mask = 1.0 mm
max_visible = 0
min_coverage = 0.90
min_precision = 0.75
```

A sample enters the review queue if it violates any threshold or has `quality_level=warning/fail/hard_fail`.

## Inputs

| Selection | Source CSV |
|---|---|
| incoming_balanced | `results/incoming_review_eval_v1_m2_51_benefitgate_sweep/best_selected_balanced.csv` |
| incoming_low_jump | `results/incoming_review_eval_v1_m2_51_benefitgate_sweep/best_selected_low_jump.csv` |
| public_balanced | `results/public_benchmark_v1_ext33_m2_51_benefitgate_sweep_sanity/best_selected_balanced.csv` |
| public_low_jump | `results/public_benchmark_v1_ext33_m2_51_benefitgate_sweep_sanity/best_selected_low_jump.csv` |

## Overall Result

Across 74 selected rows:

```text
review samples = 35
```

Flag counts:

| Flag | Count |
|---|---:|
| low_precision | 26 |
| high_jump | 6 |
| low_coverage | 5 |
| high_loss | 4 |
| high_trim | 4 |
| off_mask | 3 |
| quality_warning | 2 |

The important finding is that the dominant issue is no longer visible connectors or hard failures. It is low stitch precision, especially on line-drawing inputs.

## Selection-Level Summary

| Selection | Samples | Review Samples | Review Rate | Mean Loss | Mean Jump | Mean Coverage | Mean Precision | Worst Sample | Worst Flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| incoming_balanced | 4 | 3 | 0.750000 | 0.073831 | 7.250000 | 0.920700 | 0.788361 | pair_008 | quality_warning / high_loss / high_jump / low_precision |
| incoming_low_jump | 4 | 3 | 0.750000 | 0.073831 | 7.250000 | 0.920700 | 0.788361 | pair_008 | quality_warning / high_loss / high_jump / low_precision |
| public_balanced | 33 | 14 | 0.424242 | 0.052103 | 5.606061 | 0.982012 | 0.806207 | qd_007 | high_loss / high_jump / low_precision |
| public_low_jump | 33 | 15 | 0.454545 | 0.048809 | 4.909091 | 0.979476 | 0.793264 | qd_007 | high_loss / high_jump / low_precision |

## Source-Level Finding

QuickDraw is the major weakness:

| Selection | Source | Samples | Review Samples | Review Rate | Mean Jump | Mean Precision |
|---|---|---:|---:|---:|---:|---:|
| public_balanced | QuickDraw | 8 | 8 | 1.000000 | 7.750000 | low |
| public_low_jump | QuickDraw | 8 | 8 | 1.000000 | 7.250000 | low |

Rendered text is the second weaker source, with 3 of 5 samples needing review in both public profiles.

Openclipart and OpenMoji are much more stable:

- Openclipart review rate: 0.10 balanced, 0.20 low_jump
- OpenMoji review rate: 0.20 balanced, 0.20 low_jump

## Top Failure Cases

1. `pair_008` on incoming review:
   - loss `0.151280`
   - jump `16`
   - precision `0.670418`
   - flags: warning, high loss, high jump, low precision

2. `qd_007` QuickDraw moon:
   - balanced: loss `0.142919`, jump `15`, precision `0.606869`
   - low_jump: loss `0.128886`, jump `14`, precision `0.661752`
   - flags: high loss, high jump, low precision

3. `qd_001` QuickDraw cat:
   - balanced: jump `11`, precision `0.649960`
   - low_jump: precision `0.510740`

4. `pair_006` on incoming review:
   - selected candidate has very low jump but coverage only `0.696636`
   - flags: high trim, low coverage

## Interpretation

M2.54 changes the next optimization target.

Before this audit, the main story was:

```text
reduce jump while preserving coverage
```

After this audit, the more precise target is:

```text
preserve the current low-jump default for flat/icon inputs,
but add a line-drawing/running-stitch branch for QuickDraw-like and text-like inputs.
```

This also explains why a single low-jump policy is not enough. Line drawings need a different planner family:

- centerline / skeleton preservation;
- fewer raster fill rows;
- better stroke continuity;
- precision-aware candidate selection;
- source-aware or image-feature-aware routing.

## Next Recommended Optimization

M2.55 should add a line-domain guard:

```text
if source/image features indicate line drawing or text:
    increase precision weight
    prefer skeleton / centerline candidates
    add jump cap or jump-aware rerank
else:
    keep current low_jump robust policy
```

The immediate targets are:

- reduce QuickDraw review rate below 1.0;
- reduce `qd_007` jump below 10;
- improve QuickDraw precision above 0.75 where possible;
- keep Openclipart/OpenMoji stable.

## Artifacts

- `tools/audit_m2_policy_risk.py`
- `configs/best_current_model_m2_54_policy_risk_audit.json`
- `results/m2_54_policy_risk_audit/`

