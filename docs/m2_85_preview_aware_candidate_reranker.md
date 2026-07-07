# M2.85 Preview-Aware Candidate Reranker

Date: 2026-07-07

## Summary

M2.85 is the first step after the M2.84 guardrail that actually searches for a positive preview-texture gain.

M2.84 prevented bad promotions. M2.85 asks a stronger question:

```text
Among the whole executable M2.82 candidate pool,
is there any command-safe non-base candidate that improves the DST-derived preview score?
```

Answer:

```text
yes, but only one candidate currently passes all gates.
```

The selected candidate is:

```text
sample: ocp_005
candidate: satinrail_safeadt_light
family: satin_like
```

## Tool

New:

```text
tools/apply_preview_aware_candidate_reranker.py
```

Run:

```powershell
python tools\apply_preview_aware_candidate_reranker.py --copy-outputs
```

Inputs:

```text
results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv
results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_decision_rows.csv
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_85_preview_aware_candidate_reranker/preview_candidate_audit_rows.csv
results/public_benchmark_v1_ext33_m2_85_preview_aware_candidate_reranker/preview_aware_selected_rows.csv
results/public_benchmark_v1_ext33_m2_85_preview_aware_candidate_reranker/preview_aware_switched_rows.csv
results/public_benchmark_v1_ext33_m2_85_preview_aware_candidate_reranker/preview_aware_summary.json
```

## Candidate Audit

M2.85 audits every executable non-base candidate from the M2.82 decision table using the M2.83 DST-derived preview metric.

| Item | Count |
|---|---:|
| audited non-base rows | 627 |
| rows with positive preview score | 219 |
| rows passing command + preview gates | 1 |

Gate reasons:

| Reason | Rows |
|---|---:|
| M2.82 policy failed | 606 |
| preview gain too small | 20 |
| preview-aware candidate pass | 1 |

This is a useful diagnostic. Many candidates look better by preview score alone, but almost all of them are still rejected by command safety, learned family deployment, or no-regression constraints.

## Selected Candidate

| Field | Value |
|---|---:|
| sample | `ocp_005` |
| candidate | `satinrail_safeadt_light` |
| family | `satin_like` |
| preview delta | +0.01318776 |
| precision delta | +0.008199 |
| texture-score delta | -0.03 |
| unified loss delta | 0 |
| jump delta | 0 |
| trim delta | 0 |
| visible connector delta | 0 |
| off-mask delta | 0 |

The important part is that the improvement is measured from the actual DST stitch trajectory, not from the coarse generator texture count alone.

## Aggregate Result

| Metric | M2.84 | M2.85 | Delta |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| selected non-base switches | 3 neutral | 1 positive | - |
| hard fail | 0 | 0 | 0 |
| mean unified loss | 0.05639949 | 0.05639949 | 0 |
| mean jumps | 6.03030303 | 6.03030303 | 0 |
| mean trims | 0.60606061 | 0.60606061 | 0 |
| mean off-mask length mm | 0.12120606 | 0.12120606 | 0 |
| mean visible connectors | 0.0 | 0.0 | 0 |
| mean coverage | 0.98103933 | 0.98103933 | 0 |
| mean precision | 0.83779552 | 0.83804397 | +0.00024845 |
| mean professional preview score | 0.73007454 | 0.73047417 | +0.00039963 |
| mean generator texture score | 0.20860606 | 0.20769697 | -0.00090909 |

## Interpretation

M2.85 gives a more useful result than M2.84:

```text
M2.84 = do not get worse
M2.85 = find at least one candidate that gets better
```

The gain is small, but it is a real movement toward professional output because the selected candidate improves the DST-derived preview texture proxy while preserving all command-level safety metrics.

The negative texture-score delta is also informative. It shows that the older generator-count texture score is too coarse: it penalizes a candidate that the actual DST trajectory preview score prefers.

## Next Step

The next optimization should make the generator produce more candidates like `ocp_005 / satinrail_safeadt_light`, but for more samples.

Recommended next direction:

```text
M2.86 = preview-aware candidate generation sweep
```

Specifically:

- generate multiple satin-rail / DT-satin variants per eligible shape sample;
- evaluate each candidate with M2.83;
- select with M2.85's command + preview-positive reranker;
- use the accepted candidates as teacher rows for a future learned preview-aware stitch-family model.
