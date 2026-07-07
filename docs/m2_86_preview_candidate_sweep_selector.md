# M2.86 Preview Candidate Sweep Selector

Date: 2026-07-07

## Summary

M2.86 expands M2.85 from a fixed candidate table into a sweep over existing ext33 generator result roots.

M2.85 found only one preview-positive candidate inside the M2.82 decision table. M2.86 asks a broader question:

```text
Across all existing ext33 generator roots with command and coverage metrics,
which candidates improve the DST-derived preview score without breaking command safety?
```

Answer:

```text
7 samples can be switched safely under strict no-regression gates.
```

The selected switches are:

```text
ocp_005 -> mask_fill_edgewalk_satin_rail
ocp_006 -> mask_fill_edgewalk
ocp_007 -> mask_fill_edgewalk
ocp_008 -> mask_fill_edgewalk
omj_007 -> mask_fill_edgewalk
omj_010 -> mask_fill_edgewalk
qd_001  -> mask_fill_edgewalk
```

## Tool

New:

```text
tools/apply_preview_candidate_sweep_selector.py
```

Run:

```powershell
python tools\apply_preview_candidate_sweep_selector.py
```

The tool auto-discovers candidate roots under `results/` using:

```text
include: satin|dtsatin|edgewalk|outline|styleaware|mask_fill|nearestrow
exclude: selector|m2_81|m2_82|m2_83|m2_84|m2_85|eval_masks|repair_veto|geometry_priors
```

It requires each candidate root to contain:

```text
source_aware_hybrid_rows.csv
coverage_rows.csv
sample_id/prediction.dst
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_root_inventory.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_selected_rows.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_switched_rows.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_summary.json
```

## Candidate Audit

M2.86 scans a larger pool than M2.85.

| Item | Count |
|---|---:|
| candidate roots | 48 |
| candidate sample dirs | 1584 |
| audited candidate rows | 1584 |
| rows with positive preview score | 924 |
| rows passing command + preview gates | 34 |

Gate reasons:

| Reason | Rows |
|---|---:|
| preview gain too small | 675 |
| loss increase | 456 |
| off-mask increase | 309 |
| precision drop | 82 |
| visible connector increase | 27 |
| texture regression too large | 1 |
| preview sweep candidate pass | 34 |

This is the main difference from M2.85. Many candidates have positive preview score, but the command gates still reject most of them.

## Aggregate Result

| Metric | M2.85 | M2.86 | Delta |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| preview-positive switches | 1 | 7 | +6 |
| hard fail | 0 | 0 | 0 |
| mean unified loss | 0.05639949 | 0.05706876 | +0.00066927 |
| mean jumps | 6.03030303 | 6.15151515 | +0.12121212 |
| mean trims | 0.60606061 | 0.60606061 | 0 |
| mean off-mask length mm | 0.12120606 | 0.10619091 | -0.01501515 |
| mean visible connectors | 0.0 | 0.0 | 0 |
| mean coverage | 0.98103933 | 0.98167906 | +0.00063973 |
| mean precision | 0.83804397 | 0.83918282 | +0.00113885 |
| mean professional preview score | 0.73047417 | 0.73544111 | +0.00496694 |
| mean generator texture score | 0.20769697 | 0.22612121 | +0.01842424 |

## Interpretation

M2.86 is a stronger professional-looking step than M2.85 because it improves both:

- DST-derived professional preview score;
- generator texture score.

It does this while preserving:

- `hard_fail = 0`;
- `visible_connector_count = 0`;
- unchanged mean trim count;
- lower mean off-mask stitch length.

The cost is small but real:

- mean jump count increases by `0.1212`;
- mean unified loss increases by `0.000669`.

This is acceptable for the current objective because M2.86 is specifically optimizing toward better preview texture under strict safety gates, not purely minimizing command count.

## Current Limitation

M2.86 is still a selector over existing generator roots. It does not yet create fundamentally new stitch families or learn a preview-aware generator.

The result should be described as:

```text
a preview-aware sweep selector that finds safer professional-texture substitutions
```

not as:

```text
a complete professional digitizer
```

## Next Step

The next meaningful step is to turn the M2.86 accepted rows into teacher data:

```text
M2.87 = preview-aware teacher refresh
```

Recommended direction:

- add the 7 accepted M2.86 switches to the stitch-family teacher table;
- train a preview-aware stitch-family selector with preview gain as an explicit label;
- add a hard penalty for candidates rejected by preview, off-mask, visible connector, or precision gates;
- keep M2.86 as the deterministic deployment selector until the learned selector beats it source-held-out.
