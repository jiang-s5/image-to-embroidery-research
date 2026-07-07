# M2.82 Multifamily Executable Candidate Policy

Date: 2026-07-07

## Summary

M2.82 is the first integrated multi-family deployment audit after M2.81.

M2.81 learned to reproduce the M2.74 professional texture profile. It was still mostly a texture-control layer around `dt_satin` and `satin_rail` candidates.

M2.82 expands the executable candidate pool to include:

```text
base_keep
running_line
auto_running
auto_fill
fill_tatami_like
satin_like
outline_border
style_aware_mixed
```

The goal is not to force more texture. The goal is to ask a stricter question:

```text
Can a non-base stitch family replace the current M2.81 output
without increasing hard failure, visible connector risk, off-mask length,
jump count, trim count, or major coverage / precision loss?
```

## Tool

New:

```text
tools/apply_multifamily_executable_candidate_policy.py
```

Run command:

```powershell
python tools\apply_multifamily_executable_candidate_policy.py --copy-outputs
```

The tool uses:

```text
results/public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed/stitch_type_teacher_seed_rows.csv
results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv
results/public_benchmark_v1_ext33_m2_81_professional_gate_selector/stitch_family_selector_model.json
configs/best_current_model_m2_12_maskfill_selector.json
results/public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset/stitch_type_candidate_rows.csv
```

It writes:

```text
results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_decision_rows.csv
results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_selected_rows.csv
results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_switched_rows.csv
results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_summary.json
```

## Safety Gates

M2.82 keeps M2.81 as the baseline for every sample. A candidate can replace it only if it passes all gates:

| Gate | Threshold |
|---|---:|
| deployed family | not `reject` |
| hard fail | forbidden |
| visible connector increase | <= 0 |
| off-mask length increase | <= 0 |
| unified loss increase | <= 0.01 |
| jump increase | <= 2 |
| trim increase | <= 2 |
| coverage drop | <= 0.015 |
| precision drop | <= 0.03 |
| risky-family margin vs reject | >= 0.20 |

`base_keep` is a fallback, not a new policy switch. M2.82 only counts non-base candidates as actual switches.

## Candidate Audit

| Item | Count |
|---|---:|
| decision rows | 660 |
| executable candidate rows | 660 |
| safety-pass rows | 52 |
| promotable non-base rows | 19 |

Safety-pass rows by deployed family:

| Family | Rows |
|---|---:|
| base_keep | 31 |
| satin_like | 14 |
| auto_fill | 5 |
| running_line | 2 |

Most candidates were rejected because the learned family policy rejected them or because they increased unified loss / coverage risk.

## Selected Output Profile

M2.82 selected 5 non-base replacements across 33 public benchmark samples:

| Sample | Candidate |
|---|---|
| `ocp_005` | `satinrail_w5_s16_i5_cap45` |
| `ocp_010` | `auto_evalrepair_r10_c20` |
| `omj_001` | `dtsatin_safeadt_light` |
| `omj_003` | `dtsatin_safeadt_light` |
| `omj_005` | `satinrail_safeadt_light` |

Selected family counts:

| Family | Samples |
|---|---:|
| base_keep / M2.81 fallback | 28 |
| satin_like | 4 |
| auto_fill | 1 |

## Metrics

| Metric | M2.81 | M2.82 | Delta |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| non-base switches | 4 | 5 | +1 |
| hard fail | 0 | 0 | 0 |
| mean unified loss | 0.05639949 | 0.05640053 | +0.00000104 |
| mean jumps | 6.03030303 | 6.03030303 | 0 |
| mean trims | 0.60606061 | 0.60606061 | 0 |
| mean off-mask stitch length mm | 0.12120606 | 0.12120606 | 0 |
| mean visible connectors | 0.0 | 0.0 | 0 |
| mean coverage | 0.98103933 | 0.98105379 | +0.00001446 |
| mean precision | 0.83779552 | 0.83837373 | +0.00057821 |

## Interpretation

M2.82 is a real step, but a small one.

It proves that the M2.81 learned professional gate can be used beyond the original texture-only pool. The expanded pool contains running, fill, tatami-like, satin-like, outline, and style-aware candidates, and the policy can safely select a few non-base outputs without destabilizing command-level metrics.

However, the aggregate improvement is tiny:

```text
coverage +0.00001446
precision +0.00057821
unified loss +0.00000104
```

So M2.82 should be treated as the latest experimental multi-family candidate policy, not as proof that the system has reached professional digitizing quality.

## Promotion Decision

M2.82 supersedes M2.81 as the latest integrated learned-control experiment because it safely expands the candidate pool and selects 5 non-base outputs.

M2.74 remains the conservative promoted output profile until the 5 M2.82 switched outputs receive visual stitch-preview review. The next step is to add a stitch-preview / texture realism evaluator so that richer stitch families are promoted for visible embroidery quality, not only for command-level safety.
