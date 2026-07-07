# M2.84 Preview-Gated Multifamily Policy

Date: 2026-07-07

## Summary

M2.84 turns the M2.83 diagnostic audit into an actual deployment gate.

M2.82 showed that multi-family candidates can pass command-level safety checks. M2.83 then showed that this is not enough: two M2.82 substitutions were command-safe but worse under the DST-derived preview-texture proxy.

M2.84 fixes that by adding:

```text
command safety gate
  + preview texture non-regression gate
```

This is a conservative quality policy. It is not a new generator and it does not claim final professional digitizing quality. Its job is to prevent false progress: a candidate may not replace the baseline if the preview texture score gets worse.

## Tool

New:

```text
tools/apply_preview_gated_multifamily_policy.py
```

Run:

```powershell
python tools\apply_preview_gated_multifamily_policy.py --copy-outputs
```

Inputs:

```text
results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile/family_policy_selected_rows.csv
results/public_benchmark_v1_ext33_m2_82_multifamily_candidate_policy/multifamily_selected_rows.csv
results/public_benchmark_v1_ext33_m2_83_dst_preview_texture_audit/dst_preview_texture_delta_rows.csv
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_84_preview_gated_multifamily_policy/preview_gate_decision_rows.csv
results/public_benchmark_v1_ext33_m2_84_preview_gated_multifamily_policy/preview_gate_selected_rows.csv
results/public_benchmark_v1_ext33_m2_84_preview_gated_multifamily_policy/preview_gate_switched_rows.csv
results/public_benchmark_v1_ext33_m2_84_preview_gated_multifamily_policy/preview_gate_summary.json
```

## Gate

M2.84 starts from the M2.82 selected candidates. A non-base candidate can remain selected only if:

| Gate | Threshold |
|---|---:|
| M2.82 policy pass | true |
| preview score delta | >= 0 |
| generator texture score delta | >= -0.005 |
| visible connector delta | <= 0 |
| off-mask length delta | <= 0 |

Otherwise the sample falls back to the M2.81 baseline profile.

## Result

M2.84 keeps 3 M2.82 substitutions:

| Sample | Candidate | Family | Preview delta | Texture delta |
|---|---|---|---:|---:|
| `ocp_010` | `auto_evalrepair_r10_c20` | `auto_fill` | 0.0 | 0.0 |
| `omj_001` | `dtsatin_safeadt_light` | `satin_like` | 0.0 | 0.0 |
| `omj_005` | `dtsatin_safeadt_light` | `satin_like` | 0.0 | 0.0 |

M2.84 rejects 2 M2.82 substitutions:

| Sample | Reason |
|---|---|
| `ocp_005` | preview score regression |
| `omj_003` | preview score regression |

Aggregate metrics:

| Metric | M2.84 |
|---|---:|
| samples | 33 |
| preview-gated switches | 3 |
| hard fail | 0 |
| mean unified loss | 0.05639949 |
| mean jumps | 6.03030303 |
| mean trims | 0.60606061 |
| mean off-mask length mm | 0.12120606 |
| mean visible connectors | 0.0 |
| mean coverage | 0.98103933 |
| mean precision | 0.83779552 |
| mean professional preview score | 0.73007454 |
| mean generator texture score | 0.20860606 |

## Interpretation

M2.84 is not a headline metric improvement over M2.81. That is the point.

It blocks a misleading upgrade path:

```text
more stitch-family variety
  -> command-safe
  -> but preview texture worse
```

This is important for the professional-version goal. A professional pipeline must not select a candidate simply because it has more stitch types or slightly better coverage. It must also preserve or improve the visual stitch texture.

## Next Step

The next real improvement should generate candidates that can pass the M2.84 gate with positive preview-texture gains.

That means optimizing the generator itself:

```text
better satin/tatami geometry
better thread-direction fields
better color-region separation
real stitch-preview loss or HITL preference labels
```

M2.84 gives us the guardrail for those experiments.
