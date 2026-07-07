# M2.81 Professional-Gate Stitch-Family Selector

Date: 2026-07-07

## Summary

M2.81 fixes the main limitation discovered after M2.80.

M2.80 added a conservative deployment margin after M2.79. It removed false texture promotions, but it was too conservative for actual professional texture deployment: when applied directly to the M2.74 texture candidates, it would reject all four safe texture switches.

M2.81 adds one non-leaky expert feature:

```text
allowed_by_professional_gate
```

This feature is available from the deterministic professional gate used to build the texture candidate pool. M2.81 does **not** use:

```text
teacher_is_texture_switch
```

That matters because `teacher_is_texture_switch` is a label-like field and would leak the answer.

## Tools

Updated:

```text
tools/train_stitch_family_selector.py
```

New:

```text
tools/apply_stitch_family_policy_texture_profile.py
```

Training command:

```powershell
python tools\train_stitch_family_selector.py `
  --seed-rows results\public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed\stitch_type_teacher_seed_rows.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_81_professional_gate_selector `
  --model-id m2_81_professional_gate_selector `
  --alpha 1.0 `
  --holdout-column source_name `
  --reject-weight 4.0 `
  --include-professional-gate-feature
```

Deployment calibration command:

```powershell
python tools\calibrate_stitch_family_deployment_policy.py `
  --predictions results\public_benchmark_v1_ext33_m2_81_professional_gate_selector\stitch_family_source_heldout_predictions.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_81_professional_gate_deployment_policy `
  --model-id m2_81_professional_gate_deployment_policy `
  --min-risky-margin 0.20
```

Actual profile application command:

```powershell
python tools\apply_stitch_family_policy_texture_profile.py --copy-outputs
```

## Source-Held-Out Result

| System | Accuracy | Positive acc | Non-base positive acc | Reject recall | False texture promotions |
|---|---:|---:|---:|---:|---:|
| M2.79 rejectw4 | 0.7411 | 0.7727 | 0.5946 | 0.7333 | 6 |
| M2.80 deployment policy | 0.8423 | 0.7576 | 0.5676 | 0.8630 | 0 |
| M2.81 professional-gate policy | 0.8423 | 0.8182 | 0.6757 | 0.8481 | 0 |

M2.81 keeps the most important M2.80 property:

```text
false texture promotions = 0
```

but recovers much more useful non-base positive recall:

```text
0.5676 -> 0.6757
```

This is the right professional tradeoff: keep unsafe texture promotions at zero, but recover safe stitch-family promotions when the deterministic professional gate supports them.

## Actual Texture Profile Application

M2.81 was applied to the M2.75 professional texture candidate pool:

```text
results/public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset/stitch_type_candidate_rows.csv
```

The selected actual texture switches are:

| Sample | Candidate kind |
|---|---|
| `ocp_005` | `dt_satin` |
| `omj_001` | `dt_satin` |
| `omj_003` | `dt_satin` |
| `omj_005` | `dt_satin` |

These are the same four professional texture switches promoted by M2.74.

Command-level profile metrics:

| Metric | Value |
|---|---:|
| samples | 33 |
| texture switches | 4 |
| hard fail | 0 |
| mean unified loss | 0.05639949 |
| mean jumps | 6.03030303 |
| mean trims | 0.60606061 |
| mean off-mask stitch length mm | 0.12120606 |
| mean visible connectors | 0.0 |
| mean coverage | 0.98103933 |
| mean precision | 0.83779552 |
| mean texture score | 0.20860606 |

## DST Output Hash Check

The copied `prediction.dst` files in:

```text
results/public_benchmark_v1_ext33_m2_81_family_policy_texture_profile
```

were compared against the current promoted M2.74 profile:

```text
results/public_benchmark_v1_ext33_m2_74_professional_texture_profile
```

Result:

```text
33 samples checked
0 missing outputs
0 SHA256 DST differences
```

So M2.81 does not destabilize the promoted output profile. It explains and reproduces the M2.74 choices through a learned stitch-family selector plus a deployment gate.

## Interpretation

M2.80 was safe but too conservative for actual texture deployment.

M2.81 is the better professional direction:

```text
deterministic professional gate
  + learned stitch-family selector
  + conservative deployment margin
```

This hybrid is more credible than using the learned selector alone. It also matches how a professional digitizing system should behave: a learned model can rank and generalize, but a deterministic safety gate protects the final DST output.

## Promotion Decision

M2.74 remains the promoted DST output profile because M2.81 intentionally reproduces the same actual DST outputs.

M2.81 is promoted as the current learned-control layer for the professional texture profile. The next step is to expand the candidate pool beyond `dt_satin` / `satin_rail` and test whether the learned gate can safely choose between running, fill, tatami-like, satin-like, and outline candidates in one integrated reranking stage.
