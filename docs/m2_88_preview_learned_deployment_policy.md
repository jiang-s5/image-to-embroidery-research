# M2.88 Preview Learned Deployment Policy

Date: 2026-07-07

## Summary

M2.88 is the first step that wires the M2.87 preview-aware learned selector into the final DST deployment path.

M2.87 answered:

```text
Can preview-positive and preview-rejected candidates improve the learned selector?
```

M2.88 answers:

```text
Can that learned selector safely control final candidate deployment?
```

The answer is:

```text
yes, but only as a strict learned confirmation layer after the M2.86 command/preview gate.
```

M2.88 does not beat M2.86 on final DST metrics. It reproduces M2.86 exactly under the strict deployment policy while adding learned-selector evidence for all candidate decisions.

## Tool

New:

```text
tools/apply_preview_learned_deployment_policy.py
```

Default strict run:

```powershell
python tools\apply_preview_learned_deployment_policy.py --copy-outputs
```

Inputs:

```text
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_selected_rows.csv
results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/stitch_family_selector_model.json
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy/preview_learned_decision_rows.csv
results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy/preview_learned_selected_rows.csv
results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy/preview_learned_switched_rows.csv
results/public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy/preview_learned_summary.json
```

## Strict Deployment Policy

M2.88 strict uses:

```text
M2.86 preview_sweep_gate_pass == 1
AND
M2.87 deployed family != reject
```

Then it reranks eligible rows with:

```text
preview_sweep_score + 0.001 * selector_margin_vs_reject
```

The learned selector is used as:

```text
learned safety / texture confirmation
```

not as:

```text
exact stitch-family classifier
```

This distinction matters because the M2.87 selector maps the accepted `ocp_005` satin-like output to the broader learned `fill_tatami_like` family. Requiring exact family agreement would incorrectly reject a safe, high-preview candidate.

## Strict Result

M2.88 strict exactly reproduces M2.86 final outputs.

| Metric | M2.86 | M2.88 strict | Delta |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| switches | 7 | 7 | 0 |
| hard fail | 0 | 0 | 0 |
| mean unified loss | 0.05706876 | 0.05706876 | 0 |
| mean jumps | 6.15151515 | 6.15151515 | 0 |
| mean trims | 0.60606061 | 0.60606061 | 0 |
| mean off-mask length mm | 0.10619091 | 0.10619091 | 0 |
| mean visible connectors | 0.0 | 0.0 | 0 |
| mean coverage | 0.98167906 | 0.98167906 | 0 |
| mean precision | 0.83918282 | 0.83918282 | 0 |
| mean professional preview score | 0.73544111 | 0.73544111 | 0 |
| mean generator texture score | 0.22612121 | 0.22612121 | 0 |

The M2.88 strict selector accepts all `34` M2.86 gate-pass rows and rejects or leaves out all gate-failed rows:

| Item | Count |
|---|---:|
| audited candidate rows | 1584 |
| M2.86 preview gate pass rows | 34 |
| M2.87 selector non-reject rows | 40 |
| strict eligible rows | 34 |
| final selected switches | 7 |
| final outputs different from M2.86 | 0 |

## No-Gate Probe

I also ran the dangerous probe:

```powershell
python tools\apply_preview_learned_deployment_policy.py `
  --no-require-preview-gate `
  --output-dir results\public_benchmark_v1_ext33_m2_88_preview_learned_deployment_policy_relaxed_nogate
```

This asks whether the M2.87 learned selector can replace the M2.86 hard gate.

Result:

| Metric | M2.86 | No-gate learned probe | Delta |
|---|---:|---:|---:|
| hard fail | 0 | 3 | +3 |
| mean unified loss | 0.05706876 | 0.09428522 | +0.03721646 |
| mean jumps | 6.15151515 | 59.42424242 | +53.27272727 |
| mean trims | 0.60606061 | 8.66666667 | +8.06060606 |
| mean off-mask length mm | 0.10619091 | 4.58460303 | +4.47841212 |
| mean visible connectors | 0.0 | 1.39393939 | +1.39393939 |
| mean precision | 0.83918282 | 0.82807694 | -0.01110588 |
| mean professional preview score | 0.73544111 | 0.74095152 | +0.00551041 |
| mean generator texture score | 0.22612121 | 0.24442424 | +0.01830303 |

The no-gate probe is not promoted.

It gives small preview and texture gains, but it creates hard failures and excessive jump/trim/off-mask risk. This is the strongest evidence from M2.88:

```text
M2.87 has learned useful accept/reject structure, but it is not yet strong enough to replace deterministic command-safety and preview gates.
```

## Interpretation

M2.88 is a deployment integration result, not a new quality leap.

It upgrades the system state from:

```text
M2.86 deterministic preview sweep selector
M2.87 learned selector validated only as held-out classifier
```

to:

```text
M2.86 final output metrics preserved
M2.87 learned selector wired into final deployment as a strict confirmation layer
```

Current promoted output remains M2.86-equivalent, now with M2.88 learned-control evidence.

## Research Meaning

The important conclusion is negative-but-useful:

```text
learned preview selection cannot yet be trusted without the hard safety gate.
```

For the paper, M2.88 supports the claim that:

- learned selector evidence can be integrated without degrading final DST metrics;
- deterministic gates are still necessary for executable stitch quality;
- preview/texture-only learning can reward unsafe candidates unless paired with command-level constraints.

## Next Step

The next useful optimization is not to remove the M2.86 gate. It is to teach the model why the no-gate failures happened.

Recommended M2.89 direction:

```text
M2.89 = hard-failure-aware preview selector refresh
```

Use the no-gate false positives:

```text
ocp_001
ocp_009
qd_007
```

as explicit hard negatives with stronger features:

- jump explosion;
- trim explosion;
- off-mask increase;
- visible connector increase;
- command-safety failure reason;
- preview gain versus execution penalty.

The next target should be:

```text
keep strict M2.86/M2.88 safety,
but improve learned rejection of no-gate false positives before trying another relaxed deployment.
```
