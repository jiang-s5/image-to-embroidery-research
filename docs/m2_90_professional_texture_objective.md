# M2.90 Professional Texture Objective

Date: 2026-07-07

## Summary

M2.90 is the first profile in this line that explicitly optimizes professional stitch appearance instead of only preserving command safety.

M2.89 made the learned selector safer by feeding no-gate hard failures back as hard negatives. That was necessary, but it did not make the preview look more like a professional embroidery file.

M2.90 asks a different question:

```text
Can we accept bounded execution tradeoffs when the DST-derived preview texture improves substantially?
```

The answer is yes, as an experimental professional-texture profile.

M2.90 hard texture improves the mean professional preview score from `0.73544111` to `0.77904526` and the mean generator texture score from `0.22612121` to `0.69909091`, while keeping `0` hard_fail and `0` visible connectors. The tradeoff is higher unified loss and jumps.

## Tool

New:

```text
tools/apply_professional_texture_objective_selector.py
```

Default hard-texture run:

```powershell
python tools\apply_professional_texture_objective_selector.py
```

Balanced run:

```powershell
python tools\apply_professional_texture_objective_selector.py `
  --model-id m2_90_professional_texture_objective_balanced `
  --output-dir results\public_benchmark_v1_ext33_m2_90_professional_texture_objective_balanced `
  --max-loss-increase 0.08 `
  --max-jump-increase 8 `
  --max-trim-increase 8 `
  --max-precision-drop 0.12
```

Inputs:

```text
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv
results/public_benchmark_v1_ext33_m2_89_hard_failure_deployment_strict/preview_learned_selected_rows.csv
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_90_professional_texture_objective/professional_texture_decision_rows.csv
results/public_benchmark_v1_ext33_m2_90_professional_texture_objective/professional_texture_selected_rows.csv
results/public_benchmark_v1_ext33_m2_90_professional_texture_objective/professional_texture_switched_rows.csv
results/public_benchmark_v1_ext33_m2_90_professional_texture_objective/professional_texture_summary.json
```

## Selection Logic

M2.90 keeps hard constraints on:

```text
round-trip DST parse success
hard_fail quality level
visible connectors
absolute off-mask length
off-mask increase
coverage floor
precision floor
```

Then it scores candidates with:

```text
professional preview gain
generator texture gain
family texture gain
directional gain
rhythm gain
coverage gain
structured stitch-family bonus
minus bounded execution penalties
```

This is different from M2.86.

M2.86 selected only candidates that stayed close to the conservative command baseline. M2.90 allows a larger execution tradeoff when the preview-texture gain is large and the hard safety gates remain clean.

## Candidate Audit

The hard-texture profile audits the same `1584` candidate rows as M2.86.

| Item | Count |
|---|---:|
| candidate rows | 1584 |
| M2.90 gate-pass rows | 162 |
| selected switches | 19 |
| hard_fail final outputs | 0 |
| visible connectors final outputs | 0 |

Gate-pass rows by family:

| Family | Rows |
|---|---:|
| edgewalk_fill | 93 |
| dt_satin | 32 |
| satin_like | 32 |
| nearestrow_fill | 4 |
| style_running | 1 |

This is important because M2.90 is no longer just selecting one fill generator. It begins to pick candidates from several stitch families, especially `dt_satin`, `satin_like`, and `edgewalk_fill`.

## Hard-Texture Result

| Metric | M2.89 strict | M2.90 hard texture | Delta |
|---|---:|---:|---:|
| samples | 33 | 33 | 0 |
| switches | 7 | 19 | +12 |
| hard_fail | 0 | 0 | 0 |
| mean unified loss | 0.05706876 | 0.08998194 | +0.03291318 |
| mean jumps | 6.15151515 | 9.87878788 | +3.72727273 |
| mean trims | 0.60606061 | 1.24242424 | +0.63636363 |
| mean off-mask length mm | 0.10619091 | 0.01501515 | -0.09117576 |
| mean visible connectors | 0.0 | 0.0 | 0 |
| mean coverage | 0.98167906 | 0.99228345 | +0.01060439 |
| mean precision | 0.83918282 | 0.82633067 | -0.01285215 |
| mean professional preview score | 0.73544111 | 0.77904526 | +0.04360415 |
| mean generator texture score | 0.22612121 | 0.69909091 | +0.47296970 |
| mean family texture score | 0.25195960 | 0.59006061 | +0.33810101 |

Switched samples:

```text
ocp_001
ocp_002
ocp_003
ocp_004
ocp_005
ocp_006
ocp_007
ocp_008
ocp_009
ocp_010
omj_004
omj_006
omj_007
omj_008
omj_010
qd_005
txt_002
txt_003
txt_005
```

## Balanced Result

The balanced profile uses stricter loss and jump limits.

| Metric | M2.90 balanced |
|---|---:|
| switches | 17 |
| hard_fail | 0 |
| mean unified loss | 0.07617060 |
| mean jumps | 8.33333333 |
| mean trims | 1.03030303 |
| mean off-mask length mm | 0.03003030 |
| mean visible connectors | 0.0 |
| mean professional preview score | 0.76903936 |
| mean generator texture score | 0.60981818 |
| mean family texture score | 0.51547475 |

Compared with M2.89, balanced still gains:

```text
+0.03359825 professional preview
+0.38369697 generator texture
+0.26351515 family texture
```

with a smaller execution-cost increase than hard texture.

## Interpretation

M2.90 is the first meaningful movement toward the user's professional-preview complaint:

```text
"the output still looks like one kind of simple line stitch"
```

It starts selecting candidates with richer stitch-family texture, especially satin-like and double-satin-like outputs for Openclipart and OpenMoji samples.

However, M2.90 is still not a full professional digitizer. It does not yet synthesize new stitch plans; it selects among existing generator roots. The professional appearance improves only when a suitable candidate already exists in the candidate pool.

## Current Position

Use the profiles this way:

| Profile | Use |
|---|---|
| M2.89 strict | conservative safety baseline |
| M2.90 balanced | safer professional-texture ablation |
| M2.90 hard texture | strongest current professional-texture profile |

M2.90 should be described as:

```text
an experimental professional-texture selector with hard safety gates
```

not as:

```text
a complete commercial embroidery digitizer
```

## Next Step

M2.91 should stop relying only on selecting existing roots. The next step is to generate richer stitch-family candidates directly:

- object-aware fill direction;
- explicit satin rails for outlines and narrow components;
- outline-underlay plus fill-overlap sequencing;
- source-aware stitch-family priors;
- preview-rendered A/B visual reports for selected switches.

The key research target is:

```text
professional stitch-family generation, not just professional stitch-family selection.
```
