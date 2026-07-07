# M2.87 Preview-Aware Teacher Refresh and Selector

Date: 2026-07-07

## Summary

M2.87 turns the M2.86 preview-aware sweep result into training signal.

M2.86 was still deterministic selection:

```text
scan candidate DST roots -> audit command safety + preview texture -> pick safe preview-positive candidates
```

M2.87 asks the next question:

```text
Can the learned stitch-family selector absorb those preview-positive and preview-rejected examples?
```

Answer:

```text
yes, as a learned-control improvement, but not yet as a promoted output profile.
```

The selected M2.87 learned selector is:

```text
m2_87_preview_selector_rejectw16_margin010
```

It is selected because it keeps:

```text
false texture promotions = 0
```

while improving source-held-out accuracy and slightly improving non-base positive recall over the previous M2.81 zero-false-promotion baseline.

## Tools

New:

```text
tools/build_preview_aware_teacher_refresh.py
```

Updated:

```text
tools/train_stitch_family_selector.py
```

The training tool now supports preview-aware numeric features when present:

```text
professional_preview_score
delta_professional_preview_score
generator_texture_score
delta_generator_texture_score
preview_sweep_score
preview_sweep_gate_pass
```

## Teacher Refresh

Run:

```powershell
python tools\build_preview_aware_teacher_refresh.py
```

Inputs:

```text
results/public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed/stitch_type_teacher_seed_rows.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_candidate_rows.csv
results/public_benchmark_v1_ext33_m2_86_preview_candidate_sweep_selector/preview_sweep_selected_rows.csv
```

Outputs:

```text
results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh/preview_teacher_seed_rows.csv
results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh/preview_teacher_added_rows.csv
results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh/preview_positive_teacher_rows.csv
results/public_benchmark_v1_ext33_m2_87_preview_teacher_refresh/preview_teacher_refresh_summary.json
```

Labeling policy:

| M2.86 row type | M2.87 label |
|---|---|
| selected preview-positive switch | positive teacher |
| gate-pass but not selected | neutral `candidate_non_teacher` |
| visible / off-mask / precision / loss / texture regression | `reject` |
| preview gain too small | neutral `candidate_non_teacher` |

Teacher refresh result:

| Item | Count |
|---|---:|
| total rows | 2244 |
| added M2.86 preview rows | 1584 |
| positive teacher rows | 73 |
| added positive teacher rows | 7 |
| hard negative rows | 1149 |

The 7 added positive rows have:

| Metric | Value |
|---|---:|
| mean preview score | 0.77505593 |
| mean preview delta | +0.02529955 |
| mean texture delta | +0.08257143 |

## Selector Training

Selected run:

```powershell
python tools\train_stitch_family_selector.py `
  --seed-rows results\public_benchmark_v1_ext33_m2_87_preview_teacher_refresh\preview_teacher_seed_rows.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010 `
  --model-id m2_87_preview_selector_rejectw16_margin010 `
  --alpha 1.0 `
  --holdout-column source_name `
  --reject-weight 16.0 `
  --reject-margin 0.10 `
  --include-professional-gate-feature
```

Artifacts:

```text
results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/stitch_family_selector_model.json
results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/stitch_family_selector_summary.json
results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/stitch_family_source_heldout_predictions.csv
results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/stitch_family_source_heldout_folds.csv
results/public_benchmark_v1_ext33_m2_87_preview_selector_rejectw16_margin010/stitch_family_confusion.csv
```

## Source-Held-Out Result

| Metric | M2.81 | M2.87 selected | Delta |
|---|---:|---:|---:|
| accuracy | 0.84226190 | 0.95073892 | +0.10847702 |
| positive accuracy | 0.81818182 | 0.80821918 | -0.00996264 |
| non-base positive accuracy | 0.67567568 | 0.68181818 | +0.00614250 |
| reject recall | 0.84814815 | 0.95982533 | +0.11167718 |
| false texture promotions | 0 | 0 | 0 |

This is the useful part of M2.87:

```text
it adds preview-aware supervision without losing the zero-false-promotion safety line.
```

## Variant Selection

See:

```text
results/public_benchmark_v1_ext33_m2_87_preview_selector_model_selection/preview_selector_variant_summary.csv
```

Important tradeoff:

| Variant | Non-base positive acc | False texture promotions | Decision |
|---|---:|---:|---|
| rejectw8 margin0.10 | 0.84090909 | 6 | useful exploratory but unsafe |
| rejectw16 margin0.10 | 0.68181818 | 0 | selected |
| rejectw24 margin0.15 | 0.43181818 | 0 | too conservative |

## Interpretation

M2.87 is not a new DST output profile yet.

It is a learned-control upgrade:

```text
M2.86 = deterministic preview-aware selector
M2.87 = teacher refresh + source-held-out learned selector
```

M2.87 proves that M2.86's preview-positive and preview-rejected candidate evidence can improve the learned selector under source-held-out validation.

However, it has not yet been wired into actual deployment and rerun on the 33-sample DST output set. Therefore:

```text
M2.86 remains the current output selector.
M2.87 is the current best learned selector candidate for the next deployment step.
```

## Next Step

The next step should be:

```text
M2.88 = preview-aware learned deployment policy
```

It should apply the M2.87 selector to the executable candidate pool, then compare against M2.86 on:

- hard_fail;
- visible connectors;
- off-mask length;
- jump / trim counts;
- coverage and precision;
- DST-derived professional preview score;
- generator texture score.

Only if M2.88 matches or beats M2.86 on the final output metrics should the learned selector replace the deterministic sweep selector.
