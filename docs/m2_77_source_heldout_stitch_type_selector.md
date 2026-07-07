# M2.77 Source-Held-Out Stitch-Type Selector Audit

Date: 2026-07-03

## Summary

M2.77 makes the M2.76 stitch-type selector evaluation stricter.

M2.76 used sample-level leave-one-out:

```text
hold out one image
train on the other 32 images
predict the held-out image
```

M2.77 uses source-held-out validation:

```text
hold out one entire source family
train on the other source families
predict every image in the held-out source
```

The held-out source families are:

- Openclipart
- OpenMoji
- QuickDraw
- Rendered text

This is a stronger test because it asks whether the selector can transfer stitch-type decisions across input domains.

## Tooling Change

`tools/train_stitch_type_selector.py` now supports:

```powershell
python tools\train_stitch_type_selector.py `
  --candidate-rows results\public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset\stitch_type_candidate_rows.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_77_source_heldout_stitch_type_selector_gate `
  --model-id m2_77_source_heldout_stitch_type_selector_gate `
  --include-gate-features `
  --use-professional-gate-at-inference `
  --validation-mode source-held-out `
  --holdout-column source_name
```

## Results

| Variant | Gate features | Inference gate | Exact match | Texture recall | Texture precision | Predicted texture switches |
|---|---:|---:|---:|---:|---:|---:|
| gate-assisted | yes | yes | 1.0000 | 1.0000 | 1.0000 | 4 |
| metric-only + safety gate | no | yes | 0.8788 | 0.0000 | 0.0000 | 0 |
| metric-only, no gate | no | no | 0.6061 | 0.0000 | 0.0000 | 9 |

The gate-assisted version still reproduces M2.74 exactly:

| Metric | Gate-assisted M2.77 |
|---|---:|
| mean unified loss | 0.05639949 |
| mean jump count | 6.03030303 |
| mean trim count | 0.60606061 |
| mean off-mask length | 0.12120606 |
| mean visible connector count | 0 |
| mean coverage | 0.98103933 |
| mean strict precision | 0.83779552 |
| mean texture score | 0.20860606 |

## What This Means

M2.77 is a useful negative result.

It shows that the current learned selector can reproduce M2.74 only when the professional gate signal is available. When the model is forced to rely on metric features alone:

- with the inference safety gate, it becomes too conservative and selects no texture promotions;
- without the gate, it over-promotes texture candidates, especially on text and flat icons;
- it fails to recover all four teacher texture switches under source-held-out validation.

So the current stitch-type selector is not yet a fully learned professional digitizer. It is a constrained selector whose best behavior still depends on gate-derived supervision.

## Research Boundary

The correct claim is:

```text
M2.77 verifies that professional gate features are currently necessary for reliable stitch-type selection under source-held-out validation.
```

The incorrect claim would be:

```text
M2.77 proves that the model has learned source-general stitch-type planning.
```

## Why This Still Moves The Project Forward

This audit prevents an overclaim. It identifies the next real bottleneck:

```text
the dataset does not yet contain enough independent stitch-type teacher signal
```

The next optimization should focus on richer labels, not more ranker tuning:

- parse more real DST/PES files with known satin/fill/running behavior;
- add segment-level stitch-type labels;
- add hard negatives where texture looks attractive but harms precision, off-mask, or jump cost;
- add fill/tatami candidates, not only dt-satin and satin-rail candidates;
- repeat source-held-out validation after the teacher set is expanded.

## Promotion Decision

M2.74 remains the promoted DST output profile.

M2.77 is a generalization audit and should be cited as evidence for why the next stage must prioritize professional stitch-type teacher data.
