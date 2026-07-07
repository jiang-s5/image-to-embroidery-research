# M2.79 Stitch-Family Selector

Date: 2026-07-07

## Summary

M2.79 is the first multi-family stitch-type selector trained on the broader M2.78 teacher seed table.

It moves beyond the earlier question:

```text
should this sample use the M2.74 satin-like texture profile?
```

and starts testing a more professional digitizing question:

```text
for this candidate, should the planner treat it as running, fill/tatami-like,
satin-like, base, outline, or unsafe/reject?
```

M2.79 is still not a promoted DST output profile. It is a learning and risk-control experiment.

## Tool

```text
tools/train_stitch_family_selector.py
```

Example:

```powershell
python tools\train_stitch_family_selector.py `
  --seed-rows results\public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed\stitch_type_teacher_seed_rows.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_79_stitch_family_selector_rejectw4 `
  --model-id m2_79_stitch_family_selector_rejectw4 `
  --alpha 1.0 `
  --holdout-column source_name `
  --reject-weight 4.0
```

The validation mode is source-held-out: one full source family is held out at a time.

## Labels

M2.79 uses these labels:

| Label | Meaning |
|---|---|
| `base_keep` | keep the current base planner output |
| `running_line` | line/skeleton running-stitch style |
| `auto_running` | auto branch selected for line/text-like structure |
| `auto_fill` | auto branch selected for flat fill-like structure |
| `fill_tatami_like` | dense fill/tatami-like candidate |
| `satin_like` | satin or satin-rail-like border/texture candidate |
| `outline_border` | outline/border candidate |
| `reject` | unsafe or non-promoted candidate |

## Main Results

| Variant | Accuracy | Positive acc | Non-base positive acc | Reject recall | False texture promotions |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6458 | 0.9394 | 0.8919 | 0.5741 | 51 |
| balanced negative control | 0.3363 | 0.9848 | 0.9730 | 0.1778 | 173 |
| rejectw3 + margin0.02 | 0.7321 | 0.7879 | 0.6216 | 0.7185 | 10 |
| rejectw4 | 0.7411 | 0.7727 | 0.5946 | 0.7333 | 6 |

## Interpretation

M2.79 has two important findings.

First, M2.78 really helps the model learn positive stitch-family labels. The baseline variant reaches:

```text
non-base positive accuracy = 0.8919
```

This is a meaningful improvement over M2.77's metric-only texture selector, which recovered no source-held-out texture promotions.

Second, professional digitizing cannot optimize positive recall alone. The class-balanced variant gets high positive accuracy, but it creates:

```text
false texture promotions = 173
reject recall = 0.1778
```

That is not acceptable for professional embroidery, because unsafe satin/fill promotions create unwanted stitches, overfill, and messy render-back results.

The safer professional direction is the `rejectw4` variant:

```text
accuracy = 0.7411
reject recall = 0.7333
false texture promotions = 6
```

It sacrifices some non-base positive recall, but it is much safer. This is closer to how a real digitizer should behave: avoid bad stitch-style promotions unless the geometry and metrics support them.

## Source-Level Behavior

The `rejectw4` variant has:

| Source | Accuracy | Non-base positive acc | Reject recall | False texture promotions |
|---|---:|---:|---:|---:|
| OpenMoji | 0.6429 | 0.3846 | 0.6563 | 1 |
| Openclipart | 0.8018 | 0.6364 | 0.8022 | 3 |
| QuickDraw | 0.8247 | 0.6250 | 0.8272 | 0 |
| Rendered text | 0.5909 | 1.0000 | 0.4706 | 2 |

OpenMoji remains the hardest source because it has multiple true texture/style positives and visually attractive but risky candidates. This is the next place to add more real teacher labels.

## Promotion Decision

M2.79 is not promoted as the output profile.

M2.74 remains the promoted DST generation profile. M2.79 is a learned stitch-family selector prototype and safety audit.

## What This Adds Toward Professional Quality

Before M2.79:

```text
the learning question was mostly: satin switch or not?
```

After M2.79:

```text
the learning question becomes: which stitch family is safe and appropriate?
```

That is the correct direction for professional embroidery software, where the hard part is not simply drawing more stitches, but selecting the right stitch family while avoiding unsafe promotions.

## Next Step

The next step should combine:

- more real DST/PES-derived stitch-family labels;
- cost-sensitive training where false texture promotion is explicitly expensive;
- source-held-out validation as the default gate;
- final integration into candidate reranking only after the selector reduces visual/off-mask risk.

M2.79 proves the data direction is useful, but it also shows the selector needs more teacher diversity before it should control actual DST generation.
