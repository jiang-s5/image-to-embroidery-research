# M2.78 Stitch-Type Teacher Seed

Date: 2026-07-07

## Summary

M2.78 addresses the bottleneck exposed by M2.77.

M2.77 showed that the learned stitch-type selector can reproduce M2.74 when professional gate features are present, but metric-only source-held-out variants recover `0/4` texture promotions. That means the system does not yet have enough independent stitch-type teacher signal.

M2.78 therefore builds a broader teacher seed table for professional stitch-type planning.

It combines:

- M2.18 multi-candidate planner oracle rows;
- M2.75 professional texture teacher rows.

The result is a candidate-level seed dataset with running, fill/tatami-like, satin-like, outline, base, reject, and non-teacher candidate labels.

## Tool

```text
tools/build_stitch_type_teacher_seed.py
```

Run:

```powershell
python tools\build_stitch_type_teacher_seed.py `
  --candidate-selector-csv results\public_benchmark_v1_ext33_m2_18_styleaware_selector_v2\candidate_selector_dataset.csv `
  --stitch-type-csv results\public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset\stitch_type_candidate_rows.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed `
  --max-off-mask-mm 0.5 `
  --max-visible-connectors 0
```

## Generated Artifacts

```text
results/public_benchmark_v1_ext33_m2_78_stitch_type_teacher_seed/
  stitch_type_teacher_seed_rows.csv
  stitch_type_positive_teacher_rows.csv
  stitch_type_seed_by_candidate_family.csv
  stitch_type_seed_by_teacher_family.csv
  stitch_type_seed_by_source.csv
  stitch_type_teacher_seed_summary.json
```

## Dataset Size

| Item | Count |
|---|---:|
| candidate rows | 660 |
| samples | 33 |
| positive teacher rows | 66 |
| hard negative rows | 274 |

This is four times wider than the M2.75 teacher table, which had 165 rows.

## Candidate Families

| Candidate family | Rows |
|---|---:|
| running_line | 198 |
| satin_like | 132 |
| auto_fill | 95 |
| auto_running | 70 |
| fill_tatami_like | 66 |
| base_keep | 33 |
| outline_border | 33 |
| style_aware_mixed | 33 |

## Positive Teacher Families

| Teacher family | Positive rows |
|---|---:|
| base_keep | 29 |
| running_line | 11 |
| auto_fill | 10 |
| fill_tatami_like | 6 |
| auto_running | 5 |
| satin_like | 4 |
| outline_border | 1 |

The most important change is that the teacher signal is no longer satin-only. M2.78 now exposes positive supervision for:

- line/running style;
- fill/tatami-like style;
- satin-like border texture;
- base keep decisions;
- hard negatives.

## Source Coverage

| Source | Positive teacher families |
|---|---|
| OpenMoji | auto_fill, auto_running, base_keep, fill_tatami_like, outline_border, satin_like |
| Openclipart | auto_fill, base_keep, fill_tatami_like, running_line, satin_like |
| QuickDraw | auto_running, base_keep, running_line |
| Rendered text | auto_running, base_keep, fill_tatami_like, running_line |

This gives the next selector a better chance to learn source-aware stitch-type decisions without relying entirely on a hand-authored professional gate.

## What It Solves

M2.78 directly responds to the M2.77 failure mode:

```text
Metric-only source-held-out selector:
  texture recall = 0/4
```

The failure happened because the teacher table was too narrow:

```text
mostly keep-base decisions
only four satin-like promotions
no broad running/fill/tatami positives
```

M2.78 expands the supervision space so the next model can learn:

```text
image / geometry / candidate metrics
  -> running-line
  -> fill/tatami-like
  -> satin-like
  -> base keep
  -> reject unsafe candidate
```

## Boundary

M2.78 is not a new DST output profile. M2.74 remains the promoted generation profile.

M2.78 is a data-layer upgrade. It prepares the ground for a true multi-class stitch-type planner, which is necessary before the project can approach professional digitizing behavior.

## Next Step

The next model should train a source-held-out multi-family selector on this seed table.

The success condition should not be only:

```text
reproduce M2.74 texture choices
```

It should be:

```text
recover running/fill/satin/base/reject family choices under source-held-out validation
avoid unsafe texture promotions
improve render-back stitch texture without increasing visible/off-mask risk
```
