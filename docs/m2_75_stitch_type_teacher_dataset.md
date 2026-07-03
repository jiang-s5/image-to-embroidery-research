# M2.75 Stitch-Type Teacher Dataset

Date: 2026-07-03

## Summary

M2.75 turns the M2.74 professional texture gate into a supervised training table for the next learned stitch-type planner.

This is not a new DST output profile. M2.74 remains the current promoted output profile. M2.75 is the data layer that lets the next model learn:

```text
sample geometry + candidate metrics + texture value
  -> keep current fill/running output
  -> or promote dt-satin / satin-rail / future fill-tatami candidate
```

This matters because the project cannot reach professional digitizing quality by hand-tuning one gate forever. A real digitizer needs a learned or rule-assisted stitch-type planner that decides where to use running, satin, fill/tatami, and future underlay.

## Generated Artifacts

```text
results/public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset/
  stitch_type_candidate_rows.csv
  stitch_type_teacher_choices.csv
  stitch_type_by_candidate_kind.csv
  stitch_type_by_source.csv
  stitch_type_by_gate_reason.csv
  stitch_type_teacher_summary.json
```

The tool is:

```text
tools/build_stitch_type_planner_dataset.py
```

## What The Table Contains

Each row is a `(sample, candidate)` pair. For the current benchmark:

| Item | Count |
|---|---:|
| samples | 33 |
| candidate rows | 165 |
| base keep teacher choices | 29 |
| texture promotion teacher choices | 4 |
| teacher dt-satin choices | 4 |

The table includes:

- source/category/split;
- geometry features from mask and skeleton;
- planner branch and line score;
- candidate kind: `base_keep`, `dt_satin`, `satin_rail`, etc.;
- base and candidate command metrics;
- deltas versus the base output;
- texture counts: fill rows, satin rail segments, dt-satin segments;
- `professional_utility`;
- `allowed_by_professional_gate`;
- `gate_reason`;
- `is_teacher_choice`.

## Teacher Choices

The teacher choices reproduce M2.74:

| Sample | Source | Category | Teacher |
|---|---|---|---|
| ocp_005 | Openclipart | public_domain_logo | dtsatin_safeadt_light |
| omj_001 | OpenMoji | heart | dtsatin_safeadt_light |
| omj_003 | OpenMoji | butterfly | dtsatin_safeadt_light |
| omj_005 | OpenMoji | sun | dtsatin_safeadt_light |

All other samples keep the current base profile.

The teacher output metrics match M2.74:

| Metric | Value |
|---|---:|
| mean unified loss | 0.05639949 |
| mean jump count | 6.03030303 |
| mean trim count | 0.60606061 |
| mean off-mask length | 0.12120606 |
| mean visible connector count | 0 |
| mean coverage | 0.98103933 |
| mean strict precision | 0.83779552 |
| mean texture score | 0.20860606 |

## Gate Failure Analysis

The main rejection reasons are still informative:

| Pattern | Meaning |
|---|---|
| `excluded_source` | QuickDraw and text are intentionally excluded from generic texture promotion. |
| `loss_increase` | candidate adds texture but costs too much under current utility. |
| `off_mask_increase` | candidate produces stitches outside the mask. |
| `precision_drop_too_large` | candidate adds texture but overfills or reduces precision. |
| `jump_increase` | candidate creates too much extra routing cost. |

This gives direct training signal for a future model:

```text
accept dt-satin when geometry supports it
reject satin-like texture when it causes off-mask / precision / jump risk
route QuickDraw and text into specialized running/text planners
```

## Why This Is Progress

Before M2.75, the system had a hand-tuned rule:

```text
if candidate passes gate:
    promote texture
else:
    keep base
```

After M2.75, the same decision is represented as supervised data:

```text
features -> teacher choice
```

That is the bridge from deterministic candidate gating to a learned stitch-type planner.

## Boundary

M2.75 does not claim that the model has learned professional digitizing yet. It creates the training substrate needed for that step.

The next experiment should train a constrained selector from `stitch_type_candidate_rows.csv`, then evaluate whether it can reproduce or improve M2.74 under leave-one-out or source-held-out validation.
