# M2.76 Learned Stitch-Type Selector

Date: 2026-07-03

## Summary

M2.76 is the first learned-selector prototype for the stitch-type planning layer.

It trains a small pairwise ridge ranker on the M2.75 teacher table:

```text
sample features + candidate metrics + texture deltas
  -> rank the teacher stitch-type candidate above alternatives
```

This is not a new promoted DST output profile. M2.74 remains the current best output profile. M2.76 answers a narrower research question:

```text
Can the M2.74 professional texture choices be learned from candidate-level features?
```

## Tool

```text
tools/train_stitch_type_selector.py
```

The script reads:

```text
results/public_benchmark_v1_ext33_m2_75_stitch_type_teacher_dataset/stitch_type_candidate_rows.csv
```

and writes leave-one-out selector artifacts:

```text
stitch_type_selector_loo_choices.csv
stitch_type_selector_loo_candidate_scores.csv
stitch_type_selector_summary.json
stitch_type_selector_model.json
```

## Leave-One-Out Variants

| Variant | Gate features | Inference safety gate | Exact teacher match | Texture recall | Texture precision |
|---|---:|---:|---:|---:|---:|
| gate-assisted | yes | yes | 1.0000 | 1.0000 | 1.0000 |
| metric-only + safety gate | no | yes | 0.9091 | 0.2500 | 1.0000 |
| metric-only, no gate | no | no | 0.8788 | 0.2500 | 0.5000 |

The gate-assisted variant reproduces the M2.74 teacher choices exactly:

| Metric | Gate-assisted M2.76 |
|---|---:|
| predicted texture switches | 4 |
| mean unified loss | 0.05639949 |
| mean jump count | 6.03030303 |
| mean trim count | 0.60606061 |
| mean off-mask length | 0.12120606 |
| mean visible connector count | 0 |
| mean coverage | 0.98103933 |
| mean strict precision | 0.83779552 |
| mean texture score | 0.20860606 |

## Interpretation

The good news:

- The pipeline has moved from a hand gate to an explicit supervised selector experiment.
- Leave-one-out training can reproduce the M2.74 professional texture decisions when gate features are included.
- The output metrics match M2.74 exactly in the gate-assisted setting.

The important limitation:

- Metric-only variants recover only one of four texture promotions.
- This means the current texture selection signal is still heavily shaped by the professional gate labels.
- M2.76 should be treated as a bridge toward learned stitch-type planning, not as proof that the system has fully learned professional digitizing behavior.

## Research Meaning

M2.76 is useful because it separates two claims:

```text
Claim A: The system can generate a better professional-texture tradeoff.
  -> supported by M2.74.

Claim B: The system can learn when to choose that tradeoff without hand-authored gate labels.
  -> only partially supported by M2.76.
```

This is a cleaner research position than claiming that a rule has become a model.

## Next Step

The next model should reduce dependence on gate labels by adding more teacher examples:

- more true satin candidates;
- more fill/tatami candidates;
- more professional DST-derived stitch-type labels;
- source-held-out validation, not only sample-level leave-one-out;
- hard negatives where texture improves appearance but creates jump/off-mask/precision risk.

Only after metric-only or weak-gate variants recover most texture promotions should a learned stitch-type selector replace M2.74 as the promoted output profile.
