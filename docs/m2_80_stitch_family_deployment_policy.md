# M2.80 Stitch-Family Deployment Policy

Date: 2026-07-07

## Summary

M2.80 adds a conservative deployment policy on top of the M2.79 multi-family stitch selector.

M2.79 proved that the broader M2.78 teacher seed can learn multiple stitch families, but it still produced unsafe texture promotions:

```text
false texture promotions = 6
```

M2.80 keeps the same selector and adds a post-selector margin rule:

```text
if predicted_family is risky and score(predicted_family) - score(reject) < 0.20:
    deploy reject
else:
    keep selector prediction
```

This matches the professional digitizing rule that complex stitch families should only be used when the selector is clearly more confident than the reject option.

## Tool

```text
tools/calibrate_stitch_family_deployment_policy.py
```

Example:

```powershell
python tools\calibrate_stitch_family_deployment_policy.py `
  --predictions results\public_benchmark_v1_ext33_m2_79_stitch_family_selector_rejectw4\stitch_family_source_heldout_predictions.csv `
  --output-dir results\public_benchmark_v1_ext33_m2_80_stitch_family_deployment_policy `
  --model-id m2_80_stitch_family_deployment_policy `
  --min-risky-margin 0.20
```

## Risky Families

The deployment margin applies to:

| Family | Reason |
|---|---|
| `running_line` | can create unwanted path structure if promoted incorrectly |
| `auto_running` | same risk for line/text-like routing |
| `auto_fill` | can overfill or choose the wrong fill branch |
| `fill_tatami_like` | visible texture promotion risk |
| `satin_like` | visible satin/rail promotion risk |
| `outline_border` | border/outline promotion risk |

`base_keep` and `reject` are not risky promotions.

## Main Result

| System | Accuracy | Positive acc | Non-base positive acc | Reject recall | False texture promotions |
|---|---:|---:|---:|---:|---:|
| M2.79 rejectw4 raw selector | 0.7411 | 0.7727 | 0.5946 | 0.7333 | 6 |
| M2.80 deployment policy | 0.8423 | 0.7576 | 0.5676 | 0.8630 | 0 |

M2.80 improves:

```text
accuracy:              +0.1012
reject recall:          +0.1296
false texture promos:   6 -> 0
```

The tradeoff is:

```text
positive accuracy:      -0.0152
non-base positive acc:  -0.0270
```

This is a reasonable professional tradeoff because unsafe complex texture promotions are more damaging than missing a small number of optional non-base promotions.

## Margin Sweep

Important sweep points:

| Margin | Accuracy | Non-base positive acc | Reject recall | False texture promotions |
|---:|---:|---:|---:|---:|
| 0.08 | 0.8006 | 0.5676 | 0.8111 | 0 |
| 0.10 | 0.8065 | 0.5676 | 0.8185 | 0 |
| 0.15 | 0.8214 | 0.5676 | 0.8370 | 0 |
| 0.20 | 0.8423 | 0.5676 | 0.8630 | 0 |
| 0.25 | 0.8601 | 0.5405 | 0.8889 | 0 |

The selected point is `0.20`, because it removes false texture promotions while preserving more non-base recall than stricter margins.

## Source-Level Result

| Source | Accuracy | Positive acc | Non-base positive acc | Reject recall | False texture promotions |
|---|---:|---:|---:|---:|---:|
| OpenMoji | 0.7500 | 0.6000 | 0.3846 | 0.7969 | 0 |
| Openclipart | 0.8739 | 0.8000 | 0.6364 | 0.8901 | 0 |
| QuickDraw | 0.8454 | 0.7500 | 0.5000 | 0.8642 | 0 |
| Rendered text | 0.9318 | 1.0000 | 1.0000 | 0.9118 | 0 |

OpenMoji remains the hardest source, but M2.80 removes the unsafe texture promotions there as well.

## Promotion Decision

M2.80 is not promoted as a new DST output profile.

M2.74 remains the promoted output profile. M2.80 is the recommended safety gate before allowing learned stitch-family predictions to control actual DST generation.

## What This Adds Toward Professional Quality

M2.79 answered:

```text
which stitch family does the selector prefer?
```

M2.80 adds:

```text
is that family prediction confident enough to deploy?
```

That distinction matters for professional embroidery. A visually attractive but wrong satin/fill promotion can create messy texture, unwanted density, and bad render-back behavior. M2.80 makes the learned stitch-family layer more conservative and safer.

## Next Step

The next step is to connect this deployment policy into actual candidate reranking:

```text
candidate pool
  -> M2.79 family selector
  -> M2.80 deployment margin gate
  -> M2.74-style professional texture profile
  -> DST export and render-back evaluation
```

Only after this integration improves actual command-level metrics should the learned stitch-family selector be promoted from an audit layer to a real output layer.
