# M2.1 / M2.2 Decode Loop Comparison

Lower is better for unified loss, hard score, jump count, and visual-risk metrics.

| Method | Unified | Hard score | Exec | Visual | Jumps | Trims | Jump path mm | Off-mask mm | Visible | Safe repairs | Hard fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_1_globalrepair20 | 0.580058 | 37.152 | 0.307427 | 0.272631 | 83.00 | 15.50 | 497.7 | 173.1 | 32.50 | 94.50 | 4 |
| m2_2_retrained_globalrepair20 | 0.581322 | 37.188 | 0.308691 | 0.272631 | 82.00 | 15.75 | 490.4 | 173.6 | 32.50 | 95.25 | 4 |
| m2_1_globalrepair12 | 0.598647 | 37.688 | 0.326016 | 0.272631 | 94.00 | 19.75 | 563.0 | 172.1 | 32.50 | 89.25 | 4 |
| m2_1_globalrepair8 | 0.602398 | 38.335 | 0.329768 | 0.272631 | 109.00 | 21.00 | 633.4 | 171.9 | 32.50 | 81.75 | 4 |
| m2_edge_policy_top4 | 0.607582 | 41.652 | 0.334952 | 0.272631 | 192.75 | 20.25 | 842.2 | 171.9 | 32.50 | 0.00 | 4 |
| m2_1_balanced | 0.609459 | 41.652 | 0.336828 | 0.272631 | 191.75 | 21.00 | 854.0 | 171.9 | 32.50 | 1.00 | 4 |

## Current Best

`m2_1_globalrepair20` is the current recommended M2.1 decode setting. It beats M2 top4 and the retrained M2.2 policy on mean unified loss in this 4-sample loop.

## Training Result

`m2_2_retrained_globalrepair20` learns the hard-mined route decisions, but its DST score is slightly worse than the deterministic M2.1 repair20 decode. Keep it as an artifact, not the promoted setting.

## Remaining Limitation

All variants are still hard_fail because off-mask and visible-connector thresholds are much stricter than the current mask/geometry quality. M2.1 improves executable routing and jump behavior, not upstream mask quality.