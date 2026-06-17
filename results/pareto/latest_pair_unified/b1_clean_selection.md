# B1 Clean Selection

Lower is better. The recommendation is selected from the Pareto front using the lowest mean unified loss.

## Recommendation

- recommended method: `model13_relation_fixed_thread_v2_no_training`
- mean unified loss: `0.615568`
- mean visual risk: `0.277759`

## Pareto Front

| Method | Samples | Unified Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Jump Path mm | Hard Fail | Warning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model13_relation_fixed_thread_v2_no_training | 4 | 0.615568 | 125.250 | 24.000 | 126.252 | 19.750 | 744.033 | 4 | 0 |
| model13_relation_fixed_v2_adaptive_safehatch_no_training | 4 | 0.618034 | 104.500 | 18.250 | 748.807 | 89.000 | 641.679 | 4 | 0 |
| model13_v2_routed_best_no_training | 4 | 0.621986 | 216.000 | 38.250 | 116.281 | 16.750 | 1164.793 | 4 | 0 |
| model13_relation_fixed_v2_adaptive_no_training | 4 | 0.623767 | 104.000 | 19.250 | 749.278 | 88.500 | 633.330 | 4 | 0 |
| model13_relation_fixed_v2_adaptive_no_connectors_no_training | 4 | 0.634834 | 409.250 | 19.250 | 744.776 | 88.500 | 979.180 | 4 | 0 |

## All Methods

| Method | Unified Loss | Exec Score | Visual Risk | Hard Fail | Warning | Pass | Excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model13_relation_fixed_thread_v2_no_training | 0.615568 | 0.337809 | 0.277759 | 4 | 0 | 0 | 0 |
| model13_relation_fixed_v2_adaptive_safehatch_no_training | 0.618034 | 0.318034 | 0.300000 | 4 | 0 | 0 | 0 |
| model13_v2_routed_best_no_training | 0.621986 | 0.351273 | 0.270712 | 4 | 0 | 0 | 0 |
| model13_relation_fixed_v2_adaptive_no_training | 0.623767 | 0.323767 | 0.300000 | 4 | 0 | 0 | 0 |
| model13_relation_fixed_v2_selected_no_training | 0.633506 | 0.355690 | 0.277817 | 4 | 0 | 0 | 0 |
| model13_relation_fixed_v2_adaptive_no_connectors_no_training | 0.634834 | 0.334834 | 0.300000 | 4 | 0 | 0 | 0 |
| model13_relation_fixed_v2_selected_strictjump_no_training | 0.643507 | 0.365691 | 0.277817 | 4 | 0 | 0 | 0 |
| model13_relation_fixed_thread_v2_dense_no_training | 0.646889 | 0.368945 | 0.277944 | 4 | 0 | 0 | 0 |