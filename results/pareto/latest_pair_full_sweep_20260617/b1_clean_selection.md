# B1 Clean Selection

Lower is better. The recommendation is selected from the Pareto front.
Selection policy: `hard`.

## Recommendation

- recommended method: `b1_conservative`
- mean unified loss: `0.636772`
- mean visual risk: `0.294706`

## Pareto Front

| Method | Samples | Unified Loss | Jump | Trim | Off-Mask mm | Visible Connectors | Jump Path mm | Hard Fail | Warning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b1_low_connect | 4 | 0.628786 | 203.500 | 26.750 | 201.238 | 46.250 | 1014.621 | 4 | 0 |
| b1_conservative | 4 | 0.636772 | 130.750 | 25.250 | 185.828 | 37.000 | 783.331 | 4 | 0 |

## All Methods

| Method | Unified Loss | Exec Score | Visual Risk | Hard Fail | Warning | Pass | Excellent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b1_low_connect | 0.628786 | 0.350968 | 0.277818 | 4 | 0 | 0 | 0 |
| b1_region_fill | 0.633435 | 0.355619 | 0.277817 | 4 | 0 | 0 | 0 |
| b1_conservative | 0.636772 | 0.342067 | 0.294706 | 4 | 0 | 0 | 0 |
| b1_dense_thread | 0.649885 | 0.364587 | 0.285297 | 4 | 0 | 0 | 0 |