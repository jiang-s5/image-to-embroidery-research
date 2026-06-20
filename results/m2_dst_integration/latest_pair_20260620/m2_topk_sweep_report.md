# M2 Top-k Sweep

Lower is better. `top_k` controls how many deterministic Graph-TSP candidate edges M2 may rerank.

| Method | Samples | Unified | Hard | Exec | Visual | Jumps | Trims | Off-mask mm | Visible | Decisions | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_edge_policy_top4 | 4 | 0.607582 | 41.652422 | 0.334952 | 0.272631 | 192.750 | 20.250 | 171.912 | 32.500 | 262.000 | 95.750 |
| m2_edge_policy_top2 | 4 | 0.608222 | 41.888061 | 0.335591 | 0.272631 | 196.750 | 21.750 | 171.912 | 32.500 | 262.000 | 68.000 |
| m2_edge_policy_top8 | 4 | 0.610504 | 42.110343 | 0.337873 | 0.272631 | 203.500 | 20.750 | 171.912 | 32.500 | 262.000 | 111.500 |