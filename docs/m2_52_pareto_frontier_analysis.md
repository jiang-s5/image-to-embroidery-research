# M2.52 Pareto Frontier Analysis

Date: 2026-06-25

M2.52 continues from M2.51. M2.51 selected a sweep-tuned benefit gate with a single conservative objective. That is useful for a default policy, but it can hide the tradeoff between low jump count, low unified loss, and high stitch coverage.

M2.52 adds a Pareto-frontier analysis layer:

```text
M2.51 sweep rows
  -> collapse duplicate metric outcomes
  -> compute non-dominated tradeoff frontiers
  -> recommend best objective / low jump / high coverage / balanced knee points
```

This is not a new neural model. It is an evaluation and decision layer for choosing the current planner policy more honestly.

## Tool

```text
tools/analyze_m2_pareto_frontier.py
```

The tool reads a sweep CSV and writes:

- `pareto_frontier_rows.csv`
- `pareto_recommendations.csv`
- `pareto_summary.json`

It minimizes:

- mean unified loss
- jump count
- trim count
- off-mask stitch length
- visible connector count

It maximizes:

- coverage ratio
- stitch precision ratio

The analysis also collapses duplicate metric outcomes before computing the frontier. This matters because the M2.51 threshold grid can contain hundreds of equivalent parameter settings that produce the same selected outputs.

## Incoming Review Frontier

The incoming review set has 4 held-out paired review samples. Under the M2.51 sweep, most parameter settings collapse to only a few distinct outcomes.

| Profile | Unique Outcomes | Pareto Rows | Best Objective Loss | Best Objective Jump | Best Objective Coverage | High Coverage Loss | High Coverage Jump | High Coverage Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 3 | 3 | 0.073831 | 7.250000 | 0.920700 | 0.074710 | 8.250000 | 0.996541 |
| low_jump | 2 | 2 | 0.073831 | 7.250000 | 0.920700 | 0.074710 | 8.250000 | 0.996541 |

Interpretation:

- The best objective and best low-jump point are the same as M2.51 on incoming review.
- A high-coverage alternative exists, but it costs about +1 jump per sample and slightly higher loss.
- Both choices have zero hard failures, zero visible connectors, and zero off-mask stitch length on this set.

## Public ext33 Sanity Frontier

The public ext33 run is still a sanity check, not a clean generalization claim, because the selector was developed using public ext33 candidate tables.

| Profile | Unique Outcomes | Pareto Rows | Best Objective Loss | Best Objective Jump | Best Objective Coverage | High Coverage Loss | High Coverage Jump | High Coverage Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 8 | 8 | 0.052103 | 5.606061 | 0.982012 | 0.053560 | 5.757576 | 0.991577 |
| low_jump | 24 | 23 | 0.048809 | 4.909091 | 0.979476 | 0.051126 | 5.121212 | 0.993914 |

Key low-jump public points:

| Recommendation | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_objective | 0.048809 | 4.909091 | 0.606061 | 0.199076 | 0.000000 | 0.979476 | 0.793264 |
| best_low_jump | 0.049139 | 4.818182 | 0.787879 | 0.199076 | 0.000000 | 0.970612 | 0.794352 |
| best_high_coverage | 0.051126 | 5.121212 | 0.696970 | 0.199076 | 0.000000 | 0.993914 | 0.785532 |
| balanced_knee | 0.049667 | 5.090909 | 0.606061 | 0.169776 | 0.000000 | 0.977453 | 0.804424 |

Interpretation:

- The public low-jump frontier shows a real tradeoff surface rather than one obvious winner.
- The lowest-jump point saves about 0.09 jumps per sample versus best objective, but loses coverage and increases trim.
- The high-coverage point reaches 0.993914 coverage with no visible connectors, but costs higher loss and jump count.
- The balanced-knee point gives the best precision and lower off-mask length, but is not the lowest loss or lowest jump point.

## Recommended Current Default

For the current repository default, keep the M2.51 best-objective policy:

```text
profile = low_jump
profile_rerank_strength = 0.90
profile_coverage_penalty_weight = 0.20
profile_min_coverage_by_name = balanced=0.85,low_jump=0.85,strict_low_jump=0.85
profile_benefit_min_coverage = 0.60
profile_benefit_min_jump_gain = 2
profile_benefit_min_precision = 0.78
profile_benefit_max_loss_slack = 0.00
```

Reason:

- It is the best objective on public ext33 sanity.
- It keeps incoming review unchanged from M2.51.
- It avoids hard failures.
- It gives a clean default before we add a true learned policy selector.

For paper reporting, M2.52 should be described as:

```text
Instead of reporting only one scalar-selected configuration, we compute the Pareto frontier over executability, visual-risk, coverage, and precision metrics. This exposes the tradeoff between jump minimization and stitch coverage and prevents the planner selection from being over-interpreted as a single optimum.
```

## Artifacts

- `tools/analyze_m2_pareto_frontier.py`
- `configs/best_current_model_m2_52_pareto_frontier_analysis.json`
- `results/incoming_review_eval_v1_m2_52_pareto_frontier/`
- `results/public_benchmark_v1_ext33_m2_52_pareto_frontier_sanity/`

