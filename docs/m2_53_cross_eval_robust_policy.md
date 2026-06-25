# M2.53 Cross-Eval Robust Policy

Date: 2026-06-25

M2.53 continues from M2.52.

M2.52 exposed the Pareto frontier for each sweep output, but it still analyzed each evaluation set separately. M2.53 adds a cross-evaluation robust policy selector:

```text
incoming review sweep
public ext33 sanity sweep
        ↓
align identical benefit-gate configurations
        ↓
require every evaluation set to satisfy coverage / hard-fail constraints
        ↓
rank by average score, worst-case score, and score stability
```

This is still not a new neural model. It is a safer policy-selection protocol. Its purpose is to reduce the risk that the current default policy is picked only because it works well on one benchmark.

## Tool

```text
tools/select_m2_robust_policy.py
```

Inputs:

- one or more `benefit_gate_sweep_rows.csv` files;
- profile list, usually `balanced,low_jump`;
- hard constraints: `min_mean_coverage` and `max_hard_fail`.

Outputs:

- `robust_policy_rows.csv`
- `robust_policy_recommendations.csv`
- `robust_policy_summary.json`

## Robust Score

Each sweep/profile first gets a normalized dataset score:

```text
0.40 * objective
+ 0.18 * unified loss
+ 0.16 * jump
+ 0.06 * trim
+ 0.08 * off-mask
+ 0.08 * coverage reward
+ 0.04 * precision reward
```

Then M2.53 ranks a shared configuration by:

```text
0.55 * mean_dataset_score
+ 0.30 * worst_dataset_score
+ 0.15 * dataset_score_range
```

So the selected policy must be good on average, not too bad on the worst dataset, and reasonably stable across datasets.

## Evaluation Inputs

This run uses:

| Name | Sweep CSV |
|---|---|
| incoming | `results/incoming_review_eval_v1_m2_51_benefitgate_sweep/benefit_gate_sweep_rows.csv` |
| public_ext33 | `results/public_benchmark_v1_ext33_m2_51_benefitgate_sweep_sanity/benefit_gate_sweep_rows.csv` |

Constraints:

```text
min_mean_coverage = 0.90
max_hard_fail = 0
```

The public ext33 set is still a sanity check, not a clean generalization benchmark, because public ext33 candidate tables were used during selector development. The value of M2.53 is that it avoids selecting from public ext33 alone.

## Result

### Balanced Profile

| Recommendation | Strength | Coverage Floor | Min Coverage | Loss Avg | Jump Avg | Min Coverage Ratio | Robust Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| best robust | 0.90 | 0.95 | 0.60 | 0.062974 | 6.428031 | 0.920700 | 0.113874 |
| best objective / low jump | 0.90 | 0.85 | 0.60 | 0.062967 | 6.428031 | 0.920700 | 0.114006 |
| best min coverage | 0.90 | 0.95 | 0.70 | 0.064135 | 7.003788 | 0.991577 | 0.236785 |

Interpretation:

- The robust balanced policy prefers `coverage_floor=0.95`.
- The gain over `coverage_floor=0.85` is small but consistent under the robustness score.
- The high-coverage option is available, but it increases jump and loss.

### Low-Jump Profile

| Recommendation | Strength | Coverage Floor | Min Coverage | Max Loss Slack | Loss Avg | Jump Avg | Min Coverage Ratio | Robust Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best robust / best objective | 0.90 | 0.85 | 0.60 | 0.00 | 0.061320 | 6.079546 | 0.920700 | 0.148253 |
| best low jump | 0.90 | 0.85 | 0.60 | 0.02 | 0.061485 | 6.034091 | 0.920700 | 0.211325 |
| best min coverage | 0.90 | 0.95 | 0.70 | 0.00 | 0.062918 | 6.685606 | 0.993914 | 0.642237 |

Interpretation:

- The M2.53 low-jump robust optimum is exactly the M2.51 recommended default.
- Allowing more loss slack reduces average jump slightly, but it is less stable and has worse robust score.
- Chasing maximum coverage costs too much jump and is not a good default for the low-jump profile.

## Current Default

Keep the M2.51 / M2.53 low-jump robust default:

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

For a more conservative balanced profile, M2.53 recommends:

```text
profile = balanced
profile_rerank_strength = 0.90
profile_coverage_penalty_weight = 0.20
profile_min_coverage_by_name = balanced=0.95,low_jump=0.85,strict_low_jump=0.85
profile_benefit_min_coverage = 0.60
profile_benefit_min_jump_gain = 2
profile_benefit_min_precision = 0.78
profile_benefit_max_loss_slack = 0.00
```

## Research Interpretation

M2.53 is useful for the paper because it gives a stricter answer to the overfitting concern:

```text
The default low-jump benefit-gate policy is not selected from a single sweep table. It remains the robust optimum when the same parameter configuration must satisfy both the paired incoming review set and the public ext33 sanity set.
```

This does not prove full generalization. The next stronger step is to run the same cross-eval selector with a truly external benchmark that was not used during selector development.

## Artifacts

- `tools/select_m2_robust_policy.py`
- `configs/best_current_model_m2_53_cross_eval_robust_policy.json`
- `results/m2_53_cross_eval_robust_policy/`

