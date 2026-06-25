# M2.50 Benefit-Aware Coverage Gate

Date: 2026-06-25

M2.50 continues from M2.49. M2.49 made the coverage risk explicit by adding soft and hard coverage controls to semantic reranking. However, a fixed coverage floor is still too blunt for embroidery planning: sometimes a lower-coverage candidate is a justified tradeoff if it greatly reduces jumps and keeps high precision.

M2.50 adds an auditable benefit gate:

```text
semantic rerank
  -> coverage-safe candidate baseline
  -> below-floor candidate allowed only if:
       coverage >= absolute minimum
       jump gain >= threshold
       precision >= threshold
       loss increase <= threshold
```

The new apply-time option is:

```text
--profile-coverage-benefit-gate
```

with these controls:

```text
--profile-benefit-min-coverage
--profile-benefit-min-jump-gain
--profile-benefit-min-precision
--profile-benefit-max-loss-slack
```

The selector now writes per-sample audit fields:

- `coverage_gate_mode`
- `coverage_gate_baseline_candidate`
- `coverage_gate_baseline_coverage`
- `coverage_gate_baseline_jump`
- `coverage_gate_jump_gain`
- `coverage_gate_loss_slack`

## Recommended Apply-Time Settings

```text
--profile-semantic-rerank
--profile-rerank-strength 0.90
--profile-coverage-soft-penalty
--profile-coverage-penalty-weight 0.35
--profile-min-coverage-by-name balanced=0.90,low_jump=0.90,strict_low_jump=0.90
--profile-coverage-benefit-gate
--profile-benefit-min-coverage 0.65
--profile-benefit-min-jump-gain 3
--profile-benefit-min-precision 0.80
--profile-benefit-max-loss-slack 0.02
```

## Incoming Review Result

The incoming review set contains 4 held-out paired review samples.

| Method / Profile | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M2.42 default | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.000000 | 0.996541 | 0.753519 |
| M2.49 balanced / low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |
| M2.50 balanced | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |
| M2.50 low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 |
| M2.50 coverage | 0 | 0.076681 | 8.500000 | 1.000000 | 0.000000 | 0.000000 | 0.994577 | 0.772159 |
| M2.50 strict_coverage | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.000000 | 0.995549 | 0.781068 |

The key difference from M2.49 is not the average score. It is the auditability of the low-coverage tradeoff:

| Sample | Chosen Candidate | Mode | Coverage | Jump | Precision | Baseline Candidate | Baseline Coverage | Baseline Jump | Jump Gain |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| pair_006 | auto_evalrepair_r10_c20 | benefit_tradeoff | 0.696636 | 1 | 0.967827 | mask_fill_edgewalk_nearestrow_safeadt_t2m5_r055_a64_rows16_p40 | 1.000000 | 5 | 4 |

So M2.50 can explain why this below-floor candidate was accepted:

- it stays above the absolute coverage floor (`0.65`);
- it reduces jump count by `4`;
- it has high precision (`0.967827`);
- it does not increase loss relative to the coverage-safe baseline.

## Public ext33 Sanity Check

This is not a generalization claim because the selector was trained from public ext33 candidate tables. It is a sanity check that the gate does not introduce hard failures.

| Profile | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision | Gate Modes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M2.49 balanced | 33 | 0 | 0.052103 | 5.606061 | 0.757576 | 0.047591 | 0.000000 | 0.982012 | 0.806207 | n/a |
| M2.50 balanced | 33 | 0 | 0.052103 | 5.606061 | 0.757576 | 0.047591 | 0.000000 | 0.982012 | 0.806207 | benefit_tradeoff=1, safe_coverage=32 |
| M2.49 low_jump | 33 | 0 | 0.046240 | 4.515152 | 0.727273 | 0.199076 | 0.000000 | 0.925772 | 0.799349 | n/a |
| M2.50 low_jump | 33 | 0 | 0.049587 | 4.848485 | 0.818182 | 0.199076 | 0.000000 | 0.974050 | 0.789339 | benefit_tradeoff=2, safe_coverage=31 |

This is the intended behavior: M2.50 gives up a small amount of low-jump performance on public low_jump, but recovers coverage from `0.925772` to `0.974050`.

## Interpretation

M2.50 should be treated as a decision safety layer, not a new neural architecture.

- M2.42 remains the safest high-coverage default.
- M2.49 is a lower-jump semantic branch.
- M2.50 is the more defensible research branch because every below-floor coverage decision is auditable.

For paper writing, M2.50 is useful because it directly addresses reward-hacking risk:

```text
The model is allowed to trade coverage for fewer jumps only when the tradeoff satisfies explicit benefit constraints.
```

## Artifacts

- `configs/best_current_model_m2_50_benefit_aware_coverage_gate.json`
- `results/incoming_review_eval_v1_m2_50_benefitgate_s090_f090_w035_*_applied/`
- `results/incoming_review_eval_v1_m2_50_benefitgate_s090_f090_w035_summary.csv`
- `results/incoming_review_eval_v1_m2_50_benefitgate_s090_f090_w035_summary.json`
- `results/public_benchmark_v1_ext33_m2_50_benefitgate_sanity_summary.csv`
- `results/public_benchmark_v1_ext33_m2_50_benefitgate_sanity_summary.json`

