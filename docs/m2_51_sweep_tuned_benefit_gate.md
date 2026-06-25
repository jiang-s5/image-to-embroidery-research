# M2.51 Sweep-Tuned Benefit Gate

Date: 2026-06-25

M2.51 continues from M2.50. M2.50 added an auditable benefit-aware coverage gate, but its thresholds were still chosen manually:

```text
absolute min coverage
min jump gain
min precision
max loss slack
coverage floor
coverage penalty weight
```

M2.51 adds a reproducible threshold-sweep tool:

```text
tools/sweep_m2_benefit_gate.py
```

The tool evaluates a grid of benefit-gate settings without copying DST outputs. It writes:

- `benefit_gate_sweep_rows.csv`
- `benefit_gate_best_rows.csv`
- `best_selected_<profile>.csv`
- `benefit_gate_sweep_summary.json`

This moves the system from manual parameter picking toward loop engineering:

```text
candidate planner outputs
  -> selector + benefit gate
  -> evaluator metrics
  -> threshold sweep
  -> recommended gate policy
```

## Search Objective

The sweep objective is intentionally conservative:

```text
loss
+ 0.003 * jump
+ 0.002 * trim
+ 0.010 * off_mask
+ 0.002 * visible
+ 0.500 * max(0, min_mean_coverage - coverage)
+ hard_fail penalty
```

For this run:

```text
min_mean_coverage = 0.90
max_hard_fail = 0
```

So the search does not simply chase lower jump count. It penalizes coverage collapse and hard failures.

## Recommended Parameters

The sweep supports multiple equivalent solutions on the 4-sample incoming set. The public ext33 sanity sweep selected a stable shared setting:

```text
--profile-rerank-strength 0.90
--profile-coverage-penalty-weight 0.20
--profile-min-coverage-by-name balanced=0.85,low_jump=0.85,strict_low_jump=0.85
--profile-benefit-min-coverage 0.60
--profile-benefit-min-jump-gain 2
--profile-benefit-min-precision 0.78
--profile-benefit-max-loss-slack 0.00
```

This is the recommended M2.51 setting.

## Incoming Review Result

The incoming review set contains 4 held-out paired review samples.

| Method / Profile | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision | Gate Modes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M2.42 default | 0 | 0.074710 | 8.250000 | 1.000000 | 0.000000 | 0.000000 | 0.996541 | 0.753519 | n/a |
| M2.50 balanced | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 | benefit_tradeoff=1, safe_coverage=3 |
| M2.51 balanced | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 | benefit_tradeoff=1, safe_coverage=3 |
| M2.51 low_jump | 0 | 0.073831 | 7.250000 | 2.250000 | 0.000000 | 0.000000 | 0.920700 | 0.788361 | benefit_tradeoff=1, safe_coverage=3 |

The incoming score is unchanged from M2.50, but now the threshold policy is produced by a reproducible sweep rather than manual selection.

## Public ext33 Sanity Result

This is not a generalization claim because the selector was trained from public ext33 candidate tables. It is a sanity check that the tuned thresholds do not introduce hard failures.

| Method / Profile | Samples | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Visible | Coverage | Precision | Gate Modes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M2.50 balanced | 33 | 0 | 0.052103 | 5.606061 | 0.757576 | 0.047591 | 0.000000 | 0.982012 | 0.806207 | benefit_tradeoff=1, safe_coverage=32 |
| M2.51 balanced | 33 | 0 | 0.052103 | 5.606061 | 0.757576 | 0.047591 | 0.000000 | 0.982012 | 0.806207 | benefit_tradeoff=1, safe_coverage=32 |
| M2.50 low_jump | 33 | 0 | 0.049587 | 4.848485 | 0.818182 | 0.199076 | 0.000000 | 0.974050 | 0.789339 | benefit_tradeoff=2, safe_coverage=31 |
| M2.51 low_jump | 33 | 0 | 0.048809 | 4.909091 | 0.606061 | 0.199076 | 0.000000 | 0.979476 | 0.793264 | benefit_tradeoff=1, safe_coverage=32 |

M2.51 low_jump trades a tiny jump increase for better loss, trim, coverage, and precision:

- loss: `0.049587 -> 0.048809`
- jump: `4.848485 -> 4.909091`
- trim: `0.818182 -> 0.606061`
- coverage: `0.974050 -> 0.979476`
- precision: `0.789339 -> 0.793264`

## Interpretation

M2.51 is not a new neural model. It is a reproducible search layer for the M2.50 decision-safety gate.

Its value is:

- lower manual tuning risk;
- repeatable gate-threshold selection;
- direct evidence for why a below-floor candidate is allowed;
- safer low-jump profile behavior on public sanity checks.

For the paper narrative, M2.51 is a better experimental protocol than M2.50:

```text
We do not hand-pick safety thresholds; we sweep them under a coverage-constrained objective and then report the selected policy.
```

## Artifacts

- `tools/sweep_m2_benefit_gate.py`
- `configs/best_current_model_m2_51_sweep_tuned_benefit_gate.json`
- `results/incoming_review_eval_v1_m2_51_benefitgate_sweep/`
- `results/public_benchmark_v1_ext33_m2_51_benefitgate_sweep_sanity/`
- `results/incoming_review_eval_v1_m2_51_sweeptuned_gate_*_applied/`
- `results/public_benchmark_v1_ext33_m2_51_sweeptuned_gate_*_applied/`
- `results/incoming_review_eval_v1_m2_51_sweeptuned_gate_summary.csv`
- `results/public_benchmark_v1_ext33_m2_51_sweeptuned_gate_summary.csv`

