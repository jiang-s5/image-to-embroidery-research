# M2.35 Adaptive Fill-Inset Ablation

Date: 2026-06-22

M2.35 tests the next idea after M2.34: replace fixed 1px/2px fill inset with a component-level adaptive inset. The goal is to avoid over-eroding thin strokes while still pulling wide fill rows away from noisy mask boundaries.

This experiment is recorded as an ablation, not promoted as the current best model/system.

## Implementation

The mask-fill generator now supports adaptive fill inset:

- Compute a distance transform for each connected component.
- Use no inset for thin components.
- Use 1px inset for medium components.
- Use up to 2px inset for wide components.
- Fall back to a smaller inset if erosion removes the whole component.

New CLI flags:

- `--adaptive-fill-inset`
- `--adaptive-fill-inset-max-px`
- `--adaptive-fill-inset-thin-max-distance-px`
- `--adaptive-fill-inset-mid-max-distance-px`

The adaptive inset is applied only to fill-row generation. Connector validation still uses the evaluation mask, matching the M2.34 design.

## Tested Candidates

All candidates use the M2.34 nearest-row mask-fill settings:

- `row_spacing_mm = 1.6`
- `max_connect_mm = 6.0`
- `min_connect_inside_fraction = 0.95`
- `min_component_pixels = 64`
- `min_run_mm = 1.0`
- `use_mask_path_connectors = true`
- `max_mask_path_mm = 40.0`
- `row_order = nearest_endpoint`
- `component_order = area`

Adaptive variants:

| Candidate | Thin max distance | Mid max distance | Max inset |
|---|---:|---:|---:|
| `adapt_t2_m5` | 2px | 5px | 2px |
| `adapt_t3_m6` | 3px | 6px | 2px |
| `adapt_t4_m8` | 4px | 8px | 2px |

## Public Benchmark Single-Candidate Results

Public benchmark: `datasets/public_benchmark_v1_ext33`, 33 samples.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 fixed inset2 | 0 | 0.071837 | 7.696970 | 1.000000 | 0.081876 | 0.982203 | 0.854755 |
| adaptive `t2_m5` | 0 | 0.074424 | 7.939394 | 1.090909 | 0.081876 | 0.984220 | 0.853786 |
| adaptive `t3_m6` | 0 | 0.074171 | 7.909091 | 1.090909 | 0.081876 | 0.985045 | 0.852444 |
| adaptive `t4_m8` | 0 | 0.076787 | 8.151515 | 1.121212 | 0.100309 | 0.987465 | 0.840308 |

Adaptive inset slightly improves coverage in some settings, but it increases jump/trim and does not improve loss. The best fixed M2.34 candidate remains stronger.

## Public Benchmark Selector Results

The adaptive candidates were added to the full M2.34 candidate pool and evaluated with leave-one-out selection.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 public LOO | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.968881 | 0.819738 |
| M2.35 adaptive public LOO | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.969006 | 0.817916 |

Chosen candidates for M2.35 public LOO:

- fixed inset2: 14
- fixed inset1: 7
- raw nearest-row: 5
- auto/eval repair: 1
- edgewalk rows16: 2
- adaptive `t4_m8`: 1
- skeleton variants: 3

The selector chooses an adaptive candidate only once out of 33 samples. The tiny coverage gain is offset by a precision drop.

## Public Core-To-Full Selector Results

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 core-to-full | 0 | 0.051748 | 5.750000 | 0.437500 | 0.061937 | 0.951650 | 0.827483 |
| M2.35 adaptive core-to-full | 0 | 0.051747 | 5.750000 | 0.437500 | 0.061937 | 0.951908 | 0.823725 |

The loss change is negligible. Coverage increases slightly, but precision decreases. This is not enough to promote M2.35.

## Incoming Review Evaluation

Incoming review set: 4 private evaluation-only samples.

Single-candidate adaptive results:

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| adaptive `t2_m5` | 0 | 0.095775 | 10.250000 | 1.750000 | 0.000000 | 0.938636 | 0.851462 |
| adaptive `t3_m6` | 0 | 0.095717 | 10.250000 | 1.750000 | 0.000000 | 0.940280 | 0.847268 |
| adaptive `t4_m8` | 0 | 0.093669 | 10.000000 | 1.750000 | 0.000000 | 0.951701 | 0.836955 |

Leave-one-out selector result:

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.34 incoming LOO | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |
| M2.35 adaptive incoming LOO | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |

The best adaptive single candidate helps on incoming, but the selector still falls back to fixed/older candidates:

- fixed inset2: 2
- raw nearest-row: 1
- edgewalk rows16: 1

## Decision

M2.35 is not promoted.

M2.34 remains the current best model/system because it is simpler and has stronger or equal validation results:

1. Fixed inset2 is still the best single mask-fill candidate on public ext33.
2. Adaptive inset is selected only rarely in leave-one-out validation.
3. Public and incoming selector metrics are ties or near-ties, with slightly worse precision for M2.35.

## Research Takeaway

The hypothesis is directionally useful, but fixed thresholds are too crude. Component width alone is not enough to decide the right inset.

The next adaptive version should use a learned or selector-driven policy with richer features:

- component area,
- skeleton length,
- local stroke width distribution,
- edge confidence,
- source family,
- estimated stitch style,
- expected coverage loss,
- expected boundary spill risk.

In other words, adaptive inset should become a learned candidate-selection feature, not just a hand-coded distance-transform threshold.
