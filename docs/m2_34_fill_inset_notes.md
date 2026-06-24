# M2.34 Fill-Inset Selector Notes

Status note: M2.41 later improves the unified execution objective by adding a `safe_dt` adaptive fill-inset candidate to the calibrated selector. M2.34 remains the precision-safer baseline; see `docs/m2_41_safe_adaptive_inset_selector.md`.

Date: 2026-06-22

M2.34 promotes a new high-coverage mask-fill candidate family: nearest-endpoint fill rows with a small fill inset. The inset is applied only to the fill-row mask, while connector validation still uses the evaluation mask. This keeps the planner from stitching exactly on the noisy mask boundary and reduces boundary spill after DST quantization.

## Why This Was Tried

M2.32 quantized validation showed that hard-validating every mask-path segment can reduce off-mask length, but it increases jump count and worsens unified loss. The next hypothesis was that most boundary spill comes from fill rows that lie too close to the mask edge, not from long unsafe connectors.

The M2.34 change tests that hypothesis with a softer geometric change:

| Candidate | Change |
|---|---|
| `mask_fill_edgewalk_nearestrow_inset1_rows16_p40` | Erode fill-row mask by 1px |
| `mask_fill_edgewalk_nearestrow_inset2_rows16_p40` | Erode fill-row mask by 2px |

Common settings:

- `row_spacing_mm = 1.6`
- `max_connect_mm = 6.0`
- `min_connect_inside_fraction = 0.95`
- `min_run_mm = 1.0`
- `use_mask_path_connectors = true`
- `max_mask_path_mm = 40.0`
- `row_order = nearest_endpoint`
- `component_order = area`

## Single-Candidate Result

Compared with raw nearest-row fill, fill-inset keeps high coverage while lowering off-mask and improving precision.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| nearest-row p40 | 0 | 0.077217 | 7.515152 | 0.909091 | 0.412527 | 0.999689 | 0.749214 |
| nearest-row inset1 p40 | 0 | 0.075848 | 7.606061 | 0.969697 | 0.301779 | 0.994877 | 0.797562 |
| nearest-row inset2 p40 | 0 | 0.071837 | 7.696970 | 1.000000 | 0.081876 | 0.982203 | 0.854755 |

## Selector Results

### Public Benchmark Leave-One-Out

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.30 public LOO | 0 | 0.068519 | 6.636364 | 1.848485 | 0.105806 | 0.911108 | 0.798703 |
| M2.34 public LOO | 0 | 0.051174 | 5.636364 | 0.424242 | 0.082527 | 0.968881 | 0.819738 |

M2.34 chosen candidate counts:

- `mask_fill_edgewalk_nearestrow_inset2_rows16_p40`: 15
- `mask_fill_edgewalk_nearestrow_inset1_rows16_p40`: 7
- raw nearest-row p40: 5
- other candidates: 6

### Public Core-To-Full Holdout

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.30 full split | 0 | 0.074071 | 6.937500 | 2.062500 | 0.187256 | 0.927512 | 0.801813 |
| M2.34 full split | 0 | 0.051748 | 5.750000 | 0.437500 | 0.061937 | 0.951650 | 0.827483 |

### Incoming Review Evaluation-Only Holdout

The incoming review set is not used for training. It is private/local evaluation data.

| Variant | Hard Fail | Loss | Jump | Trim | Off-Mask mm | Coverage | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.30 incoming LOO | 0 | 0.084419 | 6.750000 | 4.750000 | 0.000000 | 0.856344 | 0.793664 |
| M2.34 incoming LOO | 0 | 0.078290 | 8.500000 | 1.250000 | 0.000000 | 0.995549 | 0.781068 |

Incoming improves loss, trim, and coverage, but jump increases. This is acceptable for promotion because the public benchmark improvements are broad and incoming remains hard-fail free with much higher coverage.

## Decision

M2.34 is promoted as the current best model/system.

The improvement is not from adding a larger neural network. It comes from a better geometry candidate inside the M2 candidate-selection framework:

1. Use fill-inset rows to avoid boundary spill.
2. Keep nearest-endpoint ordering for continuity.
3. Let the calibrated selector choose between inset1, inset2, raw nearest-row, skeleton, and auto-repair candidates.

## Remaining Boundary

M2.34 improves executable DST path quality, but it is still not a professional digitizer. It does not yet solve:

- true satin/tatami stitch texture,
- underlay,
- pull compensation,
- color-layer sequencing,
- professional 3D thread rendering,
- semantic stitch-type selection for portraits and complex cartoons.

The next best research direction is to make fill-inset adaptive per component, using distance-transform statistics or local boundary confidence instead of fixed 1px/2px erosion.
