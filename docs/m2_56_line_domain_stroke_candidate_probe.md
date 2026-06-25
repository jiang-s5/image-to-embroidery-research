# M2.56 Line-Domain Stroke Candidate Probe

Date: 2026-06-25

M2.56 follows the M2.55 line-domain guard.

M2.55 showed that reranking can only make very small safe improvements because the current candidate pool does not contain a better low-jump line-domain alternative for difficult QuickDraw samples such as `qd_007`.

M2.56 tests whether a dedicated skeleton/running-stitch candidate can fill that gap.

## Tool

```text
tools/run_line_domain_stroke_candidate.py
```

The tool generates a candidate directory compatible with the M2 selector:

- `source_aware_hybrid_rows.csv`
- `coverage_rows.csv`
- per-sample `prediction.dst`
- per-sample `eval_executability.json`
- per-sample `coverage_report.json`
- per-sample `generator_report.json`

For `QuickDraw` and `Rendered text`, it traces the skeleton as a running-stitch graph. For non-line sources, it inherits the fallback low-jump selection metrics so that the candidate can be passed into the same selector interface.

## Tested Variants

Ten line-stroke variants were tested on `datasets/public_benchmark_v1_ext33`.

The probe sweeps:

- connector dilation;
- maximum component connection distance;
- minimum connector inside-mask fraction;
- short-gap connection relaxation;
- minimum skeleton component size.

Summary artifact:

```text
results/m2_56_line_domain_stroke_candidate_probe/variant_summary.csv
```

## Best Probe Variant

The best variant by QuickDraw loss among the tested line-stroke candidates was:

```text
d0_c12_i090_min30
```

It uses:

```text
connector_dilation_px = 0
max_connect_mm = 12
min_connect_inside_fraction = 0.90
min_component_pixels = 30
```

## Comparison To Current Baseline

Current baseline is M2.51/M2.53 low-jump selection.

| Metric | M2.51 low_jump baseline | Best M2.56 stroke candidate |
|---|---:|---:|
| mean loss | 0.048809 | 0.080492 |
| mean jump | 4.909091 | 8.454545 |
| mean coverage | 0.979476 | 0.883959 |
| mean precision | 0.793264 | 0.847001 |
| QuickDraw loss | 0.064707 | 0.160356 |
| QuickDraw jump | 7.250000 | 17.750000 |
| QuickDraw coverage | 0.969279 | 0.837139 |
| QuickDraw precision | 0.605459 | 0.691860 |
| `qd_007` loss | 0.128886 | 0.203187 |
| `qd_007` jump | 14 | 23 |

The line-stroke candidate does improve precision, especially for rendered text, but it loses too much on jump count and coverage.

## Selector Probe

The best line-stroke candidate was added to the M2.55 selector pool as:

```text
line_stroke_d0_c12_min30
```

The selector-level metrics did not improve. The apparent candidate switches were mostly non-line samples inheriting fallback metrics under the new candidate name, not true quality gains.

For actual line-domain samples, M2.55 still only changed `qd_006`, the same small safe change observed before M2.56.

## Interpretation

M2.56 is an important negative result.

It shows that:

```text
pure skeleton tracing is not enough for line-domain embroidery planning.
```

Why:

- QuickDraw skeletons split into many disconnected components;
- tracing every component produces too many jumps;
- aggressively connecting components creates visible/off-mask stitches;
- filtering small components reduces hard failures but loses coverage;
- text receives high precision but low coverage because stroke tracing under-fills thick glyph regions.

## Research Decision

M2.56 should not replace the current best model.

Current default remains:

```text
M2.53/M2.51 low-jump robust policy
+ M2.55 bounded line-domain guard
```

The next real direction is not another simple skeleton tracer. It should be:

```text
M2.57 stroke grouping and line topology planner
```

with:

- skeleton component clustering;
- endpoint pairing before stitching;
- contour-to-centerline stroke grouping;
- minimum-jump stroke ordering;
- explicit line-domain reward for coverage, precision, and jump together;
- separate text strategy for thick glyphs.

## Artifacts

- `tools/run_line_domain_stroke_candidate.py`
- `configs/best_current_model_m2_56_line_domain_stroke_candidate_probe.json`
- `results/m2_56_line_domain_stroke_candidate_probe/variant_summary.csv`
- `results/m2_56_line_domain_stroke_candidate_probe/probe_summary.json`

