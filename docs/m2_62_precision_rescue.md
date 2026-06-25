# M2.62 Precision Rescue

Date: 2026-06-25

## Summary

M2.62 adds a precision-rescue gate to the line-domain selector. It targets samples where the generated stitch footprint covers too much area outside the target line mask, even when the file has good executability metrics.

This is a precision-aware tradeoff:

- stitch precision improves
- component-aware risk improves
- off-mask length decreases slightly
- mean loss and jump count increase slightly

Because the research goal is realistic embroidery rather than only minimum jump count, M2.62 is useful as the current precision-aware default.

## What Changed

Updated selector:

```text
tools/select_m2_line_domain_guard.py
```

New optional gate:

```text
--line-precision-rescue
```

The rescue gate activates only when a line-domain sample has low stitch precision. It accepts a replacement candidate only if it:

- improves precision by at least a configured margin
- keeps coverage above a minimum
- does not introduce visible connectors
- does not increase off-mask stitch length
- keeps loss and jump increase within bounded slack

## Main Results

Selector output:

```text
results/public_benchmark_v1_ext33_m2_62_precision_rescue_selector/
```

Component-aware audit:

```text
results/public_benchmark_v1_ext33_m2_62_precision_rescue_component_audit/
```

Probe summary:

```text
results/m2_62_precision_rescue_probe/
```

## M2.60 vs M2.62

| Metric | M2.60 | M2.62 | Delta |
|---|---:|---:|---:|
| mean unified loss | 0.04800653 | 0.04916026 | +0.00115347 |
| mean jump count | 4.81818182 | 4.96969697 | +0.15151518 |
| mean trim count | 0.60606061 | 0.63636364 | +0.03030339 |
| mean off-mask stitch mm | 0.19907576 | 0.18479091 | -0.01428476 |
| mean visible connectors | 0.0 | 0.0 | 0.0 |
| mean coverage | 0.98090367 | 0.97895385 | -0.00194967 |
| mean stitch precision | 0.79361039 | 0.80950997 | +0.01589961 |
| component-aware mean risk | 0.09893281 | 0.09653159 | -0.00240122 |
| hard fail | 0 | 0 | 0 |

## Changed Samples

Only two samples changed metrics.

| Sample | M2.60 Candidate | M2.62 Candidate | Loss Delta | Jump Delta | Coverage Delta | Precision Delta | Risk Delta |
|---|---|---|---:|---:|---:|---:|---:|
| `qd_001` | `mask_fill_edgewalk_nearestrow_rows16_p40` | `inset2` | -0.00191936 | +1 | -0.051661 | +0.167014 | -0.05781486 |
| `qd_003` | `mask_fill_edgewalk_nearestrow_safeadt_t2m5_r055_a64_rows16_p40` | `adapt_t2m5` | +0.03999255 | +4 | -0.012683 | +0.357672 | -0.02142545 |

`qd_003` is the most important case: the previous candidate had perfect coverage but very low precision because it overfilled around the line drawing. M2.62 sacrifices a small amount of coverage and increases jump count, but it greatly reduces the overfilled stitch footprint.

## Interpretation

M2.62 should not be described as a raw-loss improvement. It is a quality tradeoff that aligns better with visual embroidery realism:

```text
less overfilled stitch footprint
more precise line-domain rendering
same hard-fail count
same visible connector count
slightly higher jump/loss
```

This matters because previous low-loss candidates sometimes achieved high coverage by stitching too broadly around thin line drawings.

## Current Recommendation

Use M2.62 when evaluating or generating line-domain / hand-drawn / QuickDraw-like images where visual precision matters.

Use M2.60 when the priority is strict lowest unified loss and jump count.

For the research direction, M2.62 is the better default because the project is moving toward realistic embroidery quality, not only command minimization.

## Next Direction

M2.63 should address why the low-precision flag remains even after large precision gains:

```text
M2.63 line-domain adaptive precision metric
```

Recommended components:

- compute precision against a stroke-width-normalized mask
- estimate expected stitch footprint width from line thickness
- separate overfill outside local stroke band from legitimate thread width
- report both strict precision and adaptive line precision

