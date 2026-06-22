# M2.16 Satin-Rail Exploratory Result

Date: 2026-06-22

M2.16 tested a paired-rail satin candidate:

`mask_fill_edgewalk_satinrail_w8_s7_i4`

Unlike M2.15's independent centroid-normal columns, this candidate stitches a continuous zigzag sequence:

```text
outer_0 -> inner_0 -> outer_1 -> inner_1 -> ...
```

The goal was to better approximate satin stitching by moving between paired outer/inner rails instead of drawing isolated columns.

## Public Benchmark v1 ext33 Leave-One-Out

| Model | Candidates | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 14 | 0 | 0.096228 | 8.848485 | 3.484848 | 0.855224 | 0.837897 |
| M2.15 satin-column selector | 15 | 0 | 0.095029 | 8.848485 | 3.212121 | 0.866435 | 0.830423 |
| M2.16 satin-rail selector | 16 | 0 | 0.095029 | 8.848485 | 3.212121 | 0.866435 | 0.830484 |

M2.16 is effectively tied with M2.15 on public leave-one-out. It does not provide a material additional gain over the previous satin-column probe.

## Core-to-Full Holdout

| Model | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 0 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.843477 |
| M2.16 satin-rail selector | 0 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.843501 |

Core-to-full is unchanged from M2.15 and materially unchanged from M2.14.

## Incoming Review Holdout

| Model | Samples | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|
| M2.14 outline selector | 4 | 0 | 0.090709 | 7.5000 | 0.856321 |
| M2.16 satin-rail selector | 4 | 0 | 0.090709 | 7.5000 | 0.856321 |

Incoming review is unchanged. The selector did not choose the satin-rail candidate:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 2 |
| mask_fill_edgewalk_rows16_p40 | 2 |

## Standalone Satin-Rail Candidate

Standalone public ext33 candidates:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| satin rail w6 s6 i3 | 20/33 | 0.284219 | 97.151515 | 17.575758 | 1.327842 | 0.999743 | 0.718540 |
| satin rail w8 s7 i4 | 20/33 | 0.282264 | 94.878788 | 17.212121 | 1.348388 | 0.999726 | 0.719233 |
| satin rail w10 s8 i5 | 20/33 | 0.283077 | 95.303030 | 17.333333 | 1.364176 | 0.999722 | 0.719361 |

Standalone incoming review:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|
| satin rail w8 s7 i4 | 1/4 | 0.185879 | 34.0000 | 6.7500 | 0.997118 |

## Decision

M2.16 is **not promoted**.

M2.14 remains the current best. M2.16 validates that the project needs a better rail-construction method, not merely a different ordering of centroid-normal columns.

## Research Interpretation

The paired-rail idea is conceptually right, but this implementation still builds rails from centroid-normal projections. That is too crude for concave shapes, letters, and multi-lobed icons.

The next satin direction should use distance-transform level sets:

1. derive an outer rail from a small positive distance-to-boundary contour;
2. derive an inner rail from a deeper distance contour;
3. match rail points by local nearest-neighbor or geodesic order, not centroid direction;
4. reject rail pairs that cross holes or thin negative spaces;
5. evaluate satin borders with a dedicated render-back border-thickness metric.
