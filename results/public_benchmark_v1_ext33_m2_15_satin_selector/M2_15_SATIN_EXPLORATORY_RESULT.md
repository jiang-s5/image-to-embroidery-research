# M2.15 Satin-Column Exploratory Result

Date: 2026-06-22

M2.15 tested a first satin-like border candidate:

`mask_fill_edgewalk_satin_w8_s7_i4`

The candidate samples the inset external contour and creates short column stitches from the outer rail toward the component center. The goal was to move beyond a single contour outline and approximate satin-border behavior.

## Public Benchmark v1 ext33 Leave-One-Out

| Model | Candidates | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 14 | 0 | 0.096228 | 8.848485 | 3.484848 | 0.855224 | 0.837897 |
| M2.15 satin selector | 15 | 0 | 0.095029 | 8.848485 | 3.212121 | 0.866435 | 0.830423 |

The LOO result is weakly positive: M2.15 improves unified loss, trim count, and coverage, but slightly reduces stitch precision and increases off-mask length from `0.090794` to `0.110715`.

## Core-to-Full Holdout

| Model | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|
| M2.14 outline selector | 0 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.843477 |
| M2.15 satin selector | 0 | 0.111060 | 10.1875 | 3.7500 | 0.826159 | 0.843501 |

Core-to-full does not show a material improvement. The selector chose the satin candidate on one test sample, but aggregate metrics are essentially unchanged.

## Incoming Review Holdout

| Model | Samples | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|
| M2.14 outline selector | 4 | 0 | 0.090709 | 7.5000 | 0.856321 |
| M2.15 satin selector | 4 | 0 | 0.090709 | 7.5000 | 0.856321 |

Incoming review is unchanged. The selector did not choose the satin candidate:

| Candidate | Count |
|---|---:|
| auto_evalrepair_r10_c20 | 2 |
| mask_fill_edgewalk_rows16_p40 | 2 |

## Standalone Satin Candidate

Standalone satin-column candidates on public ext33:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Off-Mask mm | Mean Coverage | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| satin w4 s6 i3 | 20/33 | 0.283898 | 97.151515 | 17.575758 | 1.313558 | 0.999747 | 0.718296 |
| satin w6 s5 i3 | 20/33 | 0.283649 | 97.090909 | 17.575758 | 1.313558 | 0.999747 | 0.718061 |
| satin w8 s7 i4 | 20/33 | 0.282392 | 94.878788 | 17.212121 | 1.343558 | 0.999726 | 0.718969 |

Standalone satin-column candidate on incoming review:

| Candidate | Hard Fail | Mean Unified Loss | Mean Jump Count | Mean Trim Count | Mean Coverage |
|---|---:|---:|---:|---:|---:|
| satin w8 s7 i4 | 1/4 | 0.193626 | 34.7500 | 7.0000 | 0.997118 |

## Decision

M2.15 is **not promoted**.

M2.14 remains the current best because M2.15 only improves public LOO and does not provide a material core-to-full or incoming-review gain.

## Research Interpretation

This experiment is still useful. It shows that a simple centroid-normal satin approximation is not enough. It increases coverage and gives a satin-like local texture, but it still carries too much jump/trim and precision cost.

The next satin direction should be:

1. build paired inner/outer rails from distance-transform level sets;
2. stitch a continuous zigzag between the rails instead of independent columns;
3. classify only suitable border regions as satin, leaving complex thin regions to running stitch;
4. evaluate satin-specific realism, not only coverage and command safety.
