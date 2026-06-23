# Public Benchmark v1

This benchmark is for public, style-diverse evaluation of image-to-embroidery
DST/PES generation. It is not a professional DST ground-truth benchmark.

The goal is to test whether planner changes generalize beyond the small paired
held-out set:

## Current ext33 Status

The initial `datasets/public_benchmark_v1` smoke set contains 7 samples. The current external validation set used by the M2.30-M2.40 experiments is `datasets/public_benchmark_v1_ext33`, with 33 samples: 10 Openclipart, 10 OpenMoji, 8 QuickDraw, and 5 rendered text samples. The ext33 benchmark does not yet include Oxford-IIIT Pet or another real-photo-with-mask subset, so real-photo generalization remains a future validation gap.

See [public_benchmark_ext33_validation.md](public_benchmark_ext33_validation.md) for current M2.1/M2.34/M2.39/M2.40 results and the geometry-prior ablation summary.

- DST/PES parse success,
- jump/trim count,
- visible connector count/length,
- off-mask stitch length,
- unified executability score,
- hard-fail rate.

## Recommended Composition

| Subset | Target Count | Purpose |
| --- | ---: | --- |
| Openclipart | 10 | Flat vector/logo-like shapes |
| OpenMoji | 10 | Multi-color cartoon icons |
| QuickDraw | 8 | Running-stitch/line-drawing structure |
| Oxford-IIIT Pet | 8 | Real photos with trimap masks |
| Rendered text | 4 | Letters, holes, thin structures |

Start with 20 samples if runtime is tight:

| Subset | Count |
| --- | ---: |
| Openclipart | 5 |
| OpenMoji | 5 |
| QuickDraw | 5 |
| Oxford-IIIT Pet | 3 |
| Rendered text | 2 |

## Build Command

Minimal local smoke benchmark:

```powershell
python tools\prepare_public_benchmark_v1.py `
  --output-dir datasets\public_benchmark_v1 `
  --clean `
  --download-openmoji `
  --openmoji-limit 5 `
  --text-limit 2
```

With manually curated local assets:

```powershell
python tools\prepare_public_benchmark_v1.py `
  --output-dir datasets\public_benchmark_v1 `
  --clean `
  --openclipart-dir external\public_sources\openclipart_selected `
  --quickdraw-dir external\public_sources\quickdraw_rendered `
  --oxford-pet-dir external\public_sources\oxford_pet_selected `
  --download-openmoji `
  --text-limit 4
```

For Oxford-IIIT Pet, point --oxford-pet-dir at either the dataset root containing images/ and nnotations/trimaps/, or a selected-image directory with matching trimap PNGs discoverable nearby. Images without a trimap are skipped, because this subset is meant to test real-photo input with an external foreground mask rather than a nonwhite heuristic mask.

The script writes:

```text
datasets/public_benchmark_v1/
  manifest.csv
  README.md
  source_licenses.json
  openmoji_10/
  text_render_4/
  ...
```

`manifest.csv` columns:

```text
sample_id, source, category, license, image_path, mask_path, source_url, split, notes
```

## Reproducible ext33 Validation

The `datasets/public_benchmark_v1_ext33` image files are not stored in normal Git history. To reproduce the current external-validation tables, rebuild or restore the ext33 benchmark under `datasets/public_benchmark_v1_ext33`, then run the documented evaluation commands below.

Run the 20-sample B0-B4 geometry-prior follow-up:

```powershell
python tools\run_geometry_priors_ablation.py `
  --input-dir datasets\public_benchmark_v1_ext33 `
  --checkpoint checkpoints\best_model13_multiformat_all_vector_continuity.pt `
  --m2-edge-policy checkpoints\m2_edge_policy_m2_1_globalrepair20.json `
  --output-dir results\public_benchmark_v1_ext33_repair_veto_core20 `
  --limit 20 `
  --max-components 40 `
  --safe-connect-repair-max-mm 20.0 `
  --m2-edge-policy-top-k 4 `
  --m2-jump-aware-weight 0.20 `
  --m2-visible-weight 0.30 `
  --m2-offmask-weight 0.25
```

This produces aggregate, per-domain, and per-sample CSV/JSON files under `results/public_benchmark_v1_ext33_repair_veto_core20/`. The current conclusion is negative for hard EDT/Canny repair vetoes and still negative for promoting B4 soft geometry rerank as the main planner.

The current balanced best system remains M2.34, not M2.1. M2.1 is retained for historical comparison because it was the strongest early paired-holdout decode-loop result, but it fails on most ext33 public samples.

## Evaluation

Run the geometry-prior ablation on the generated benchmark:

```powershell
python tools\run_geometry_priors_ablation.py `
  --input-dir datasets\public_benchmark_v1 `
  --checkpoint checkpoints\best_model13_multiformat_all_vector_continuity.pt `
  --output-dir results\public_benchmark_v1_geometry_priors `
  --cpu
```

For formal reporting, use the 4 paired held-out samples and this public benchmark
as two separate test blocks:

| Benchmark | Purpose |
| --- | --- |
| Paired held-out | Preserve historical M2.1/M2.2 comparability |
| Public benchmark v1 | Public style-diverse generalization |

## Source Notes

- Openclipart: public-domain/CC0 clipart, suitable for vector/logo-like tests.
- OpenMoji: CC BY-SA 4.0 graphics; attribution required.
- Quick, Draw!: Google Creative Lab dataset under CC BY 4.0.
- Oxford-IIIT Pet: real photos with pixel-level trimap segmentation annotations;
  preserve dataset attribution and terms.
- Rendered text: generated locally; record the exact font and font license before
  publishing the benchmark.

## Paper Wording

Use this benchmark carefully:

> Public benchmark images are used for executability and visual-risk evaluation,
> not for professional DST ground-truth matching.

Do not claim these images prove the generated DST matches professional digitizer
output. They support claims about executable quality and failure-risk reduction.
