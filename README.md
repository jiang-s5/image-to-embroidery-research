# Image-to-Embroidery Stitch Planning

Research prototype for converting raster images into machine-embroidery structure and DST/PES outputs.

For a minimal run, start with [QUICKSTART.md](QUICKSTART.md).

The project is not a simple image filter. The current pipeline is:

```text
input image / DST-rendered preview
  -> geometry prediction
  -> mask, density, direction, boundary, centerline
  -> stitch type, endpoints, path order
  -> vector-continuity and jump-risk supervision
  -> retrieval-augmented planner priors
  -> relation-aware transition cost
  -> graph/path planning
  -> DST/PES export and render-back evaluation
```

## Current Direction

The latest stable research direction is a geometry-to-planner model with vector-continuity supervision:

- predict embroidery regions and geometry;
- preserve centerline and boundary structure;
- predict entry/exit endpoints and path order;
- penalize disconnected vector paths and long jumps;
- export legal DST/PES files through `pyembroidery`;
- evaluate both model maps and machine-executable stitch metrics.

## Important Models

The current promoted checkpoint is model13:

```text
model13_multiformat_all_vector_continuity_20260514
```

Model14B is kept as a later geometry-continuity experiment, but it was not promoted as the main checkpoint.

See [MODEL_CARD.md](MODEL_CARD.md) for metrics and checkpoint notes.

## Research Contributions

1. DST-derived supervision dataset: real embroidery files are parsed/rendered into dense geometry, planning, and continuity labels rather than treated as ordinary image pairs.
2. Multi-task embroidery representation: the model predicts mask, density, direction, boundary, centerline, stitch type, endpoints, path order, segment structure, and continuity maps.
3. Path-continuity learning: vector-continuity labels and planner-side penalties target broken outlines, long jumps, and disconnected stitch traces before DST/PES export.
4. Conservative continuity-aware planning: fixed planner priors for component filtering, near-connect stitching, short-stitch limits, and path ordering substantially reduce jump/trim commands in command-level ablations.
5. Anti-overfit planner controls: leave-one-out, external-index, random-prior, shuffled-prior, and fixed-prior controls are used to separate true retrieval gains from planner-parameter effects.
6. Executability-first evaluation: DST/PES outputs are scored with command-level jump, trim, long-stitch, round-trip parse, and render-back visual checks.

## Architecture

![Architecture](docs/architecture.svg)

```text
Input Image / DST Render
        |
Shared Encoder
        |
Geometry Heads
(mask / density / direction / boundary / centerline)
        |
Planner Heads
(stitch type / endpoints / path order / segment map)
        |
Vector-Continuity Heads
(stitch trace / near-connect / jump endpoint)
        |
Graph & Path Planner
        |
DST / PES Export
        |
Render-back Evaluation
```

## Dataset

The recommended main training dataset is:

```text
datasets/dataset4_multiformat_all_geometry_graph
```

It is a DST-derived multi-task supervision dataset. It contains:

- rendered preview/label images;
- mask, density, direction, boundary, centerline;
- stitch type labels;
- endpoint heatmaps;
- segment maps;
- path graph JSON;
- vector-continuity labels.

The full dataset is too large for normal Git history. This repository package includes manifests and instructions. Put the full datasets under `datasets/` after cloning.

For a lightweight smoke test, use the mini demo dataset under:

```text
datasets/mini_demo
```

See [DATASET_CARD.md](DATASET_CARD.md).

## Optimization Tools

The current high-priority optimization path is implemented as three practical tools.

### Enhanced DST-Derived Labels

Build richer labels directly from DST/PES command streams:

```powershell
python build_dst_label_v2.py `
  --dst-dir path/to/embroidery_files `
  --output-dir datasets/dst_label_v2 `
  --limit 100
```

This creates `stitch_trace`, `same_color_near_connect`, `long_jump_endpoint`, `trim_endpoint`, `color_change_endpoint`, `closure_gap_endpoint`, and `path_order` maps, plus segment and path-event JSON files.

### Conservative and Retrieval-Augmented Planner

Create a small planner index from demo or training images:

```powershell
python retrieval_augmented_planner.py `
  --image-dir datasets/mini_demo/inputs `
  --output datasets/mini_demo/retrieval_planner_index.json
```

Use the index during inference:

```powershell
python infer_model3_portrait_hybrid.py inputs/your_image.png `
  --checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --output-dir outputs/your_image_model13 `
  --geometry-planner `
  --model-path-order `
  --use-continuity-planner `
  --retrieval-index datasets/mini_demo/retrieval_planner_index.json `
  --planner-config configs/relation_planner.yaml `
  --serpentine-fill
```

The `summary.json` records the retrieved matches and the final planner values applied to DST/PES export. Current ablations show that the strongest improvement comes from a conservative fixed planner prior, while retrieval-specific gains require controls before they should be claimed.

### Relation-Aware Planner Cost

`configs/relation_planner.yaml` enables the A1/A2 planner-only ablation path from the research report. It adds normalized distance, jump/trim risk, lock risk, near-connect bonus, direction alignment, endpoint compatibility, and retrieval-prior terms to the transition cost.

Run command-level executability evaluation after export:

```powershell
python tools/eval_executability.py `
  --pred outputs/your_image_model13/embroidery_output.dst `
  --report outputs/your_image_model13/executability_eval.json
```

Run the full A0/A1/A2 mini-demo ablation:

```powershell
python tools/run_planner_ablation.py `
  --input-dir datasets/mini_demo/inputs `
  --output-dir outputs/planner_ablation_mini_demo `
  --limit 12 `
  --cpu
```

This writes `metrics.csv`, `summary.json`, and `comparison.md`. Use `--reuse` to skip samples that already have an `executability_eval.json`.

Run the anti-overfit controls:

```powershell
python tools/run_planner_ablation.py `
  --input-dir datasets/mini_demo/inputs `
  --output-dir outputs/planner_ablation_mini_demo_controls `
  --limit 12 `
  --include-validation-controls `
  --external-retrieval-index outputs/retrieval_indices/dataset4_multiformat_all_index.json `
  --cpu
```

The control report includes `A2_fixed_params`, `A2_leave_one_out`, `A2_external_index`, `A2_random_prior`, and `A2_shuffled_prior`. If these match `A2_retrieval_relation`, the result should be interpreted as a planner-parameter improvement rather than proof that retrieval itself generalizes.

Create a render-back visual report for the same ablation folder:

```powershell
python tools/make_planner_visual_report.py `
  --ablation-dir outputs/planner_ablation_mini_demo_controls `
  --output-dir outputs/planner_visual_report
```

Use the contact sheet to check whether lower `jump_count` and `trim_count` are achieved by adding visible stitch connectors across blank regions.

### Render Augmentation

Create fabric, lighting, blur, color, and noise variants without changing geometry labels:

```powershell
python augment_render_inputs.py `
  --dataset-dir datasets/dataset4_multiformat_all_geometry_graph `
  --manifest manifest_dataset2.csv `
  --output-dir datasets/dataset4_render_augmented `
  --variants 2
```

Use this for robustness experiments before moving to heavier generative models such as CVAE, cGAN, diffusion, or autoregressive DST sequence modeling.

### Real-Image Canonicalization

Preprocess a real or panel-style input into an embroidery-design-like image before inference:

```powershell
python tools/preprocess_real_image.py inputs/your_image.png `
  --output-dir outputs/your_image_preprocess `
  --colors 10
```

The tool writes `canonical_input.png`, `foreground_mask.png`, `edge_map.png`, and `color_quantized.png`. For wide diagnostic panels it automatically crops the left input tile before resizing. Use both the raw-cropped image and the canonical image in ablations: raw crops may preserve fill/satin behavior better, while canonicalized images can reduce noisy fragments and emphasize outlines.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch is required by both training and inference code. For explicit installs:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

or, for a CUDA 12.1 example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

If your CUDA version differs, install the matching PyTorch wheel from the official PyTorch selector, then install the rest of this repository's requirements.

## Train

Example continuation training:

```powershell
python train_model10_vector_continuity.py `
  --dataset-dir datasets/dataset4_multiformat_all_geometry_graph `
  --output-dir models/model13_multiformat_all_vector_continuity `
  --init-checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --epochs 24
```

## Inference

Example:

```powershell
python infer_model3_portrait_hybrid.py inputs/your_image.png `
  --checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --output-dir outputs/your_image_model13 `
  --geometry-planner `
  --model-path-order `
  --use-continuity-planner `
  --retrieval-index datasets/mini_demo/retrieval_planner_index.json `
  --planner-config configs/relation_planner.yaml `
  --serpentine-fill
```

## Artifact Policy

Normal Git should contain code, documentation, small examples, and manifests. Large artifacts should be uploaded separately:

- checkpoints: Git LFS or GitHub Releases;
- full datasets: GitHub Releases, DVC remote, Hugging Face Dataset, or external storage;
- generated outputs: keep local unless they are selected figures.

See [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md).
