# Image-to-Embroidery Stitch Planning

Research prototype for converting raster images into machine-embroidery structure and DST/PES outputs.

The project is not a simple image filter. The current pipeline is:

```text
input image / DST-rendered preview
  -> geometry prediction
  -> mask, density, direction, boundary, centerline
  -> stitch type, endpoints, path order
  -> vector-continuity and jump-risk supervision
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

## Dataset

The main training dataset is a DST-derived multi-task supervision dataset. It contains:

- rendered preview/label images;
- mask, density, direction, boundary, centerline;
- stitch type labels;
- endpoint heatmaps;
- segment maps;
- path graph JSON;
- vector-continuity labels.

The full dataset is too large for normal Git history. This repository package includes manifests and instructions. Put the full datasets under `datasets/` after cloning.

See [DATASET_CARD.md](DATASET_CARD.md).

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build matching your CUDA environment before running the training scripts.

## Train

Example continuation training:

```powershell
python train_model10_vector_continuity.py `
  --dataset-dir datasets/dataset4_multiformat_all_geometry_graph `
  --output-dir models/model13_multiformat_all_vector_continuity `
  --epochs 24
```

## Inference

Example:

```powershell
python infer_model3_portrait_hybrid.py inputs/your_image.png `
  --checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --output-dir outputs/your_image_model13 `
  --use-continuity-planner
```

## Artifact Policy

Normal Git should contain code, documentation, small examples, and manifests. Large artifacts should be uploaded separately:

- checkpoints: Git LFS or GitHub Releases;
- full datasets: GitHub Releases, DVC remote, Hugging Face Dataset, or external storage;
- generated outputs: keep local unless they are selected figures.

See [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md).
