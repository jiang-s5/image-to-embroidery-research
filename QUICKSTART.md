# Quickstart

This guide is for a quick smoke test. It does not require the full 10GB+ training dataset.

## 1. Clone

```powershell
git clone https://github.com/jiang-s5/image-to-embroidery-research.git
cd image-to-embroidery-research
```

## 2. Install

CPU example:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
```

GPU example for CUDA 12.1:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

If your CUDA version is different, install the matching PyTorch wheel first, then install `requirements.txt`.

## 3. Checkpoint

The promoted checkpoint path expected by the examples is:

```text
checkpoints/best_model13_multiformat_all_vector_continuity.pt
```

If this file is not present after cloning, download the release asset:

```text
image-to-embroidery-research_model_artifacts_20260609.zip
```

from the GitHub Release and extract it into the repository root so that the `checkpoints/` folder is restored.

## 4. Mini Demo Dataset

The repository includes a small inference demo dataset:

```text
datasets/mini_demo/inputs
```

It contains 12 sample PNG files. These are intended for quick inference and packaging checks, not for benchmark training.

## 5. Run One Image

```powershell
.\.venv\Scripts\python.exe infer_model3_portrait_hybrid.py `
  datasets/mini_demo/inputs/dst3322_00007_CEJ88_model10_continuity_prediction.png `
  --checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --output-dir outputs/quickstart_ce_demo `
  --geometry-planner `
  --model-path-order `
  --use-continuity-planner `
  --serpentine-fill `
  --cpu
```

Expected output files include:

```text
outputs/quickstart_ce_demo/embroidery_output.dst
outputs/quickstart_ce_demo/embroidery_output.pes
outputs/quickstart_ce_demo/prediction_panel.png
outputs/quickstart_ce_demo/summary.json
```

## 6. Train on the Main Dataset

After downloading and extracting the main training dataset:

```text
datasets/dataset4_multiformat_all_geometry_graph
```

continue training with:

```powershell
.\.venv\Scripts\python.exe train_model10_vector_continuity.py `
  --dataset-dir datasets/dataset4_multiformat_all_geometry_graph `
  --output-dir models/model13_multiformat_all_vector_continuity_continue `
  --init-checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --epochs 24
```
