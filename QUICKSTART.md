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

The quickstart also uses a small retrieval planner index:

```text
datasets/mini_demo/retrieval_planner_index.json
```

If it is missing, rebuild it with:

```powershell
.\.venv\Scripts\python.exe retrieval_augmented_planner.py `
  --image-dir datasets/mini_demo/inputs `
  --output datasets/mini_demo/retrieval_planner_index.json
```

## 5. Run One Image

```powershell
.\.venv\Scripts\python.exe infer_model3_portrait_hybrid.py `
  datasets/mini_demo/inputs/dst3322_00007_CEJ88_model10_continuity_prediction.png `
  --checkpoint checkpoints/best_model13_multiformat_all_vector_continuity.pt `
  --output-dir outputs/quickstart_ce_demo `
  --geometry-planner `
  --model-path-order `
  --use-continuity-planner `
  --retrieval-index datasets/mini_demo/retrieval_planner_index.json `
  --planner-config configs/relation_planner.yaml `
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

The `summary.json` should include `planner_values` and `retrieval_planner`, which show the retrieved nearest examples and the final planner settings used for export.

Run the command-level executability check:

```powershell
.\.venv\Scripts\python.exe tools/eval_executability.py `
  --pred outputs/quickstart_ce_demo/embroidery_output.dst `
  --report outputs/quickstart_ce_demo/executability_eval.json
```

The report includes parse success, stitch/jump/trim counts, high-risk jump count, illegal long-stitch count, and max jump length.

## 6. Optional Label and Augmentation Builders

Enhanced DST/PES labels:

```powershell
.\.venv\Scripts\python.exe build_dst_label_v2.py `
  --dst-dir path/to/embroidery_files `
  --output-dir datasets/dst_label_v2 `
  --limit 20
```

Render augmentation:

```powershell
.\.venv\Scripts\python.exe augment_render_inputs.py `
  --input-dir datasets/mini_demo/inputs `
  --output-dir outputs/mini_demo_render_augmented `
  --variants 1 `
  --limit 3
```

Real-image preprocessing:

```powershell
.\.venv\Scripts\python.exe tools/preprocess_real_image.py `
  datasets/mini_demo/inputs/dst3322_00058_ANIB1037_model10_continuity_prediction.png `
  --output-dir outputs/bird_preprocess `
  --colors 10
```

Then run inference on either `outputs/bird_preprocess/input_resized.png` for a raw crop or `outputs/bird_preprocess/canonical_input.png` for the color-simplified, edge-enhanced version.

## 7. Train on the Main Dataset

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
