# GitHub Upload Guide

## Recommended Repository Layout

```text
image-to-embroidery-research/
  README.md
  requirements.txt
  *.py
  DATASET_CARD.md
  MODEL_CARD.md
  artifact_manifest.csv
  checkpoints/
    best_model13_multiformat_all_vector_continuity.pt
  sample_artifacts/
    previews/
```

## What To Commit Normally

Commit:

- source code;
- training/inference scripts;
- documentation;
- small previews and examples;
- artifact manifests;
- one small current checkpoint if needed.

Avoid committing:

- full datasets;
- generated output folders;
- virtual environments;
- raw `.7z`/`.zip` archives;
- large batches of model checkpoints.

## Large Artifacts

The full local datasets are tens of GB. Put them in one of these places:

- GitHub Releases;
- Git LFS;
- DVC remote storage;
- Hugging Face Datasets/Models;
- institutional storage or cloud drive.

Then keep only the manifest and download instructions in Git.

## Safe Data Collection Policy

For public embroidery websites:

- crawl slowly;
- use a clear User-Agent;
- save cached pages;
- avoid concurrent requests;
- never guess download IDs;
- do not bypass login or access control;
- if login is required, download through a normal user session and import local files.

## Local Commands

Initialize repository:

```powershell
git init
git add .
git commit -m "Initial image-to-embroidery research package"
```

If you choose Git LFS for checkpoints:

```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add checkpoints/*.pt
git commit -m "Add promoted checkpoint through LFS"
```

If you do not use LFS, keep checkpoints out of Git and upload them as Release assets.
