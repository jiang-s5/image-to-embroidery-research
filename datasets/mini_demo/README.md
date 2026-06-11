# Mini Demo Dataset

This folder is a lightweight inference demo set. It is included so a new user can run the pipeline without downloading the full training corpus.

Contents:

- `inputs/`: 12 PNG images for quick inference tests.
- `manifest.csv`: file list and intended use.
- `retrieval_planner_index.json`: tiny similarity index for retrieval-augmented planner smoke tests.

Use this set for:

- installation checks;
- checkpoint loading checks;
- output-file checks for DST/PES export;
- quick screenshots for reports.
- checking that retrieval planner priors are read and recorded in `summary.json`.

Do not use it as a formal benchmark. The full recommended training dataset is:

```text
datasets/dataset4_multiformat_all_geometry_graph
```
