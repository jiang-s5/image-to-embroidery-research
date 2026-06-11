# Dataset Card

## Main Dataset

Recommended primary training dataset:

```text
datasets/dataset4_multiformat_all_geometry_graph
```

This is the most complete geometry-graph package in the current release. Download it from GitHub Releases and extract it under `datasets/`.

Older or smaller datasets remain useful for ablation and debugging, but new users should start with `dataset4_multiformat_all_geometry_graph`.

## Mini Demo Dataset

A lightweight inference demo set is included in Git:

```text
datasets/mini_demo
```

It contains 12 PNG inputs for quick installation and inference checks. It is not a formal training or benchmark dataset.

## Composition

The full dataset is built from real embroidery files, mainly DST and other machine-embroidery formats.

The supervision chain is:

```text
source embroidery file
  -> render/parse stitches
  -> raster labels
  -> geometry labels
  -> graph labels
  -> vector-continuity labels
```

Core sample fields:

| Field | Meaning |
| --- | --- |
| `pair_id` | Stable sample id |
| `source_dst` | Original embroidery file path |
| `input_png` | Rendered or paired input image |
| `mask_png` | Embroidery foreground mask |
| `density_npy` | Local stitch density |
| `direction_xy_npy` | Directed stitch vector field |
| `direction_axis_npy` | Undirected stitch axis field |
| `direction_confidence_npy` | Direction confidence |
| `boundary_png` | Boundary map |
| `centerline_png` | Centerline/skeleton map |
| `color_layer_png` | Color layer map |
| `stitch_type_heuristic_png` | Heuristic running/satin/fill label |
| `endpoint_heatmap_npy` | Entry/exit endpoint heatmap |
| `segment_id_map_npy` | Segment id map |
| `path_graph_json` | Segment graph/path metadata |
| `*_vector_continuity.npy` | Continuity, near-connect, and jump-risk supervision |

## Current Split

For older `dataset2_collection_20260512_geometry_graph` experiments:

```text
total: 3033
train: 2206
val:   381
test:  446
```

For `dataset4_multiformat_all_geometry_graph`, read the split counts from the extracted dataset manifest because this package may be rebuilt as new DST/PES sources are imported.

## Why This Dataset Is Different

This is not an image-classification dataset. It is a multi-task embroidery-planning dataset. A model trained on it should learn:

- where embroidery exists;
- how dense the stitches are;
- where boundaries and centerlines are;
- what stitch family is likely needed;
- where segments enter and exit;
- how path order should behave;
- where long jumps or continuity breaks are risky.

## Known Limitations

- Some direction labels are weak because DST does not always encode high-level digitizer intent.
- Some stitch-type labels are heuristic, not manually annotated.
- Pixel-level path order is useful but still weaker than true segment-level planning.
- Professional satin/fill style still needs stronger rule-guided or segment-level supervision.

## External Public-Page Dataset

The Embroideres public-page collector stores page metadata and public preview images:

```text
datasets/embroideres_free_raw_20260513_category_probe
datasets/embroideres_free_1_raw_public_manifest
```

If DST/PES downloads require login, do not bypass access control. Download files through a normal logged-in session and then import them into the dataset builder.
