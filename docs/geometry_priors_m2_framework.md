# Geometry Priors for M2 Planner Decoding

This note documents the next planner optimization direction:

```text
Graph-TSP candidates
  -> hard-safe filter
  -> M2 risk-aware ranking
  -> jump-aware rerank
  -> mask-safe connect repair
  -> EDT/Canny repair veto
  -> DST/PES export
  -> executable render-back evaluation
```

## Why This Exists

The current M2.1 global-repair branch already reduced jump and trim counts, but
remaining hard failures are dominated by:

- visible connectors crossing strong visual boundaries,
- stitches leaving the active mask,
- fragile mask holes/noise that make safe-connect repair unreliable.

The next useful step is therefore not a larger model first. It is a classical
geometry-prior layer that gives the planner stable local facts:

- **EDT distance transform**: how far a connector stays from the nearest mask
  boundary.
- **Sobel/Canny edge maps**: whether a connector crosses a strong visual edge.
- **Morphological cleanup**: a cleaner planning mask for repair decisions.

Pilot experiments showed that strong Sobel/Canny routing penalties can
over-constrain the planner and cause jump explosions. The later core20 public
benchmark follow-up shows that even hard EDT/Canny repair vetoes are not a
promoted main method: they do not reduce hard-fail rate and they increase jump
count. Geometry priors should therefore be treated as diagnostics or weak
optional reranking features unless a future experiment proves a better tradeoff.

## New Code

- `planner/geometry_priors.py`
  - `build_geometry_priors(...)`
  - `connector_geometry_stats(...)`
  - `geometry_stats_are_safe(...)`
  - `save_geometry_prior_artifacts(...)`
- `planner/graph_tsp.py`
  - `GraphTSPConfig` now accepts geometry prior weights and thresholds.
  - selected edges record `gp_dt_*`, `gp_sobel_cross_mean`,
    `gp_canny_cross_frac`, and geometry risk flags.
- `infer_model3_portrait_hybrid.py`
  - new `--geometry-priors` switch.
  - new `--gp-repair-veto-dt`, `--gp-repair-veto-sobel`, and
    `--gp-repair-veto-canny` switches.
  - saves `geometry_priors/clean_mask.png`, `dt_vis.png`, `sobel_vis.png`,
    `canny_edges.png`, `dt_map.npy`, `sobel_mag.npy`, and metadata.
- `tools/score_unified.py`
  - keeps the original score unchanged.
  - adds geometry diagnostic columns when present.

## Recommended Command

```powershell
python infer_model3_portrait_hybrid.py inputs\your_image.png `
  --checkpoint checkpoints\best_model13_multiformat_all_vector_continuity.pt `
  --output-dir outputs\your_image_m2_geometry_priors `
  --geometry-planner `
  --model-path-order `
  --graph-tsp-planner `
  --m2-edge-policy checkpoints\m2_edge_policy_m2_1_globalrepair20.json `
  --m2-edge-policy-top-k 4 `
  --m2-hard-safe-filter `
  --m2-jump-aware-weight 0.20 `
  --m2-visible-weight 0.30 `
  --m2-offmask-weight 0.25 `
  --safe-connect-repair `
  --safe-connect-repair-max-mm 20 `
  --safe-connect-repair-global-mask `
  --geometry-priors `
  --m2-dt-weight 0 `
  --m2-sobel-weight 0 `
  --m2-canny-weight 0 `
  --gp-repair-veto-dt `
  --gp-repair-veto-canny
```

## Default Geometry Prior Settings

```text
gp_dt_method = opencv_precise_l2
gp_sample_step_px = 1.0
gp_morph_open = 3
gp_morph_close = 5
gp_remove_small_objects = 64
gp_remove_small_holes = 128
gp_gaussian_sigma = 1.0
gp_sobel_ksize = 3
gp_canny_low_ratio = 0.4
gp_canny_high_quantile = 0.90
m2_dt_weight = 0.0
m2_sobel_weight = 0.0
m2_canny_weight = 0.0
gp_repair_veto_dt = true
gp_repair_veto_sobel = false
gp_repair_veto_canny = true
safe_connect_repair_dt_min_px = 1.0
safe_connect_repair_dt_q05_px = 1.5
safe_connect_repair_canny_frac_max = 0.12
safe_connect_repair_sobel_mean_max = 0.18
```

## Ablation Plan

Use the same held-out samples and the same unified score definition.

| Variant | Change |
| --- | --- |
| B0 | M2.1 globalrepair20 baseline |
| B1 | B0 + EDT repair veto |
| B2 | B0 + Canny repair veto |
| B3 | B0 + EDT + Canny repair veto |
| B4 | B0 + weak EDT/Sobel/Canny Graph-TSP cost terms, no repair veto |
| Legacy | Strong EDT/Sobel/Canny routing penalty, for negative-control analysis only |

Primary success criteria:

- unified loss does not regress,
- hard score does not regress,
- visible connector count/length decreases,
- off-mask stitch length decreases,
- jump count does not rebound by more than 10%.

Secondary evidence:

- `graph_tsp_trace.json` shows lower geometry-risk edge selection,
- `geometry_priors/dt_vis.png` and `canny_edges.png` explain failure cases,
- safe-connect repairs do not cross strong Canny edges.

Core20 finding:

> Strong edge-based penalties and hard EDT/Canny repair vetoes can over-constrain
> the route planner and cause excessive jumps. B4 soft geometry rerank lowers
> jump count, but worsens unified loss and off-mask/visual risk. Visual-edge
> priors should be kept as diagnostics or very weak optional features, not as the
> primary routing or repair objective.

## Research Positioning

This is a hybrid decoding layer:

```text
learned geometry maps + M2 edge policy + classical geometric safety priors
```

It should be described as a planner-safety improvement, not as the final answer
to realistic stitch rendering. Realistic embroidery still needs richer stitch
type modeling, density planning, underlay, pull compensation, and thread-level
rendering.
