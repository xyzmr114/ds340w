# Experiment protocol (fixed 2026-10-14, before any model training)

Purpose: decide splits, metrics, baselines and tests **before** seeing results, so nothing is tuned to the test set. Any later deviation is logged at the bottom with a reason.

## Data
- 20 m cubes over Vidarbha (76.00–80.75°E, 19.00–21.75°N), 209 blocks of 0.25°, built by `pipelines/cube20.py` from Sentinel-2 L2A (2018+) and Landsat 8/9 C2 L2 (2015–2017). The scene IDs used are stored in each cube's attributes.
- Cropland = ESA WorldCover 2021 class 40. Weather = CHIRPS v2.0 rainfall plus ERA5 surface soil moisture and 2 m temperature.

## Tasks
- **A. Residue cover.** Inputs: 6 dekads of the 6 bands + clear count + phenology features (+ weather in ablations). Target: NDTI at leads 1, 2, 3 dekads. Scored only where NDTI was observed and the pixel is cropland.
- **B. Dry-season water.** Inputs: monthly Jun–Oct indices + last summer's state (+ weather). Targets: April–May irrigated-cropland fraction and water fraction.

## Splits (spatial and temporal blocking)
- Blocks are grouped into 2×2 super-blocks (~55 km). Super-blocks are assigned by fixed seed 340: 70% train, 10% val, 20% test (`stage20.split_of`).
- Task A seasons: train 2018–2022, val 2023, test 2024 and 2025.
- Task B water years: test 2018 (→ summer 2019, post-drought), 2023 (→ 2024), 2025 (→ 2026); val 2022; train the rest.
- **Primary test = held-out seasons in held-out super-blocks.** A secondary "temporal-only" score (held-out seasons, training super-blocks) is reported to show the spatial-generalisation gap.
- Test data is touched once per final model. Model selection, early stopping and hyperparameters use **val only**.

## Leakage rules
- Inputs end at the forecast issue time. No future weather, and no future NDVI or NDTI.
- Phenology features use only the current season up to issue time.
- Crop-type clusters are fit on training seasons only.
- Normalisation statistics are computed from the training split only.

## Models (same data, same splits)
- **Baselines:** persistence (last observation); climatology (per-pixel training quantiles for that dekad or season); ridge; LightGBM (per pixel).
- **Spatiotemporal:** ConvLSTM, PredRNN-v2, SimVP, TAU (OpenSTL reference implementations); official Earthformer (amazon-science); Prithvi-EO-2.0-300M-TL fine-tuned.
- All neural models: quantile outputs (P10/P50/P90), pinball loss, AdamW, cosine schedule, early stopping on val pinball, **3 seeds**. The hyperparameter search budget is equal per model family (documented).

## Metrics
- Median forecast: RMSE, MAE, MASE (scaled by in-sample naive).
- Probabilistic: mean pinball loss, 80% interval coverage, shortfall rate below P10 (target ≈ 10%).
- Catchment level (25 km): correlation and error of forecast vs observed totals.
- Task B is a rare-event target (only ~1–10% of cropland stays irrigated in May; e.g. block 146 near Kuhi: 0.2–1.9% in most years, ~10% in the rain-green 2023/2024). So pinball/RMSE alone would reward predicting "nothing irrigated". We additionally report **precision–recall AUC and Brier score** for "cell irrigated" (truth ≥ 0.5) using the P50 forecast as the score, plus the patch-total correlation.

## Statistical tests
- 95% bootstrap CIs over test patches (1,000 resamples), clustered by super-block.
- Each model vs persistence and vs climatology: paired Wilcoxon signed-rank test on per-block mean pinball loss, plus win rate across blocks.
- Weather ablation: one variable at a time (rain, soil moisture, temperature), then all together; report the change in pinball loss with CI.

## Error analysis
- By region and super-block, by crop-type cluster, by clear-observation rate, by cropland fraction, by lead time.

## Reporting rules
- NDTI is reported as a residue-cover index, not biomass. Tonnage estimates come only from the separate statistical layer (district crop statistics × residue-to-product ratios), clearly labelled with sources and uncertainty.
- Negative results (e.g. a model not beating persistence, weather not helping) are reported as found.
- Every number in the report comes from files in `results_v2/` produced by code in `pipelines/`.

## Residue target definition
NDTI is scored only on clear cropland pixels that are **not green** in the target dekad (NDVI < 0.3). Green canopy water absorbs in SWIR and raises NDTI, so unmasked NDTI tracks crop greenness rather than residue (visible in the Kuhi block, 2024: NDTI high on 1 Oct (green crops), lower on 11 Nov (post-harvest), high again on 21 Dec (rabi green-up)). This is the standard green-vegetation mask in residue remote sensing (Zheng et al., 2014).

## Deviation log
- 2026-10-14, before training: added PR-AUC / Brier for Task B (rare-event target), after a data sanity check showed the irrigated share is ~1% in most years. Test years are now described by observed rainfall (e.g. 2018 Jun–Sep ≈ 1,100 mm, near normal; its deficit was Sep–Oct) rather than labelled "drought years".
- 2026-10-15, before training: added a **transfer test** on 294 rest-of-Maharashtra blocks (ids ≥ 1000, `grid.py`): western sugarcane belt, Marathwada, Khandesh, Konkan. Never trained on or used for selection. Same test seasons/years; every model and baseline is scored there after training. Question: do models trained on Vidarbha generalise to other farming systems?
- 2026-10-14, before training: residue target restricted to non-green cropland (NDVI < 0.3) after the Kuhi block figure showed raw NDTI following crop greenness (see "Residue target definition").
