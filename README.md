# Forecasting a crop-residue index (NDTI) across Maharashtra

DS 340W (Penn State), Group 9: Harsh Rathi and Tianhao Wang.

Code, result files and the final report for forecasting the Normalized Difference Tillage Index (NDTI), a
Sentinel-2 residue-cover proxy, one to three dekads ahead on post-harvest cropland across Maharashtra. The
report is `report/DS340W_Group9_Final_Report.pdf`.

## Layout

| Path | Contents |
|---|---|
| `pipelines/mh_grid.py` | 0.25 degree statewide grid (479 cells), super-block split (seed 340) |
| `pipelines/ndti_plan.py` | block selection (cropland share at least 30%, per division) -> `results_v2/ndti_local/plan.json` |
| `pipelines/ndti_cube_v2.py` | dekadal Sentinel-2 cubes (Earth Search L2A, 12.8 km window per block, 20 m) |
| `pipelines/fetch_weather_v2.py`, `fetch_chirps.py` | ERA5-family daily weather (Open-Meteo) and CHIRPS (ClimateSERV) per block |
| `pipelines/final_eval.py` | baselines (persistence, non-green persistence, 6-dekad mean, seasonal persistence, climatology), ridge, quantile GBM and weather variants -> `results_v2/final/rows.csv`, `fit.json` |
| `pipelines/final_neural.py` | ConvLSTM and the official Earthformer, seeds 340-342 -> `neural_rows.csv`, `neural_runs.json` |
| `pipelines/final_analyze.py` | pooled metrics, block-level Wilcoxon tests, super-block bootstrap -> `summary.json`, `final_numbers.tex`, results figure |
| `pipelines/final_extra.py` | additional test-block checks, per-lead tests, calibration by lead, green-last terciles, training details -> `final_numbers_extra.tex`, `extra.json` |
| `report/` | LaTeX source and PDF; `report/scripts/` regenerates the report-only macros (`final_numbers_report.tex`), the pixel-level green-last split (`green_gain.py`) and both figures |
| `results_v2/final/` | per-unit result rows for every model and seed, run metadata, summaries and macro files |
| `results_v2/final/archive_earthformer_lr1e-3/` | archived first Earthformer configuration (diverged; excluded from all results) |
| `logs/` | evaluation, training and analysis logs of the final runs |
| `EXPERIMENT_PROTOCOL.md` | the protocol fixed before training |

Every number in the report is a LaTeX macro written by `final_analyze.py`, `final_extra.py` or
`report/scripts/report_macros.py` from the files in `results_v2/final/`.

## Reproducing

Python 3.13; packages in `requirements.txt`. Earthformer is used from the official repository
(amazon-science/earth-forecasting-transformer, commit 7732b03), cloned to `ext/earthformer`.

```bash
cd pipelines
python mh_grid.py                       # grid and split
python ndti_plan.py                     # block selection, writes ../local_data/harvest20/tasks.txt
N=5 bash ../tools/run_ndti_cubes_v2.sh ../local_data/harvest20/tasks.txt   # cubes (large download)
python fetch_weather_v2.py && python fetch_chirps.py
python final_eval.py
python final_neural.py --seeds 340 341 342 --epochs 12 --resume --skip-done
python final_analyze.py && python final_extra.py
cd ../report/scripts && python report_macros.py && python fig_results.py && python fig_study_area.py
cd .. && tectonic DS340W_Group9_Final_Report.tex
```

`tasks.txt` holds one "block season" pair per line. The `SEASONS` table in `ndti_plan.py` omits 2023 for test
blocks; add those pairs, because seasonal persistence reads each test block's previous season.

The Sentinel-2 cubes, tile caches and prediction files (about 45 GB) are not included; they are rebuilt from
the open archives by the commands above. The analysis steps (`final_analyze.py`, `final_extra.py`,
`report_macros.py`, the figures and the report) run from the shipped result files alone. `green_gain.py` needs
the local test caches and `models.pkl` (see below).

## Trained models

The trained models are attached to the
[`weights-v1` release](https://github.com/xyzmr114/ds340w/releases/tag/weights-v1) rather than committed:

- `convlstm_s34{0,1,2}_best.pt` and `earthformer_s34{0,1,2}_best.pt`: the checkpoints tested in the report
  (lowest validation pinball loss per run). Put them in `results_v2/final/neural_ckpt/`.
- `models.pkl`: the quantile GBMs, weather variants, ridge model and regional climatology from `final_eval.py`
  (scikit-learn 1.9.1). Put it in `results_v2/final/`.
- `SHA256SUMS.txt`: checksums for all of the above.

The release notes describe the checkpoint contents and show how to load them.

## Use of AI tools

Generative AI tools were used for copy editing, LaTeX formatting, and routine code assistance. The authors
designed the study, reviewed all outputs, and are responsible for the content.
