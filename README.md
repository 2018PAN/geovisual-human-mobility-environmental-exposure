# Geovisual Exploration of Human Mobility and Environmental Exposure

This repository contains the non-confidential code, configuration, data-source documentation, and selected aggregate results for the master's thesis *Geovisual Exploration of Human Mobility and Environmental Exposure From a Geospatial Analysis Perspective*.

The study examines population redistribution and ambient PM2.5 exposure during the 2018 Chinese Spring Festival (1 February–12 March 2018, Beijing time) on a fixed 10 km grid. The analytical workflow covers temporal correction, spatial assignment, population calibration, exposure calculation, period aggregation, spatial autocorrelation, exact exposure decomposition, Random Forest/XGBoost modelling, spatial cross-validation, SHAP analysis, and publication graphics.

## Repository structure

- `Code/`: current end-to-end processing and analysis scripts.
- `Code/public_analysis/`: the compact public scripts from the earlier repository version.
- `Code/data_acquisition/`: Google Earth Engine scripts used to prepare the GHSL and night-time-light covariates.
- `Config/`: study periods, field definitions, analytical settings, and province-name mapping.
- `Data/`: data-source inventory, disclosure policy, and selected aggregate publication results.
- `requirements.txt`: Python package requirements.

## Data availability and exclusions

The commercial mobile-application records and all direct or grid-level population data are excluded. This includes raw activity records, official calibration totals, calibrated population surfaces, hourly/daily/period population grids, model inputs containing population features, prediction tables, SHAP value tables, and point- or grid-level exposure records.

Third-party source rasters and NetCDF files are also not redistributed here. Several exceed the repository's conservative 25 MiB file policy, and all remain subject to their providers' access and reuse terms. `Data/source_inventory.csv` records every non-population source file found in the controlled workspace and points users to the source documentation in `Data/README.md`.

Only small, publication-level aggregate outputs are included under `Data/derived_results/`. They contain model-performance summaries, feature-importance summaries, spatial-statistical summaries, and national/provincial decomposition summaries. They do not contain record-level or grid-level population values.

Run `python Code/repository_guard.py` before committing. It rejects confidential data locations, population-bearing table schemas, large files, and raw binary data formats.

## Reproducibility scope

The code documents the complete analytical procedure, but an independent end-to-end rerun requires separately authorised access to the commercial population source and to the third-party datasets listed in `Data/README.md`. The public aggregate results can be inspected without those restricted inputs.

The main current workflow is ordered by the numbered directories in `Code/`. Shared path and field definitions are in `Code/00_common/`, while the formal analysis settings are in `Config/downstream_analysis.json` and `Config/chunyun_periods.yaml`.

## Environment

Python 3.11 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python Code/repository_guard.py
```

Administrative boundaries and licensed source datasets must be placed in the local paths expected by the scripts or supplied through their command-line options. They are intentionally absent from this repository.
