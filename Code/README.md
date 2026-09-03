# Analysis code

The numbered directories follow the thesis workflow:

1. `01_population_time_correction`: validate source partitions, convert timestamps to Beijing time, and audit temporal coverage.
2. `02_population_province_assignment`: validate boundaries and assign observations to provincial units.
3. `03_population_calibration`: calibrate the platform signal to official provincial totals and validate the result.
4. `04_exposure_time_corrected`: match daily CHAP PM2.5 and build hourly 10 km exposure grids.
5. `05_hourly_grid` and `05_aggregation`: build hourly, daily, and festival-period analysis tables.
6. `06_spatial_analysis`: Global Moran's I and Local Moran/LISA analysis.
7. `07_decomposition`: exact population, pollution, and interaction decomposition.
8. `08_modeling`: model-input construction, Random Forest/XGBoost fitting, spatial cross-validation, and SHAP values.
9. `09_visualization`: time series, maps, summary figures, and figure validation.

Shared paths, fields, projection definitions, validation helpers, spatial functions, and plotting conventions are in `00_common`. The PowerShell runners execute the downstream and component-analysis stages only when invoked explicitly.

`public_analysis` preserves the compact scripts from the first public repository version. They accept user-supplied CSV or Parquet inputs and remain useful for inspecting the core decomposition, spatial-statistical, modelling, and SHAP procedures.

The source code contains field names and processing logic for the restricted population data, but no population observations or calibration values are included. Some stages require authorised local inputs and therefore cannot run from a clean clone alone.
