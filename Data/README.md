# Data documentation

This directory is a public disclosure package, not a copy of the controlled research workspace.

## Included files

- `source_inventory.csv`: file-level inventory of the non-population source data present in the controlled workspace, with size and GitHub inclusion status.
- `derived_results_manifest.csv`: SHA-256 manifest for every included aggregate result.
- `derived_results/`: small publication-level summaries used to verify reported model, spatial-statistical, and decomposition results.
- `excluded_groups.csv`: group-level record of confidential, licensed, large, or grid-level material deliberately excluded from GitHub.

The included result tables do not contain record-level or grid-level population values. Feature names may refer to population-related variables because those names are necessary to interpret the published models.

## Principal source datasets

| Data | Analytical role | Source |
|---|---|---|
| Commercial mobile-application activity | Dynamic population reconstruction | Restricted; provider and raw records are not disclosed or redistributed |
| Provincial population totals | Calibration targets | China Statistical Yearbook 2019 and official 2018 regional statistics; values are not included here |
| Administrative boundaries | Screening and province assignment | National Fundamental Geographic Information System, National Geomatics Center of China |
| ChinaHighPM2.5 / CHAP V4 | Daily 1 km ambient PM2.5 | Wei and Li (2024), [dataset DOI](https://doi.org/10.5281/zenodo.3539349) |
| ERA5-Land | Temperature, precipitation, and wind | Munoz-Sabater (2019), [dataset DOI](https://doi.org/10.24381/cds.e2161bac) |
| Gridded GDP | 2018 economic covariate | Chen et al. (2021), [Figshare dataset](https://doi.org/10.6084/m9.figshare.17004523.v1) |
| GRIP v4 | Road-density covariate | Meijer et al. (2018), [GRIP download page](https://www.globio.info/download-grip-dataset) |
| GHSL GHS-BUILT-S | 2015 built-up ratio | Pesaresi and Politis (2023), [dataset DOI](https://doi.org/10.2905/9F06F36F-4B11-47EC-ABB0-4F8B7B1D72EA) |
| NASA Black Marble VNP46A2 | February 2018 mean night-time lights | NASA LP DAAC (2021), [dataset DOI](https://doi.org/10.5067/VIIRS/VNP46A2.002) |

Provider licences and access conditions remain controlling. Users should obtain source data directly from the providers and verify the applicable terms.

## Repository size policy

No single repository file may exceed 25 MiB. Raw raster, NetCDF, Parquet, model-binary, and archive formats are excluded from `Data/` even when smaller. Large or licensed sources are documented instead of duplicated.
