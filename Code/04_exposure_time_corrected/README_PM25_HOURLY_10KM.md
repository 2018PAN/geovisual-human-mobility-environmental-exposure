# Formal PM2.5 matching and hourly 10 km exposure

## Scope

`01_match_pm25_and_aggregate_hourly_10km.py` reads only formal calibrated
population partitions:

`Output/Population/calibrated_daily_parquet`

It does not read or recreate province-assigned, time-corrected, or raw daily
population parquet. PM2.5 is matched at each original App coordinate before
projection and spatial aggregation. The normal output is only the hourly
China-LCC 10 km grid; point exposure is disabled unless
`--save-point-exposure` is supplied.

## Legacy code used

- `match_pm25_to_population_utc_aligned.py` (controlled predecessor workspace)
  supplied the exact CHAP filename convention, NetCDF variable and coordinate
  discovery, nearest-coordinate matching, and UTC source-date metadata.
- `build_hourly_grid_utc_aligned_direct.py` (controlled predecessor workspace)
  supplied the vectorized nearest-index implementation, including ascending
  and descending coordinate arrays, range checks, exact-tie behavior, and the
  direct path that avoids a mandatory point parquet.
- `aggregate_hourly_exposure_grid_time_corrected.py` (controlled predecessor workspace)
  supplied the rule that non-finite PM2.5/exposure is excluded rather than
  filled with zero, Beijing-local output grouping, and the
  exposure/population definition of population-weighted PM2.5.
- `plot_pm25_exposure_and_population_redistribution.py` (controlled predecessor workspace)
  supplied the established China LCC string, 10,000 m grid size, `(0, 0)`
  origin, and floor-based grid assignment.
- `../05_hourly_grid/01_aggregate_population_hourly_10km.py`
  supplied the already-tested five-minute forward-duration weighting method.

The old Chunyun exposure builders themselves use a 0.1-degree output grid.
They do not define an LCC 10 km grid. For this reason, PM2.5 matching and
missing-value behavior come from `Code\Chunyun`, while the exact requested LCC
10 km definition comes from the existing old-project redistribution script
listed above. No new projection, origin, or grid rounding rule was invented.

## Changes required by formal calibrated population

- The only formal population weight is `estimated_population`.
- Original platform activity remains `app_count`.
- At point level:
  `calibrated_exposure = estimated_population * pm25`.
- The optional diagnostic exposure is:
  `app_exposure = app_count * pm25`.
- `utc_time`, `local_time`, `utc_date`, and hour fields are read and validated
  as already audited. The script never applies another UTC offset.
- CHAP files are selected from `utc_date`, never from `local_date`.
- Output partitions and hour groups use Beijing `local_date` and `local_time`.

## Snapshot and hourly calculation

Within every real five-minute timestamp and 10 km grid:

```text
snapshot_population = sum(estimated_population with finite PM2.5)
snapshot_app_count = sum(app_count with finite PM2.5)
snapshot_exposure = sum(estimated_population * pm25)
snapshot_app_exposure = sum(app_count * pm25)
```

The finite-PM2.5 restriction is the old Chunyun missing-value rule. Missing
records and their excluded `estimated_population` are reported by local date
and hour.

For a regular `00, 05, ..., 55` hour, twelve equal five-minute weights make
the result exactly the arithmetic mean. Otherwise, each observed snapshot
represents time until the next observation in that hour; the last represents
time until the hour boundary. A single duration is capped at 10 minutes.
Time before the first observation and long uncovered gaps remain missing.
There is no zero fill and no interpolation.

```text
population_weighted_pm25 = hourly_exposure / hourly_population
```

## Atomic and restart behavior

Each local date is written to a hidden temporary file and atomically renamed
only after success. A stale temporary file for the current date is deleted on
restart. Timestamp/grid chunk totals are temporarily spooled into at most 24
hour-specific parquet files. This makes the result independent of physical
row order in the calibrated parquet while bounding memory to one hour during
final aggregation. The hourly temporary directory is removed after a
successful atomic commit and is cleared before retrying an interrupted date.
Existing completed outputs are skipped unless `--overwrite` is used.

## Commands

Single validation date:

```bat
python 01_match_pm25_and_aggregate_hourly_10km.py ^
  --start-date 2018-02-02 ^
  --end-date 2018-02-02
```

Full period, to be started manually only after diagnostics are accepted:

```bat
python 01_match_pm25_and_aggregate_hourly_10km.py ^
  --start-date 2018-02-01 ^
  --end-date 2018-03-12
```

Useful options:

```text
--overwrite
--save-point-exposure
--chunk-size 1000000
--date-basis local
```

Outputs:

- `Output\Exposure\hourly_grid\hourly_exposure_grid_YYYY-MM-DD.parquet`
- `Output\Exposure\diagnostics\pm25_matching_summary_YYYY-MM-DD.csv`
- `Output\Exposure\diagnostics\pm25_missing_by_hour_YYYY-MM-DD.csv`
- `Output\Exposure\diagnostics\hourly_snapshot_coverage_YYYY-MM-DD.csv`

The optional point output is:

- `Output\Exposure\point_exposure\point_exposure_YYYY-MM-DD.parquet`
