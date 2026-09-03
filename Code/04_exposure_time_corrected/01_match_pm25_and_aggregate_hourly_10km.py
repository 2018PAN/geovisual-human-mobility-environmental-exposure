from __future__ import annotations

"""Match calibrated population snapshots to CHAP PM2.5 and build hourly grids.

Legacy equivalence and deliberate changes
-----------------------------------------
* ``Code/Chunyun/match_pm25_to_population_utc_aligned.py``:
  exact CHAP filename discovery, NetCDF variable/coordinate discovery, and
  nearest-coordinate point matching.
* ``Code/Chunyun/build_hourly_grid_utc_aligned_direct.py``:
  vectorized ascending/descending nearest-index logic, raster-range rejection,
  missing-PM2.5 exclusion, and direct aggregation without a mandatory point
  parquet.
* ``Code/Chunyun/aggregate_hourly_exposure_grid_time_corrected.py``:
  Beijing-local output grouping and exposure/population PM2.5 ratio.
* ``Code/plot_pm25_exposure_and_population_redistribution.py``:
  the established China LCC string, 10,000 m grid size, zero-metre grid
  origin, and ``floor(projected_coordinate / 10000) * 10000`` assignment.
* ``New/Code/05_hourly_grid/01_aggregate_population_hourly_10km.py``:
  the already-tested five-minute snapshot weighting rule.

The old Chunyun scripts treated ``count`` as the population weight and summed
records directly to an hour.  This script instead reads only formal calibrated
files, uses ``estimated_population`` and ``calibrated_exposure``, first sums
points at each real five-minute timestamp/grid, and then averages snapshots.
It trusts the audited ``utc_time``/``local_time`` fields and never applies
another timezone conversion.
"""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from pyproj import Transformer


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    CALIBRATED_DIR,
    CHINA_LCC,
    GRID_SIZE_M,
    NEW_ROOT,
    add_date_arguments,
    iter_dates,
    validate_date_range,
)


PM25_DIR = (
    NEW_ROOT.parent / "RawData" / "Environment" / "PM2.5" / "2018"
)
OUTPUT_DIR = NEW_ROOT / "Output" / "Exposure" / "hourly_grid"
POINT_OUTPUT_DIR = NEW_ROOT / "Output" / "Exposure" / "point_exposure"
DIAGNOSTICS_DIR = NEW_ROOT / "Output" / "Exposure" / "diagnostics"

POPULATION_COLUMN = "estimated_population"
APP_COUNT_COLUMN = "app_count"
EXPOSURE_COLUMN = "calibrated_exposure"

EXPECTED_SAMPLE_COUNT = 12
EXPECTED_INTERVAL_MINUTES = 5.0
MAX_SNAPSHOT_DURATION_MINUTES = 10.0

SNAPSHOT_VALUE_COLUMNS = [
    "snapshot_population",
    "snapshot_app_count",
    "snapshot_exposure",
    "snapshot_app_exposure",
]
HOURLY_VALUE_MAP = {
    "snapshot_population": "hourly_population",
    "snapshot_app_count": "hourly_app_count",
    "snapshot_exposure": "hourly_exposure",
    "snapshot_app_exposure": "hourly_app_exposure",
}


@dataclass(frozen=True)
class PM25Raster:
    values: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    variable_name: str
    source_path: Path
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


def require_columns(
    available: Iterable[str], required: Iterable[str], path: Path
) -> None:
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def find_pm25_file(utc_date: str, pm25_dir: Path) -> Path:
    """Use the exact legacy CHAP UTC-date filename/search rule."""
    yyyymmdd = utc_date.replace("-", "")
    filename = f"CHAP_PM2.5_D1K_{yyyymmdd}_V4.nc"
    direct_path = pm25_dir / filename
    if direct_path.exists():
        return direct_path.resolve()

    matches = sorted(pm25_dir.rglob(filename)) if pm25_dir.exists() else []
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(
        f"PM2.5 UTC-date file not found for {utc_date}. "
        f"Expected {filename} under {pm25_dir}"
    )


def find_coord_name(
    dataset: xr.Dataset, candidates: Iterable[str]
) -> str:
    all_names = list(dataset.coords) + list(dataset.dims)
    lookup = {name.lower(): name for name in all_names}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        "Could not identify a PM2.5 coordinate. "
        f"Candidates={list(candidates)}, coords={list(dataset.coords)}, "
        f"dims={list(dataset.dims)}"
    )


def open_pm25_raster(
    nc_path: Path, variable_name: str | None = None
) -> PM25Raster:
    """Load one daily raster exactly in (latitude, longitude) order."""
    dataset = xr.open_dataset(nc_path)
    try:
        selected_variable = variable_name
        if selected_variable is None:
            if not dataset.data_vars:
                raise ValueError(f"No data variables found in {nc_path}")
            selected_variable = list(dataset.data_vars)[0]
        if selected_variable not in dataset.data_vars:
            raise ValueError(
                f"PM2.5 variable {selected_variable!r} not found in "
                f"{nc_path}; variables={list(dataset.data_vars)}"
            )

        lat_name = find_coord_name(
            dataset, ["lat", "latitude", "y"]
        )
        lon_name = find_coord_name(
            dataset, ["lon", "longitude", "x"]
        )
        data_array = dataset[selected_variable].squeeze(drop=True)
        for dimension in list(data_array.dims):
            if dimension not in (lat_name, lon_name):
                data_array = data_array.isel({dimension: 0})
        if data_array.dims != (lat_name, lon_name):
            data_array = data_array.transpose(lat_name, lon_name)

        values = np.asarray(data_array.to_numpy(), dtype="float32").copy()
        latitudes = np.asarray(
            dataset[lat_name].to_numpy(), dtype="float64"
        ).copy()
        longitudes = np.asarray(
            dataset[lon_name].to_numpy(), dtype="float64"
        ).copy()
    finally:
        dataset.close()

    if values.shape != (len(latitudes), len(longitudes)):
        raise ValueError(
            f"Unexpected PM2.5 raster shape in {nc_path}: "
            f"values={values.shape}, lat={len(latitudes)}, "
            f"lon={len(longitudes)}"
        )
    return PM25Raster(
        values=values,
        latitudes=latitudes,
        longitudes=longitudes,
        variable_name=selected_variable,
        source_path=nc_path.resolve(),
        latitude_min=float(np.nanmin(latitudes)),
        latitude_max=float(np.nanmax(latitudes)),
        longitude_min=float(np.nanmin(longitudes)),
        longitude_max=float(np.nanmax(longitudes)),
    )


class PM25Matcher:
    """Cached legacy-equivalent nearest-neighbour CHAP matcher."""

    def __init__(
        self, pm25_dir: Path, variable_name: str | None = None
    ) -> None:
        self.pm25_dir = pm25_dir
        self.variable_name = variable_name
        self.cache: dict[str, PM25Raster] = {}

    def get(self, utc_date: str) -> PM25Raster:
        if utc_date not in self.cache:
            source_path = find_pm25_file(utc_date, self.pm25_dir)
            raster = open_pm25_raster(
                source_path, variable_name=self.variable_name
            )
            self.cache[utc_date] = raster
            direction = (
                "ascending"
                if raster.latitudes[-1] > raster.latitudes[0]
                else "descending"
            )
            print(
                f"    opened UTC PM2.5 {utc_date}: "
                f"{source_path.name}, variable={raster.variable_name}, "
                f"shape={raster.values.shape}, latitude={direction}"
            )
        return self.cache[utc_date]

    @staticmethod
    def nearest_indices(
        query: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        """Return legacy nearest indices; exact ties select the left value."""
        if len(coordinates) < 2:
            return np.zeros(len(query), dtype=np.int64)

        ascending = coordinates[1] > coordinates[0]
        ordered = coordinates if ascending else coordinates[::-1]
        positions = np.searchsorted(ordered, query)
        positions = np.clip(positions, 1, len(ordered) - 1)
        left = ordered[positions - 1]
        right = ordered[positions]
        choose_right = np.abs(right - query) < np.abs(query - left)
        ordered_indices = (
            positions - 1 + choose_right.astype(np.int64)
        ).astype(np.int64)
        if ascending:
            return ordered_indices
        return (len(coordinates) - 1 - ordered_indices).astype(np.int64)

    def lookup(
        self, latitudes: np.ndarray, longitudes: np.ndarray, utc_date: str
    ) -> tuple[np.ndarray, PM25Raster]:
        raster = self.get(utc_date)
        pm25 = np.full(len(latitudes), np.nan, dtype="float32")
        in_raster = (
            np.isfinite(latitudes)
            & np.isfinite(longitudes)
            & (latitudes >= raster.latitude_min)
            & (latitudes <= raster.latitude_max)
            & (longitudes >= raster.longitude_min)
            & (longitudes <= raster.longitude_max)
        )
        if in_raster.any():
            latitude_indices = self.nearest_indices(
                latitudes[in_raster], raster.latitudes
            )
            longitude_indices = self.nearest_indices(
                longitudes[in_raster], raster.longitudes
            )
            pm25[in_raster] = raster.values[
                latitude_indices, longitude_indices
            ]
        return pm25, raster


def build_hourly_time_weights(
    observation_times: Iterable[pd.Timestamp],
    *,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    expected_interval_minutes: float = EXPECTED_INTERVAL_MINUTES,
    max_snapshot_duration_minutes: float = (
        MAX_SNAPSHOT_DURATION_MINUTES
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build forward-duration weights without zero fill or interpolation.

    Each snapshot represents time from its observation to the next observed
    snapshot in the same hour.  The last snapshot represents time to the hour
    boundary.  Every duration is capped at 10 minutes by default, so a longer
    gap remains uncovered.  Time before the first observation also remains
    uncovered.  A complete 00,05,...,55 sequence receives twelve equal
    five-minute weights and is therefore exactly an arithmetic mean.
    """
    if expected_sample_count <= 0:
        raise ValueError("expected_sample_count must be positive")
    if expected_interval_minutes <= 0:
        raise ValueError("expected_interval_minutes must be positive")
    if max_snapshot_duration_minutes <= 0:
        raise ValueError(
            "max_snapshot_duration_minutes must be positive"
        )

    times = (
        pd.DatetimeIndex(pd.to_datetime(list(observation_times)))
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if len(times) == 0:
        return pd.DataFrame(), pd.DataFrame()

    hour_values = times.floor("h")
    if hour_values.nunique() != 1:
        raise ValueError(
            "build_hourly_time_weights expects observations from one hour"
        )
    hour_start = hour_values[0]
    hour_end = hour_start + pd.Timedelta(hours=1)
    expected_times = pd.date_range(
        hour_start,
        periods=expected_sample_count,
        freq=pd.Timedelta(minutes=expected_interval_minutes),
    )
    regular_complete = bool(
        len(times) == expected_sample_count and times.equals(expected_times)
    )

    maximum_seconds = max_snapshot_duration_minutes * 60.0
    durations_seconds: list[float] = []
    for position, current in enumerate(times):
        raw_end = (
            min(times[position + 1], hour_end)
            if position + 1 < len(times)
            else hour_end
        )
        raw_seconds = max((raw_end - current).total_seconds(), 0.0)
        durations_seconds.append(min(raw_seconds, maximum_seconds))

    boundaries = (
        pd.DatetimeIndex([hour_start, *times.tolist(), hour_end])
        .drop_duplicates()
        .sort_values()
    )
    max_gap_minutes = (
        float(np.diff(boundaries.asi8).max() / 60_000_000_000)
        if len(boundaries) > 1
        else np.nan
    )
    covered_minutes = float(sum(durations_seconds) / 60.0)
    sample_count = int(len(times))
    weights = pd.DataFrame(
        {
            "local_time": times,
            "local_datetime": hour_start,
            "weight_seconds": durations_seconds,
        }
    )
    quality = pd.DataFrame(
        [
            {
                "local_datetime": hour_start,
                "sample_count": sample_count,
                "expected_sample_count": expected_sample_count,
                "coverage_ratio": (
                    sample_count / expected_sample_count
                ),
                "covered_minutes": covered_minutes,
                "duration_coverage_ratio": covered_minutes / 60.0,
                "first_observation_time": times.min(),
                "last_observation_time": times.max(),
                "is_complete_hour": (
                    sample_count == expected_sample_count
                ),
                "is_regular_expected_interval": regular_complete,
                "max_observed_gap_minutes": max_gap_minutes,
                "max_snapshot_duration_minutes": (
                    max_snapshot_duration_minutes
                ),
                "aggregation_method": (
                    "arithmetic_mean_regular_complete"
                    if regular_complete
                    else "time_weighted_forward_duration_capped"
                ),
            }
        ]
    )
    return weights, quality


def aggregate_snapshot_grids_to_hourly(
    snapshot_grid: pd.DataFrame,
    weights: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """Average snapshot grid totals with the established global time weights.

    An absent grid at an observed national snapshot contributes zero at that
    snapshot.  A nationally absent snapshot is never created or zero-filled.
    PM2.5-missing point records are excluded before snapshot aggregation, as
    in the old Chunyun exposure code, and their population is reported
    separately in diagnostics.
    """
    if snapshot_grid.empty or weights.empty:
        return pd.DataFrame()
    work = snapshot_grid.merge(
        weights[["local_time", "local_datetime", "weight_seconds"]],
        on="local_time",
        how="inner",
        validate="many_to_one",
    )
    weighted_columns: list[str] = []
    missing_columns: list[str] = []
    for source, output in HOURLY_VALUE_MAP.items():
        weighted = f"weighted__{output}"
        missing = f"missing_weight__{output}"
        work[weighted] = work[source] * work["weight_seconds"]
        work[missing] = np.where(
            work[source].isna(), work["weight_seconds"], 0.0
        )
        weighted_columns.append(weighted)
        missing_columns.append(missing)

    hourly = work.groupby(
        ["local_datetime", "grid_x", "grid_y"],
        as_index=False,
        observed=True,
        sort=True,
    )[[*weighted_columns, *missing_columns]].sum(min_count=1)
    hourly = hourly.merge(
        quality,
        on="local_datetime",
        how="left",
        validate="many_to_one",
    )
    global_denominator = hourly["covered_minutes"] * 60.0
    for source, output in HOURLY_VALUE_MAP.items():
        denominator = (
            global_denominator - hourly.pop(f"missing_weight__{output}")
        )
        hourly[output] = (
            hourly.pop(f"weighted__{output}")
            / denominator.replace(0.0, np.nan)
        )
    return hourly


def project_to_grid(
    longitude: np.ndarray,
    latitude: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the established zero-origin China-LCC 10 km grid."""
    projected_x, projected_y = transformer.transform(longitude, latitude)
    grid_x = (
        np.floor(np.asarray(projected_x) / GRID_SIZE_M) * GRID_SIZE_M
    ).astype("int64")
    grid_y = (
        np.floor(np.asarray(projected_y) / GRID_SIZE_M) * GRID_SIZE_M
    ).astype("int64")
    return grid_x, grid_y


class SpoolingHourlyAggregator:
    """Aggregate arbitrary input order through bounded hourly temp partitions.

    Formal calibrated partitions are normally chronological, but audited or
    recovered records can be appended after the main sorted block.  Requiring
    physical parquet row order would therefore reject valid timestamps.  Each
    matched chunk is first reduced to timestamp/grid totals and written to one
    temporary parquet per Beijing hour.  At end-of-day, one hour at a time is
    read, duplicate chunk contributions are merged, timestamps are sorted, and
    the established hourly weighting is applied.  Memory is bounded by a
    single hour and input row order has no effect on the result.
    """

    def __init__(
        self,
        temporary_directory: Path,
        *,
        expected_sample_count: int,
        expected_interval_minutes: float,
        max_snapshot_duration_minutes: float,
    ) -> None:
        self.temporary_directory = temporary_directory
        self.expected_sample_count = expected_sample_count
        self.expected_interval_minutes = expected_interval_minutes
        self.max_snapshot_duration_minutes = (
            max_snapshot_duration_minutes
        )
        self.temporary_directory.mkdir(parents=True, exist_ok=False)
        self.writers: dict[pd.Timestamp, pq.ParquetWriter] = {}
        self.paths: dict[pd.Timestamp, Path] = {}
        self.observation_times: dict[
            pd.Timestamp, set[pd.Timestamp]
        ] = {}
        self.utc_dates: dict[pd.Timestamp, set[str]] = {}
        self.utc_date_by_timestamp: dict[pd.Timestamp, str] = {}

    def consume_batch(
        self,
        observations: list[tuple[pd.Timestamp, str]],
        snapshot_partial: pd.DataFrame,
    ) -> None:
        for observation_time, utc_date in observations:
            observation_time = pd.Timestamp(observation_time)
            utc_date = str(utc_date)
            previous_utc_date = self.utc_date_by_timestamp.get(
                observation_time
            )
            if (
                previous_utc_date is not None
                and previous_utc_date != utc_date
            ):
                raise ValueError(
                    "One local timestamp maps to multiple utc_date values: "
                    f"{observation_time}, {previous_utc_date}, {utc_date}"
                )
            self.utc_date_by_timestamp[observation_time] = utc_date
            hour_start = observation_time.floor("h")
            self.observation_times.setdefault(hour_start, set()).add(
                observation_time
            )
            self.utc_dates.setdefault(hour_start, set()).add(utc_date)

        if snapshot_partial.empty:
            return
        hour_values = snapshot_partial["local_time"].dt.floor("h")
        for hour_start in sorted(hour_values.unique()):
            hour_start = pd.Timestamp(hour_start)
            part = snapshot_partial.loc[
                hour_values == hour_start,
                [
                    "local_time",
                    "grid_x",
                    "grid_y",
                    *SNAPSHOT_VALUE_COLUMNS,
                ],
            ]
            table = pa.Table.from_pandas(part, preserve_index=False)
            writer = self.writers.get(hour_start)
            if writer is None:
                path = (
                    self.temporary_directory
                    / f"snapshot_grid_hour_{hour_start:%H}.parquet"
                )
                writer = pq.ParquetWriter(
                    path, table.schema, compression="zstd"
                )
                self.writers[hour_start] = writer
                self.paths[hour_start] = path
            writer.write_table(table)

    def close_writers(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers = {}

    def finish(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.close_writers()
        hourly_results: list[pd.DataFrame] = []
        quality_results: list[pd.DataFrame] = []
        for hour_start in sorted(self.observation_times):
            hour_times = sorted(self.observation_times[hour_start])
            weights, quality = build_hourly_time_weights(
                hour_times,
                expected_sample_count=self.expected_sample_count,
                expected_interval_minutes=self.expected_interval_minutes,
                max_snapshot_duration_minutes=(
                    self.max_snapshot_duration_minutes
                ),
            )
            utc_dates_used = ",".join(
                sorted(self.utc_dates.get(hour_start, set()))
            )
            quality["pm25_utc_dates_used"] = utc_dates_used
            quality_results.append(quality)

            path = self.paths.get(hour_start)
            if path is None:
                continue
            partials = pd.read_parquet(path)
            snapshots = partials.groupby(
                ["local_time", "grid_x", "grid_y"],
                as_index=False,
                observed=True,
                sort=True,
            )[SNAPSHOT_VALUE_COLUMNS].sum(min_count=1)
            hourly_part = aggregate_snapshot_grids_to_hourly(
                snapshots,
                weights,
                quality.drop(columns=["pm25_utc_dates_used"]),
            )
            hourly_part["pm25_utc_dates_used"] = utc_dates_used
            hourly_results.append(hourly_part)

        hourly = (
            pd.concat(hourly_results, ignore_index=True)
            if hourly_results
            else pd.DataFrame()
        )
        quality = (
            pd.concat(quality_results, ignore_index=True)
            if quality_results
            else pd.DataFrame()
        )
        return hourly, quality

    def cleanup(self) -> None:
        self.close_writers()
        if self.temporary_directory.exists():
            shutil.rmtree(self.temporary_directory)


class MissingPM25Diagnostics:
    def __init__(self, local_day: str) -> None:
        self.local_day = local_day
        self.total_records = 0
        self.matched_records = 0
        self.missing_records = 0
        self.input_population = 0.0
        self.missing_population = 0.0
        self.by_hour: dict[int, dict[str, float | int]] = {}

    def update(
        self,
        local_hours: np.ndarray,
        population: np.ndarray,
        matched: np.ndarray,
    ) -> None:
        self.total_records += len(population)
        self.matched_records += int(matched.sum())
        self.missing_records += int((~matched).sum())
        self.input_population += float(population.sum(dtype="float64"))
        self.missing_population += float(
            population[~matched].sum(dtype="float64")
        )
        for hour in np.unique(local_hours):
            hour_mask = local_hours == hour
            hour_matched = matched[hour_mask]
            hour_population = population[hour_mask]
            row = self.by_hour.setdefault(
                int(hour),
                {
                    "input_record_count": 0,
                    "matched_record_count": 0,
                    "pm25_missing_record_count": 0,
                    "input_estimated_population": 0.0,
                    "pm25_missing_estimated_population": 0.0,
                },
            )
            row["input_record_count"] += int(hour_mask.sum())
            row["matched_record_count"] += int(hour_matched.sum())
            row["pm25_missing_record_count"] += int(
                (~hour_matched).sum()
            )
            row["input_estimated_population"] += float(
                hour_population.sum(dtype="float64")
            )
            row["pm25_missing_estimated_population"] += float(
                hour_population[~hour_matched].sum(dtype="float64")
            )

    def summary_frame(
        self, matcher: PM25Matcher, output_rows: int
    ) -> pd.DataFrame:
        missing_share = (
            self.missing_population / self.input_population
            if self.input_population > 0
            else np.nan
        )
        utc_dates = sorted(matcher.cache)
        return pd.DataFrame(
            [
                {
                    "local_date": self.local_day,
                    "input_record_count": self.total_records,
                    "pm25_matched_record_count": self.matched_records,
                    "pm25_missing_record_count": self.missing_records,
                    "input_estimated_population": self.input_population,
                    "pm25_missing_estimated_population": (
                        self.missing_population
                    ),
                    "pm25_missing_population_share": missing_share,
                    "output_grid_hour_rows": output_rows,
                    "pm25_utc_dates_used": ",".join(utc_dates),
                    "pm25_source_files": ",".join(
                        matcher.cache[item].source_path.name
                        for item in utc_dates
                    ),
                }
            ]
        )

    def hourly_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for hour, values in sorted(self.by_hour.items()):
            input_population = float(
                values["input_estimated_population"]
            )
            missing_population = float(
                values["pm25_missing_estimated_population"]
            )
            rows.append(
                {
                    "local_date": self.local_day,
                    "local_hour": hour,
                    **values,
                    "pm25_missing_record_share": (
                        int(values["pm25_missing_record_count"])
                        / int(values["input_record_count"])
                    ),
                    "pm25_missing_population_share": (
                        missing_population / input_population
                        if input_population > 0
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(rows)


def validate_batch_times(
    frame: pd.DataFrame,
    local_day: date,
) -> tuple[
    pd.Series,
    pd.Series,
    np.ndarray,
    np.ndarray,
    int,
    pd.Timestamp,
    pd.Timestamp,
]:
    """Validate audited time semantics without requiring physical row order."""
    utc_time = pd.to_datetime(frame["utc_time"], errors="coerce")
    local_time = pd.to_datetime(frame["local_time"], errors="coerce")
    if utc_time.isna().any() or local_time.isna().any():
        raise ValueError("Calibrated input contains invalid audited times")
    if not ((local_time - utc_time) == pd.Timedelta(hours=8)).all():
        raise ValueError(
            "Existing local_time is not exactly utc_time + 8 hours"
        )

    local_start = pd.Timestamp(local_day)
    local_end = local_start + pd.Timedelta(days=1)
    if not (
        local_time.ge(local_start) & local_time.lt(local_end)
    ).all():
        raise ValueError(
            f"Input partition contains records outside {local_day}"
        )

    local_date_values = frame["local_date"].astype(str).to_numpy()
    utc_date_values = frame["utc_date"].astype(str).to_numpy()
    if not np.all(local_date_values == local_day.isoformat()):
        raise ValueError(
            f"local_date field does not match partition {local_day}"
        )
    derived_utc_dates = utc_time.dt.strftime("%Y-%m-%d").to_numpy()
    if not np.array_equal(utc_date_values, derived_utc_dates):
        raise ValueError("utc_date is inconsistent with audited utc_time")

    local_hours = local_time.dt.hour.to_numpy(dtype="int16")
    utc_hours = utc_time.dt.hour.to_numpy(dtype="int16")
    if not np.array_equal(
        frame["local_hour"].to_numpy(dtype="int16"), local_hours
    ):
        raise ValueError("local_hour is inconsistent with local_time")
    if not np.array_equal(
        frame["utc_hour"].to_numpy(dtype="int16"), utc_hours
    ):
        raise ValueError("utc_hour is inconsistent with utc_time")

    time_values = local_time.to_numpy(dtype="datetime64[ns]")
    descending_transitions = int(
        np.sum(time_values[1:] < time_values[:-1])
    )
    first_time = pd.Timestamp(time_values[0])
    last_time = pd.Timestamp(time_values[-1])
    return (
        utc_time,
        local_time,
        utc_date_values,
        local_hours,
        descending_transitions,
        first_time,
        last_time,
    )


def observation_rows(
    local_time: pd.Series, utc_dates: np.ndarray
) -> list[tuple[pd.Timestamp, str]]:
    if len(local_time) == 0:
        return []
    unique_pairs = pd.DataFrame(
        {
            "local_time": local_time.to_numpy(
                dtype="datetime64[ns]"
            ),
            "utc_date": utc_dates,
        }
    ).drop_duplicates()
    return list(
        unique_pairs.itertuples(index=False, name=None)
    )


def write_csv_atomic(frame: pd.DataFrame, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.with_name(f".{final_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    frame.to_csv(
        temporary_path, index=False, encoding="utf-8-sig"
    )
    os.replace(temporary_path, final_path)


def finalize_hourly_output(
    hourly: pd.DataFrame,
    local_day: date,
    inverse_transformer: Transformer,
) -> pd.DataFrame:
    if hourly.empty:
        raise ValueError(f"No hourly 10 km output produced for {local_day}")

    hourly["local_date"] = local_day.isoformat()
    hourly["local_hour"] = (
        hourly["local_datetime"].dt.hour.astype("int8")
    )
    hourly["grid_size_m"] = GRID_SIZE_M
    hourly["grid_center_x"] = hourly["grid_x"] + GRID_SIZE_M / 2
    hourly["grid_center_y"] = hourly["grid_y"] + GRID_SIZE_M / 2
    center_lon, center_lat = inverse_transformer.transform(
        hourly["grid_center_x"].to_numpy(dtype="float64"),
        hourly["grid_center_y"].to_numpy(dtype="float64"),
    )
    hourly["grid_center_lon"] = np.asarray(center_lon)
    hourly["grid_center_lat"] = np.asarray(center_lat)
    hourly["grid_id"] = (
        hourly["grid_x"].astype(str)
        + "_"
        + hourly["grid_y"].astype(str)
    )
    hourly["population_weighted_pm25"] = (
        hourly["hourly_exposure"]
        / hourly["hourly_population"].replace(0.0, np.nan)
    )

    for column in (
        "hourly_population",
        "hourly_app_count",
        "hourly_exposure",
        "hourly_app_exposure",
    ):
        finite = hourly[column].dropna()
        if (finite < 0).any():
            raise ValueError(f"Negative values found in output {column}")

    ordered_columns = [
        "local_date",
        "local_hour",
        "local_datetime",
        "grid_id",
        "grid_x",
        "grid_y",
        "grid_center_x",
        "grid_center_y",
        "grid_center_lon",
        "grid_center_lat",
        "grid_size_m",
        "hourly_population",
        "hourly_app_count",
        "hourly_exposure",
        "hourly_app_exposure",
        "population_weighted_pm25",
        "sample_count",
        "expected_sample_count",
        "coverage_ratio",
        "covered_minutes",
        "duration_coverage_ratio",
        "first_observation_time",
        "last_observation_time",
        "is_complete_hour",
        "is_regular_expected_interval",
        "max_observed_gap_minutes",
        "max_snapshot_duration_minutes",
        "aggregation_method",
        "pm25_utc_dates_used",
    ]
    return hourly[ordered_columns].sort_values(
        ["local_datetime", "grid_x", "grid_y"]
    ).reset_index(drop=True)


def process_day(local_day: date, args: argparse.Namespace) -> bool:
    day_text = local_day.isoformat()
    input_path = (
        args.input_dir / f"population_calibrated_{day_text}.parquet"
    )
    output_path = (
        args.output_dir / f"hourly_exposure_grid_{day_text}.parquet"
    )
    output_temp = output_path.with_name(f".{output_path.name}.tmp")
    point_path = (
        args.point_output_dir / f"point_exposure_{day_text}.parquet"
    )
    point_temp = point_path.with_name(f".{point_path.name}.tmp")
    snapshot_parts_dir = output_path.with_name(
        f".{output_path.name}.snapshot_parts"
    )

    for stale_temp in (output_temp, point_temp):
        if stale_temp.exists():
            print(f"  removing stale temporary file: {stale_temp}")
            stale_temp.unlink()
    if snapshot_parts_dir.exists():
        if (
            snapshot_parts_dir.parent.resolve()
            != args.output_dir.resolve()
            or not snapshot_parts_dir.name.startswith(
                ".hourly_exposure_grid_"
            )
            or not snapshot_parts_dir.name.endswith(
                ".parquet.snapshot_parts"
            )
        ):
            raise RuntimeError(
                f"Unsafe snapshot temporary directory: "
                f"{snapshot_parts_dir}"
            )
        print(
            "  removing stale snapshot temporary directory: "
            f"{snapshot_parts_dir}"
        )
        shutil.rmtree(snapshot_parts_dir)

    if output_path.exists() and not args.overwrite:
        print(f"Skipped existing local-date output: {output_path}")
        return False
    if not input_path.exists():
        raise FileNotFoundError(
            f"Formal calibrated input not found: {input_path}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    save_point_today = bool(args.save_point_exposure)
    if save_point_today:
        args.point_output_dir.mkdir(parents=True, exist_ok=True)
        if point_path.exists() and not args.overwrite:
            print(
                "  existing optional point output retained; it will not be "
                f"rewritten: {point_path}"
            )
            save_point_today = False

    parquet_file = pq.ParquetFile(input_path)
    required_columns = [
        "utc_time",
        "local_time",
        "utc_date",
        "local_date",
        "utc_hour",
        "local_hour",
        "lat",
        "lon",
        APP_COUNT_COLUMN,
        POPULATION_COLUMN,
    ]
    require_columns(
        parquet_file.schema_arrow.names, required_columns, input_path
    )
    print(f"\nBeijing local date: {day_text}")
    print(f"  calibrated input: {input_path}")
    print(f"  input rows: {parquet_file.metadata.num_rows:,}")
    print(f"  PM2.5 UTC-date root: {args.pm25_dir}")
    print(f"  hourly output: {output_path}")
    print(
        "  point exposure: "
        + (str(point_path) if save_point_today else "disabled")
    )

    matcher = PM25Matcher(
        args.pm25_dir, variable_name=args.pm25_variable
    )
    forward_transformer = Transformer.from_crs(
        "EPSG:4326", CHINA_LCC, always_xy=True
    )
    inverse_transformer = Transformer.from_crs(
        CHINA_LCC, "EPSG:4326", always_xy=True
    )
    aggregator = SpoolingHourlyAggregator(
        snapshot_parts_dir,
        expected_sample_count=args.expected_sample_count,
        expected_interval_minutes=args.expected_interval_minutes,
        max_snapshot_duration_minutes=(
            args.max_snapshot_duration_minutes
        ),
    )
    diagnostics = MissingPM25Diagnostics(day_text)
    point_writer: pq.ParquetWriter | None = None
    previous_batch_last_time: pd.Timestamp | None = None
    input_time_order_descents = 0

    try:
        for batch_number, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=args.chunk_size,
                columns=required_columns,
            ),
            start=1,
        ):
            frame = batch.to_pandas()
            (
                utc_time,
                local_time,
                utc_dates,
                local_hours,
                within_batch_descents,
                first_time,
                last_time,
            ) = validate_batch_times(frame, local_day)
            boundary_descent = int(
                previous_batch_last_time is not None
                and first_time < previous_batch_last_time
            )
            input_time_order_descents += (
                within_batch_descents + boundary_descent
            )
            if within_batch_descents or boundary_descent:
                print(
                    "    input row-order note: "
                    f"batch={batch_number}, "
                    f"within_batch_descents={within_batch_descents}, "
                    f"boundary_descent={boundary_descent}; "
                    "valid timestamps will be merged by hour/timestamp"
                )
            previous_batch_last_time = last_time

            latitude = frame["lat"].to_numpy(dtype="float64")
            longitude = frame["lon"].to_numpy(dtype="float64")
            app_count = frame[APP_COUNT_COLUMN].to_numpy(
                dtype="float64"
            )
            population = frame[POPULATION_COLUMN].to_numpy(
                dtype="float64"
            )
            for name, values in (
                ("lat", latitude),
                ("lon", longitude),
                (APP_COUNT_COLUMN, app_count),
                (POPULATION_COLUMN, population),
            ):
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Non-finite {name} in {input_path}, "
                        f"batch {batch_number}"
                    )
            if (app_count < 0).any() or (population < 0).any():
                raise ValueError(
                    f"Negative population values in {input_path}, "
                    f"batch {batch_number}"
                )

            pm25 = np.full(len(frame), np.nan, dtype="float32")
            pm25_source_file = np.empty(len(frame), dtype=object)
            for utc_date_value in sorted(set(utc_dates)):
                date_mask = utc_dates == utc_date_value
                matched_values, raster = matcher.lookup(
                    latitude[date_mask],
                    longitude[date_mask],
                    str(utc_date_value),
                )
                pm25[date_mask] = matched_values
                pm25_source_file[date_mask] = raster.source_path.name

            matched_pm25 = np.isfinite(pm25)
            diagnostics.update(
                local_hours, population, matched_pm25
            )
            calibrated_exposure = (
                population * pm25.astype("float64")
            )
            app_exposure = app_count * pm25.astype("float64")

            if save_point_today:
                point_frame = pd.DataFrame(
                    {
                        "utc_time": utc_time,
                        "local_time": local_time,
                        "utc_date": utc_dates,
                        "local_date": day_text,
                        "utc_hour": frame["utc_hour"].to_numpy(),
                        "local_hour": local_hours,
                        "lat": latitude,
                        "lon": longitude,
                        APP_COUNT_COLUMN: app_count,
                        POPULATION_COLUMN: population,
                        "pm25": pm25,
                        EXPOSURE_COLUMN: calibrated_exposure,
                        "app_exposure": app_exposure,
                        "pm25_source_date_utc": utc_dates,
                        "pm25_source_file": pm25_source_file,
                    }
                )
                point_table = pa.Table.from_pandas(
                    point_frame, preserve_index=False
                )
                if point_writer is None:
                    point_writer = pq.ParquetWriter(
                        point_temp,
                        point_table.schema,
                        compression="zstd",
                    )
                point_writer.write_table(point_table)

            matched_indices = np.flatnonzero(matched_pm25)
            if len(matched_indices):
                grid_x, grid_y = project_to_grid(
                    longitude[matched_indices],
                    latitude[matched_indices],
                    forward_transformer,
                )
                matched_frame = pd.DataFrame(
                    {
                        "local_time": local_time.iloc[
                            matched_indices
                        ].to_numpy(),
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "snapshot_population": population[
                            matched_indices
                        ],
                        "snapshot_app_count": app_count[
                            matched_indices
                        ],
                        "snapshot_exposure": calibrated_exposure[
                            matched_indices
                        ],
                        "snapshot_app_exposure": app_exposure[
                            matched_indices
                        ],
                    }
                )
                snapshot_partial = matched_frame.groupby(
                    ["local_time", "grid_x", "grid_y"],
                    as_index=False,
                    observed=True,
                    sort=True,
                )[SNAPSHOT_VALUE_COLUMNS].sum(min_count=1)
            else:
                snapshot_partial = pd.DataFrame(
                    columns=[
                        "local_time",
                        "grid_x",
                        "grid_y",
                        *SNAPSHOT_VALUE_COLUMNS,
                    ]
                )

            observations = observation_rows(local_time, utc_dates)
            aggregator.consume_batch(observations, snapshot_partial)
            print(
                f"    batch {batch_number}: rows={len(frame):,}, "
                f"matched_pm25={int(matched_pm25.sum()):,}, "
                f"missing_pm25={int((~matched_pm25).sum()):,}, "
                f"snapshot_grid_rows={len(snapshot_partial):,}"
            )

        if point_writer is not None:
            point_writer.close()
            point_writer = None
            os.replace(point_temp, point_path)
            print(f"  atomically committed: {point_path}")

        hourly, quality = aggregator.finish()
        hourly = finalize_hourly_output(
            hourly, local_day, inverse_transformer
        )
        hourly.to_parquet(
            output_temp,
            index=False,
            compression="zstd",
        )
        os.replace(output_temp, output_path)

        summary = diagnostics.summary_frame(
            matcher, output_rows=len(hourly)
        )
        summary["input_time_order_descents"] = (
            input_time_order_descents
        )
        hourly_missing = diagnostics.hourly_frame()
        coverage_path = (
            args.diagnostics_dir
            / f"hourly_snapshot_coverage_{day_text}.csv"
        )
        summary_path = (
            args.diagnostics_dir
            / f"pm25_matching_summary_{day_text}.csv"
        )
        hourly_missing_path = (
            args.diagnostics_dir
            / f"pm25_missing_by_hour_{day_text}.csv"
        )
        write_csv_atomic(quality, coverage_path)
        write_csv_atomic(summary, summary_path)
        write_csv_atomic(hourly_missing, hourly_missing_path)
        aggregator.cleanup()

        missing_share = summary[
            "pm25_missing_population_share"
        ].iat[0]
        print(f"  atomically committed: {output_path}")
        print(f"  output grid-hour rows: {len(hourly):,}")
        print(
            f"  PM2.5 matched records: "
            f"{diagnostics.matched_records:,}"
        )
        print(
            f"  PM2.5 missing records: "
            f"{diagnostics.missing_records:,}"
        )
        print(
            "  PM2.5-missing estimated_population: "
            f"{diagnostics.missing_population:,.6f} "
            f"({missing_share:.8%})"
        )
        print(
            f"  local hours represented: "
            f"{sorted(hourly['local_hour'].unique().tolist())}"
        )
        print(
            "  physical input time-order descents handled: "
            f"{input_time_order_descents:,}"
        )
        print(f"  diagnostics: {summary_path}")
        print(f"  diagnostics: {hourly_missing_path}")
        print(f"  diagnostics: {coverage_path}")
        return True
    except Exception:
        if point_writer is not None:
            point_writer.close()
        aggregator.close_writers()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match formal calibrated five-minute population records to "
            "UTC-date CHAP PM2.5 at original coordinates, aggregate each "
            "snapshot to the established China-LCC 10 km grid, then compute "
            "Beijing-local hourly snapshot means."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=CALIBRATED_DIR)
    parser.add_argument("--pm25-dir", type=Path, default=PM25_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--point-output-dir", type=Path, default=POINT_OUTPUT_DIR
    )
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR
    )
    parser.add_argument(
        "--pm25-variable",
        default=None,
        help=(
            "Optional NetCDF variable. Default: first data variable, "
            "matching the old Chunyun code."
        ),
    )
    parser.add_argument(
        "--save-point-exposure",
        action="store_true",
        help=(
            "Also save a large point-level parquet. Disabled by default."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Maximum calibrated rows per read/match chunk.",
    )
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=EXPECTED_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--expected-interval-minutes",
        type=float,
        default=EXPECTED_INTERVAL_MINUTES,
    )
    parser.add_argument(
        "--max-snapshot-duration-minutes",
        type=float,
        default=MAX_SNAPSHOT_DURATION_MINUTES,
    )
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "local":
        raise ValueError(
            "--start-date and --end-date select Beijing local-date "
            "partitions; use --date-basis local."
        )
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    validate_date_range(args.start_date, args.end_date)
    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Formal calibrated input directory not found: {args.input_dir}"
        )
    if not args.pm25_dir.exists():
        raise FileNotFoundError(
            f"PM2.5 directory not found: {args.pm25_dir}"
        )

    print("Formal calibrated population -> PM2.5 -> hourly LCC 10 km")
    print(
        f"  Beijing local dates: {args.start_date} to {args.end_date}"
    )
    print(f"  population input: {args.input_dir}")
    print(f"  PM2.5 file selection: existing utc_date/utc_time")
    print(f"  population column: {POPULATION_COLUMN}")
    print(f"  exposure column: {EXPOSURE_COLUMN}")
    print(f"  China LCC: {CHINA_LCC}")
    print(
        f"  grid: {GRID_SIZE_M:,} m, origin=(0, 0), floor assignment"
    )
    print(
        "  hourly method: regular 12-snapshot arithmetic mean; "
        "otherwise forward-duration weighting capped at "
        f"{args.max_snapshot_duration_minutes:g} minutes"
    )
    print(
        "  missing PM2.5: retained in diagnostics, excluded from exposure "
        "aggregation, never filled with zero"
    )

    processed = 0
    skipped = 0
    for local_day in iter_dates(args.start_date, args.end_date):
        if process_day(local_day, args):
            processed += 1
        else:
            skipped += 1
    print(
        f"\nFinished: processed local dates={processed}, "
        f"skipped existing={skipped}"
    )


if __name__ == "__main__":
    main()
