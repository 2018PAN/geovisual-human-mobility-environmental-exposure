from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    BEIJING_OFFSET_HOURS,
    CALIBRATED_DIR,
    CHINA_LCC,
    CHUNYUN_END,
    CHUNYUN_START,
    DIAGNOSTICS_DIR,
    GRID_SIZE_M,
    NEW_ROOT,
    add_date_arguments,
    iter_dates,
    validate_date_range,
)


OUTPUT_DIR = NEW_ROOT / "Output" / "Population" / "hourly_grid_10km"
DEFAULT_EXPECTED_SAMPLE_COUNT = 12
DEFAULT_EXPECTED_INTERVAL_MINUTES = 5.0
DEFAULT_MAX_SNAPSHOT_DURATION_MINUTES = 10.0


@dataclass(frozen=True)
class ValueSpec:
    source: str
    output: str


def require_columns(columns: set[str], required: set[str], path: Path) -> None:
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def build_hourly_time_weights(
    observation_times: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    expected_sample_count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    expected_interval_minutes: float = DEFAULT_EXPECTED_INTERVAL_MINUTES,
    max_snapshot_duration_minutes: float = (
        DEFAULT_MAX_SNAPSHOT_DURATION_MINUTES
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one forward-duration weight per unique snapshot.

    A snapshot represents the interval from its timestamp to the next observed
    timestamp in the same hour.  The last snapshot represents the interval to
    the hour boundary.  Each duration is clipped to
    ``max_snapshot_duration_minutes``; a long data gap therefore remains
    uncovered instead of being filled or interpolated.  Time before the first
    observation is also left uncovered.

    For the regular complete sequence 00, 05, ..., 55 minutes, every weight is
    five minutes and the weighted result is exactly the arithmetic mean.
    """
    if expected_sample_count <= 0:
        raise ValueError("expected_sample_count must be positive")
    if expected_interval_minutes <= 0:
        raise ValueError("expected_interval_minutes must be positive")
    if max_snapshot_duration_minutes <= 0:
        raise ValueError("max_snapshot_duration_minutes must be positive")

    times = (
        pd.DatetimeIndex(pd.to_datetime(observation_times, errors="coerce"))
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if len(times) == 0:
        empty_weights = pd.DataFrame(
            columns=["local_time", "local_hour_start", "weight_seconds"]
        )
        empty_quality = pd.DataFrame(
            columns=[
                "local_hour_start",
                "sample_count",
                "expected_sample_count",
                "coverage_ratio",
                "first_observation_time",
                "last_observation_time",
                "is_complete_hour",
                "is_regular_expected_interval",
                "weighted_duration_minutes",
                "duration_coverage_ratio",
                "max_observed_gap_minutes",
                "max_snapshot_duration_minutes",
                "aggregation_method",
            ]
        )
        return empty_weights, empty_quality

    weight_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    time_frame = pd.DataFrame({"local_time": times})
    time_frame["local_hour_start"] = time_frame["local_time"].dt.floor("h")

    interval = pd.Timedelta(minutes=expected_interval_minutes)
    max_duration_seconds = max_snapshot_duration_minutes * 60.0

    for hour_start, part in time_frame.groupby(
        "local_hour_start", sort=True, observed=True
    ):
        hour_times = pd.DatetimeIndex(part["local_time"]).sort_values()
        hour_end = hour_start + pd.Timedelta(hours=1)
        expected_times = pd.date_range(
            hour_start,
            periods=expected_sample_count,
            freq=interval,
        )
        regular_complete = bool(
            len(hour_times) == expected_sample_count
            and hour_times.equals(expected_times)
        )

        durations_seconds: list[float] = []
        for position, current in enumerate(hour_times):
            if position + 1 < len(hour_times):
                raw_end = min(hour_times[position + 1], hour_end)
            else:
                raw_end = hour_end
            raw_seconds = max((raw_end - current).total_seconds(), 0.0)
            duration_seconds = min(raw_seconds, max_duration_seconds)
            durations_seconds.append(duration_seconds)
            weight_rows.append(
                {
                    "local_time": current,
                    "local_hour_start": hour_start,
                    "weight_seconds": duration_seconds,
                }
            )

        boundary_times = pd.DatetimeIndex(
            [hour_start, *hour_times.tolist(), hour_end]
        ).drop_duplicates().sort_values()
        if len(boundary_times) > 1:
            max_gap_minutes = float(
                np.diff(boundary_times.asi8).max() / 60_000_000_000
            )
        else:
            max_gap_minutes = np.nan
        total_duration_minutes = float(sum(durations_seconds) / 60.0)
        sample_count = int(len(hour_times))
        quality_rows.append(
            {
                "local_hour_start": hour_start,
                "sample_count": sample_count,
                "expected_sample_count": expected_sample_count,
                "coverage_ratio": sample_count / expected_sample_count,
                "first_observation_time": hour_times.min(),
                "last_observation_time": hour_times.max(),
                "is_complete_hour": sample_count == expected_sample_count,
                "is_regular_expected_interval": regular_complete,
                "weighted_duration_minutes": total_duration_minutes,
                "duration_coverage_ratio": total_duration_minutes / 60.0,
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
        )

    weights = pd.DataFrame(weight_rows)
    quality = pd.DataFrame(quality_rows)
    return weights, quality


def aggregate_snapshot_grids_to_hourly(
    snapshot_grid: pd.DataFrame,
    weights: pd.DataFrame,
    quality: pd.DataFrame,
    value_specs: list[ValueSpec],
) -> pd.DataFrame:
    """Average snapshot grid totals using global within-hour time weights.

    The denominator is the sum of weights for every available snapshot in the
    hour, not merely the snapshots on which a particular grid has a row.
    Consequently, an absent grid at an otherwise available snapshot represents
    zero population in that grid, while an entirely missing snapshot is never
    inserted as zero.
    """
    if snapshot_grid.empty or weights.empty:
        return pd.DataFrame()
    required = {"local_time", "grid_x", "grid_y"}
    missing = required.difference(snapshot_grid.columns)
    if missing:
        raise ValueError(f"Snapshot grid is missing columns: {sorted(missing)}")

    work = snapshot_grid.merge(
        weights,
        on="local_time",
        how="inner",
        validate="many_to_one",
    )
    if work.empty:
        return pd.DataFrame()
    for spec in value_specs:
        work[f"weighted__{spec.output}"] = (
            work[spec.source].astype("float64") * work["weight_seconds"]
        )
        work[f"missing_weight__{spec.output}"] = np.where(
            work[spec.source].isna(),
            work["weight_seconds"],
            0.0,
        )
    numeric_columns = [
        column
        for spec in value_specs
        for column in (
            f"weighted__{spec.output}",
            f"missing_weight__{spec.output}",
        )
    ]
    hourly = work.groupby(
        ["local_hour_start", "grid_x", "grid_y"],
        as_index=False,
        observed=True,
    )[numeric_columns].sum(min_count=1)
    hourly = hourly.merge(
        quality,
        on="local_hour_start",
        how="left",
        validate="many_to_one",
    )
    global_denominator = hourly["weighted_duration_minutes"] * 60.0
    for spec in value_specs:
        denominator = (
            global_denominator
            - hourly.pop(f"missing_weight__{spec.output}")
        )
        hourly[spec.output] = (
            hourly.pop(f"weighted__{spec.output}")
            / denominator.replace(0.0, np.nan)
        )
    return hourly


def aggregate_input_day(
    input_path: Path,
    local_day: pd.Timestamp,
    *,
    batch_size: int,
    population_column: str,
    app_count_column: str,
    exposure_column: str,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[ValueSpec], bool]:
    parquet_file = pq.ParquetFile(input_path)
    available = set(parquet_file.schema_arrow.names)
    require_columns(
        available,
        {"local_time", "lat", "lon", population_column},
        input_path,
    )
    has_app_count = app_count_column in available
    has_exposure = exposure_column in available
    columns = ["local_time", "lat", "lon", population_column]
    if has_app_count:
        columns.append(app_count_column)
    if has_exposure:
        columns.append(exposure_column)

    transformer = Transformer.from_crs(
        "EPSG:4326", CHINA_LCC, always_xy=True
    )
    local_start = local_day
    local_end = local_day + pd.Timedelta(days=1)
    partials: list[pd.DataFrame] = []
    observation_parts: list[np.ndarray] = []

    value_specs = [
        ValueSpec(population_column, "hourly_population"),
    ]
    if has_app_count:
        value_specs.append(
            ValueSpec(app_count_column, "hourly_app_count_snapshot_mean")
        )
    if has_exposure:
        value_specs.append(
            ValueSpec(exposure_column, "hourly_calibrated_exposure")
        )

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=batch_size, columns=columns),
        start=1,
    ):
        frame = batch.to_pandas()
        local_time_values = pd.to_datetime(
            frame["local_time"], errors="coerce", cache=True
        )
        valid = (
            local_time_values.notna()
            & local_time_values.ge(local_start)
            & local_time_values.lt(local_end)
        )
        if not valid.any():
            continue
        work = frame.loc[valid, columns[1:]].copy()
        work["local_time"] = local_time_values.loc[valid].to_numpy()
        for column in ["lat", "lon", *[spec.source for spec in value_specs]]:
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(
            subset=["local_time", "lat", "lon", population_column]
        )
        if work.empty:
            continue
        if (work[population_column] < 0).any():
            raise ValueError(
                f"Negative {population_column} in {input_path}, "
                f"batch {batch_number}"
            )

        observation_parts.append(
            work["local_time"].drop_duplicates().to_numpy()
        )
        x, y = transformer.transform(
            work["lon"].to_numpy(dtype="float64", copy=False),
            work["lat"].to_numpy(dtype="float64", copy=False),
        )
        work["grid_x"] = (
            np.floor(np.asarray(x) / GRID_SIZE_M) * GRID_SIZE_M
        ).astype("int64")
        work["grid_y"] = (
            np.floor(np.asarray(y) / GRID_SIZE_M) * GRID_SIZE_M
        ).astype("int64")
        value_columns = [spec.source for spec in value_specs]
        grouped = work.groupby(
            ["local_time", "grid_x", "grid_y"],
            as_index=False,
            observed=True,
        )[value_columns].sum(min_count=1)
        partials.append(grouped)
        print(
            f"    batch {batch_number}: input={len(frame):,}, "
            f"selected={len(work):,}, snapshot-grid={len(grouped):,}"
        )

    if not partials:
        raise ValueError(f"No valid calibrated records found in {input_path}")
    value_columns = [spec.source for spec in value_specs]
    snapshot_grid = pd.concat(partials, ignore_index=True).groupby(
        ["local_time", "grid_x", "grid_y"],
        as_index=False,
        observed=True,
    )[value_columns].sum(min_count=1)
    observations = pd.DatetimeIndex(
        np.concatenate(observation_parts)
    ).drop_duplicates().sort_values()
    return snapshot_grid, observations, value_specs, has_exposure


def add_grid_coordinates(hourly: pd.DataFrame) -> pd.DataFrame:
    inverse = Transformer.from_crs(
        CHINA_LCC, "EPSG:4326", always_xy=True
    )
    center_x = hourly["grid_x"].to_numpy(dtype="float64") + GRID_SIZE_M / 2
    center_y = hourly["grid_y"].to_numpy(dtype="float64") + GRID_SIZE_M / 2
    center_lon, center_lat = inverse.transform(center_x, center_y)
    hourly["grid_size_m"] = GRID_SIZE_M
    hourly["grid_center_lon"] = np.asarray(center_lon)
    hourly["grid_center_lat"] = np.asarray(center_lat)
    hourly["utc_hour_start"] = hourly["local_hour_start"] - pd.Timedelta(
        hours=BEIJING_OFFSET_HOURS
    )
    hourly["local_date"] = hourly["local_hour_start"].dt.strftime("%Y-%m-%d")
    hourly["local_hour"] = hourly["local_hour_start"].dt.hour.astype("int8")
    ordered = [
        "utc_hour_start",
        "local_date",
        "local_hour_start",
        "local_hour",
        "grid_x",
        "grid_y",
        "grid_size_m",
        "grid_center_lon",
        "grid_center_lat",
        "hourly_population",
    ]
    optional = [
        column
        for column in (
            "hourly_app_count_snapshot_mean",
            "hourly_calibrated_exposure",
        )
        if column in hourly.columns
    ]
    quality = [
        "sample_count",
        "expected_sample_count",
        "coverage_ratio",
        "first_observation_time",
        "last_observation_time",
        "is_complete_hour",
        "is_regular_expected_interval",
        "weighted_duration_minutes",
        "duration_coverage_ratio",
        "max_observed_gap_minutes",
        "max_snapshot_duration_minutes",
        "aggregation_method",
    ]
    return hourly[[*ordered, *optional, *quality]].sort_values(
        ["local_hour_start", "grid_x", "grid_y"]
    )


def process_day(
    day: datetime.date,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, bool]:
    day_text = day.isoformat()
    input_path = (
        args.input_dir / f"population_calibrated_{day_text}.parquet"
    )
    output_path = (
        args.output_dir / f"population_hourly_10km_{day_text}.parquet"
    )
    coverage_path = (
        args.diagnostics_dir
        / f"hourly_population_coverage_{day_text}.csv"
    )
    if output_path.exists() and coverage_path.exists() and not args.overwrite:
        print(f"Skipped existing hourly output: {output_path}")
        return pd.read_csv(coverage_path, parse_dates=[
            "local_hour_start",
            "first_observation_time",
            "last_observation_time",
        ]), "hourly_calibrated_exposure" in pq.ParquetFile(
            output_path
        ).schema_arrow.names
    if not input_path.exists():
        raise FileNotFoundError(f"Calibrated input not found: {input_path}")

    print(f"Local date {day_text}")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    snapshot_grid, observations, value_specs, has_exposure = (
        aggregate_input_day(
            input_path,
            pd.Timestamp(day),
            batch_size=args.batch_size,
            population_column=args.population_column,
            app_count_column=args.app_count_column,
            exposure_column=args.exposure_column,
        )
    )
    weights, quality = build_hourly_time_weights(
        observations,
        expected_sample_count=args.expected_sample_count,
        expected_interval_minutes=args.expected_interval_minutes,
        max_snapshot_duration_minutes=(
            args.max_snapshot_duration_minutes
        ),
    )
    hourly = aggregate_snapshot_grids_to_hourly(
        snapshot_grid,
        weights,
        quality,
        value_specs,
    )
    hourly = add_grid_coordinates(hourly)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_path, index=False, compression="zstd")
    quality.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    print(f"  snapshot times: {len(observations):,}")
    print(f"  complete hours: {int(quality['is_complete_hour'].sum()):,}")
    print(
        f"  incomplete hours: "
        f"{int((~quality['is_complete_hour']).sum()):,}"
    )
    print(f"  created: {output_path}")
    print(f"  created: {coverage_path}")
    return quality, has_exposure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate calibrated five-minute population snapshots to the "
            "established China-LCC 10 km grid using a capped time-weighted "
            "mean. Missing snapshots are not zero-filled or interpolated."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=CALIBRATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR
    )
    parser.add_argument(
        "--population-column", default="estimated_population"
    )
    parser.add_argument("--app-count-column", default="app_count")
    parser.add_argument(
        "--exposure-column", default="calibrated_exposure"
    )
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        default=DEFAULT_EXPECTED_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--expected-interval-minutes",
        type=float,
        default=DEFAULT_EXPECTED_INTERVAL_MINUTES,
    )
    parser.add_argument(
        "--max-snapshot-duration-minutes",
        type=float,
        default=DEFAULT_MAX_SNAPSHOT_DURATION_MINUTES,
    )
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "local":
        raise ValueError(
            "Hourly 10 km outputs are partitioned by Beijing local_date; "
            "use --date-basis local."
        )
    validate_date_range(args.start_date, args.end_date)
    coverage_frames: list[pd.DataFrame] = []
    exposure_present = False
    for day in iter_dates(args.start_date, args.end_date):
        quality, day_has_exposure = process_day(day, args)
        coverage_frames.append(quality)
        exposure_present = exposure_present or day_has_exposure

    consolidated = pd.concat(coverage_frames, ignore_index=True).sort_values(
        "local_hour_start"
    )
    consolidated_path = (
        args.diagnostics_dir / "hourly_population_time_coverage.csv"
    )
    if consolidated_path.exists() and not args.overwrite:
        print(
            "Consolidated diagnostic already exists and was not overwritten: "
            f"{consolidated_path}"
        )
    else:
        consolidated.to_csv(
            consolidated_path, index=False, encoding="utf-8-sig"
        )
        print(f"Created: {consolidated_path}")
    print(
        "Exposure aggregation column: "
        + (
            args.exposure_column
            if exposure_present
            else "not present; population-only output"
        )
    )
    print(
        "Hourly values are means of available snapshots, never sums across "
        "the approximately 12 five-minute observations."
    )


if __name__ == "__main__":
    main()
