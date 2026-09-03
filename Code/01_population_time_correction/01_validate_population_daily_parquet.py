from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    BEIJING_OFFSET_HOURS,
    CHUNYUN_END,
    CHUNYUN_START,
    DIAGNOSTICS_DIR,
    POPULATION_INPUT_DIR,
    add_date_arguments,
    iter_dates,
    validate_date_range,
)


FILENAME_RE = re.compile(r"^population_china_(\d{4}-\d{2}-\d{2})\.parquet$")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _column_statistics(parquet_file: pq.ParquetFile, column: str) -> dict[str, object]:
    index = parquet_file.schema_arrow.names.index(column)
    minima: list[object] = []
    maxima: list[object] = []
    null_count = 0
    statistics_complete = True
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = parquet_file.metadata.row_group(row_group_index).column(index).statistics
        if statistics is None or not statistics.has_min_max:
            statistics_complete = False
            continue
        minima.append(statistics.min)
        maxima.append(statistics.max)
        null_count += int(statistics.null_count or 0)
    return {
        "min": min(minima) if minima else None,
        "max": max(maxima) if maxima else None,
        "null_count": null_count if statistics_complete else None,
        "complete": statistics_complete,
    }


def _hours_from_time_statistics(
    parquet_file: pq.ParquetFile,
) -> tuple[set[int], int]:
    time_index = parquet_file.schema_arrow.names.index("time")
    hours: set[int] = set()
    invalid_stat_row_groups = 0
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = (
            parquet_file.metadata.row_group(row_group_index)
            .column(time_index)
            .statistics
        )
        if statistics is None or not statistics.has_min_max:
            continue
        try:
            start = datetime.strptime(str(statistics.min), TIME_FORMAT)
            end = datetime.strptime(str(statistics.max), TIME_FORMAT)
        except (TypeError, ValueError):
            invalid_stat_row_groups += 1
            continue
        cursor = start.replace(minute=0, second=0)
        while cursor <= end:
            hours.add(cursor.hour)
            cursor += timedelta(hours=1)
    return hours, invalid_stat_row_groups


def inspect_metadata(path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected population parquet filename: {path.name}")
    file_date = match.group(1)
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    required = {"date", "time", "lat", "lon", "count"}
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    time_stats = _column_statistics(parquet_file, "time")
    date_stats = _column_statistics(parquet_file, "date")
    lat_stats = _column_statistics(parquet_file, "lat")
    lon_stats = _column_statistics(parquet_file, "lon")
    count_stats = _column_statistics(parquet_file, "count")
    hours_set, invalid_time_stat_row_groups = _hours_from_time_statistics(
        parquet_file
    )
    hours = sorted(hours_set)
    invalid_time_value_count = 0
    blank_time_value_count = 0
    time_partition_mismatch_value_count = 0
    time_index = parquet_file.schema_arrow.names.index("time")
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        statistics = (
            parquet_file.metadata.row_group(row_group_index)
            .column(time_index)
            .statistics
        )
        row_group_stats_valid = True
        row_group_partition_matches = True
        if statistics is None or not statistics.has_min_max:
            row_group_stats_valid = False
        else:
            for value in (statistics.min, statistics.max):
                try:
                    parsed = datetime.strptime(str(value), TIME_FORMAT)
                    if parsed.strftime("%Y-%m-%d") != file_date:
                        row_group_partition_matches = False
                except (TypeError, ValueError):
                    row_group_stats_valid = False
        if row_group_stats_valid and row_group_partition_matches:
            continue
        values = (
            parquet_file.read_row_group(row_group_index, columns=["time"])
            .column("time")
            .to_pandas()
        )
        text = values.astype("string")
        parsed_values = pd.to_datetime(text, errors="coerce")
        blank_time_value_count += int(text.fillna("").str.strip().eq("").sum())
        invalid_time_value_count += int(parsed_values.isna().sum())
        time_partition_mismatch_value_count += int(
            (
                parsed_values.notna()
                & (parsed_values.dt.strftime("%Y-%m-%d") != file_date)
            ).sum()
        )

    time_format_valid = True
    for value in (time_stats["min"], time_stats["max"]):
        try:
            datetime.strptime(str(value), TIME_FORMAT)
        except (TypeError, ValueError):
            time_format_valid = False
    time_date_matches_file = (
        str(time_stats["min"]).startswith(file_date)
        and str(time_stats["max"]).startswith(file_date)
        and str(date_stats["min"]) == file_date
        and str(date_stats["max"]) == file_date
    )

    return {
        "file": str(path.resolve()),
        "file_date": file_date,
        "file_size_bytes": path.stat().st_size,
        "rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.metadata.num_row_groups,
        "columns": "|".join(schema.names),
        "schema": "|".join(f"{field.name}:{field.type}" for field in schema),
        "time_type": str(schema.field("time").type),
        "time_min": time_stats["min"],
        "time_max": time_stats["max"],
        "time_format_valid": time_format_valid,
        "invalid_time_stat_row_groups": invalid_time_stat_row_groups,
        "invalid_time_value_count": invalid_time_value_count,
        "blank_time_value_count": blank_time_value_count,
        "time_partition_mismatch_value_count": time_partition_mismatch_value_count,
        "time_date_matches_utc_partition": time_date_matches_file,
        "utc_hours": ",".join(f"{hour:02d}" for hour in hours),
        "utc_hour_count": len(hours),
        "date_min": date_stats["min"],
        "date_max": date_stats["max"],
        "lat_type": str(schema.field("lat").type),
        "lat_min": lat_stats["min"],
        "lat_max": lat_stats["max"],
        "lat_null_count": lat_stats["null_count"],
        "lon_type": str(schema.field("lon").type),
        "lon_min": lon_stats["min"],
        "lon_max": lon_stats["max"],
        "lon_null_count": lon_stats["null_count"],
        "count_type": str(schema.field("count").type),
        "count_min": count_stats["min"],
        "count_max": count_stats["max"],
        "count_null_count": count_stats["null_count"],
        "count_negative_possible": (
            count_stats["min"] is None or int(count_stats["min"]) < 0
        ),
        "count_zero_possible": (
            count_stats["min"] is None or int(count_stats["min"]) == 0
        ),
        "metadata_statistics_complete": all(
            stats["complete"]
            for stats in (time_stats, date_stats, lat_stats, lon_stats, count_stats)
        ),
        "duplicate_scan_status": "NOT_SCANNED",
        "duplicate_coordinate_records": pd.NA,
        "duplicate_exact_records": pd.NA,
        "time_order_monotonic": pd.NA,
    }


def _packed_coord(lat_raw: np.ndarray, lon_raw: np.ndarray) -> np.ndarray:
    return (
        (lat_raw.astype(np.int64) + 32768) * 65536
        + lon_raw.astype(np.int64)
        + 32768
    )


def exhaustive_duplicate_scan(
    path: Path, batch_size: int
) -> dict[str, object]:
    parquet_file = pq.ParquetFile(path)
    required = {"time", "lat_raw", "lon_raw", "count"}
    missing = sorted(required.difference(parquet_file.schema_arrow.names))
    if missing:
        return {
            "duplicate_scan_status": f"UNAVAILABLE_MISSING_{','.join(missing)}",
            "duplicate_coordinate_records": pd.NA,
            "duplicate_exact_records": pd.NA,
            "time_order_monotonic": pd.NA,
            "unique_timestamps": [],
        }

    current_time: str | None = None
    current_coord_keys: list[np.ndarray] = []
    current_counts: list[np.ndarray] = []
    coordinate_duplicates = 0
    exact_duplicates = 0
    monotonic = True
    unique_timestamps: list[str] = []
    total_rows = 0

    def finalize_timestamp() -> tuple[int, int]:
        if not current_coord_keys:
            return 0, 0
        coords = np.concatenate(current_coord_keys)
        counts = np.concatenate(current_counts)
        coord_unique = np.unique(coords).size
        coord_dupes = len(coords) - coord_unique
        exact = np.empty(
            len(coords), dtype=[("coord", np.int64), ("count", np.int32)]
        )
        exact["coord"] = coords
        exact["count"] = counts.astype(np.int32, copy=False)
        exact_dupes = len(exact) - np.unique(exact).size
        return int(coord_dupes), int(exact_dupes)

    columns = ["time", "lat_raw", "lon_raw", "count"]
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=batch_size, columns=columns), start=1
    ):
        frame = batch.to_pandas()
        total_rows += len(frame)
        times = frame["time"].astype(str).to_numpy()
        if not len(times):
            continue
        changes = np.flatnonzero(times[1:] != times[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(frame)]))
        for start, end in zip(starts, ends, strict=True):
            timestamp = times[start]
            if current_time is not None and timestamp != current_time:
                coord_dupes, exact_dupes = finalize_timestamp()
                coordinate_duplicates += coord_dupes
                exact_duplicates += exact_dupes
                current_coord_keys.clear()
                current_counts.clear()
                if timestamp < current_time:
                    monotonic = False
                current_time = timestamp
                unique_timestamps.append(timestamp)
            elif current_time is None:
                current_time = timestamp
                unique_timestamps.append(timestamp)

            current_coord_keys.append(
                _packed_coord(
                    frame["lat_raw"].to_numpy()[start:end],
                    frame["lon_raw"].to_numpy()[start:end],
                )
            )
            current_counts.append(frame["count"].to_numpy()[start:end])
        print(
            f"  duplicate scan batch {batch_number}: "
            f"{len(frame):,} rows; cumulative {total_rows:,}"
        )

    coord_dupes, exact_dupes = finalize_timestamp()
    coordinate_duplicates += coord_dupes
    exact_duplicates += exact_dupes
    return {
        "duplicate_scan_status": "EXHAUSTIVE_NATURAL_KEY_TIME_LAT_LON",
        "duplicate_coordinate_records": coordinate_duplicates,
        "duplicate_exact_records": exact_duplicates,
        "time_order_monotonic": monotonic,
        "unique_timestamps": unique_timestamps,
    }


def build_local_hour_coverage(inventory: pd.DataFrame) -> pd.DataFrame:
    available_local: set[datetime] = set()
    for row in inventory.itertuples(index=False):
        for hour_text in str(row.utc_hours).split(","):
            if not hour_text:
                continue
            utc_hour = datetime.strptime(
                f"{row.file_date} {hour_text}", "%Y-%m-%d %H"
            )
            available_local.add(utc_hour + timedelta(hours=BEIJING_OFFSET_HOURS))

    if not available_local:
        return pd.DataFrame(
            columns=[
                "local_date",
                "expected_hours",
                "available_hours",
                "missing_hours",
                "is_complete_local_day",
            ]
        )
    start = min(available_local).date()
    end = max(available_local).date()
    rows = []
    for local_date in iter_dates(start, end):
        present = sorted(
            point.hour for point in available_local if point.date() == local_date
        )
        missing = sorted(set(range(24)).difference(present))
        rows.append(
            {
                "local_date": local_date.isoformat(),
                "expected_hours": 24,
                "available_hours": len(present),
                "missing_hours": ",".join(f"{hour:02d}" for hour in missing),
                "is_complete_local_day": len(present) == 24,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only validation of existing App population daily parquet files. "
            "The source time field is interpreted as UTC and is never modified."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    add_date_arguments(parser, default_basis="utc")
    parser.add_argument(
        "--duplicate-scan-date",
        type=lambda value: datetime.strptime(value, "%Y-%m-%d").date(),
        default=CHUNYUN_START,
        help=(
            "UTC date for exhaustive natural-key duplicate scanning. Other files "
            "are checked from complete Parquet metadata only."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "utc":
        raise ValueError(
            "Raw daily parquet partitions use UTC dates; validation requires "
            "--date-basis utc."
        )
    validate_date_range(args.start_date, args.end_date)
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Population daily parquet directory not found: {args.input_dir}")

    all_files = sorted(args.input_dir.glob("population_china_*.parquet"))
    selected_files = []
    for path in all_files:
        match = FILENAME_RE.match(path.name)
        if match:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            if args.start_date <= file_date <= args.end_date:
                selected_files.append(path)

    expected_dates = set(iter_dates(args.start_date, args.end_date))
    actual_dates = {
        datetime.strptime(FILENAME_RE.match(path.name).group(1), "%Y-%m-%d").date()
        for path in selected_files
    }
    missing_dates = sorted(expected_dates.difference(actual_dates))
    extra_dates = sorted(actual_dates.difference(expected_dates))

    print("Population daily parquet validation")
    print(f"  input: {args.input_dir.resolve()}")
    print(f"  date basis: UTC")
    print(f"  selected UTC dates: {args.start_date} to {args.end_date}")
    print(f"  selected files: {len(selected_files)}")
    print(f"  missing UTC dates: {[value.isoformat() for value in missing_dates] or 'none'}")
    print(f"  extra UTC dates: {[value.isoformat() for value in extra_dates] or 'none'}")

    inventory_rows = []
    for index, path in enumerate(selected_files, start=1):
        row = inspect_metadata(path)
        inventory_rows.append(row)
        print(
            f"  [{index:02d}/{len(selected_files):02d}] {path.name}: "
            f"rows={row['rows']:,}, time={row['time_min']}..{row['time_max']}, "
            f"hours={row['utc_hour_count']}, count={row['count_min']}..{row['count_max']}"
            f", invalid-time-values={row['invalid_time_value_count']:,}"
            f", cross-partition-values={row['time_partition_mismatch_value_count']:,}"
        )

    duplicate_target = args.input_dir / (
        f"population_china_{args.duplicate_scan_date.isoformat()}.parquet"
    )
    duplicate_result: dict[str, object] | None = None
    if duplicate_target in selected_files:
        print(f"Exhaustive duplicate scan: {duplicate_target}")
        duplicate_result = exhaustive_duplicate_scan(
            duplicate_target, args.batch_size
        )
        for row in inventory_rows:
            if row["file_date"] == args.duplicate_scan_date.isoformat():
                for key in (
                    "duplicate_scan_status",
                    "duplicate_coordinate_records",
                    "duplicate_exact_records",
                    "time_order_monotonic",
                ):
                    row[key] = duplicate_result[key]

    inventory = pd.DataFrame(inventory_rows)
    coverage = build_local_hour_coverage(inventory)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.diagnostics_dir / "population_daily_parquet_inventory.csv"
    coverage_path = args.diagnostics_dir / "population_time_coverage.csv"
    summary_path = args.diagnostics_dir / "population_validation_summary.md"
    outputs = [inventory_path, coverage_path, summary_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Diagnostics already exist; rerun with --overwrite: "
            + ", ".join(map(str, existing))
        )

    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    feb1 = coverage.loc[coverage["local_date"] == CHUNYUN_START.isoformat()]
    feb1_complete = (
        bool(feb1.iloc[0]["is_complete_local_day"]) if len(feb1) else False
    )
    feb1_missing = feb1.iloc[0]["missing_hours"] if len(feb1) else "00-23"
    cadence = "not exhaustively scanned"
    timestamps_per_hour = "unknown"
    if duplicate_result and duplicate_result["unique_timestamps"]:
        timestamps = [
            datetime.strptime(value, TIME_FORMAT)
            for value in duplicate_result["unique_timestamps"]
        ]
        delta_values = pd.Series(
            [
                int((right - left).total_seconds() // 60)
                for left, right in zip(timestamps, timestamps[1:])
                if right > left
            ],
            dtype="int64",
        )
        base_cadence = int(delta_values.mode().iloc[0])
        gap_deltas = sorted(
            int(value)
            for value in delta_values.unique()
            if int(value) > base_cadence
        )
        cadence = (
            f"{base_cadence} minutes; larger observed gaps={gap_deltas or 'none'}"
        )
        timestamps_per_hour = (
            f"{len(timestamps) / len({value.hour for value in timestamps}):.1f}"
        )

    summary = f"""# Population daily parquet validation

- Input directory: `{args.input_dir.resolve()}`
- UTC files: {len(selected_files)}
- UTC file date range: {args.start_date} to {args.end_date}
- Missing UTC dates: {", ".join(value.isoformat() for value in missing_dates) or "none"}
- Total rows (metadata): {int(inventory["rows"].sum()):,}
- Source `time` storage type: {", ".join(sorted(inventory["time_type"].unique()))}
- Source time semantics: UTC. This is validated against the UTC source partition date and the unchanged source `time`; no `-8 hour` transformation is used.
- Observed exhaustive test-date cadence: {cadence}
- Observations per covered hour on exhaustive test date: {timestamps_per_hour}
- All selected files cover UTC hours 00-23: {bool((inventory["utc_hour_count"] == 24).all())}
- Files with invalid/blank time statistics: {int((~inventory["time_format_valid"]).sum())}
- Row groups whose time statistics contain invalid/blank values: {int(inventory["invalid_time_stat_row_groups"].sum())}
- Invalid time values counted in affected row groups: {int(inventory["invalid_time_value_count"].sum()):,}
- Blank time values counted in affected row groups: {int(inventory["blank_time_value_count"].sum()):,}
- Valid timestamps stored under a different filename UTC date: {int(inventory["time_partition_mismatch_value_count"].sum()):,}
- All selected `count` row-group minima are positive: {bool((inventory["count_min"] > 0).all())}
- Total `count` nulls from complete row-group statistics: {int(inventory["count_null_count"].fillna(0).sum())}
- Duplicate scan scope: exhaustive for UTC {args.duplicate_scan_date}; other dates are not exhaustively scanned because the collection contains {int(inventory["rows"].sum()):,} records.
- Duplicate coordinate records on UTC {args.duplicate_scan_date}: {duplicate_result["duplicate_coordinate_records"] if duplicate_result else "not scanned"}
- Exact duplicate records on UTC {args.duplicate_scan_date}: {duplicate_result["duplicate_exact_records"] if duplicate_result else "not scanned"}
- Beijing local 2018-02-01 complete: {feb1_complete}
- Missing Beijing local hours on 2018-02-01: {feb1_missing or "none"}
- Important: missing local hours are not filled and are not interpreted as zero.
"""
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Created: {inventory_path}")
    print(f"Created: {coverage_path}")
    print(f"Created: {summary_path}")
    print(
        "Beijing local 2018-02-01 complete: "
        f"{feb1_complete}; missing hours: {feb1_missing or 'none'}"
    )


if __name__ == "__main__":
    main()
