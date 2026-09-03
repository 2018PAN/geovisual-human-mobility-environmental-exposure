from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    DIAGNOSTICS_DIR,
    POPULATION_INPUT_DIR,
    add_date_arguments,
    iter_dates,
    validate_date_range,
)


def dictionary_values(array: pa.ChunkedArray) -> set[str]:
    values: set[str] = set()
    for chunk in array.chunks:
        if isinstance(chunk, pa.DictionaryArray):
            values.update(
                str(value)
                for value in chunk.dictionary.to_pylist()
                if value is not None
            )
        else:
            values.update(
                str(value)
                for value in chunk.unique().to_pylist()
                if value is not None
            )
    return values


def scan_unique_time_texts(paths: list[Path]) -> set[str]:
    unique_values: set[str] = set()
    for file_number, path in enumerate(paths, start=1):
        parquet_file = pq.ParquetFile(
            path,
            read_dictionary=["time"],
        )
        if "time" not in parquet_file.schema_arrow.names:
            raise ValueError(f"Missing time column: {path}")
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_index,
                columns=["time"],
                use_threads=False,
            )
            unique_values.update(dictionary_values(table["time"]))
        print(
            f"Time dictionary scan [{file_number}/{len(paths)}] "
            f"{path.name}: cumulative unique text values="
            f"{len(unique_values):,}"
        )
    return unique_values


def load_recovered_times(audit_path: Path) -> pd.DatetimeIndex:
    if not audit_path.exists():
        raise FileNotFoundError(f"Blank-time audit not found: {audit_path}")
    audit = pd.read_csv(audit_path, dtype="string")
    recoverable = (
        audit["is_recoverable"]
        .fillna("")
        .str.strip()
        .str.casefold()
        .isin(["true", "1", "yes"])
    )
    values = pd.to_datetime(
        audit.loc[recoverable, "recovered_utc_time"],
        errors="coerce",
    ).dropna()
    return pd.DatetimeIndex(values).drop_duplicates().sort_values()


def duration_quality(
    hour_start: pd.Timestamp,
    hour_times: pd.DatetimeIndex,
    *,
    expected_sample_count: int,
    expected_interval_minutes: float,
    max_snapshot_duration_minutes: float,
) -> dict[str, object]:
    hour_end = hour_start + pd.Timedelta(hours=1)
    interval = pd.Timedelta(minutes=expected_interval_minutes)
    expected_times = pd.date_range(
        hour_start,
        periods=expected_sample_count,
        freq=interval,
    )
    available = set(hour_times)
    missing = [
        timestamp.strftime("%H:%M:%S")
        for timestamp in expected_times
        if timestamp not in available
    ]
    regular_complete = bool(
        len(hour_times) == expected_sample_count
        and hour_times.equals(expected_times)
    )

    max_duration_seconds = max_snapshot_duration_minutes * 60.0
    durations: list[float] = []
    for position, current in enumerate(hour_times):
        if position + 1 < len(hour_times):
            raw_end = min(hour_times[position + 1], hour_end)
        else:
            raw_end = hour_end
        raw_seconds = max((raw_end - current).total_seconds(), 0.0)
        durations.append(min(raw_seconds, max_duration_seconds))

    if len(hour_times):
        boundary_times = pd.DatetimeIndex(
            [hour_start, *hour_times.tolist(), hour_end]
        ).drop_duplicates().sort_values()
        max_gap_minutes = float(
            np.diff(boundary_times.asi8).max() / 60_000_000_000
        )
        first_observation = hour_times.min()
        last_observation = hour_times.max()
    else:
        max_gap_minutes = 60.0
        first_observation = pd.NaT
        last_observation = pd.NaT
    duration_minutes = float(sum(durations) / 60.0)
    sample_count = int(len(hour_times))
    return {
        "local_date": hour_start.strftime("%Y-%m-%d"),
        "local_hour": int(hour_start.hour),
        "local_hour_start": hour_start,
        "utc_hour_start": hour_start - pd.Timedelta(hours=8),
        "sample_count": sample_count,
        "expected_sample_count": expected_sample_count,
        "coverage_ratio": sample_count / expected_sample_count,
        "first_observation_time": first_observation,
        "last_observation_time": last_observation,
        "missing_expected_times": "|".join(missing),
        "is_complete_hour": sample_count == expected_sample_count,
        "is_regular_expected_interval": regular_complete,
        "weighted_duration_minutes": duration_minutes,
        "duration_coverage_ratio": duration_minutes / 60.0,
        "max_observed_gap_minutes": max_gap_minutes,
        "max_snapshot_duration_minutes": max_snapshot_duration_minutes,
        "aggregation_method": (
            "no_observations"
            if sample_count == 0
            else (
                "arithmetic_mean_regular_complete"
                if regular_complete
                else "time_weighted_forward_duration_capped"
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract actual unique UTC timestamps from parquet dictionary "
            "values and report five-minute snapshot completeness for every "
            "formal Beijing local hour. Population records are not converted."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument(
        "--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR
    )
    parser.add_argument(
        "--blank-time-audit",
        type=Path,
        default=DIAGNOSTICS_DIR / "blank_time_audit.csv",
    )
    parser.add_argument("--expected-sample-count", type=int, default=12)
    parser.add_argument(
        "--expected-interval-minutes", type=float, default=5.0
    )
    parser.add_argument(
        "--max-snapshot-duration-minutes", type=float, default=10.0
    )
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "local":
        raise ValueError(
            "This diagnostic is defined on Beijing local hours; use "
            "--date-basis local."
        )
    validate_date_range(args.start_date, args.end_date)
    if args.expected_sample_count <= 0:
        raise ValueError("--expected-sample-count must be positive")
    if args.expected_interval_minutes <= 0:
        raise ValueError("--expected-interval-minutes must be positive")
    if args.max_snapshot_duration_minutes <= 0:
        raise ValueError(
            "--max-snapshot-duration-minutes must be positive"
        )

    paths = []
    for day in iter_dates(
        args.start_date - timedelta(days=1),
        args.end_date,
    ):
        path = args.input_dir / f"population_china_{day:%Y-%m-%d}.parquet"
        if path.exists():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No population parquet files found in {args.input_dir}"
        )

    output_path = (
        args.diagnostics_dir
        / "population_hourly_snapshot_coverage.csv"
    )
    summary_path = (
        args.diagnostics_dir
        / "population_hourly_snapshot_coverage_summary.csv"
    )
    existing = [
        path for path in (output_path, summary_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs exist; rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    raw_texts = scan_unique_time_texts(paths)
    parsed_utc = pd.to_datetime(
        pd.Series(sorted(raw_texts), dtype="string"),
        errors="coerce",
    ).dropna()
    recovered_utc = load_recovered_times(args.blank_time_audit)
    utc_times = pd.DatetimeIndex(
        np.concatenate(
            [
                parsed_utc.to_numpy(dtype="datetime64[ns]"),
                recovered_utc.to_numpy(dtype="datetime64[ns]"),
            ]
        )
    ).drop_duplicates().sort_values()
    local_times = utc_times + pd.Timedelta(hours=8)
    local_start = pd.Timestamp(
        datetime.combine(args.start_date, time.min)
    )
    local_end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    local_times = local_times[
        (local_times >= local_start) & (local_times < local_end)
    ]

    rows = []
    for hour_start in pd.date_range(
        local_start,
        local_end - pd.Timedelta(hours=1),
        freq="h",
    ):
        hour_end = hour_start + pd.Timedelta(hours=1)
        hour_times = local_times[
            (local_times >= hour_start) & (local_times < hour_end)
        ]
        rows.append(
            duration_quality(
                hour_start,
                hour_times,
                expected_sample_count=args.expected_sample_count,
                expected_interval_minutes=args.expected_interval_minutes,
                max_snapshot_duration_minutes=(
                    args.max_snapshot_duration_minutes
                ),
            )
        )
    coverage = pd.DataFrame(rows)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_path, index=False, encoding="utf-8-sig")

    complete_hours = int(coverage["is_complete_hour"].sum())
    incomplete_hours = int((~coverage["is_complete_hour"]).sum())
    zero_observation_hours = int((coverage["sample_count"] == 0).sum())
    summary = pd.DataFrame(
        [
            {
                "date_basis": "local",
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "formal_hours": len(coverage),
                "complete_hours": complete_hours,
                "incomplete_hours": incomplete_hours,
                "zero_observation_hours": zero_observation_hours,
                "hours_with_1_to_11_snapshots": int(
                    coverage["sample_count"].between(1, 11).sum()
                ),
                "hours_with_more_than_expected_snapshots": int(
                    (
                        coverage["sample_count"]
                        > args.expected_sample_count
                    ).sum()
                ),
                "actual_valid_utc_timestamp_min": (
                    utc_times.min() if len(utc_times) else pd.NaT
                ),
                "actual_valid_utc_timestamp_max": (
                    utc_times.max() if len(utc_times) else pd.NaT
                ),
                "actual_selected_local_timestamp_min": (
                    local_times.min() if len(local_times) else pd.NaT
                ),
                "actual_selected_local_timestamp_max": (
                    local_times.max() if len(local_times) else pd.NaT
                ),
                "recoverable_blank_time_timestamps_added": len(
                    recovered_utc
                ),
                "invalid_or_blank_time_dictionary_values": (
                    len(raw_texts) - int(parsed_utc.nunique())
                ),
                "expected_sample_count": args.expected_sample_count,
                "expected_interval_minutes": (
                    args.expected_interval_minutes
                ),
                "max_snapshot_duration_minutes": (
                    args.max_snapshot_duration_minutes
                ),
            }
        ]
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Created: {output_path}")
    print(f"Created: {summary_path}")
    print(f"Formal Beijing hours: {len(coverage):,}")
    print(f"Complete hours: {complete_hours:,}")
    print(f"Incomplete hours: {incomplete_hours:,}")
    print(f"Zero-observation hours: {zero_observation_hours:,}")


if __name__ == "__main__":
    main()
