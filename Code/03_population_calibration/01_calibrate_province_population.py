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
    CALIBRATED_DIR,
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    PROVINCE_ASSIGNED_DIR,
    add_date_arguments,
    read_and_validate_official_population,
    selected_input_local_dates,
    validate_date_range,
)


INPUT_PREFIX = "population_province_assigned_"
OUTPUT_PREFIX = "population_calibrated_"


def selected_mask(
    frame: pd.DataFrame,
    *,
    date_basis: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    column = "utc_time" if date_basis == "utc" else "local_time"
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Invalid timestamps in {column}")
    return (values >= start) & (values < end)


def robust_anomaly_flags(values: pd.Series) -> tuple[pd.Series, pd.Series, float, float]:
    median = float(values.median())
    absolute_deviation = (values - median).abs()
    mad = float(absolute_deviation.median())
    if mad > 0:
        spread = 1.4826 * mad
        low_threshold = max(0.0, median - 6.0 * spread)
        high_threshold = median + 6.0 * spread
    else:
        low_threshold = max(0.0, median * 0.25)
        high_threshold = median * 4.0
    return (
        values < low_threshold,
        values > high_threshold,
        low_threshold,
        high_threshold,
    )


def infer_cadence_minutes(times: pd.Series) -> int:
    unique = pd.Series(pd.to_datetime(times).drop_duplicates().sort_values())
    if len(unique) < 2:
        return 60
    minutes = unique.diff().dropna().dt.total_seconds().div(60)
    minutes = minutes[minutes > 0]
    if minutes.empty:
        return 60
    return int(minutes.mode().iloc[0])


def scan_reference_counts(
    input_paths: list[Path],
    *,
    date_basis: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    province_partials: list[pd.DataFrame] = []
    national_partials: list[pd.DataFrame] = []
    scan_stats = {
        "selected_records": 0,
        "selected_app_count": 0,
        "unmatched_records": 0,
        "unmatched_app_count": 0,
    }
    for input_path in input_paths:
        parquet_file = pq.ParquetFile(input_path)
        required = {"utc_time", "local_time", "province", "count"}
        missing = sorted(required.difference(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"{input_path} is missing columns: {missing}")
        print(f"Reference pass reading: {input_path}")
        for batch_number, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["utc_time", "local_time", "province", "count"],
            ),
            start=1,
        ):
            frame = batch.to_pandas()
            mask = selected_mask(
                frame, date_basis=date_basis, start=start, end=end
            )
            frame = frame.loc[mask].copy()
            if frame.empty:
                continue
            frame["utc_time"] = pd.to_datetime(frame["utc_time"])
            frame["local_time"] = pd.to_datetime(frame["local_time"])
            frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
            if frame["count"].isna().any() or (frame["count"] < 0).any():
                raise ValueError(f"{input_path} contains invalid count values")

            unmatched = frame["province"].isna()
            scan_stats["selected_records"] += len(frame)
            scan_stats["selected_app_count"] += int(frame["count"].sum())
            scan_stats["unmatched_records"] += int(unmatched.sum())
            scan_stats["unmatched_app_count"] += int(
                frame.loc[unmatched, "count"].sum()
            )
            national_partials.append(
                frame.groupby(["utc_time", "local_time"], as_index=False)["count"]
                .sum()
                .rename(columns={"count": "raw_app_total"})
            )
            matched = frame.loc[~unmatched]
            if not matched.empty:
                province_partials.append(
                    matched.groupby(
                        ["utc_time", "local_time", "province"],
                        as_index=False,
                        observed=True,
                    )["count"]
                    .sum()
                    .rename(columns={"count": "raw_app_count"})
                )
            print(
                f"  reference batch {batch_number}: selected={len(frame):,}, "
                f"unmatched={int(unmatched.sum()):,}"
            )

    if not national_partials or not province_partials:
        raise ValueError("No selected records available for calibration")
    national = (
        pd.concat(national_partials, ignore_index=True)
        .groupby(["utc_time", "local_time"], as_index=False)["raw_app_total"]
        .sum()
        .sort_values("utc_time")
    )
    province = (
        pd.concat(province_partials, ignore_index=True)
        .groupby(["utc_time", "local_time", "province"], as_index=False)[
            "raw_app_count"
        ]
        .sum()
        .sort_values(["utc_time", "province"])
    )
    return national, province, scan_stats


def build_diagnostics_and_weights(
    national: pd.DataFrame,
    province: pd.DataFrame,
    official: pd.DataFrame,
    *,
    official_total: int,
    exclude_anomalies: bool,
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_basis: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
    float,
    float,
    pd.DataFrame,
]:
    low, high, low_threshold, high_threshold = robust_anomaly_flags(
        national["raw_app_total"]
    )
    national = national.copy()
    national["is_anomalous_low"] = low
    national["is_anomalous_high"] = high
    national["is_excluded_from_reference"] = (
        low | high if exclude_anomalies else False
    )
    national["utc_hour"] = national["utc_time"].dt.hour.astype("int8")
    national["local_hour"] = national["local_time"].dt.hour.astype("int8")

    cadence_minutes = infer_cadence_minutes(national["utc_time"])
    if date_basis == "local":
        expected_utc_start = start - pd.Timedelta(hours=8)
        expected_utc_end = end - pd.Timedelta(hours=8)
    else:
        expected_utc_start = start
        expected_utc_end = end
    expected_times = pd.date_range(
        expected_utc_start,
        expected_utc_end - pd.Timedelta(minutes=cadence_minutes),
        freq=f"{cadence_minutes}min",
    )
    available_times = set(national["utc_time"])
    missing_times = pd.DataFrame(
        {"utc_time": [value for value in expected_times if value not in available_times]}
    )
    if not missing_times.empty:
        missing_times["local_time"] = missing_times["utc_time"] + pd.Timedelta(
            hours=8
        )

    valid_times = set(
        national.loc[
            ~national["is_excluded_from_reference"], "utc_time"
        ].tolist()
    )
    province = province.copy()
    province["is_valid_reference_time"] = province["utc_time"].isin(valid_times)

    expected_index = pd.MultiIndex.from_product(
        [expected_times, official["province"].tolist()],
        names=["utc_time", "province"],
    )
    present_index = pd.MultiIndex.from_frame(province[["utc_time", "province"]])
    missing_province = expected_index.difference(present_index).to_frame(index=False)
    if not missing_province.empty:
        missing_province["local_time"] = missing_province["utc_time"] + pd.Timedelta(
            hours=8
        )

    reference = (
        province.loc[province["is_valid_reference_time"]]
        .groupby("province", as_index=False, observed=True)
        .agg(
            reference_app_count=("raw_app_count", "median"),
            valid_reference_timestamps=("utc_time", "nunique"),
        )
    )
    weights = official[
        ["province", "official_population_2018"]
    ].merge(reference, on="province", how="left", validate="one_to_one")
    if weights["reference_app_count"].isna().any():
        missing = weights.loc[
            weights["reference_app_count"].isna(), "province"
        ].tolist()
        raise ValueError(
            f"No valid reference App count for official provinces: {missing}"
        )
    if (weights["reference_app_count"] <= 0).any():
        bad = weights.loc[weights["reference_app_count"] <= 0, "province"].tolist()
        raise ValueError(f"Non-positive province reference App counts: {bad}")
    weights["province_expansion_weight"] = (
        weights["official_population_2018"] / weights["reference_app_count"]
    )

    province_calibrated = province.merge(
        weights[
            [
                "province",
                "official_population_2018",
                "reference_app_count",
                "province_expansion_weight",
            ]
        ],
        on="province",
        how="left",
        validate="many_to_one",
    )
    province_calibrated["preliminary_population"] = (
        province_calibrated["raw_app_count"]
        * province_calibrated["province_expansion_weight"]
    )
    preliminary = (
        province_calibrated.groupby(
            ["utc_time", "local_time"], as_index=False
        )["preliminary_population"]
        .sum()
        .rename(
            columns={"preliminary_population": "preliminary_population_total"}
        )
    )
    timestamp_totals = national.merge(
        preliminary,
        on=["utc_time", "local_time"],
        how="left",
        validate="one_to_one",
    )
    if (
        timestamp_totals["preliminary_population_total"].isna().any()
        or (timestamp_totals["preliminary_population_total"] <= 0).any()
    ):
        raise ValueError("Invalid preliminary national population totals")
    timestamp_totals["national_timestamp_scaling_factor"] = (
        official_total / timestamp_totals["preliminary_population_total"]
    )
    timestamp_totals["final_estimated_population_total"] = (
        timestamp_totals["preliminary_population_total"]
        * timestamp_totals["national_timestamp_scaling_factor"]
    )
    timestamp_totals["target_population_total"] = official_total
    timestamp_totals["absolute_error"] = (
        timestamp_totals["final_estimated_population_total"]
        - timestamp_totals["target_population_total"]
    ).abs()
    timestamp_totals["relative_error"] = (
        timestamp_totals["absolute_error"]
        / timestamp_totals["target_population_total"]
    )

    province_calibrated = province_calibrated.merge(
        timestamp_totals[
            ["utc_time", "national_timestamp_scaling_factor"]
        ],
        on="utc_time",
        how="left",
        validate="many_to_one",
    )
    province_calibrated["estimated_population"] = (
        province_calibrated["preliminary_population"]
        * province_calibrated["national_timestamp_scaling_factor"]
    )
    return (
        weights,
        timestamp_totals,
        province_calibrated,
        missing_times,
        cadence_minutes,
        low_threshold,
        high_threshold,
        missing_province,
    )


def write_calibrated_outputs(
    input_paths: list[Path],
    *,
    output_dir: Path,
    date_basis: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    weights: pd.DataFrame,
    timestamp_totals: pd.DataFrame,
    overwrite: bool,
    batch_size: int,
    compression: str,
) -> tuple[list[Path], int]:
    weight_by_province = weights.set_index("province")
    reference_lookup = weight_by_province["reference_app_count"].to_dict()
    expansion_lookup = weight_by_province["province_expansion_weight"].to_dict()
    scale_lookup = timestamp_totals.set_index("utc_time")[
        "national_timestamp_scaling_factor"
    ].to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    written_rows = 0

    for input_path in input_paths:
        local_date_text = input_path.stem.removeprefix(INPUT_PREFIX)
        final_path = output_dir / f"{OUTPUT_PREFIX}{local_date_text}.parquet"
        if final_path.exists() and not overwrite:
            print(f"Skipping existing calibrated output: {final_path}")
            continue
        temp_path = final_path.with_name(f".{final_path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        writer: pq.ParquetWriter | None = None
        parquet_file = pq.ParquetFile(input_path)
        print(f"Calibration write pass reading: {input_path}")
        input_columns = [
            "time",
            "utc_date",
            "utc_time",
            "utc_hour",
            "local_date",
            "local_time",
            "local_hour",
            "lat",
            "lon",
            "province",
            "count",
            "official_population_2018",
            "source_file",
            "line_no",
            "province_assignment_method",
            "province_assignment_method_code",
        ]
        missing_input_columns = sorted(
            set(input_columns).difference(parquet_file.schema_arrow.names)
        )
        if missing_input_columns:
            raise ValueError(
                f"{input_path} is missing calibration columns: "
                f"{missing_input_columns}"
            )
        try:
            for batch_number, batch in enumerate(
                parquet_file.iter_batches(
                    batch_size=batch_size,
                    columns=input_columns,
                ),
                start=1,
            ):
                frame = batch.to_pandas()
                mask = selected_mask(
                    frame, date_basis=date_basis, start=start, end=end
                )
                frame = frame.loc[mask].copy()
                if frame.empty:
                    continue
                frame["utc_time"] = pd.to_datetime(frame["utc_time"])
                frame["local_time"] = pd.to_datetime(frame["local_time"])
                frame["app_count"] = pd.to_numeric(
                    frame["count"], errors="coerce"
                )
                frame["province_reference_app_count"] = frame["province"].map(
                    reference_lookup
                )
                frame["province_expansion_weight"] = frame["province"].map(
                    expansion_lookup
                )
                frame["preliminary_population"] = (
                    frame["app_count"] * frame["province_expansion_weight"]
                )
                frame["national_timestamp_scaling_factor"] = frame[
                    "utc_time"
                ].map(scale_lookup)
                frame["estimated_population"] = (
                    frame["preliminary_population"]
                    * frame["national_timestamp_scaling_factor"]
                )

                required_order = [
                    "time",
                    "utc_date",
                    "utc_time",
                    "utc_hour",
                    "local_date",
                    "local_time",
                    "local_hour",
                    "lat",
                    "lon",
                    "province",
                    "count",
                    "app_count",
                    "official_population_2018",
                    "province_reference_app_count",
                    "province_expansion_weight",
                    "preliminary_population",
                    "national_timestamp_scaling_factor",
                    "estimated_population",
                    "source_file",
                    "line_no",
                    "province_assignment_method",
                    "province_assignment_method_code",
                ]
                missing = [
                    column for column in required_order if column not in frame.columns
                ]
                if missing:
                    raise ValueError(
                        f"{input_path} is missing output-required columns: {missing}"
                    )
                frame = frame[required_order]
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path, table.schema, compression=compression
                    )
                writer.write_table(table)
                written_rows += len(frame)
                print(
                    f"  write batch {batch_number}: rows={len(frame):,}; "
                    f"cumulative={written_rows:,}"
                )
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"No selected calibration rows from {input_path}")
        temp_path.replace(final_path)
        created.append(final_path)
        print(f"Created: {final_path}")
    return created, written_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate App population with fixed province expansion weights and "
            "a time-varying national scaling factor."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=PROVINCE_ASSIGNED_DIR)
    parser.add_argument("--output-dir", type=Path, default=CALIBRATED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument(
        "--spatial-quality-gate",
        type=Path,
        default=DIAGNOSTICS_DIR / "spatial_assignment_quality_gate.csv",
        help=(
            "Formal calibration is refused unless this full-range spatial "
            "quality gate reports quality_gate_pass=True."
        ),
    )
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    add_date_arguments(parser, default_basis="local")
    parser.add_argument(
        "--reference-method",
        choices=("median",),
        default="median",
    )
    parser.add_argument(
        "--exclude-anomalies",
        action="store_true",
        help=(
            "Exclude robustly flagged national low/high timestamps from the "
            "province reference median. Default reports flags without excluding."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    if not args.spatial_quality_gate.exists():
        raise FileNotFoundError(
            "Spatial assignment quality gate is required before formal "
            f"calibration: {args.spatial_quality_gate}"
        )
    gate = pd.read_csv(args.spatial_quality_gate)
    required_gate_columns = {
        "quality_gate_pass",
        "in_china_unmatched_share",
        "final_unmatched_app_count_share",
        "maximum_allowed_final_unmatched_share",
    }
    if len(gate) != 1 or not required_gate_columns.issubset(gate.columns):
        raise ValueError(
            f"Invalid spatial quality gate: {args.spatial_quality_gate}"
        )
    gate_pass = (
        str(gate.iloc[0]["quality_gate_pass"])
        .strip()
        .casefold()
        in {"true", "1", "yes"}
    )
    if not gate_pass:
        final_share = float(
            gate.iloc[0]["final_unmatched_app_count_share"]
        )
        maximum = float(
            gate.iloc[0]["maximum_allowed_final_unmatched_share"]
        )
        raise RuntimeError(
            "Formal calibration blocked by spatial quality gate: final "
            f"unmatched App count share={final_share:.10%}, "
            f"maximum={maximum:.2%}."
        )
    if not np.isclose(
        float(gate.iloc[0]["final_unmatched_app_count_share"]),
        float(gate.iloc[0]["in_china_unmatched_share"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Spatial quality gate does not use the China-only unmatched "
            "share"
        )
    official, official_total, official_path = read_and_validate_official_population(
        args.official_population, args.mapping
    )
    local_dates = selected_input_local_dates(
        args.start_date, args.end_date, args.date_basis
    )
    input_paths = [
        args.input_dir / f"{INPUT_PREFIX}{local_date.isoformat()}.parquet"
        for local_date in local_dates
    ]
    missing_inputs = [path for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Selected province-assigned inputs are missing: "
            + ", ".join(map(str, missing_inputs))
        )

    start = pd.Timestamp(datetime.combine(args.start_date, time.min))
    end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    print("Province population calibration")
    print(f"  input: {args.input_dir.resolve()}")
    print(f"  output: {args.output_dir.resolve()}")
    print(f"  date basis: {args.date_basis}")
    print(f"  selected dates: {args.start_date} to {args.end_date}")
    print(f"  reference method: {args.reference_method}")
    print(f"  anomaly exclusion enabled: {args.exclude_anomalies}")
    print(f"  official population file: {official_path}")
    print(f"  direct 34-region target total: {official_total:,}")

    national, province, scan_stats = scan_reference_counts(
        input_paths,
        date_basis=args.date_basis,
        start=start,
        end=end,
        batch_size=args.batch_size,
    )
    if scan_stats["unmatched_records"] or scan_stats["unmatched_app_count"]:
        raise RuntimeError(
            "Formal province-assigned inputs still contain unmatched or "
            "outside-China records. Rebuild them with "
            "05_apply_final_coordinate_assignment.py before calibration. "
            f"unmatched_records={scan_stats['unmatched_records']:,}, "
            f"unmatched_app_count={scan_stats['unmatched_app_count']:,}"
        )
    (
        weights,
        timestamp_totals,
        province_calibrated,
        missing_times,
        cadence_minutes,
        low_threshold,
        high_threshold,
        missing_province,
    ) = build_diagnostics_and_weights(
        national,
        province,
        official,
        official_total=official_total,
        exclude_anomalies=args.exclude_anomalies,
        start=start,
        end=end,
        date_basis=args.date_basis,
    )

    created, written_rows = write_calibrated_outputs(
        input_paths,
        output_dir=args.output_dir,
        date_basis=args.date_basis,
        start=start,
        end=end,
        weights=weights,
        timestamp_totals=timestamp_totals,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        compression=args.compression,
    )

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.diagnostics_dir / "province_calibration_weights.csv"
    hourly_path = args.diagnostics_dir / "hourly_calibration_totals.csv"
    province_path = (
        args.diagnostics_dir / "province_hourly_calibrated_totals.csv"
    )
    national_path = args.diagnostics_dir / "national_app_count_diagnostics.csv"
    province_app_path = args.diagnostics_dir / "province_app_count_diagnostics.csv"
    missing_time_path = args.diagnostics_dir / "calibration_missing_times.csv"
    missing_province_path = (
        args.diagnostics_dir / "calibration_missing_province_times.csv"
    )
    summary_path = args.diagnostics_dir / "calibration_summary.md"
    diagnostic_paths = [
        weights_path,
        hourly_path,
        province_path,
        national_path,
        province_app_path,
        missing_time_path,
        missing_province_path,
        summary_path,
    ]
    existing = [path for path in diagnostic_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Calibration diagnostics exist; rerun with --overwrite: "
            + ", ".join(map(str, existing))
        )

    weights.rename(
        columns={
            "reference_app_count": "reference_app_count",
            "province_expansion_weight": "province_expansion_weight",
        }
    )[
        [
            "province",
            "official_population_2018",
            "reference_app_count",
            "province_expansion_weight",
            "valid_reference_hours",
        ]
    ].to_csv(weights_path, index=False, encoding="utf-8-sig")
    hourly[
        [
            "utc_time",
            "local_time",
            "raw_app_total",
            "preliminary_population_total",
            "national_scaling_factor",
            "final_estimated_population_total",
            "target_population_total",
            "relative_error",
        ]
    ].to_csv(hourly_path, index=False, encoding="utf-8-sig")
    province_calibrated[
        [
            "utc_time",
            "local_time",
            "province",
            "raw_app_count",
            "preliminary_population",
            "estimated_population",
        ]
    ].to_csv(province_path, index=False, encoding="utf-8-sig")
    hourly[
        [
            "utc_time",
            "local_time",
            "utc_hour",
            "local_hour",
            "raw_app_total",
            "is_anomalous_low",
            "is_anomalous_high",
            "is_excluded_from_reference",
        ]
    ].to_csv(national_path, index=False, encoding="utf-8-sig")
    province_calibrated[
        [
            "utc_time",
            "local_time",
            "province",
            "raw_app_count",
            "is_valid_reference_time",
        ]
    ].to_csv(province_app_path, index=False, encoding="utf-8-sig")
    missing_times.to_csv(missing_time_path, index=False, encoding="utf-8-sig")
    missing_province.to_csv(
        missing_province_path, index=False, encoding="utf-8-sig"
    )

    coverage_path = DIAGNOSTICS_DIR / "population_time_coverage.csv"
    first_local_day_complete = "unknown"
    first_local_day_missing = "unknown"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path, dtype={"local_date": "string"})
        row = coverage.loc[coverage["local_date"] == "2018-02-01"]
        if len(row):
            first_local_day_complete = str(
                bool(row.iloc[0]["is_complete_local_day"])
            )
            first_local_day_missing = str(row.iloc[0]["missing_hours"])

    assignment_path = args.diagnostics_dir / "province_assignment_summary.csv"
    assignment_line = "Province assignment summary not found."
    if assignment_path.exists():
        assignment = pd.read_csv(assignment_path)
        if len(assignment):
            assignment_line = (
                f"Unmatched records: {int(assignment.iloc[0]['unmatched_records']):,}; "
                f"unmatched unique coordinates: "
                f"{int(assignment.iloc[0]['unmatched_unique_coordinates']):,}; "
                f"unmatched App count: "
                f"{int(assignment.iloc[0]['unmatched_app_count']):,}; "
                f"share: {float(assignment.iloc[0]['unmatched_count_share']):.8%}."
            )

    max_error = float(hourly["relative_error"].max())
    summary = f"""# Province population calibration summary

- Date selection: {args.start_date} to {args.end_date} (`{args.date_basis}` basis).
- Exact UTC time range used: {national["utc_time"].min()} to {national["utc_time"].max()}.
- Source observation cadence: {cadence_minutes} minutes. Formulas are applied at every source timestamp; requested diagnostic filenames retain “hourly” for compatibility.
- Province reference App count: median across valid source timestamps in the selected range.
- Anomaly rule: national App totals below {low_threshold:.6f} or above {high_threshold:.6f} are flagged using median ± 6 × 1.4826 × MAD.
- Anomalous timestamps excluded from reference: {args.exclude_anomalies}.
- Missing selected source timestamps: {len(missing_times)}.
- Missing province-timestamp combinations: {len(missing_province)}. Missing observations are not converted to zero.
- Official population file: `{official_path}`.
- Official 34-region direct sum: {official_total:,} persons.
- Province expansion weights are fixed; national scaling factors vary by source timestamp.
- Maximum national calibration relative error: {max_error:.12g}.
- Selected input records: {scan_stats["selected_records"]:,}.
- Selected unmatched records: {scan_stats["unmatched_records"]:,}.
- {assignment_line}
- Beijing local 2018-02-01 complete: {first_local_day_complete}.
- Beijing local 2018-02-01 missing hours: {first_local_day_missing}.
- Calibrated row outputs created: {len(created)} files, {written_rows:,} records.
"""
    summary_path.write_text(summary, encoding="utf-8")

    for path in diagnostic_paths:
        print(f"Created: {path}")
    print("Province calibration weight summary:")
    print(
        weights[
            [
                "province",
                "reference_app_count",
                "province_expansion_weight",
                "valid_reference_hours",
            ]
        ].to_string(index=False)
    )
    print(f"Maximum national relative error: {max_error:.12g}")


if __name__ == "__main__":
    # Keep this filename as the stable public entry point while the formal
    # runner owns the national-boundary gate, timestamp terminology, and full
    # diagnostic suite. The functions above remain the reusable streaming core.
    import runpy

    runpy.run_path(
        str(
            Path(__file__).resolve().with_name(
                "02_run_formal_full_period_calibration.py"
            )
        ),
        run_name="__main__",
    )
