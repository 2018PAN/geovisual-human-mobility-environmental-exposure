from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_DIR = SCRIPT_DIR.parent / "00_common"
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


CORE_PATH = SCRIPT_DIR / "01_calibrate_province_population.py"
CORE_SPEC = importlib.util.spec_from_file_location(
    "formal_calibration_core", CORE_PATH
)
if CORE_SPEC is None or CORE_SPEC.loader is None:
    raise ImportError(f"Cannot load calibration core: {CORE_PATH}")
core = importlib.util.module_from_spec(CORE_SPEC)
CORE_SPEC.loader.exec_module(core)


INPUT_PREFIX = "population_province_assigned_"
OUTPUT_PREFIX = "population_calibrated_"
EXPECTED_OFFICIAL_TOTAL = 1_428_272_732


def parse_boolean(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def require_china_only_gate(path: Path) -> pd.Series:
    gate = pd.read_csv(path)
    required = {
        "quality_gate_pass",
        "in_china_unmatched_share",
        "final_unmatched_app_count_share",
        "inside_china_records",
        "inside_china_app_count",
    }
    missing = sorted(required.difference(gate.columns))
    if len(gate) != 1 or missing:
        raise ValueError(
            f"Invalid China-only spatial gate; rows={len(gate)}, missing={missing}"
        )
    row = gate.iloc[0]
    if not parse_boolean(row["quality_gate_pass"]):
        raise RuntimeError("China-only spatial gate did not pass")
    if not np.isclose(
        float(row["in_china_unmatched_share"]),
        float(row["final_unmatched_app_count_share"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Spatial gate denominator is not China-only")
    if float(row["in_china_unmatched_share"]) > 0.01:
        raise RuntimeError("China-only unmatched share exceeds 1%")
    return row


def scaling_distribution(values: pd.Series) -> pd.DataFrame:
    values = values.astype(float)
    return pd.DataFrame(
        [
            {
                "timestamp_count": len(values),
                "minimum": values.min(),
                "p01": values.quantile(0.01),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "mean": values.mean(),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "maximum": values.max(),
                "standard_deviation": values.std(),
            }
        ]
    )


def missing_intervals(
    expected: pd.DatetimeIndex,
    observed: pd.DatetimeIndex,
    cadence_minutes: int,
) -> tuple[pd.DatetimeIndex, str]:
    missing = expected.difference(observed)
    if not len(missing):
        return missing, ""
    groups: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    group_start = missing[0]
    previous = missing[0]
    cadence = pd.Timedelta(minutes=cadence_minutes)
    for value in missing[1:]:
        if value - previous != cadence:
            groups.append((group_start, previous))
            group_start = value
        previous = value
    groups.append((group_start, previous))
    text = " | ".join(
        f"{left:%H:%M}-{right:%H:%M}" for left, right in groups
    )
    return missing, text


def build_local_completeness(
    timestamp_totals: pd.DataFrame,
    *,
    start_date,
    end_date,
    cadence_minutes: int,
) -> pd.DataFrame:
    local_times = pd.DatetimeIndex(
        pd.to_datetime(timestamp_totals["local_time"])
    )
    rows = []
    for day in pd.date_range(start_date, end_date, freq="D"):
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1)
        observed = pd.DatetimeIndex(
            local_times[(local_times >= day_start) & (local_times < day_end)]
        ).drop_duplicates().sort_values()
        expected = pd.date_range(
            day_start,
            day_end - pd.Timedelta(minutes=cadence_minutes),
            freq=f"{cadence_minutes}min",
        )
        missing, intervals = missing_intervals(
            expected, observed, cadence_minutes
        )
        rows.append(
            {
                "local_date": day_start.strftime("%Y-%m-%d"),
                "expected_five_minute_timestamps": len(expected),
                "available_five_minute_timestamps": len(observed),
                "missing_five_minute_timestamps": len(missing),
                "coverage_ratio": len(observed) / len(expected),
                "first_observation_time": (
                    observed.min() if len(observed) else pd.NaT
                ),
                "last_observation_time": (
                    observed.max() if len(observed) else pd.NaT
                ),
                "missing_intervals": intervals,
                "is_complete_local_date": len(missing) == 0,
            }
        )
    return pd.DataFrame(rows)


def add_weight_flags(weights: pd.DataFrame) -> pd.DataFrame:
    result = weights.copy()
    values = result["province_expansion_weight"].astype(float)
    q10 = float(values.quantile(0.10))
    q90 = float(values.quantile(0.90))
    log_values = np.log(values)
    q1 = float(log_values.quantile(0.25))
    q3 = float(log_values.quantile(0.75))
    iqr = q3 - q1
    result["weight_size_flag"] = np.select(
        [
            values <= q10,
            values >= q90,
        ],
        ["small_bottom_10pct", "large_top_10pct"],
        default="middle_80pct",
    )
    result["is_log_iqr_outlier"] = (
        (log_values < q1 - 1.5 * iqr)
        | (log_values > q3 + 1.5 * iqr)
    )
    return result


def format_weight_rows(frame: pd.DataFrame) -> str:
    return ", ".join(
        f"{row.province}={row.province_expansion_weight:.6g}"
        for row in frame.itertuples(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal full-period five-minute province population "
            "calibration after the national-boundary-first quality gate."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=PROVINCE_ASSIGNED_DIR)
    parser.add_argument("--output-dir", type=Path, default=CALIBRATED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument(
        "--spatial-quality-gate",
        type=Path,
        default=DIAGNOSTICS_DIR / "spatial_assignment_quality_gate.csv",
    )
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--compression", default="zstd")
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    if args.date_basis != "local":
        raise ValueError("Formal calibration requires --date-basis local")
    gate = require_china_only_gate(args.spatial_quality_gate)
    official, official_total, official_path = read_and_validate_official_population(
        args.official_population, args.mapping
    )
    if official_total != EXPECTED_OFFICIAL_TOTAL:
        raise ValueError(
            f"Official total {official_total:,} differs from confirmed "
            f"{EXPECTED_OFFICIAL_TOTAL:,}"
        )

    local_dates = selected_input_local_dates(
        args.start_date, args.end_date, args.date_basis
    )
    input_paths = [
        args.input_dir / f"{INPUT_PREFIX}{day.isoformat()}.parquet"
        for day in local_dates
    ]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Formal province partitions missing: "
            + ", ".join(map(str, missing))
        )

    start = pd.Timestamp(datetime.combine(args.start_date, time.min))
    end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    print("Formal full-period five-minute population calibration")
    print(f"  Beijing dates: {args.start_date} to {args.end_date}")
    print(f"  official file: {official_path}")
    print(f"  official target: {official_total:,}")
    print("  reference: full-period province timestamp medians")
    print("  anomaly timestamps: report only; do not exclude")
    print("  obsolete pre-national-filter unmatched outputs: not used")

    national, province, scan_stats = core.scan_reference_counts(
        input_paths,
        date_basis="local",
        start=start,
        end=end,
        batch_size=args.batch_size,
    )
    if scan_stats["unmatched_records"] or scan_stats["unmatched_app_count"]:
        raise RuntimeError(
            "Formal inputs contain unmatched records: "
            f"records={scan_stats['unmatched_records']:,}, "
            f"count={scan_stats['unmatched_app_count']:,}"
        )
    if scan_stats["selected_records"] != int(gate["inside_china_records"]):
        raise RuntimeError(
            "Formal input row count differs from the passing China-only gate"
        )
    if scan_stats["selected_app_count"] != int(gate["inside_china_app_count"]):
        raise RuntimeError(
            "Formal input App count differs from the passing China-only gate"
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
    ) = core.build_diagnostics_and_weights(
        national,
        province,
        official,
        official_total=official_total,
        exclude_anomalies=False,
        start=start,
        end=end,
        date_basis="local",
    )
    if cadence_minutes != 5:
        raise RuntimeError(
            f"Expected five-minute cadence, inferred {cadence_minutes} minutes"
        )

    created, written_rows = core.write_calibrated_outputs(
        input_paths,
        output_dir=args.output_dir,
        date_basis="local",
        start=start,
        end=end,
        weights=weights,
        timestamp_totals=timestamp_totals,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        compression=args.compression,
    )
    if written_rows != scan_stats["selected_records"]:
        raise RuntimeError(
            f"Written rows {written_rows:,} differ from selected "
            f"{scan_stats['selected_records']:,}"
        )

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "weights": args.diagnostics_dir / "province_calibration_weights.csv",
        "timestamps": args.diagnostics_dir / "timestamp_calibration_totals.csv",
        "timestamps_compat": (
            args.diagnostics_dir / "hourly_calibration_totals.csv"
        ),
        "province_timestamps": (
            args.diagnostics_dir
            / "province_timestamp_calibrated_totals.csv"
        ),
        "province_timestamps_compat": (
            args.diagnostics_dir / "province_hourly_calibrated_totals.csv"
        ),
        "national_app": (
            args.diagnostics_dir / "national_app_count_diagnostics.csv"
        ),
        "province_app": (
            args.diagnostics_dir / "province_app_count_diagnostics.csv"
        ),
        "missing_times": (
            args.diagnostics_dir / "calibration_missing_times.csv"
        ),
        "missing_province": (
            args.diagnostics_dir
            / "calibration_missing_province_times.csv"
        ),
        "scale_distribution": (
            args.diagnostics_dir
            / "national_timestamp_scaling_factor_distribution.csv"
        ),
        "province_series": (
            args.diagnostics_dir
            / "province_calibrated_timeseries_summary.csv"
        ),
        "completeness": (
            args.diagnostics_dir / "calibrated_local_date_completeness.csv"
        ),
        "validation_day": (
            args.diagnostics_dir / "calibration_validation_2018-02-02.csv"
        ),
        "validation_day_province": (
            args.diagnostics_dir
            / "calibration_validation_2018-02-02_by_province.csv"
        ),
        "summary": args.diagnostics_dir / "calibration_summary.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Calibration diagnostics already exist; use --overwrite: "
            + ", ".join(map(str, existing))
        )

    weights = add_weight_flags(weights)
    weights_columns = [
        "province",
        "official_population_2018",
        "reference_app_count",
        "province_expansion_weight",
        "valid_reference_timestamps",
        "weight_size_flag",
        "is_log_iqr_outlier",
    ]
    weights[weights_columns].to_csv(
        paths["weights"], index=False, encoding="utf-8-sig"
    )
    timestamp_columns = [
        "utc_time",
        "local_time",
        "raw_app_total",
        "preliminary_population_total",
        "national_timestamp_scaling_factor",
        "final_estimated_population_total",
        "target_population_total",
        "absolute_error",
        "relative_error",
    ]
    for key in ("timestamps", "timestamps_compat"):
        timestamp_totals[timestamp_columns].to_csv(
            paths[key], index=False, encoding="utf-8-sig"
        )
    province_timestamp_columns = [
        "utc_time",
        "local_time",
        "province",
        "raw_app_count",
        "preliminary_population",
        "national_timestamp_scaling_factor",
        "estimated_population",
    ]
    for key in ("province_timestamps", "province_timestamps_compat"):
        province_calibrated[province_timestamp_columns].to_csv(
            paths[key], index=False, encoding="utf-8-sig"
        )
    timestamp_totals[
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
    ].to_csv(paths["national_app"], index=False, encoding="utf-8-sig")
    province_calibrated[
        [
            "utc_time",
            "local_time",
            "province",
            "raw_app_count",
            "is_valid_reference_time",
        ]
    ].to_csv(paths["province_app"], index=False, encoding="utf-8-sig")
    missing_times.to_csv(
        paths["missing_times"], index=False, encoding="utf-8-sig"
    )
    missing_province.to_csv(
        paths["missing_province"], index=False, encoding="utf-8-sig"
    )
    scaling_distribution(
        timestamp_totals["national_timestamp_scaling_factor"]
    ).to_csv(
        paths["scale_distribution"], index=False, encoding="utf-8-sig"
    )

    province_series = (
        province_calibrated.groupby("province", as_index=False, observed=True)
        .agg(
            timestamp_count=("utc_time", "nunique"),
            raw_app_count_min=("raw_app_count", "min"),
            raw_app_count_mean=("raw_app_count", "mean"),
            raw_app_count_median=("raw_app_count", "median"),
            raw_app_count_max=("raw_app_count", "max"),
            estimated_population_min=("estimated_population", "min"),
            estimated_population_mean=("estimated_population", "mean"),
            estimated_population_median=("estimated_population", "median"),
            estimated_population_max=("estimated_population", "max"),
            estimated_population_std=("estimated_population", "std"),
        )
        .merge(
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
            validate="one_to_one",
        )
    )
    province_series.to_csv(
        paths["province_series"], index=False, encoding="utf-8-sig"
    )

    completeness = build_local_completeness(
        timestamp_totals,
        start_date=args.start_date,
        end_date=args.end_date,
        cadence_minutes=cadence_minutes,
    )
    completeness.to_csv(
        paths["completeness"], index=False, encoding="utf-8-sig"
    )
    validation_start = pd.Timestamp("2018-02-02")
    validation_end = validation_start + pd.Timedelta(days=1)
    validation_mask = (
        (timestamp_totals["local_time"] >= validation_start)
        & (timestamp_totals["local_time"] < validation_end)
    )
    timestamp_totals.loc[
        validation_mask, timestamp_columns
    ].to_csv(paths["validation_day"], index=False, encoding="utf-8-sig")
    validation_province_mask = (
        (province_calibrated["local_time"] >= validation_start)
        & (province_calibrated["local_time"] < validation_end)
    )
    province_calibrated.loc[
        validation_province_mask, province_timestamp_columns
    ].to_csv(
        paths["validation_day_province"],
        index=False,
        encoding="utf-8-sig",
    )

    first_day = completeness.loc[
        completeness["local_date"].eq("2018-02-01")
    ].iloc[0]
    detail_day = completeness.loc[
        completeness["local_date"].eq("2018-02-02")
    ].iloc[0]
    if bool(first_day["is_complete_local_date"]):
        raise RuntimeError("Beijing 2018-02-01 must be marked incomplete")
    if "00:00-07:55" not in str(first_day["missing_intervals"]):
        raise RuntimeError(
            "Unexpected Beijing 2018-02-01 missing interval: "
            f"{first_day['missing_intervals']}"
        )
    validation_timestamp_count = int(validation_mask.sum())
    if validation_timestamp_count != int(
        detail_day["available_five_minute_timestamps"]
    ):
        raise RuntimeError(
            "2018-02-02 detailed validation row count differs from its "
            "observed completeness count"
        )

    max_relative_error = float(timestamp_totals["relative_error"].max())
    mean_relative_error = float(timestamp_totals["relative_error"].mean())
    max_absolute_error = float(timestamp_totals["absolute_error"].max())
    mean_absolute_error = float(timestamp_totals["absolute_error"].mean())
    bottom_five = weights.nsmallest(
        5, "province_expansion_weight"
    )[["province", "province_expansion_weight"]]
    top_five = weights.nlargest(
        5, "province_expansion_weight"
    )[["province", "province_expansion_weight"]]
    robust_outliers = weights.loc[
        weights["is_log_iqr_outlier"],
        ["province", "province_expansion_weight"],
    ]
    summary = f"""# Formal full-period province population calibration

- Result: COMPLETE; PM2.5 matching was not run.
- Beijing local dates: {args.start_date} through {args.end_date}.
- Exact UTC timestamps: {national["utc_time"].min()} through {national["utc_time"].max()}.
- Source cadence: {cadence_minutes} minutes.
- Formal records require audited valid UTC time, the requested Beijing date, coverage by the 34-region national boundary, and a valid province assignment.
- Obsolete pre-national-filter unmatched outputs were not used.
- Province reference: median of each province's App-count sum across all available valid timestamps.
- Reference anomaly flags were reported but not excluded.
- Official 34-region target: {official_total:,}.
- Valid source timestamps: {len(timestamp_totals):,}.
- Missing requested timestamps: {len(missing_times):,}; no zero filling or interpolation was applied.
- Missing province-timestamp combinations: {len(missing_province):,}.
- Maximum national-total absolute error: {max_absolute_error:.12g}.
- Mean national-total absolute error: {mean_absolute_error:.12g}.
- Maximum national-total relative error: {max_relative_error:.12g}.
- Mean national-total relative error: {mean_relative_error:.12g}.
- Beijing 2018-02-01: incomplete, {int(first_day["available_five_minute_timestamps"])}/288 timestamps, missing `{first_day["missing_intervals"]}`.
- Beijing 2018-02-02: {"complete" if bool(detail_day["is_complete_local_date"]) else "incomplete"}, {int(detail_day["available_five_minute_timestamps"])}/288 timestamps, missing `{detail_day["missing_intervals"]}`.
- Smallest five weights: {format_weight_rows(bottom_five)}.
- Largest five weights: {format_weight_rows(top_five)}.
- Log-IQR weight outliers: {format_weight_rows(robust_outliers) or "none"}.
- Calibrated outputs: {len(created)} local-date files, {written_rows:,} records.
"""
    paths["summary"].write_text(summary, encoding="utf-8")
    for path in paths.values():
        print(f"Created: {path}")
    print("Formal province weights:")
    print(
        weights[
            [
                "province",
                "reference_app_count",
                "province_expansion_weight",
                "valid_reference_timestamps",
            ]
        ].to_string(index=False)
    )
    print(
        "National relative error: "
        f"maximum={max_relative_error:.12g}, "
        f"mean={mean_relative_error:.12g}"
    )


if __name__ == "__main__":
    main()
