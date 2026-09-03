from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    EXPOSURE_COLUMN,
    HOURLY_ANALYSIS_DIR,
    HOURLY_EXPOSURE_DIR,
    POPULATION_COLUMN,
    PW_PM25_COLUMN,
    add_grid_center_lonlat,
    all_chunyun_dates,
    assign_period,
    atomic_csv,
    atomic_parquet,
    load_analysis_config,
    numeric_finite,
    read_completeness,
    require_columns,
    require_file,
    stage_log,
)


REQUIRED_COLUMNS = [
    "local_date",
    "local_hour",
    "local_datetime",
    "grid_id",
    "grid_x",
    "grid_y",
    "grid_center_x",
    "grid_center_y",
    POPULATION_COLUMN,
    EXPOSURE_COLUMN,
    PW_PM25_COLUMN,
    "sample_count",
    "expected_sample_count",
    "coverage_ratio",
    "covered_minutes",
    "first_observation_time",
    "last_observation_time",
    "is_complete_hour",
    "aggregation_method",
]


def parse_args() -> argparse.Namespace:
    config = load_analysis_config()["coverage"]
    parser = argparse.ArgumentParser(
        description=(
            "Validate and standardize formal Beijing-hour 10 km exposure "
            "files for the downstream Chunyun workflow."
        )
    )
    parser.add_argument("--start-date", default="2018-02-01")
    parser.add_argument("--end-date", default="2018-03-12")
    parser.add_argument(
        "--minimum-hour-coverage-ratio",
        type=float,
        default=config["minimum_hour_coverage_ratio_default"],
    )
    parser.add_argument(
        "--include-incomplete-hours",
        action=argparse.BooleanOptionalAction,
        default=config["include_incomplete_hours_default"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_dates(start: str, end: str) -> list[str]:
    allowed = all_chunyun_dates()
    dates = [day for day in allowed if start <= day <= end]
    if not dates:
        raise ValueError(
            f"No configured Chunyun dates selected by {start} to {end}"
        )
    if start < allowed[0] or end > allowed[-1]:
        raise ValueError(
            f"Requested dates must be within {allowed[0]} to {allowed[-1]}"
        )
    return dates


def prepare_day(
    frame: pd.DataFrame,
    day: str,
    *,
    minimum_coverage: float,
    include_incomplete: bool,
    date_completeness: dict,
) -> pd.DataFrame:
    require_columns(frame, REQUIRED_COLUMNS, f"hourly input {day}")
    frame = frame.copy()
    frame["local_date"] = frame["local_date"].astype(str)
    observed_dates = set(frame["local_date"].dropna().unique())
    if observed_dates != {day}:
        raise ValueError(
            f"{day} input contains unexpected local_date values: "
            f"{sorted(observed_dates)}"
        )
    frame["local_datetime"] = pd.to_datetime(
        frame["local_datetime"], errors="raise"
    )
    timestamp_dates = frame["local_datetime"].dt.strftime("%Y-%m-%d")
    if not timestamp_dates.eq(day).all():
        raise ValueError(
            f"{day} input contains local_datetime values outside that "
            "Beijing date"
        )
    supplied_hours = pd.to_numeric(
        frame["local_hour"], errors="coerce"
    )
    if not supplied_hours.eq(frame["local_datetime"].dt.hour).all():
        raise ValueError(
            f"local_hour disagrees with local_datetime for {day}"
        )
    duplicate_count = int(
        frame.duplicated(["local_datetime", "grid_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"{day} contains {duplicate_count:,} duplicate "
            "(local_datetime, grid_id) rows"
        )
    for column in [
        POPULATION_COLUMN,
        EXPOSURE_COLUMN,
        PW_PM25_COLUMN,
        "coverage_ratio",
        "covered_minutes",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    numeric_finite(
        frame,
        [
            POPULATION_COLUMN,
            EXPOSURE_COLUMN,
            PW_PM25_COLUMN,
            "coverage_ratio",
            "covered_minutes",
        ],
    )
    if (frame[[POPULATION_COLUMN, EXPOSURE_COLUMN]] < 0).any().any():
        raise ValueError(f"Negative formal population/exposure in {day}")
    ratio_expected = np.divide(
        frame[EXPOSURE_COLUMN].to_numpy(dtype="float64"),
        frame[POPULATION_COLUMN].to_numpy(dtype="float64"),
        out=np.full(len(frame), np.nan),
        where=frame[POPULATION_COLUMN].to_numpy(dtype="float64") > 0,
    )
    supplied = frame[PW_PM25_COLUMN].to_numpy(dtype="float64")
    finite = np.isfinite(ratio_expected) & np.isfinite(supplied)
    if finite.any() and not np.allclose(
        supplied[finite], ratio_expected[finite], rtol=1e-8, atol=1e-10
    ):
        raise ValueError(
            f"{PW_PM25_COLUMN} is not {EXPOSURE_COLUMN}/{POPULATION_COLUMN} "
            f"for {day}"
        )
    keep = frame["coverage_ratio"].ge(minimum_coverage)
    if not include_incomplete:
        keep &= frame["is_complete_hour"].astype(bool)
    frame["included_by_hour_coverage_rule"] = keep
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError(f"No hourly-grid rows remain after filtering {day}")

    frame = add_grid_center_lonlat(frame)
    frame["period"] = assign_period(day)
    frame["local_date_is_complete"] = bool(
        date_completeness["is_complete_local_date"]
    )
    frame["local_date_coverage_ratio"] = float(
        date_completeness["coverage_ratio"]
    )
    frame["minimum_hour_coverage_ratio_used"] = minimum_coverage
    frame["included_incomplete_hours"] = include_incomplete

    # Compatibility aliases used by the legacy Chunyun scripts. These now
    # represent formal calibrated population and exposure, not raw App count.
    frame["date"] = frame["local_date"]
    frame["hour"] = frame["local_hour"]
    frame["total_count"] = frame[POPULATION_COLUMN]
    frame["total_exposure"] = frame[EXPOSURE_COLUMN]
    frame["weighted_pm25"] = frame[PW_PM25_COLUMN]
    return frame.sort_values(["local_datetime", "grid_id"]).reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.minimum_hour_coverage_ratio <= 1:
        raise ValueError("--minimum-hour-coverage-ratio must be in [0, 1]")
    completeness = read_completeness().set_index("local_date")
    manifest_rows: list[dict] = []

    with stage_log("05_01_build_hourly_analysis_dataset"):
        for day in selected_dates(args.start_date, args.end_date):
            input_path = (
                HOURLY_EXPOSURE_DIR
                / f"hourly_exposure_grid_{day}.parquet"
            )
            output_path = (
                HOURLY_ANALYSIS_DIR / f"hourly_analysis_{day}.parquet"
            )
            require_file(input_path)
            if day not in completeness.index:
                raise ValueError(
                    f"Missing local-date completeness record for {day}"
                )
            if output_path.exists() and not args.overwrite:
                print(f"SKIP existing: {output_path}")
                continue

            print(f"READ {input_path}")
            raw = pd.read_parquet(input_path)
            result = prepare_day(
                raw,
                day,
                minimum_coverage=args.minimum_hour_coverage_ratio,
                include_incomplete=args.include_incomplete_hours,
                date_completeness=completeness.loc[day].to_dict(),
            )
            atomic_parquet(result, output_path)
            hours = sorted(result["local_hour"].dropna().unique().tolist())
            manifest_rows.append(
                {
                    "local_date": day,
                    "period": assign_period(day),
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "input_rows": len(raw),
                    "output_rows": len(result),
                    "valid_hour_count": len(hours),
                    "valid_hours": ",".join(map(str, hours)),
                    "local_date_is_complete": bool(
                        completeness.loc[day, "is_complete_local_date"]
                    ),
                    "local_date_coverage_ratio": float(
                        completeness.loc[day, "coverage_ratio"]
                    ),
                }
            )
            print(f"WRITE {output_path} rows={len(result):,}")

        if manifest_rows:
            manifest_path = (
                HOURLY_ANALYSIS_DIR / "hourly_analysis_manifest.csv"
            )
            manifest = pd.DataFrame(manifest_rows)
            if manifest_path.exists() and not args.overwrite:
                old = pd.read_csv(manifest_path)
                manifest = pd.concat([old, manifest], ignore_index=True)
                manifest = manifest.drop_duplicates(
                    "local_date", keep="last"
                ).sort_values("local_date")
            atomic_csv(manifest, manifest_path)
            print(f"WRITE {manifest_path}")


if __name__ == "__main__":
    main()
