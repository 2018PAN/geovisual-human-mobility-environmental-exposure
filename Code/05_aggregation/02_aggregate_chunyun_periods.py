from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    DAILY_ANALYSIS_DIR,
    EXPOSURE_COLUMN,
    GRID_COLUMNS,
    HOURLY_ANALYSIS_DIR,
    POPULATION_COLUMN,
    PW_PM25_COLUMN,
    all_chunyun_dates,
    atomic_parquet,
    load_analysis_config,
    period_dates,
    periods,
    require_columns,
    require_file,
    stage_log,
)


GRID_KEYS = GRID_COLUMNS


def parse_args() -> argparse.Namespace:
    coverage = load_analysis_config()["coverage"]
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate formal hourly 10 km data to legacy-compatible daily "
            "and Chunyun-period means."
        )
    )
    parser.add_argument(
        "--minimum-hour-coverage-ratio",
        type=float,
        default=coverage["minimum_hour_coverage_ratio_default"],
    )
    parser.add_argument(
        "--include-incomplete-hours",
        action=argparse.BooleanOptionalAction,
        default=coverage["include_incomplete_hours_default"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def aggregate_day(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    require_columns(
        frame,
        GRID_KEYS
        + [
            "local_date",
            "local_hour",
            POPULATION_COLUMN,
            EXPOSURE_COLUMN,
            PW_PM25_COLUMN,
            "coverage_ratio",
            "covered_minutes",
            "is_complete_hour",
            "local_date_is_complete",
            "local_date_coverage_ratio",
        ],
        f"hourly analysis {day}",
    )
    grouped = frame.groupby(GRID_KEYS, as_index=False, observed=True)
    daily = grouped.agg(
        daily_population=(POPULATION_COLUMN, "sum"),
        daily_exposure=(EXPOSURE_COLUMN, "sum"),
        daily_mean_pm25=(PW_PM25_COLUMN, "mean"),
        valid_hour_count=("local_hour", "nunique"),
        complete_hour_count=("is_complete_hour", "sum"),
        mean_hour_coverage_ratio=("coverage_ratio", "mean"),
        minimum_hour_coverage_ratio=("coverage_ratio", "min"),
        covered_minutes=("covered_minutes", "sum"),
    )
    daily["daily_weighted_pm25"] = np.divide(
        daily["daily_exposure"],
        daily["daily_population"],
        out=np.full(len(daily), np.nan),
        where=daily["daily_population"].to_numpy(dtype="float64") > 0,
    )
    daily["local_date"] = day
    daily["date"] = day
    daily["period"] = frame["period"].iloc[0]
    daily["local_date_is_complete"] = bool(
        frame["local_date_is_complete"].iloc[0]
    )
    daily["local_date_coverage_ratio"] = float(
        frame["local_date_coverage_ratio"].iloc[0]
    )
    # Legacy compatibility aliases; count means calibrated population.
    daily["daily_total_count"] = daily["daily_population"]
    daily["daily_total_exposure"] = daily["daily_exposure"]
    return daily


def aggregate_period(
    daily: pd.DataFrame, name: str, expected_days: int
) -> pd.DataFrame:
    grouped = daily.groupby(GRID_KEYS, as_index=False, observed=True)
    result = grouped.agg(
        **{
            f"{name}_mean_population": ("daily_population", "mean"),
            f"{name}_mean_exposure": ("daily_exposure", "mean"),
            f"{name}_mean_pm25": ("daily_mean_pm25", "mean"),
            f"{name}_weighted_pm25": ("daily_weighted_pm25", "mean"),
            f"{name}_valid_day_count": ("local_date", "nunique"),
            f"{name}_valid_hour_count": ("valid_hour_count", "sum"),
            f"{name}_complete_hour_count": (
                "complete_hour_count",
                "sum",
            ),
            f"{name}_mean_hour_coverage_ratio": (
                "mean_hour_coverage_ratio",
                "mean",
            ),
            f"{name}_minimum_hour_coverage_ratio": (
                "minimum_hour_coverage_ratio",
                "min",
            ),
            f"{name}_mean_local_date_coverage_ratio": (
                "local_date_coverage_ratio",
                "mean",
            ),
        }
    )
    result[f"{name}_expected_day_count"] = expected_days
    result[f"{name}_day_coverage_ratio"] = (
        result[f"{name}_valid_day_count"] / expected_days
    )
    result[f"{name}_expected_hour_count"] = expected_days * 24
    result[f"{name}_hour_coverage_ratio"] = (
        result[f"{name}_valid_hour_count"]
        / result[f"{name}_expected_hour_count"]
    )
    incomplete = (
        ~daily.groupby(GRID_KEYS, observed=True)[
            "local_date_is_complete"
        ].all()
    ).rename(f"{name}_contains_incomplete_local_date")
    result = result.merge(
        incomplete.reset_index(),
        on=GRID_KEYS,
        how="left",
        validate="one_to_one",
    )
    result[f"{name}_mean_count"] = result[
        f"{name}_mean_population"
    ]
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.minimum_hour_coverage_ratio <= 1:
        raise ValueError("--minimum-hour-coverage-ratio must be in [0, 1]")
    config = periods()
    period_outputs = {
        name: AGGREGATED_DIR / f"{name}_period_grid.parquet"
        for name in config
    }
    summary_path = AGGREGATED_DIR / "chunyun_period_grid_summary.parquet"

    with stage_log("05_02_aggregate_chunyun_periods"):
        daily_frames: list[pd.DataFrame] = []
        for day in all_chunyun_dates():
            input_path = HOURLY_ANALYSIS_DIR / f"hourly_analysis_{day}.parquet"
            output_path = DAILY_ANALYSIS_DIR / f"daily_grid_{day}.parquet"
            require_file(input_path)
            print(f"READ {input_path}")
            hourly = pd.read_parquet(input_path)
            keep = hourly["coverage_ratio"].ge(
                args.minimum_hour_coverage_ratio
            )
            if not args.include_incomplete_hours:
                keep &= hourly["is_complete_hour"].astype(bool)
            # Legacy daily aggregation excludes an hourly grid cell when its
            # exposure/concentration match is missing. Missing PM2.5 is never
            # filled with zero, and its population is not silently retained
            # in that grid-day denominator.
            finite_formal = (
                pd.to_numeric(
                    hourly[POPULATION_COLUMN], errors="coerce"
                ).notna()
                & pd.to_numeric(
                    hourly[EXPOSURE_COLUMN], errors="coerce"
                ).notna()
                & pd.to_numeric(
                    hourly[PW_PM25_COLUMN], errors="coerce"
                ).notna()
            )
            finite_formal &= np.isfinite(
                hourly[
                    [
                        POPULATION_COLUMN,
                        EXPOSURE_COLUMN,
                        PW_PM25_COLUMN,
                    ]
                ].to_numpy(dtype="float64")
            ).all(axis=1)
            removed_invalid = int((keep & ~finite_formal).sum())
            keep &= finite_formal
            hourly = hourly.loc[keep].copy()
            if hourly.empty:
                raise ValueError(
                    f"No rows remain after hour filtering for {day}"
                )
            print(
                f"  excluded hourly-grid rows with missing formal "
                f"exposure/PM2.5: {removed_invalid:,}"
            )
            daily = aggregate_day(hourly, day)
            if output_path.exists() and not args.overwrite:
                existing = pd.read_parquet(output_path)
                daily_frames.append(existing)
                print(f"SKIP existing and reuse: {output_path}")
            else:
                atomic_parquet(daily, output_path)
                daily_frames.append(daily)
                print(f"WRITE {output_path} rows={len(daily):,}")

        all_daily = pd.concat(daily_frames, ignore_index=True)
        period_frames: dict[str, pd.DataFrame] = {}
        for name, item in config.items():
            selected = all_daily.loc[
                all_daily["local_date"].isin(period_dates(name))
            ].copy()
            observed = set(selected["local_date"].unique())
            expected = set(period_dates(name))
            if observed != expected:
                raise ValueError(
                    f"{name} daily inputs differ from configured dates; "
                    f"missing={sorted(expected-observed)}, "
                    f"extra={sorted(observed-expected)}"
                )
            result = aggregate_period(
                selected, name, int(item["expected_days"])
            )
            path = period_outputs[name]
            if path.exists() and not args.overwrite:
                result = pd.read_parquet(path)
                print(f"SKIP existing and reuse: {path}")
            else:
                atomic_parquet(result, path)
                print(f"WRITE {path} rows={len(result):,}")
            period_frames[name] = result

        merged: pd.DataFrame | None = None
        for name in ["pre", "festival", "post"]:
            part = period_frames[name]
            merged = (
                part
                if merged is None
                else merged.merge(
                    part,
                    on=GRID_KEYS,
                    how="outer",
                    validate="one_to_one",
                )
            )
        assert merged is not None
        merged["valid_hour_count"] = merged[
            [
                "pre_valid_hour_count",
                "festival_valid_hour_count",
                "post_valid_hour_count",
            ]
        ].sum(axis=1, min_count=1)
        merged["complete_hour_count"] = merged[
            [
                "pre_complete_hour_count",
                "festival_complete_hour_count",
                "post_complete_hour_count",
            ]
        ].sum(axis=1, min_count=1)
        if summary_path.exists() and not args.overwrite:
            print(f"SKIP existing: {summary_path}")
        else:
            atomic_parquet(merged, summary_path)
            print(f"WRITE {summary_path} rows={len(merged):,}")


if __name__ == "__main__":
    main()
