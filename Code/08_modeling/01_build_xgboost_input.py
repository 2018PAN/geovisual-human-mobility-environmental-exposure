from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    DAILY_ANALYSIS_DIR,
    DECOMPOSITION_DIR,
    HOURLY_ANALYSIS_DIR,
    MODELING_DIR,
    WORKSPACE_ROOT,
    add_legacy_01_degree_keys,
    analysis_target_choices,
    atomic_csv,
    atomic_parquet,
    load_analysis_config,
    modeling_spec,
    period_dates,
    require_columns,
    require_file,
    select_comparisons,
    stage_log,
)


PERIOD_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
LEGACY_ROOT = WORKSPACE_ROOT / "Output_Chunyun_TimeCorrected"
WEATHER_PATH = (
    LEGACY_ROOT
    / "Analysis"
    / "weather"
    / "chunyun_weather_phase_means.parquet"
)
LEGACY_FESTIVAL_INPUT = (
    LEGACY_ROOT
    / "Analysis"
    / "xgboost"
    / "inputs"
    / "chunyun_xgboost_input_total_exposure_extended.parquet"
)
LEGACY_POST_INPUT = (
    LEGACY_ROOT
    / "Analysis"
    / "xgboost"
    / "inputs"
    / "chunyun_xgboost_input_post_festival_total_exposure.parquet"
)

STATIC_FEATURES = [
    "road_density_m_per_km2",
    "built_up_ratio",
    "nighttime_light_2018",
    "gdp_2018",
]
MIN_HOURS_PER_DAY = 18
MIN_VALID_DAYS = {"pre": 10, "festival": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build formal 10 km XGBoost inputs while preserving the legacy "
            "feature definitions and transformations."
        )
    )
    parser.add_argument(
        "--comparison",
        choices=["festival_pre", "post_festival", "all"],
        default="all",
    )
    parser.add_argument(
        "--analysis-target",
        choices=analysis_target_choices(),
        default="total_exposure",
        help=(
            "Formal analysis target. The default preserves the existing "
            "total-exposure output names."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(
    comparison: str, analysis_target: str
) -> tuple[Path, Path]:
    suffix = (
        comparison
        if analysis_target == "total_exposure"
        else f"{comparison}_{analysis_target}"
    )
    return (
        MODELING_DIR / "inputs" / f"xgboost_input_{suffix}.parquet",
        MODELING_DIR
        / "diagnostics"
        / f"xgboost_input_{suffix}_merge_diagnostics.csv",
    )


def read_phase_hourly(phase: str) -> pd.DataFrame:
    frames = []
    for day in period_dates(phase):
        path = HOURLY_ANALYSIS_DIR / f"hourly_analysis_{day}.parquet"
        require_file(path)
        frame = pd.read_parquet(
            path,
            columns=[
                "grid_id",
                "local_date",
                "local_hour",
                "hourly_population",
            ],
        )
        require_columns(
            frame,
            [
                "grid_id",
                "local_date",
                "local_hour",
                "hourly_population",
            ],
            path,
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def read_phase_daily(phase: str) -> pd.DataFrame:
    frames = []
    for day in period_dates(phase):
        path = DAILY_ANALYSIS_DIR / f"daily_grid_{day}.parquet"
        require_file(path)
        frame = pd.read_parquet(
            path,
            columns=["grid_id", "local_date", "daily_population"],
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def dynamic_features(
    phase: str, *, overwrite_cache: bool = False
) -> pd.DataFrame:
    if phase not in MIN_VALID_DAYS:
        raise ValueError(f"No legacy dynamic-feature definition for {phase}")
    cache_path = (
        MODELING_DIR
        / "inputs"
        / f"xgboost_dynamic_features_{phase}.parquet"
    )
    if cache_path.exists() and not overwrite_cache:
        print(f"REUSE dynamic-feature cache: {cache_path}")
        return pd.read_parquet(cache_path)
    prefix = phase
    minimum_days = MIN_VALID_DAYS[phase]
    daily = read_phase_daily(phase)
    daily["daily_population"] = pd.to_numeric(
        daily["daily_population"], errors="coerce"
    )
    daily = daily.loc[daily["daily_population"].ge(0)].copy()
    daily_group = daily.groupby("grid_id")["daily_population"]
    features = daily_group.agg(
        **{
            f"{prefix}_daily_mean_count_computed": "mean",
            f"{prefix}_daily_count_std": lambda values: values.std(
                ddof=1
            ),
            f"{prefix}_valid_daily_count_days": "count",
        }
    ).reset_index()
    features[f"{prefix}_count_cv"] = (
        features[f"{prefix}_daily_count_std"]
        / features[f"{prefix}_daily_mean_count_computed"].where(
            features[f"{prefix}_daily_mean_count_computed"] != 0
        )
    )
    features.loc[
        features[f"{prefix}_valid_daily_count_days"] < minimum_days,
        f"{prefix}_count_cv",
    ] = np.nan

    hourly = read_phase_hourly(phase)
    hourly["hourly_population"] = pd.to_numeric(
        hourly["hourly_population"], errors="coerce"
    )
    hourly = hourly.loc[hourly["hourly_population"].ge(0)].copy()
    hourly["day_period"] = np.where(
        hourly["local_hour"].between(8, 19), "day", "night"
    )
    means = hourly.pivot_table(
        index="grid_id",
        columns="day_period",
        values="hourly_population",
        aggfunc="mean",
    ).rename(
        columns={
            "day": f"{prefix}_day_mean_hourly_count",
            "night": f"{prefix}_night_mean_hourly_count",
        }
    )
    counts = hourly.pivot_table(
        index="grid_id",
        columns="day_period",
        values="hourly_population",
        aggfunc="count",
    ).rename(
        columns={
            "day": f"{prefix}_day_hour_observations",
            "night": f"{prefix}_night_hour_observations",
        }
    )
    day_night = means.join(counts, how="outer").reset_index()
    for name in ["day", "night"]:
        for kind in ["mean_hourly_count", "hour_observations"]:
            column = f"{prefix}_{name}_{kind}"
            if column not in day_night:
                day_night[column] = np.nan
    day_night[f"{prefix}_day_night_ratio"] = (
        day_night[f"{prefix}_day_mean_hourly_count"]
        / day_night[f"{prefix}_night_mean_hourly_count"].where(
            day_night[f"{prefix}_night_mean_hourly_count"] != 0
        )
    )
    observations = (
        day_night[f"{prefix}_day_hour_observations"].fillna(0)
        + day_night[f"{prefix}_night_hour_observations"].fillna(0)
    )
    day_night.loc[
        observations < minimum_days * MIN_HOURS_PER_DAY,
        f"{prefix}_day_night_ratio",
    ] = np.nan

    grid_day = (
        hourly.groupby(["grid_id", "local_date"], as_index=False)
        .agg(
            valid_hours=("local_hour", "nunique"),
            daily_min_hourly_count=("hourly_population", "min"),
            daily_max_hourly_count=("hourly_population", "max"),
        )
    )
    grid_day["daily_amplitude"] = (
        grid_day["daily_max_hourly_count"]
        - grid_day["daily_min_hourly_count"]
    )
    amplitude = (
        grid_day.loc[grid_day["valid_hours"] >= MIN_HOURS_PER_DAY]
        .groupby("grid_id", as_index=False)
        .agg(
            **{
                f"{prefix}_daily_amplitude": (
                    "daily_amplitude",
                    "mean",
                ),
                f"{prefix}_valid_hourly_days": (
                    "local_date",
                    "nunique",
                ),
                f"{prefix}_min_hours_in_valid_day": (
                    "valid_hours",
                    "min",
                ),
            }
        )
    )
    hourly_mean = (
        hourly.groupby("grid_id", as_index=False)["hourly_population"]
        .mean()
        .rename(
            columns={
                "hourly_population": f"{prefix}_mean_hourly_count"
            }
        )
    )
    amplitude = amplitude.merge(
        hourly_mean, on="grid_id", how="outer", validate="one_to_one"
    )
    amplitude[f"{prefix}_daily_relative_amplitude"] = (
        amplitude[f"{prefix}_daily_amplitude"]
        / amplitude[f"{prefix}_mean_hourly_count"].where(
            amplitude[f"{prefix}_mean_hourly_count"] != 0
        )
    )
    amplitude.loc[
        amplitude[f"{prefix}_valid_hourly_days"] < minimum_days,
        [
            f"{prefix}_daily_amplitude",
            f"{prefix}_daily_relative_amplitude",
        ],
    ] = np.nan
    result = (
        features.merge(
            day_night, on="grid_id", how="outer", validate="one_to_one"
        )
        .merge(
            amplitude, on="grid_id", how="outer", validate="one_to_one"
        )
    )
    atomic_parquet(result, cache_path)
    print(f"WRITE dynamic-feature cache: {cache_path}")
    return result


def legacy_source(
    path: Path, fields: list[str]
) -> pd.DataFrame:
    require_file(path)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = ["grid_lon", "grid_lat"] + [
        column
        for column in fields
        if column in available
    ]
    source = pd.read_parquet(path, columns=columns)
    source["legacy_grid_lon"] = pd.to_numeric(
        source["grid_lon"], errors="coerce"
    ).round(4)
    source["legacy_grid_lat"] = pd.to_numeric(
        source["grid_lat"], errors="coerce"
    ).round(4)
    values = [column for column in columns if column not in {"grid_lon", "grid_lat"}]
    return (
        source.groupby(
            ["legacy_grid_lon", "legacy_grid_lat"], as_index=False
        )[values]
        .mean()
    )


def merge_source(
    base: pd.DataFrame,
    source: pd.DataFrame,
    fields: list[str],
    source_path: Path | str,
    diagnostics: list[dict],
) -> pd.DataFrame:
    merged = base.merge(
        source,
        on=["legacy_grid_lon", "legacy_grid_lat"],
        how="left",
        validate="many_to_one",
    )
    for field in fields:
        valid = int(merged[field].notna().sum()) if field in merged else 0
        diagnostics.append(
            {
                "variable": field,
                "source_path": str(source_path),
                "target_grid_count": len(base),
                "matched_grid_count": valid,
                "missing_count_after_merge": len(base) - valid,
                "coverage_percentage": (
                    valid / len(base) * 100 if len(base) else np.nan
                ),
            }
        )
    return merged


def build(
    comparison: str,
    analysis_target: str,
    config: dict,
    *,
    overwrite_dynamic_cache: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = (
        PERIOD_PATH
        if analysis_target == "total_exposure"
        else DECOMPOSITION_DIR / "grid_level_decomposition.parquet"
    )
    require_file(source_path)
    period = add_legacy_01_degree_keys(pd.read_parquet(source_path))
    comparison_config = modeling_spec(
        config, comparison, analysis_target
    )
    target = comparison_config["target"]
    all_features = list(
        dict.fromkeys(
            comparison_config["b0_features_with_gdp"]
            + comparison_config["b0_features_without_gdp"]
            + comparison_config["b1_features"]
            + comparison_config.get("accounting_features", [])
        )
    )
    phase = "pre" if comparison == "festival_pre" else "festival"
    dynamics = dynamic_features(
        phase, overwrite_cache=overwrite_dynamic_cache
    )
    base_columns = [
        "grid_id",
        "grid_x",
        "grid_y",
        "grid_center_x",
        "grid_center_y",
        "grid_center_lon",
        "grid_center_lat",
        "legacy_grid_lon",
        "legacy_grid_lat",
        target,
    ]
    period_features = [
        field
        for field in all_features
        if field in period.columns
    ]
    base = period[base_columns + period_features].copy()
    base["grid_lon"] = base["grid_center_lon"]
    base["grid_lat"] = base["grid_center_lat"]
    base = base.merge(
        dynamics, on="grid_id", how="left", validate="one_to_one"
    )
    diagnostics: list[dict] = []

    weather_fields = [
        field for field in all_features if "temperature" in field
        or "precipitation" in field
        or "wind_speed" in field
    ]
    weather = legacy_source(WEATHER_PATH, weather_fields)
    base = merge_source(
        base, weather, weather_fields, WEATHER_PATH, diagnostics
    )

    legacy_inputs = [LEGACY_FESTIVAL_INPUT, LEGACY_POST_INPUT]
    static_frames = []
    for path in legacy_inputs:
        require_file(path)
        available = set(pq.ParquetFile(path).schema_arrow.names)
        selected = [field for field in STATIC_FEATURES if field in available]
        if selected:
            static_frames.append(legacy_source(path, selected))
    static = pd.concat(static_frames, ignore_index=True)
    static = (
        static.groupby(
            ["legacy_grid_lon", "legacy_grid_lat"], as_index=False
        )[STATIC_FEATURES]
        .mean()
    )
    base = merge_source(
        base,
        static,
        STATIC_FEATURES,
        f"{LEGACY_FESTIVAL_INPUT} | {LEGACY_POST_INPUT}",
        diagnostics,
    )

    missing = [feature for feature in all_features if feature not in base]
    if missing:
        raise ValueError(
            f"Unable to construct configured features for {comparison}: "
            f"{missing}"
        )
    nonnegative = [
        field
        for field in all_features
        if field
        not in {
            f"{comparison}_count_change",
            f"{comparison}_pm25_change",
            f"{comparison}_temperature_change",
            f"{comparison}_precipitation_change",
            f"{comparison}_wind_speed_change",
        }
    ]
    for field in nonnegative:
        base.loc[pd.to_numeric(base[field], errors="coerce") < 0, field] = np.nan
    for field in all_features:
        if field.endswith("_mean_pm25"):
            base.loc[
                pd.to_numeric(base[field], errors="coerce") <= 0,
                field,
            ] = np.nan
    base = base.replace([np.inf, -np.inf], np.nan)
    required_complete = list(
        dict.fromkeys(
            [target]
            + comparison_config["b1_features"]
            + comparison_config.get("accounting_features", [])
        )
    )
    b1_complete = base.dropna(subset=required_complete)
    complete_rate = len(b1_complete) / len(base) if len(base) else 0
    diagnostics.append(
        {
            "variable": "B1_complete_case",
            "source_path": "combined",
            "target_grid_count": len(base),
            "matched_grid_count": len(b1_complete),
            "missing_count_after_merge": len(base) - len(b1_complete),
            "coverage_percentage": complete_rate * 100,
        }
    )
    if analysis_target == "pollution":
        baseline_population = comparison_config[
            "accounting_baseline_population"
        ]
        pm25_change = comparison_config["accounting_pm25_change"]
        reconstructed = (
            b1_complete[baseline_population]
            * b1_complete[pm25_change]
        )
        error = b1_complete[target] - reconstructed
        diagnostics.append(
            {
                "variable": "pollution_accounting_identity",
                "source_path": str(source_path),
                "target_grid_count": len(b1_complete),
                "matched_grid_count": len(b1_complete),
                "missing_count_after_merge": 0,
                "coverage_percentage": 100.0,
                "maximum_absolute_error": error.abs().max(),
                "mean_absolute_error": error.abs().mean(),
            }
        )
    keep = list(
        dict.fromkeys(
            base_columns
            + ["grid_lon", "grid_lat"]
            + all_features
            + [
                column
                for column in base.columns
                if column.startswith(f"{phase}_valid_")
            ]
        )
    )
    return b1_complete[keep], pd.DataFrame(diagnostics)


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    with stage_log("08_01_build_xgboost_input"):
        for comparison in select_comparisons(args.comparison):
            output, diagnostic = output_paths(
                comparison, args.analysis_target
            )
            if output.exists() and diagnostic.exists() and not args.overwrite:
                print(
                    f"SKIP existing input set: {comparison}/"
                    f"{args.analysis_target}"
                )
                continue
            if (output.exists() or diagnostic.exists()) and not args.overwrite:
                raise FileExistsError(
                    f"Partial XGBoost input set exists for {comparison}/"
                    f"{args.analysis_target}"
                )
            print(
                f"BUILD XGBoost input: {comparison}/"
                f"{args.analysis_target}"
            )
            result, diagnostics = build(
                comparison,
                args.analysis_target,
                config,
                overwrite_dynamic_cache=args.overwrite,
            )
            atomic_csv(diagnostics, diagnostic)
            print(f"WRITE {diagnostic}")
            complete_row = diagnostics.loc[
                diagnostics["variable"].eq("B1_complete_case")
            ].iloc[0]
            complete_rate = float(
                complete_row["coverage_percentage"]
            ) / 100
            if (
                complete_rate
                < config["modeling"]["minimum_complete_case_rate"]
            ):
                raise ValueError(
                    f"{comparison} B1 complete-case coverage "
                    f"{complete_rate:.2%} is below configured safeguard; "
                    f"review {diagnostic}"
                )
            atomic_parquet(result, output)
            print(f"WRITE {output} rows={len(result):,}")


if __name__ == "__main__":
    main()
