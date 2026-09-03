from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    FIGURES_DIR,
    MODELING_DIR,
    analysis_target_choices,
    load_analysis_config,
    modeling_spec,
    require_file,
    select_comparisons,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    apply_nature_style,
    normalization,
    plot_grid_map,
    plot_shap_multipanel,
    plot_shap_multipanel_individual_scale,
    pooled_normalization,
)


NATIONAL_GRID_PATH = (
    AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
)

FEATURE_LABELS = {
    "built_up_ratio": "Built-up ratio",
    "festival_count_cv": "Festival population CV",
    "festival_daily_relative_amplitude": (
        "Festival daily population amplitude"
    ),
    "festival_day_night_ratio": "Festival day–night population ratio",
    "festival_mean_count": "Festival mean population",
    "festival_mean_pm25": "Festival mean PM2.5",
    "festival_pre_count_change": (
        "Population change: festival − pre-festival"
    ),
    "festival_pre_pm25_change": (
        "PM2.5 change: festival − pre-festival"
    ),
    "festival_pre_precipitation_change": (
        "Precipitation change: festival − pre-festival"
    ),
    "festival_pre_temperature_change": (
        "Temperature change: festival − pre-festival"
    ),
    "festival_pre_wind_speed_change": (
        "Wind-speed change: festival − pre-festival"
    ),
    "nighttime_light_2018": "Night-time light",
    "post_festival_count_change": (
        "Population change: post-festival − festival"
    ),
    "post_festival_pm25_change": (
        "PM2.5 change: post-festival − festival"
    ),
    "post_festival_precipitation_change": (
        "Precipitation change: post-festival − festival"
    ),
    "post_festival_temperature_change": (
        "Temperature change: post-festival − festival"
    ),
    "post_festival_wind_speed_change": (
        "Wind-speed change: post-festival − festival"
    ),
    "pre_count_cv": "Pre-festival population CV",
    "pre_daily_relative_amplitude": (
        "Pre-festival daily population amplitude"
    ),
    "pre_day_night_ratio": "Pre-festival day–night population ratio",
    "pre_mean_count": "Pre-festival mean population",
    "pre_mean_pm25": "Pre-festival mean PM2.5",
    "road_density_m_per_km2": "Road density",
}

SUMMARY_FEATURE_LABELS = {
    "built_up_ratio": "Built-up ratio",
    "festival_count_cv": "Festival population CV",
    "festival_daily_relative_amplitude": "Festival daily amplitude",
    "festival_day_night_ratio": "Festival day-night ratio",
    "festival_mean_count": "Festival mean population",
    "festival_mean_pm25": "Festival mean PM2.5",
    "festival_pre_count_change": "Population change",
    "festival_pre_pm25_change": "PM2.5 change",
    "festival_pre_precipitation_change": "Precipitation change",
    "festival_pre_temperature_change": "Temperature change",
    "festival_pre_wind_speed_change": "Wind-speed change",
    "nighttime_light_2018": "Night-time light (2018)",
    "post_festival_count_change": "Population change",
    "post_festival_pm25_change": "PM2.5 change",
    "post_festival_precipitation_change": "Precipitation change",
    "post_festival_temperature_change": "Temperature change",
    "post_festival_wind_speed_change": "Wind-speed change",
    "pre_count_cv": "Pre-festival population CV",
    "pre_daily_relative_amplitude": "Pre-festival daily amplitude",
    "pre_day_night_ratio": "Pre-festival day-night ratio",
    "pre_mean_count": "Pre-festival mean population",
    "pre_mean_pm25": "Pre-festival mean PM2.5",
    "road_density_m_per_km2": "Road density (m/km²)",
}

SUMMARY_FONT = "Arial"
SUMMARY_LABEL_SIZE = 7.6
SUMMARY_AXIS_SIZE = 8.4
SUMMARY_TICK_SIZE = 7.5
SUMMARY_SCALE = 1_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot legacy XGBoost, residual, importance, and SHAP figures."
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
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Regenerate residual and SHAP spatial maps only.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Regenerate only the SHAP bar and beeswarm summaries using "
            "publication-readable feature labels."
        ),
    )
    return parser.parse_args()


def save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_summary(fig, path: Path, dpi: int, size: tuple[float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*size)
    fig.savefig(path, dpi=dpi, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def style_summary_axis(axis, *, beeswarm: bool = False) -> None:
    """Apply a compact, thesis-ready style at the final Word figure size."""
    if beeswarm:
        axis.axvline(0, color="#7B8792", linewidth=0.65, zorder=0)
    axis.grid(axis="y" if beeswarm else "x", color="#D8DEE4", linewidth=0.45, linestyle=(0, (2, 4)))
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#9AA4AE")
    axis.spines["bottom"].set_color("#9AA4AE")
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(axis="x", labelsize=SUMMARY_TICK_SIZE, width=0.6, colors="#334155")
    axis.tick_params(axis="y", labelsize=SUMMARY_LABEL_SIZE, length=0, pad=5, colors="#243447")
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_fontfamily(SUMMARY_FONT)
        label.set_fontweight("normal")
    axis.xaxis.label.set_fontfamily(SUMMARY_FONT)
    axis.xaxis.label.set_fontsize(SUMMARY_AXIS_SIZE)
    axis.xaxis.label.set_fontweight("normal")
    axis.xaxis.label.set_color("#243447")
    axis.xaxis.set_label_coords(0.5, -0.075)


def comparison_label(comparison: str) -> str:
    if comparison == "festival_pre":
        return "Festival − pre-festival"
    if comparison == "post_festival":
        return "Post-festival − festival"
    raise ValueError(comparison)


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(
        feature,
        feature.replace("_", " ").capitalize(),
    )


def summary_feature_label(feature: str) -> str:
    return SUMMARY_FEATURE_LABELS.get(feature, feature_label(feature))


def plot_comparison(
    comparison: str,
    analysis_target: str,
    config: dict,
    overwrite: bool,
    maps_only: bool,
    summary_only: bool,
) -> None:
    comparison_config = modeling_spec(
        config, comparison, analysis_target
    )
    target = comparison_config["target"]
    features = comparison_config["b1_features"]
    suffix = (
        comparison
        if analysis_target == "total_exposure"
        else f"{comparison}_{analysis_target}"
    )
    source_dir = (
        MODELING_DIR / comparison
        if analysis_target == "total_exposure"
        else MODELING_DIR / comparison / analysis_target
    )
    prediction_path = source_dir / "prediction_results.csv"
    importance_path = source_dir / "feature_importance.csv"
    shap_path = source_dir / "shap_values.parquet"
    shap_metadata_path = source_dir / "shap_metadata.json"
    snapshot_path = source_dir / "configuration_snapshot.json"
    for path in [
        prediction_path,
        importance_path,
        shap_path,
        shap_metadata_path,
        snapshot_path,
        NATIONAL_GRID_PATH,
    ]:
        require_file(path)
    prediction = pd.read_csv(prediction_path)
    importance = pd.read_csv(importance_path)
    shap_frame = pd.read_parquet(shap_path)
    with shap_metadata_path.open("r", encoding="utf-8") as handle:
        shap_metadata = json.load(handle)
    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if shap_metadata.get("coverage_scope") != (
        "all_complete_modeling_grids"
    ):
        raise ValueError(
            f"SHAP output is not full coverage: {shap_path}"
        )
    if len(shap_frame) != int(snapshot["common_complete_rows"]):
        raise ValueError(
            "Full SHAP row count differs from formal model support: "
            f"{len(shap_frame):,}/"
            f"{int(snapshot['common_complete_rows']):,}"
        )
    if shap_frame["grid_id"].duplicated().any():
        raise ValueError("Full SHAP output contains duplicate grid_id")
    shap_columns = [f"shap_{feature}" for feature in features]
    if not np.isfinite(
        shap_frame[shap_columns].to_numpy(dtype="float64")
    ).all():
        raise ValueError("Non-finite SHAP value inside model support")
    national_grid = pd.read_parquet(
        NATIONAL_GRID_PATH,
        columns=[
            "grid_id",
            "grid_x",
            "grid_y",
            "grid_center_x",
            "grid_center_y",
            "grid_center_lon",
            "grid_center_lat",
        ],
    )
    figure_dir = (
        FIGURES_DIR / "XGBoost" / comparison
        if analysis_target == "total_exposure"
        else FIGURES_DIR
        / "XGBoost"
        / comparison
        / analysis_target
    )
    shap_dir = (
        FIGURES_DIR / "SHAP" / comparison
        if analysis_target == "total_exposure"
        else FIGURES_DIR / "SHAP" / comparison / analysis_target
    )
    formal_input_path = (
        MODELING_DIR / "inputs" / f"xgboost_input_{suffix}.parquet"
    )
    dpi = int(config["plotting"]["dpi_models"])
    map_dpi = int(config["plotting"]["dpi_maps"])
    apply_nature_style(dpi)

    metric_path = source_dir / "model_metrics.csv"
    require_file(metric_path)
    metrics = pd.read_csv(metric_path)
    path = figure_dir / "model_metric_comparison.png"
    if not maps_only and (overwrite or not path.exists()):
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
        for axis, measure in zip(axes, ["r2", "rmse", "mae"]):
            axis.bar(
                metrics["model"],
                metrics[measure],
                color=["#A0CBE8", "#FFBE7D", "#4E79A7", "#F28E2B"],
                edgecolor="black",
                linewidth=0.25,
            )
            axis.set_title(measure.upper())
            axis.tick_params(axis="x", rotation=55)
        fig.suptitle(
            f"{comparison.replace('_', ' ').title()} model evaluation"
        )
        fig.tight_layout()
        save(fig, path, dpi)

    cv_path = source_dir / "spatial_cv_metrics.csv"
    path = figure_dir / "spatial_cv_error.png"
    if (
        not maps_only
        and cv_path.exists()
        and (overwrite or not path.exists())
    ):
        cv = pd.read_csv(cv_path)
        detail = cv.loc[cv["fold"].astype(str).ne("mean_std")].copy()
        fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
        for axis, measure in zip(axes, ["r2", "rmse", "mae"]):
            groups = [
                detail.loc[detail["model"].eq(model), measure].dropna()
                for model in metrics["model"]
            ]
            axis.boxplot(
                groups,
                tick_labels=metrics["model"],
                showfliers=False,
            )
            axis.set_title(measure.upper())
            axis.tick_params(axis="x", rotation=55)
        fig.suptitle(
            f"{comparison.replace('_', ' ').title()} spatial CV"
        )
        fig.tight_layout()
        save(fig, path, dpi)

    path = figure_dir / "observed_vs_predicted.png"
    if not maps_only and (overwrite or not path.exists()):
        observed = prediction[target].to_numpy()
        predicted = prediction["B1_XGBoost_prediction"].to_numpy()
        fig, axis = plt.subplots(figsize=(5.0, 4.5))
        axis.scatter(observed, predicted, s=5, alpha=0.35, linewidths=0)
        limits = [
            min(observed.min(), predicted.min()),
            max(observed.max(), predicted.max()),
        ]
        axis.plot(limits, limits, "k--", linewidth=0.8)
        axis.set_xlabel(
            f"Observed {analysis_target.replace('_', ' ')}"
        )
        axis.set_ylabel(
            f"Predicted {analysis_target.replace('_', ' ')}"
        )
        axis.set_title(
            f"{comparison.replace('_', ' ').title()} "
            f"{analysis_target.replace('_', ' ')}: "
            "observed vs predicted"
        )
        save(fig, path, dpi)

    path = figure_dir / "residual_distribution.png"
    if not maps_only and (overwrite or not path.exists()):
        residual = prediction["B1_XGBoost_residual"]
        fig, axis = plt.subplots(figsize=(5.0, 3.6))
        axis.hist(residual, bins=60, color="#4C78A8", alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Residual (observed - predicted)")
        axis.set_ylabel("Test-grid count")
        save(fig, path, dpi)

    path = figure_dir / "feature_importance.png"
    if not maps_only and (overwrite or not path.exists()):
        ordered = importance.sort_values("xgboost_importance")
        fig, axis = plt.subplots(figsize=(7.5, 5.3))
        axis.barh(
            ordered["feature"],
            ordered["xgboost_importance"],
            color="#4C78A8",
        )
        axis.set_xlabel("XGBoost feature importance")
        save(fig, path, dpi)

    path = figure_dir / "residual_map.png"
    if overwrite or not path.exists():
        map_frame = prediction.rename(
            columns={
                "grid_lon": "grid_center_lon",
                "grid_lat": "grid_center_lat",
            }
        )
        # Grid coordinates are recovered from the formal input to ensure the
        # established 10 km LCC geometry is used for mapping.
        formal = pd.read_parquet(
            formal_input_path,
            columns=[
                "grid_id",
                "grid_center_x",
                "grid_center_y",
            ],
        )
        map_frame = map_frame.merge(
            formal, on="grid_id", how="left", validate="one_to_one"
        )
        plot_grid_map(
            map_frame,
            "B1_XGBoost_residual",
            path,
            title=(
                f"{comparison_label(comparison)}: "
                f"{analysis_target.replace('_', ' ')} residual"
            ),
            colorbar_label=(
                "Observed - predicted "
                f"{analysis_target.replace('_', ' ')}"
            ),
            diverging=True,
            national_grid=national_grid,
            support_label="Outside test sample",
            dpi=map_dpi,
        )

    x_all = shap_frame[features]
    shap_values = shap_frame[
        [f"shap_{feature}" for feature in features]
    ].to_numpy()
    path = shap_dir / "shap_summary_beeswarm.png"
    display_labels = [summary_feature_label(feature) for feature in features]
    if not maps_only and (
        overwrite or summary_only or not path.exists()
    ):
        shap.summary_plot(
            shap_values / SUMMARY_SCALE,
            x_all,
            feature_names=display_labels,
            max_display=len(features),
            plot_size=(7.09, 5.72),
            color_bar_label="Feature value",
            show=False,
        )
        fig = plt.gcf()
        axis = fig.axes[0]
        axis.set_xlabel("SHAP value (million exposure-load units)")
        style_summary_axis(axis, beeswarm=True)
        if len(fig.axes) > 1:
            colorbar_axis = fig.axes[-1]
            colorbar_axis.tick_params(labelsize=7.2, length=0)
            colorbar_axis.yaxis.label.set_fontfamily(SUMMARY_FONT)
            colorbar_axis.yaxis.label.set_fontsize(8.2)
            for label in colorbar_axis.get_yticklabels():
                label.set_fontfamily(SUMMARY_FONT)
                label.set_fontsize(7.2)
        fig.subplots_adjust(left=0.335, right=0.915, top=0.98, bottom=0.12)
        save_summary(fig, path, dpi, (7.09, 5.72))
    path = shap_dir / "shap_bar_importance.png"
    if not maps_only and (
        overwrite or summary_only or not path.exists()
    ):
        importance_values = np.abs(shap_values).mean(axis=0) / SUMMARY_SCALE
        order = np.argsort(importance_values)
        fig, axis = plt.subplots(figsize=(7.09, 5.62))
        axis.barh(
            np.asarray(display_labels)[order],
            importance_values[order],
            color="#3E7CB1",
            edgecolor="#2D5F87",
            linewidth=0.35,
            height=0.62,
        )
        axis.set_xlabel("Mean absolute SHAP value (million exposure-load units)")
        style_summary_axis(axis)
        fig.subplots_adjust(left=0.335, right=0.975, top=0.98, bottom=0.13)
        save_summary(fig, path, dpi, (7.09, 5.62))
    if summary_only:
        print(f"Refreshed publication-readable SHAP summaries: {shap_dir}")
        return
    if not maps_only:
        for feature in features:
            path = shap_dir / f"shap_dependence_{feature}.png"
            if path.exists() and not overwrite:
                continue
            shap.dependence_plot(
                feature, shap_values, x_all, show=False
            )
            save(plt.gcf(), path, dpi)

    required_spatial = [
        "grid_id",
        "grid_center_x",
        "grid_center_y",
        "grid_center_lon",
        "grid_center_lat",
    ]
    missing_spatial = [
        column
        for column in required_spatial
        if column not in shap_frame
    ]
    if missing_spatial:
        raise ValueError(
            f"Full SHAP output lacks spatial columns: {missing_spatial}"
        )
    for feature in features:
        path = shap_dir / f"shap_spatial_{feature}.png"
        if path.exists() and not overwrite:
            continue
        shap_column = f"shap_{feature}"
        map_frame = shap_frame[
            [*required_spatial, shap_column]
        ].copy()
        plot_grid_map(
            map_frame,
            shap_column,
            path,
            title=f"SHAP contribution: {feature_label(feature)}",
            colorbar_label="SHAP contribution",
            diverging=True,
            norm=normalization(
                shap_frame[shap_column],
                diverging=True,
            ),
            national_grid=national_grid,
            support_label="Outside model support",
            dpi=map_dpi,
        )

    ranked_features = sorted(
        features,
        key=lambda feature: float(
            np.abs(
                shap_frame[f"shap_{feature}"].to_numpy(
                    dtype="float64"
                )
            ).mean()
        ),
        reverse=True,
    )
    top_features = ranked_features[:6]
    top_columns = [f"shap_{feature}" for feature in top_features]
    shared_norm = pooled_normalization(
        [shap_frame[column] for column in top_columns],
        diverging=True,
    )
    multipanel_path = shap_dir / "shap_spatial_top6_shared_scale.png"
    if overwrite or not multipanel_path.exists():
        plot_shap_multipanel(
            shap_frame[
                [*required_spatial, *top_columns]
            ],
            top_columns,
            [feature_label(feature) for feature in top_features],
            multipanel_path,
            figure_title=(
                f"{comparison_label(comparison)}: "
                f"{analysis_target.replace('_', ' ')} SHAP contributions"
            ),
            national_grid=national_grid,
            norm=shared_norm,
            dpi=map_dpi,
        )

    individual_path = (
        shap_dir / "shap_spatial_top6_individual_scale.png"
    )
    if overwrite or not individual_path.exists():
        plot_shap_multipanel_individual_scale(
            shap_frame[
                [*required_spatial, *top_columns]
            ],
            top_columns,
            [feature_label(feature) for feature in top_features],
            individual_path,
            figure_title=(
                f"{comparison_label(comparison)}: "
                f"{analysis_target.replace('_', ' ')} "
                "detailed SHAP contributions"
            ),
            national_grid=national_grid,
            dpi=map_dpi,
        )


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    with stage_log("08_04_plot_xgboost_shap"):
        for comparison in select_comparisons(args.comparison):
            plot_comparison(
                comparison,
                args.analysis_target,
                config,
                args.overwrite,
                args.maps_only,
                args.summary_only,
            )


if __name__ == "__main__":
    main()
