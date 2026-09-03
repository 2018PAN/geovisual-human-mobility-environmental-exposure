from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    FIGURES_DIR,
    MODELING_DIR,
    SPATIAL_ANALYSIS_DIR,
    load_analysis_config,
    require_file,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    normalization,
    plot_grid_map,
    plot_lisa_map,
    plot_shap_multipanel,
    plot_shap_multipanel_individual_scale,
    pooled_normalization,
)


OUTPUT_DIR = FIGURES_DIR / "_publication_preview"
NATIONAL_GRID_PATH = (
    AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
)
CHANGE_PATH = NATIONAL_GRID_PATH
LISA_PATH = (
    SPATIAL_ANALYSIS_DIR
    / "LISA"
    / "lisa_festival_pre_exposure_change.parquet"
)
SHAP_PATH = MODELING_DIR / "festival_pre" / "shap_values.parquet"
SHAP_METADATA_PATH = (
    MODELING_DIR / "festival_pre" / "shap_metadata.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the five-map publication preview batch before overwrite."
        )
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Validation output exists; use --overwrite: {path}"
        )


def feature_label(feature: str) -> str:
    labels = {
        "festival_pre_count_change": "Population change",
        "pre_mean_count": "Pre-festival mean population",
        "pre_count_cv": "Pre-festival population CV",
        "pre_day_night_ratio": "Day–night population ratio",
        "pre_daily_relative_amplitude": "Daily population amplitude",
        "pre_mean_pm25": "Pre-festival mean PM2.5",
        "festival_pre_pm25_change": "PM2.5 change",
        "festival_pre_temperature_change": "Temperature change",
        "festival_pre_precipitation_change": "Precipitation change",
        "festival_pre_wind_speed_change": "Wind-speed change",
        "road_density_m_per_km2": "Road density",
        "built_up_ratio": "Built-up ratio",
        "nighttime_light_2018": "Night-time light",
    }
    return labels.get(feature, feature.replace("_", " ").capitalize())


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    dpi = int(config["plotting"]["dpi_maps"])
    for path in [
        NATIONAL_GRID_PATH,
        CHANGE_PATH,
        LISA_PATH,
        SHAP_PATH,
        SHAP_METADATA_PATH,
    ]:
        require_file(path)
    with SHAP_METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("coverage_scope") != "all_complete_modeling_grids":
        raise ValueError("Validation requires full-coverage SHAP output")

    with stage_log("09_05_validate_publication_map_batch"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        national_grid = pd.read_parquet(NATIONAL_GRID_PATH)
        if national_grid["grid_id"].duplicated().any():
            raise ValueError("National validation grid_id is not unique")

        change_output = OUTPUT_DIR / "01_exposure_change.png"
        require_output(change_output, args.overwrite)
        plot_grid_map(
            national_grid,
            "festival_pre_exposure_change",
            change_output,
            title="Exposure change: festival − pre-festival",
            colorbar_label="Calibrated exposure change",
            diverging=True,
            dpi=dpi,
            diagnostic_path=change_output.with_suffix(".layout.json"),
            thumbnail_path=change_output.with_name(
                f"{change_output.stem}_thumbnail.png"
            ),
        )

        shap_frame = pd.read_parquet(SHAP_PATH)
        shap_columns = [
            column
            for column in shap_frame
            if column.startswith("shap_")
        ]
        if len(shap_frame) != int(metadata["rows"]):
            raise ValueError("SHAP metadata row count mismatch")
        if shap_frame["grid_id"].duplicated().any():
            raise ValueError("Full SHAP grid_id is not unique")
        if not np.isfinite(
            shap_frame[shap_columns].to_numpy(dtype="float64")
        ).all():
            raise ValueError("Non-finite SHAP value inside model support")

        single_feature = "built_up_ratio"
        single_column = f"shap_{single_feature}"
        shap_output = OUTPUT_DIR / "02_full_shap.png"
        require_output(shap_output, args.overwrite)
        plot_grid_map(
            shap_frame,
            single_column,
            shap_output,
            title="SHAP contribution: built-up ratio",
            colorbar_label="SHAP contribution",
            diverging=True,
            norm=normalization(
                shap_frame[single_column],
                diverging=True,
            ),
            national_grid=national_grid,
            support_label="Outside model support",
            dpi=dpi,
            diagnostic_path=shap_output.with_suffix(".layout.json"),
            thumbnail_path=shap_output.with_name(
                f"{shap_output.stem}_thumbnail.png"
            ),
        )

        lisa_output = OUTPUT_DIR / "05_lisa.png"
        require_output(lisa_output, args.overwrite)
        lisa = pd.read_parquet(LISA_PATH)
        plot_lisa_map(
            lisa,
            "lisa_cluster",
            lisa_output,
            title=(
                "LISA clusters of exposure change: "
                "festival − pre-festival"
            ),
            order=config["spatial_analysis"]["cluster_order"],
            dpi=dpi,
            diagnostic_path=lisa_output.with_suffix(".layout.json"),
            thumbnail_path=lisa_output.with_name(
                f"{lisa_output.stem}_thumbnail.png"
            ),
        )

        feature_names = [
            column.removeprefix("shap_")
            for column in shap_columns
        ]
        ranked = sorted(
            feature_names,
            key=lambda feature: float(
                np.abs(
                    shap_frame[f"shap_{feature}"].to_numpy(
                        dtype="float64"
                    )
                ).mean()
            ),
            reverse=True,
        )
        top_features = ranked[:6]
        top_columns = [f"shap_{feature}" for feature in top_features]
        shared_output = (
            OUTPUT_DIR / "03_shap_top6_shared_scale.png"
        )
        require_output(shared_output, args.overwrite)
        plot_shap_multipanel(
            shap_frame,
            top_columns,
            [feature_label(feature) for feature in top_features],
            shared_output,
            figure_title=(
                "Festival − pre-festival: top SHAP contributions"
            ),
            national_grid=national_grid,
            norm=pooled_normalization(
                [shap_frame[column] for column in top_columns],
                diverging=True,
            ),
            dpi=dpi,
            diagnostic_path=shared_output.with_suffix(".layout.json"),
            thumbnail_path=shared_output.with_name(
                f"{shared_output.stem}_thumbnail.png"
            ),
        )

        individual_output = (
            OUTPUT_DIR / "04_shap_top6_individual_scale.png"
        )
        require_output(individual_output, args.overwrite)
        plot_shap_multipanel_individual_scale(
            shap_frame,
            top_columns,
            [feature_label(feature) for feature in top_features],
            individual_output,
            figure_title=(
                "Festival − pre-festival: detailed SHAP contributions"
            ),
            national_grid=national_grid,
            dpi=dpi,
            diagnostic_path=individual_output.with_suffix(
                ".layout.json"
            ),
            thumbnail_path=individual_output.with_name(
                f"{individual_output.stem}_thumbnail.png"
            ),
        )

        test_rows = int(shap_frame["split_role"].eq("test").sum())
        train_rows = int(shap_frame["split_role"].eq("train").sum())
        summary = {
            "validation_status": "passed",
            "full_shap_rows": len(shap_frame),
            "unique_shap_grid_ids": int(
                shap_frame["grid_id"].nunique()
            ),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "nonfinite_shap_values": int(
                (~np.isfinite(
                    shap_frame[shap_columns].to_numpy(dtype="float64")
                )).sum()
            ),
            "old_test_shap_max_absolute_difference": metadata[
                "test_shap_max_absolute_difference"
            ],
            "old_test_shap_max_relative_difference": metadata[
                "test_shap_max_relative_difference"
            ],
            "top_six_features": top_features,
            "outputs": [
                str(change_output),
                str(shap_output),
                str(lisa_output),
                str(shared_output),
                str(individual_output),
            ],
            "layout_diagnostics": [
                str(path.with_suffix(".layout.json"))
                for path in [
                    change_output,
                    shap_output,
                    shared_output,
                    individual_output,
                    lisa_output,
                ]
            ],
        }
        summary_path = OUTPUT_DIR / "preview_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
