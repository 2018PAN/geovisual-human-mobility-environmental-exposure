from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    MODELING_DIR,
    analysis_target_choices,
    atomic_json,
    atomic_parquet,
    load_analysis_config,
    modeling_spec,
    require_columns,
    require_file,
    select_comparisons,
    stage_log,
)


SPATIAL_COLUMNS = [
    "grid_id",
    "grid_x",
    "grid_y",
    "grid_center_x",
    "grid_center_y",
    "grid_center_lon",
    "grid_center_lat",
    "grid_lon",
    "grid_lat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate full-model-support SHAP values from the already "
            "fitted formal B1 XGBoost model."
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
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Rows per TreeExplainer call; does not alter SHAP values.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_paths(
    comparison: str,
    analysis_target: str,
) -> tuple[Path, Path]:
    suffix = (
        comparison
        if analysis_target == "total_exposure"
        else f"{comparison}_{analysis_target}"
    )
    input_path = (
        MODELING_DIR / "inputs" / f"xgboost_input_{suffix}.parquet"
    )
    directory = (
        MODELING_DIR / comparison
        if analysis_target == "total_exposure"
        else MODELING_DIR / comparison / analysis_target
    )
    return input_path, directory


def load_snapshot(path: Path) -> dict:
    require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reconstruct_complete_model_data(
    input_path: Path,
    snapshot: dict,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    require_file(input_path)
    source = pd.read_parquet(input_path).replace(
        [np.inf, -np.inf], np.nan
    )
    baseline = list(snapshot["baseline_features"])
    accounting = list(snapshot.get("accounting_features", []))
    required = list(
        dict.fromkeys(
            [
                *SPATIAL_COLUMNS,
                target,
                *baseline,
                *features,
                *accounting,
            ]
        )
    )
    require_columns(source, required, input_path)
    used_for_complete_case = list(
        dict.fromkeys(
            [
                "grid_id",
                "grid_lon",
                "grid_lat",
                target,
                *baseline,
                *features,
                *accounting,
            ]
        )
    )
    complete_index = source[used_for_complete_case].dropna().index
    data = source.loc[complete_index, required].copy()
    if len(data) != int(snapshot["common_complete_rows"]):
        raise ValueError(
            "Reconstructed complete modelling rows differ from the fitted "
            f"model snapshot: {len(data):,} versus "
            f"{int(snapshot['common_complete_rows']):,}"
        )
    if data["grid_id"].duplicated().any():
        examples = data.loc[
            data["grid_id"].duplicated(keep=False), "grid_id"
        ].head(10)
        raise ValueError(
            "Duplicate grid_id values in complete modelling input: "
            f"{examples.tolist()}"
        )

    model_config = snapshot["modeling_config"]
    train_index, test_index = train_test_split(
        data.index,
        test_size=float(model_config["test_size"]),
        random_state=int(model_config["random_state"]),
    )
    if len(train_index) != int(snapshot["train_rows"]):
        raise ValueError("Reconstructed training-row count has changed")
    if len(test_index) != int(snapshot["test_rows"]):
        raise ValueError("Reconstructed test-row count has changed")
    return data, np.asarray(train_index), np.asarray(test_index)


def calculate_in_chunks(
    explainer: shap.TreeExplainer,
    predictors: pd.DataFrame,
    chunk_size: int,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    parts = []
    for start in range(0, len(predictors), chunk_size):
        stop = min(start + chunk_size, len(predictors))
        print(f"  SHAP rows {start + 1:,}-{stop:,}/{len(predictors):,}")
        part = np.asarray(
            explainer.shap_values(
                predictors.iloc[start:stop],
                check_additivity=False,
            )
        )
        parts.append(part)
    return np.concatenate(parts, axis=0)


def old_test_surface(
    old: pd.DataFrame,
    prediction: pd.DataFrame,
    shap_columns: list[str],
) -> pd.DataFrame:
    require_columns(old, ["grid_id", *shap_columns], "old SHAP output")
    require_columns(
        prediction,
        ["grid_id"],
        "formal test prediction output",
    )
    if old["grid_id"].duplicated().any():
        raise ValueError("The previous SHAP output contains duplicate grid_id")
    test_ids = set(prediction["grid_id"])
    subset = old.loc[old["grid_id"].isin(test_ids), ["grid_id", *shap_columns]]
    if len(subset) != len(prediction):
        raise ValueError(
            "Previous SHAP output does not contain every formal test grid: "
            f"{len(subset):,}/{len(prediction):,}"
        )
    return subset


def verify_old_test_values(
    old_test: pd.DataFrame,
    new_result: pd.DataFrame,
    shap_columns: list[str],
) -> dict[str, float | int]:
    new_test = new_result.loc[
        new_result["split_role"].eq("test"),
        ["grid_id", *shap_columns],
    ]
    comparison = old_test.merge(
        new_test,
        on="grid_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_old", "_new"),
    )
    if len(comparison) != len(old_test):
        raise ValueError("Old/new test SHAP grid IDs are not identical")
    old_values = comparison[
        [f"{column}_old" for column in shap_columns]
    ].to_numpy(dtype="float64")
    new_values = comparison[
        [f"{column}_new" for column in shap_columns]
    ].to_numpy(dtype="float64")
    difference = np.abs(old_values - new_values)
    denominator = np.maximum(np.abs(old_values), 1.0)
    max_absolute = float(difference.max(initial=0.0))
    max_relative = float(
        (difference / denominator).max(initial=0.0)
    )
    if not np.allclose(
        old_values,
        new_values,
        rtol=1e-7,
        atol=1e-6,
        equal_nan=False,
    ):
        raise ValueError(
            "Full-coverage calculation changed original test SHAP values: "
            f"max_abs={max_absolute:.12g}; "
            f"max_relative={max_relative:.12g}"
        )
    return {
        "verified_test_rows": len(comparison),
        "test_shap_max_absolute_difference": max_absolute,
        "test_shap_max_relative_difference": max_relative,
    }


def calculate(
    comparison: str,
    analysis_target: str,
    config: dict,
    overwrite: bool,
    chunk_size: int,
) -> None:
    comparison_config = modeling_spec(
        config, comparison, analysis_target
    )
    target = comparison_config["target"]
    configured_features = list(comparison_config["b1_features"])
    input_path, directory = model_paths(comparison, analysis_target)
    model_path = directory / "b1_xgboost.joblib"
    prediction_path = directory / "prediction_results.csv"
    snapshot_path = directory / "configuration_snapshot.json"
    output_path = directory / "shap_values.parquet"
    metadata_path = directory / "shap_metadata.json"

    if output_path.exists() and metadata_path.exists() and not overwrite:
        metadata = load_snapshot(metadata_path)
        if metadata.get("coverage_scope") == "all_complete_modeling_grids":
            print(
                f"SKIP existing full SHAP set: "
                f"{comparison}/{analysis_target}"
            )
            return
        raise FileExistsError(
            "Existing SHAP output is test-only; rerun with --overwrite "
            f"to create full coverage: {output_path}"
        )
    if (output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Partial SHAP set exists for {comparison}/"
            f"{analysis_target}"
        )

    for path in [
        model_path,
        prediction_path,
        snapshot_path,
        input_path,
    ]:
        require_file(path)
    if not output_path.exists():
        raise FileNotFoundError(
            "The previous test-only SHAP output is required for numerical "
            f"equivalence validation before replacement: {output_path}"
        )

    snapshot = load_snapshot(snapshot_path)
    snapshot_features = list(snapshot["b1_features"])
    if snapshot_features != configured_features:
        raise ValueError(
            "Current configuration feature order differs from the fitted "
            "model snapshot; refusing to calculate SHAP"
        )
    if snapshot["target"] != target:
        raise ValueError("Configured target differs from fitted snapshot")
    transformations = snapshot["transformations"]
    expected_transformations = {
        "winsorization": None,
        "standardization": False,
        "log_transform": False,
        "early_stopping": None,
    }
    if transformations != expected_transformations:
        raise ValueError(
            "Unsupported fitted preprocessing encountered; the SHAP "
            "workflow must explicitly reproduce it before continuing: "
            f"{transformations}"
        )

    data, train_index, test_index = reconstruct_complete_model_data(
        input_path,
        snapshot,
        target,
        snapshot_features,
    )
    prediction = pd.read_csv(prediction_path)
    require_columns(
        prediction,
        ["grid_id", "grid_lon", "grid_lat", target, *snapshot_features],
        prediction_path,
    )
    expected_test_ids = data.loc[test_index, "grid_id"].tolist()
    if prediction["grid_id"].tolist() != expected_test_ids:
        raise ValueError(
            "Reconstructed formal test split is not identical to "
            "prediction_results.csv"
        )

    model = joblib.load(model_path)
    model_features = list(model.feature_names_in_)
    if model_features != snapshot_features:
        raise ValueError(
            "Saved model feature order differs from configuration snapshot"
        )
    old = pd.read_parquet(output_path)
    shap_columns = [
        f"shap_{feature}" for feature in snapshot_features
    ]
    previous_test = old_test_surface(old, prediction, shap_columns)

    explainer = shap.TreeExplainer(model)
    values = calculate_in_chunks(
        explainer,
        data[snapshot_features],
        chunk_size,
    )
    expected_shape = (len(data), len(snapshot_features))
    if values.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Non-finite SHAP values inside model support: {bad:,}"
        )

    result_columns = [
        *SPATIAL_COLUMNS,
        target,
        *snapshot_features,
    ]
    result = data[result_columns].copy()
    result["split_role"] = "train"
    result.loc[test_index, "split_role"] = "test"
    result["split_role"] = pd.Categorical(
        result["split_role"],
        categories=["train", "test"],
        ordered=True,
    )
    for index, feature in enumerate(snapshot_features):
        result[f"shap_{feature}"] = values[:, index]
    if result["grid_id"].duplicated().any():
        raise ValueError("Duplicate grid_id after full SHAP calculation")
    if result[shap_columns].isna().any().any():
        raise ValueError("Missing SHAP values inside model support")
    split_counts = result["split_role"].value_counts()
    if int(split_counts.get("train", 0)) != len(train_index):
        raise ValueError("Unexpected train split_role count")
    if int(split_counts.get("test", 0)) != len(test_index):
        raise ValueError("Unexpected test split_role count")

    verification = verify_old_test_values(
        previous_test,
        result,
        shap_columns,
    )
    expected_value = np.asarray(explainer.expected_value).reshape(-1)
    metadata = {
        "comparison": comparison,
        "analysis_target": analysis_target,
        "coverage_scope": "all_complete_modeling_grids",
        "input_path": str(input_path),
        "model_path": str(model_path),
        "target": target,
        "features_in_model_order": snapshot_features,
        "explainer": "TreeExplainer",
        "check_additivity": False,
        "preprocessing": transformations,
        "rows": len(result),
        "unique_grid_ids": int(result["grid_id"].nunique()),
        "train_rows": int(split_counts.get("train", 0)),
        "test_rows": int(split_counts.get("test", 0)),
        "all_shap_values_finite": True,
        "expected_value": expected_value.tolist(),
        **verification,
    }

    # Only replace the test-only output after every scientific check passes.
    atomic_parquet(result, output_path)
    atomic_json(metadata, metadata_path)
    print(f"WRITE {output_path}")
    print(f"WRITE {metadata_path}")
    print(
        f"Full SHAP coverage: {len(result):,} unique grids; "
        f"train={metadata['train_rows']:,}; "
        f"test={metadata['test_rows']:,}"
    )
    print(
        "Original test SHAP equivalence: "
        f"max_abs={verification['test_shap_max_absolute_difference']:.12g}; "
        f"max_relative="
        f"{verification['test_shap_max_relative_difference']:.12g}"
    )


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    with stage_log("08_03_calculate_shap"):
        for comparison in select_comparisons(args.comparison):
            calculate(
                comparison,
                args.analysis_target,
                config,
                args.overwrite,
                args.chunk_size,
            )


if __name__ == "__main__":
    main()
