from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    MODELING_DIR,
    analysis_target_choices,
    atomic_csv,
    atomic_json,
    load_analysis_config,
    modeling_spec,
    require_columns,
    require_file,
    select_comparisons,
    stage_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train legacy-compatible B0/B1 Random Forest and XGBoost "
            "models for both Chunyun contrasts."
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
    parser.add_argument("--skip-spatial-cv", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def metrics(y_true, prediction) -> dict:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def make_spatial_folds(
    data: pd.DataFrame, block_size: float, n_folds: int
) -> pd.Series:
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    x, y = transformer.transform(
        data["grid_lon"].to_numpy(), data["grid_lat"].to_numpy()
    )
    block_x = np.floor(np.asarray(x) / block_size).astype("int32")
    block_y = np.floor(np.asarray(y) / block_size).astype("int32")
    blocks = pd.Series(
        list(zip(block_x, block_y)),
        index=data.index,
        name="spatial_block",
    )
    counts = blocks.value_counts()
    fold_sizes = np.zeros(n_folds, dtype="int64")
    block_to_fold = {}
    for block, count in counts.items():
        fold = int(np.argmin(fold_sizes))
        block_to_fold[block] = fold
        fold_sizes[fold] += count
    print(
        f"Spatial blocks={len(counts):,}; fold sizes={fold_sizes.tolist()}"
    )
    return blocks.map(block_to_fold).astype("int8")


def new_model(kind: str, model_config: dict):
    if kind == "rf":
        return RandomForestRegressor(**model_config["random_forest"])
    if kind == "xgboost":
        return XGBRegressor(**model_config["xgboost"])
    raise ValueError(kind)


def baseline_features(
    data: pd.DataFrame, comparison_config: dict, model_config: dict
) -> list[str]:
    if "gdp_2018" in data:
        coverage = data["gdp_2018"].notna().mean()
        print(f"GDP coverage: {coverage:.2%}")
        if coverage >= model_config["gdp_baseline_coverage_threshold"]:
            return comparison_config["b0_features_with_gdp"]
    return comparison_config["b0_features_without_gdp"]


def run_spatial_cv(
    data: pd.DataFrame,
    target: str,
    feature_sets: dict[str, list[str]],
    model_config: dict,
) -> pd.DataFrame:
    folds = make_spatial_folds(
        data,
        float(model_config["spatial_block_size_m"]),
        int(model_config["spatial_folds"]),
    )
    rows = []
    for name, features in feature_sets.items():
        kind = "rf" if name.endswith("_RF") else "xgboost"
        for fold in sorted(folds.unique()):
            train = data.loc[folds != fold]
            validation = data.loc[folds == fold]
            model = new_model(kind, model_config)
            model.fit(train[features], train[target])
            result = metrics(
                validation[target], model.predict(validation[features])
            )
            rows.append(
                {
                    "model": name,
                    "fold": int(fold),
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_features": len(features),
                    **result,
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("model")[["r2", "rmse", "mae"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "model" if item[0] == "model" else f"{item[0]}_{item[1]}"
        for item in summary.columns
    ]
    summary["fold"] = "mean_std"
    summary["n_train"] = np.nan
    summary["n_validation"] = np.nan
    summary["n_features"] = summary["model"].map(
        {name: len(features) for name, features in feature_sets.items()}
    )
    return pd.concat([detail, summary], ignore_index=True, sort=False)


def run_comparison(
    comparison: str,
    analysis_target: str,
    config: dict,
    skip_spatial_cv: bool,
    overwrite: bool,
) -> None:
    model_config = config["modeling"]
    comparison_config = modeling_spec(
        config, comparison, analysis_target
    )
    target = comparison_config["target"]
    suffix = (
        comparison
        if analysis_target == "total_exposure"
        else f"{comparison}_{analysis_target}"
    )
    input_path = (
        MODELING_DIR / "inputs" / f"xgboost_input_{suffix}.parquet"
    )
    comparison_dir = (
        MODELING_DIR / comparison
        if analysis_target == "total_exposure"
        else MODELING_DIR / comparison / analysis_target
    )
    metric_path = comparison_dir / "model_metrics.csv"
    importance_path = comparison_dir / "feature_importance.csv"
    prediction_path = comparison_dir / "prediction_results.csv"
    cv_path = comparison_dir / "spatial_cv_metrics.csv"
    snapshot_path = comparison_dir / "configuration_snapshot.json"
    accounting_path = comparison_dir / "accounting_benchmark.csv"
    model_paths = {
        name: comparison_dir / f"{name.lower()}.joblib"
        for name in ["B0_RF", "B0_XGBoost", "B1_RF", "B1_XGBoost"]
    }
    outputs = [
        metric_path,
        importance_path,
        prediction_path,
        snapshot_path,
        *model_paths.values(),
    ]
    if not skip_spatial_cv:
        outputs.append(cv_path)
    if analysis_target == "pollution":
        outputs.append(accounting_path)
    existing = [path for path in outputs if path.exists()]
    if len(existing) == len(outputs) and not overwrite:
        print(
            f"SKIP existing model set: {comparison}/{analysis_target}"
        )
        return
    if existing and not overwrite:
        raise FileExistsError(
            f"Partial model output set exists for {comparison}/"
            f"{analysis_target}: {existing}"
        )

    require_file(input_path)
    data = pd.read_parquet(input_path).replace([np.inf, -np.inf], np.nan)
    b0 = baseline_features(data, comparison_config, model_config)
    b1 = comparison_config["b1_features"]
    accounting_features = comparison_config.get(
        "accounting_features", []
    )
    require_columns(
        data,
        [
            "grid_id",
            "grid_lon",
            "grid_lat",
            target,
            *b0,
            *b1,
            *accounting_features,
        ],
        input_path,
    )
    used = list(
        dict.fromkeys(
            [
                "grid_id",
                "grid_lon",
                "grid_lat",
                target,
                *b0,
                *b1,
                *accounting_features,
            ]
        )
    )
    data = data[used].dropna().copy()
    if len(data) < int(model_config["minimum_complete_rows"]):
        raise ValueError(
            f"Only {len(data)} common complete rows for "
            f"{comparison}/{analysis_target}"
        )
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=float(model_config["test_size"]),
        random_state=int(model_config["random_state"]),
    )
    train = data.loc[train_indices]
    test = data.loc[test_indices]
    feature_sets = {
        "B0_RF": b0,
        "B0_XGBoost": b0,
        "B1_RF": b1,
        "B1_XGBoost": b1,
    }
    fitted = {}
    metric_rows = []
    predictions = test[
        list(
            dict.fromkeys(
                [
                    "grid_id",
                    "grid_lon",
                    "grid_lat",
                    target,
                    *b1,
                    *accounting_features,
                ]
            )
        )
    ].copy()
    for name, features in feature_sets.items():
        kind = "rf" if name.endswith("_RF") else "xgboost"
        model = new_model(kind, model_config)
        print(
            f"FIT {comparison}/{analysis_target} {name}: "
            f"{len(features)} features"
        )
        model.fit(train[features], train[target])
        prediction = model.predict(test[features])
        result = metrics(test[target], prediction)
        metric_rows.append(
            {
                "comparison": comparison,
                "analysis_target": analysis_target,
                "model": name,
                **result,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(features),
                "features": " | ".join(features),
            }
        )
        predictions[f"{name}_prediction"] = prediction
        predictions[f"{name}_residual"] = (
            test[target].to_numpy() - prediction
        )
        fitted[name] = model

    b1_xgb = fitted["B1_XGBoost"]
    b1_rf = fitted["B1_RF"]
    importance = pd.DataFrame(
        {
            "feature": b1,
            "xgboost_importance": b1_xgb.feature_importances_,
            "random_forest_importance": b1_rf.feature_importances_,
        }
    ).sort_values("xgboost_importance", ascending=False)
    importance.insert(0, "analysis_target", analysis_target)
    importance.insert(0, "comparison", comparison)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(metric_rows), metric_path)
    atomic_csv(importance, importance_path)
    atomic_csv(predictions, prediction_path)
    if analysis_target == "pollution":
        baseline_population = comparison_config[
            "accounting_baseline_population"
        ]
        pm25_change = comparison_config["accounting_pm25_change"]
        accounting_prediction = (
            test[baseline_population] * test[pm25_change]
        )
        accounting_error = (
            test[target].to_numpy()
            - accounting_prediction.to_numpy()
        )
        accounting_result = {
            "comparison": comparison,
            "analysis_target": analysis_target,
            "benchmark": "accounting_identity",
            "formula": f"{baseline_population} * {pm25_change}",
            **metrics(test[target], accounting_prediction),
            "n_test": len(test),
            "maximum_absolute_identity_error": float(
                np.abs(accounting_error).max()
            ),
            "mean_absolute_identity_error": float(
                np.abs(accounting_error).mean()
            ),
            "included_in_primary_b1": False,
        }
        atomic_csv(
            pd.DataFrame([accounting_result]), accounting_path
        )
    for name, model in fitted.items():
        temporary = model_paths[name].with_name(
            f".{model_paths[name].name}.tmp"
        )
        joblib.dump(model, temporary)
        temporary.replace(model_paths[name])
    if not skip_spatial_cv:
        spatial_cv = run_spatial_cv(
            data, target, feature_sets, model_config
        )
        spatial_cv.insert(0, "analysis_target", analysis_target)
        spatial_cv.insert(0, "comparison", comparison)
        atomic_csv(spatial_cv, cv_path)
    snapshot = {
        "comparison": comparison,
        "analysis_target": analysis_target,
        "target": target,
        "baseline_features": b0,
        "b1_features": b1,
        "accounting_features": accounting_features,
        "common_complete_rows": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "modeling_config": model_config,
        "transformations": {
            "winsorization": None,
            "standardization": False,
            "log_transform": False,
            "early_stopping": None,
        },
    }
    atomic_json(snapshot, snapshot_path)
    for path in outputs:
        print(f"WRITE {path}")


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    with stage_log("08_02_run_xgboost"):
        for comparison in select_comparisons(args.comparison):
            run_comparison(
                comparison,
                args.analysis_target,
                config,
                args.skip_spatial_cv,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
