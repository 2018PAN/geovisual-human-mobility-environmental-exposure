"""Train the baseline and full Random Forest and XGBoost models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from analysis_config import BASELINE_FEATURES, FULL_FEATURES, MODEL_SETTINGS, TARGETS
from io_utils import read_table, require_columns, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comparison", choices=TARGETS, required=True)
    parser.add_argument("--skip-spatial-cv", action="store_true")
    return parser.parse_args()


def new_model(kind: str):
    if kind == "rf":
        return RandomForestRegressor(**MODEL_SETTINGS["random_forest"])
    if kind == "xgboost":
        return XGBRegressor(**MODEL_SETTINGS["xgboost"])
    raise ValueError(kind)


def scores(observed, predicted) -> dict[str, float]:
    return {
        "r2": r2_score(observed, predicted),
        "rmse": mean_squared_error(observed, predicted) ** 0.5,
        "mae": mean_absolute_error(observed, predicted),
    }


def assign_spatial_folds(frame: pd.DataFrame) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(frame["grid_lon"], frame["grid_lat"])
    size = MODEL_SETTINGS["spatial_block_size_m"]
    blocks = pd.Series(
        list(zip(np.floor(np.asarray(x) / size), np.floor(np.asarray(y) / size))),
        index=frame.index,
    )
    counts = blocks.value_counts()
    fold_sizes = np.zeros(MODEL_SETTINGS["spatial_folds"], dtype=int)
    block_to_fold = {}
    for block, count in counts.items():
        fold = int(fold_sizes.argmin())
        block_to_fold[block] = fold
        fold_sizes[fold] += int(count)
    return blocks.map(block_to_fold).to_numpy(dtype=int)


def spatial_cross_validation(
    frame: pd.DataFrame, target: str, feature_sets: dict[str, list[str]]
) -> pd.DataFrame:
    fold_ids = assign_spatial_folds(frame)
    rows = []
    for fold in range(MODEL_SETTINGS["spatial_folds"]):
        train = frame.loc[fold_ids != fold]
        test = frame.loc[fold_ids == fold]
        if train.empty or test.empty:
            raise ValueError("A spatial fold is empty")
        for name, features in feature_sets.items():
            kind = "rf" if name.endswith("RF") else "xgboost"
            model = new_model(kind)
            model.fit(train[features], train[target])
            row = {"fold": fold, "model": name, "n_test": len(test)}
            row.update(scores(test[target], model.predict(test[features])))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    target = TARGETS[args.comparison]
    feature_sets = {
        "B0_RF": BASELINE_FEATURES[args.comparison],
        "B0_XGBoost": BASELINE_FEATURES[args.comparison],
        "B1_RF": FULL_FEATURES[args.comparison],
        "B1_XGBoost": FULL_FEATURES[args.comparison],
    }
    required = list(
        dict.fromkeys(
            ["grid_lon", "grid_lat", target]
            + [item for features in feature_sets.values() for item in features]
        )
    )
    source = read_table(args.input).replace([np.inf, -np.inf], np.nan)
    require_columns(source, required)
    data = source[required].dropna().copy()
    if len(data) < 100:
        raise ValueError("Fewer than 100 complete grid cells remain")

    train, test = train_test_split(
        data,
        test_size=MODEL_SETTINGS["test_size"],
        random_state=MODEL_SETTINGS["random_state"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    predictions = test[["grid_lon", "grid_lat", target]].copy()
    fitted = {}
    for name, features in feature_sets.items():
        kind = "rf" if name.endswith("RF") else "xgboost"
        model = new_model(kind)
        model.fit(train[features], train[target])
        predicted = model.predict(test[features])
        row = {
            "model": name,
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(features),
            "features": " | ".join(features),
        }
        row.update(scores(test[target], predicted))
        metrics_rows.append(row)
        predictions[f"{name}_prediction"] = predicted
        predictions[f"{name}_residual"] = test[target].to_numpy() - predicted
        fitted[name] = model
        joblib.dump(model, args.output_dir / f"{name.lower()}.joblib")

    b1_features = FULL_FEATURES[args.comparison]
    importance = pd.DataFrame(
        {
            "feature": b1_features,
            "random_forest_importance": fitted["B1_RF"].feature_importances_,
            "xgboost_importance": fitted["B1_XGBoost"].feature_importances_,
        }
    ).sort_values("xgboost_importance", ascending=False)
    write_table(pd.DataFrame(metrics_rows), args.output_dir / "model_metrics.csv")
    write_table(predictions, args.output_dir / "test_predictions.csv")
    write_table(importance, args.output_dir / "feature_importance.csv")
    if not args.skip_spatial_cv:
        result = spatial_cross_validation(data, target, feature_sets)
        write_table(result, args.output_dir / "spatial_cv_metrics.csv")


if __name__ == "__main__":
    main()
