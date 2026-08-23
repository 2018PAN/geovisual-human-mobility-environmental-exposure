"""Calculate full-support TreeSHAP values for a fitted B1 XGBoost model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from analysis_config import FULL_FEATURES
from io_utils import read_table, require_columns, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison", choices=FULL_FEATURES, required=True)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    features = FULL_FEATURES[args.comparison]
    frame = read_table(args.input).replace([np.inf, -np.inf], np.nan)
    require_columns(frame, features)
    complete = frame.dropna(subset=features).copy()
    model = joblib.load(args.model)
    if list(model.feature_names_in_) != features:
        raise ValueError("Saved model features do not match the selected comparison")

    explainer = shap.TreeExplainer(model)
    parts = []
    for start in range(0, len(complete), args.chunk_size):
        stop = min(start + args.chunk_size, len(complete))
        parts.append(
            np.asarray(
                explainer.shap_values(
                    complete.iloc[start:stop][features], check_additivity=False
                )
            )
        )
    values = np.concatenate(parts, axis=0)
    if values.shape != (len(complete), len(features)):
        raise ValueError("Unexpected SHAP output shape")

    identifiers = [
        column
        for column in ["grid_id", "grid_lon", "grid_lat"]
        if column in complete.columns
    ]
    result = complete[identifiers + features].copy()
    for index, feature in enumerate(features):
        result[f"shap_{feature}"] = values[:, index]
    write_table(result, args.output)


if __name__ == "__main__":
    main()
