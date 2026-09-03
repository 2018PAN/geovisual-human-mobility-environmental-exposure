from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    DECOMPOSITION_DIR,
    atomic_csv,
    atomic_parquet,
    ensure_outputs,
    load_analysis_config,
    require_columns,
    require_file,
    stage_log,
)


INPUT_PATH = DECOMPOSITION_DIR / "decomposition_input.parquet"
OUTPUT_PATH = DECOMPOSITION_DIR / "grid_level_decomposition.parquet"
NATIONAL_PATH = DECOMPOSITION_DIR / "national_summary.csv"
PROVINCE_PATH = DECOMPOSITION_DIR / "province_summary.csv"
STATISTICS_PATH = DECOMPOSITION_DIR / "component_contribution_statistics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unchanged three-term Chunyun exposure accounting "
            "decomposition (not a causal model)."
        )
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_contrast(
    frame: pd.DataFrame,
    prefix: str,
    baseline: str,
    comparison: str,
) -> None:
    p0 = frame[f"{baseline}_mean_population"]
    c0 = frame[f"{baseline}_mean_pm25"]
    p1 = frame[f"{comparison}_mean_population"]
    c1 = frame[f"{comparison}_mean_pm25"]
    delta_population = p1 - p0
    delta_pm25 = c1 - c0
    frame[f"{prefix}_mobility_component"] = c0 * delta_population
    frame[f"{prefix}_pollution_component"] = p0 * delta_pm25
    frame[f"{prefix}_interaction_component"] = (
        delta_population * delta_pm25
    )
    components = [
        f"{prefix}_mobility_component",
        f"{prefix}_pollution_component",
        f"{prefix}_interaction_component",
    ]
    frame[f"{prefix}_decomposed_total_change"] = frame[components].sum(
        axis=1
    )
    frame[f"{prefix}_actual_total_change"] = p1 * c1 - p0 * c0
    frame[f"{prefix}_check_error"] = (
        frame[f"{prefix}_decomposed_total_change"]
        - frame[f"{prefix}_actual_total_change"]
    )


def contribution_statistics(
    result: pd.DataFrame, contrasts: dict
) -> pd.DataFrame:
    rows = []
    suffixes = [
        "mobility_component",
        "pollution_component",
        "interaction_component",
        "decomposed_total_change",
        "actual_total_change",
    ]
    for contrast in contrasts:
        for suffix in suffixes:
            column = f"{contrast}_{suffix}"
            values = result[column].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            rows.append(
                {
                    "comparison": contrast,
                    "component": suffix,
                    "variable": column,
                    "count": int(values.count()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "sum": values.sum(),
                    "abs_sum": values.abs().sum(),
                    "mean_abs": values.abs().mean(),
                    "positive_grid_count": int((values > 0).sum()),
                    "negative_grid_count": int((values < 0).sum()),
                    "zero_grid_count": int((values == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def grouped_summary(
    frame: pd.DataFrame, group: str | None, contrasts: dict
) -> pd.DataFrame:
    component_columns = [
        f"{contrast}_{suffix}"
        for contrast in contrasts
        for suffix in [
            "mobility_component",
            "pollution_component",
            "interaction_component",
            "decomposed_total_change",
            "actual_total_change",
        ]
    ]
    if group is None:
        values = frame[component_columns].sum(min_count=1)
        return pd.DataFrame(
            [{"region": "China", **values.to_dict(), "grid_count": len(frame)}]
        )
    selected = frame.loc[frame[group].notna()].copy()
    summary = (
        selected.groupby(group, as_index=False)[component_columns]
        .sum(min_count=1)
    )
    counts = (
        selected.groupby(group).size().rename("grid_count").reset_index()
    )
    return summary.merge(counts, on=group, validate="one_to_one")


def main() -> None:
    args = parse_args()
    outputs = [
        OUTPUT_PATH,
        NATIONAL_PATH,
        PROVINCE_PATH,
        STATISTICS_PATH,
    ]
    config = load_analysis_config()["decomposition"]
    contrasts = config["contrasts"]
    with stage_log("07_02_run_exposure_decomposition"):
        require_file(INPUT_PATH)
        if not ensure_outputs(outputs, args.overwrite):
            return
        source = pd.read_parquet(INPUT_PATH)
        required = [
            f"{phase}_mean_{measure}"
            for phase in ["pre", "festival", "post"]
            for measure in ["population", "pm25"]
        ]
        require_columns(source, required, INPUT_PATH)
        result = source.copy()
        for name, item in contrasts.items():
            add_contrast(
                result, name, item["baseline"], item["comparison"]
            )
        errors = [
            result[f"{name}_check_error"].abs().max()
            for name in contrasts
        ]
        scales = [
            result[f"{name}_actual_total_change"].abs().max()
            for name in contrasts
        ]
        tolerances = [max(1e-8, scale * 1e-10) for scale in scales]
        if any(error > tolerance for error, tolerance in zip(errors, tolerances)):
            raise ValueError(
                f"Decomposition identity failed: errors={errors}, "
                f"tolerances={tolerances}"
            )
        national = grouped_summary(result, None, contrasts)
        province = grouped_summary(result, "province", contrasts)
        statistics = contribution_statistics(result, contrasts)
        atomic_parquet(result, OUTPUT_PATH)
        atomic_csv(national, NATIONAL_PATH)
        atomic_csv(province, PROVINCE_PATH)
        atomic_csv(statistics, STATISTICS_PATH)
        for path in outputs:
            print(f"WRITE {path}")
        print(
            "Identity check maximum absolute errors: "
            + ", ".join(
                f"{name}={error:.12g}"
                for name, error in zip(contrasts, errors)
            )
        )


if __name__ == "__main__":
    main()
