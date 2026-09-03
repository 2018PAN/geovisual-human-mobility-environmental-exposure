from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights import DistanceBand


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    SPATIAL_ANALYSIS_DIR,
    atomic_csv,
    atomic_parquet,
    load_analysis_config,
    require_columns,
    require_file,
    stage_log,
)
from downstream_spatial import (  # noqa: E402
    assign_grid_province,
    filter_grid_centers_to_china,
    spatial_weight_geodataframe,
)


INPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
OUTPUT_DIR = SPATIAL_ANALYSIS_DIR / "LISA"

QUADRANTS = {
    1: "High-High",
    2: "Low-High",
    3: "Low-Low",
    4: "High-Low",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy-compatible Local Moran/LISA."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        help="Defaults to variables in downstream_analysis.json.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=INPUT_PATH,
        help=(
            "Grid table containing the requested variables. Component "
            "analysis uses grid_level_decomposition.parquet."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(variable: str) -> list[Path]:
    return [
        OUTPUT_DIR / f"lisa_{variable}.parquet",
        OUTPUT_DIR / f"lisa_{variable}_cluster_counts.csv",
        OUTPUT_DIR / f"lisa_{variable}_province_summary.csv",
    ]


def calculate_lisa(
    source: pd.DataFrame,
    variable: str,
    config: dict,
    source_path: Path,
) -> pd.DataFrame:
    require_columns(source, [variable], source_path)
    result = source.copy()
    numeric = pd.to_numeric(result[variable], errors="coerce")
    valid = numeric.replace([np.inf, -np.inf], np.nan).notna()
    if valid.sum() < 3 or np.isclose(numeric.loc[valid].std(ddof=0), 0):
        raise ValueError(f"Insufficient nonconstant data for {variable}")

    result["lisa_i"] = np.nan
    result["lisa_p_value"] = np.nan
    result["lisa_quadrant"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["lisa_cluster"] = "NoData"

    valid_frame = result.loc[valid].copy()
    geodata = spatial_weight_geodataframe(
        valid_frame, config["projected_crs"]
    )
    coordinates = np.column_stack(
        [geodata.geometry.x, geodata.geometry.y]
    )
    weights = DistanceBand(
        coordinates,
        threshold=float(config["threshold_m"]),
        binary=True,
        silence_warnings=False,
    )
    weights.transform = config["row_standardization"]
    lisa = Moran_Local(
        geodata[variable].to_numpy(dtype="float64"),
        weights,
        permutations=int(config["permutations"]),
    )
    target_index = geodata.index
    result.loc[target_index, "lisa_i"] = lisa.Is
    result.loc[target_index, "lisa_p_value"] = lisa.p_sim
    result.loc[target_index, "lisa_quadrant"] = lisa.q
    significant = lisa.p_sim < float(config["significance_level"])
    clusters = np.full(len(geodata), "Not significant", dtype=object)
    for quadrant, label in QUADRANTS.items():
        clusters[significant & (lisa.q == quadrant)] = label
    result.loc[target_index, "lisa_cluster"] = clusters
    result["lisa_cluster"] = pd.Categorical(
        result["lisa_cluster"],
        categories=config["cluster_order"],
        ordered=True,
    )
    result["lisa_significance_level"] = float(
        config["significance_level"]
    )
    result["lisa_permutations"] = int(config["permutations"])
    result["lisa_weight_method"] = config["weight_method"]
    result["lisa_distance_threshold_m"] = float(config["threshold_m"])
    result["lisa_row_standardization"] = config[
        "row_standardization"
    ]
    return result


def summaries(
    result: pd.DataFrame, variable: str, cluster_order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_counts = (
        result["lisa_cluster"]
        .value_counts(dropna=False)
        .reindex(cluster_order, fill_value=0)
        .rename_axis("cluster")
        .reset_index(name="grid_count")
    )
    cluster_counts["share"] = cluster_counts["grid_count"] / len(result)
    cluster_counts.insert(0, "variable", variable)

    assigned = assign_grid_province(result)
    assigned["province"] = assigned["province"].fillna("Unassigned")
    province = (
        assigned.groupby(
            ["province", "lisa_cluster"],
            observed=False,
            dropna=False,
        )
        .size()
        .rename("grid_count")
        .reset_index()
    )
    totals = province.groupby("province")["grid_count"].transform("sum")
    province["share_within_province"] = province["grid_count"] / totals
    province.insert(0, "variable", variable)
    return cluster_counts, province


def main() -> None:
    args = parse_args()
    config = load_analysis_config()["spatial_analysis"]
    variables = args.variables or config["variables"]
    with stage_log("06_02_local_moran_lisa"):
        require_file(args.input_path)
        print(f"READ {args.input_path}")
        source = pd.read_parquet(args.input_path)
        if config["filter_to_china_boundary"]:
            source = filter_grid_centers_to_china(source)
        for variable in variables:
            paths = output_paths(variable)
            if all(path.exists() for path in paths) and not args.overwrite:
                print(f"SKIP existing LISA output set: {variable}")
                continue
            if any(path.exists() for path in paths) and not args.overwrite:
                raise FileExistsError(
                    f"Partial output set exists for {variable}: {paths}"
                )
            print(f"RUN Local Moran: {variable}")
            result = calculate_lisa(
                source, variable, config, args.input_path
            )
            counts, province = summaries(
                result, variable, config["cluster_order"]
            )
            atomic_parquet(result, paths[0])
            atomic_csv(counts, paths[1])
            atomic_csv(province, paths[2])
            for path in paths:
                print(f"WRITE {path}")


if __name__ == "__main__":
    main()
