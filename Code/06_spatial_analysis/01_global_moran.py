from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import DistanceBand


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    SPATIAL_ANALYSIS_DIR,
    atomic_csv,
    load_analysis_config,
    require_columns,
    require_file,
    stage_log,
)
from downstream_spatial import (  # noqa: E402
    filter_grid_centers_to_china,
    spatial_weight_geodataframe,
)


INPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
OUTPUT_DIR = SPATIAL_ANALYSIS_DIR / "GlobalMoran"
OUTPUT_PATH = OUTPUT_DIR / "global_moran_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legacy-compatible Global Moran's I."
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
    parser.add_argument(
        "--output-name",
        default=OUTPUT_PATH.name,
        help="CSV filename written inside Output/SpatialAnalysis/GlobalMoran.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def comparison_from_variable(variable: str) -> str:
    if variable.startswith("festival_pre_"):
        return "festival_pre"
    if variable.startswith("post_festival_"):
        return "post_festival"
    return "other"


def run_variable(
    frame: pd.DataFrame,
    variable: str,
    config: dict,
    source_path: Path,
) -> dict:
    require_columns(frame, [variable], source_path)
    work = frame.replace([np.inf, -np.inf], np.nan)
    work = work.loc[pd.to_numeric(work[variable], errors="coerce").notna()]
    work = work.copy()
    work[variable] = pd.to_numeric(work[variable], errors="raise")
    if len(work) < 3 or np.isclose(work[variable].std(ddof=0), 0):
        raise ValueError(f"Insufficient nonconstant data for {variable}")

    geodata = spatial_weight_geodataframe(
        work, config["projected_crs"]
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
    moran = Moran(
        geodata[variable].to_numpy(dtype="float64"),
        weights,
        permutations=int(config["permutations"]),
    )
    island_count = sum(len(neighbours) == 0 for neighbours in weights.neighbors.values())
    return {
        "variable": variable,
        "comparison": comparison_from_variable(variable),
        "moran_i": float(moran.I),
        "expected_i": float(moran.EI),
        "z_score": float(moran.z_sim),
        "p_value": float(moran.p_sim),
        "permutations": int(config["permutations"]),
        "n_grids": len(geodata),
        "weight_method": config["weight_method"],
        "projected_crs": config["projected_crs"],
        "distance_threshold_m": float(config["threshold_m"]),
        "row_standardization": config["row_standardization"],
        "island_count": island_count,
        "random_seed": config["random_seed"],
    }


def main() -> None:
    args = parse_args()
    config = load_analysis_config()["spatial_analysis"]
    variables = args.variables or config["variables"]
    if Path(args.output_name).name != args.output_name:
        raise ValueError("--output-name must be a filename, not a path")
    output_path = OUTPUT_DIR / args.output_name
    with stage_log("06_01_global_moran"):
        require_file(args.input_path)
        if output_path.exists() and not args.overwrite:
            print(f"SKIP existing: {output_path}")
            return
        print(f"READ {args.input_path}")
        frame = pd.read_parquet(args.input_path)
        if config["filter_to_china_boundary"]:
            frame = filter_grid_centers_to_china(frame)
        results = []
        for variable in variables:
            print(f"RUN Global Moran: {variable}")
            results.append(
                run_variable(
                    frame, variable, config, args.input_path
                )
            )
        output = pd.DataFrame(results)
        atomic_csv(output, output_path)
        print(f"WRITE {output_path}")
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
