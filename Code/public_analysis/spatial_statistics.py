"""Global Moran's I and Local Moran cluster analysis on grid centroids."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import DistanceBand
from pyproj import Transformer

from analysis_config import SPATIAL_SETTINGS
from io_utils import read_table, require_columns, write_table


QUADRANTS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--longitude", default="grid_lon")
    parser.add_argument("--latitude", default="grid_lat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [args.longitude, args.latitude, args.variable]
    frame = read_table(args.input).replace([np.inf, -np.inf], np.nan)
    require_columns(frame, required)
    work = frame.dropna(subset=required).copy().reset_index(drop=True)
    if len(work) < 3 or np.isclose(work[args.variable].std(ddof=0), 0):
        raise ValueError("The selected variable has insufficient variation")

    transformer = Transformer.from_crs(
        "EPSG:4326", SPATIAL_SETTINGS["projected_crs"], always_xy=True
    )
    x, y = transformer.transform(
        work[args.longitude].to_numpy(), work[args.latitude].to_numpy()
    )
    coordinates = np.column_stack([x, y])
    weights = DistanceBand(
        coordinates,
        threshold=SPATIAL_SETTINGS["distance_threshold_m"],
        binary=True,
        silence_warnings=False,
    )
    weights.transform = "r"

    np.random.seed(SPATIAL_SETTINGS["random_seed"])
    values = work[args.variable].to_numpy(dtype="float64")
    global_moran = Moran(
        values, weights, permutations=SPATIAL_SETTINGS["permutations"]
    )
    global_result = pd.DataFrame(
        [
            {
                "variable": args.variable,
                "moran_i": global_moran.I,
                "expected_i": global_moran.EI,
                "z_score": global_moran.z_sim,
                "pseudo_p_value": global_moran.p_sim,
                "n_grids": len(work),
                "distance_threshold_m": SPATIAL_SETTINGS[
                    "distance_threshold_m"
                ],
                "permutations": SPATIAL_SETTINGS["permutations"],
                "row_standardization": "r",
                "island_count": sum(
                    len(item) == 0 for item in weights.neighbors.values()
                ),
            }
        ]
    )

    np.random.seed(SPATIAL_SETTINGS["random_seed"])
    local_moran = Moran_Local(
        values, weights, permutations=SPATIAL_SETTINGS["permutations"]
    )
    work["local_moran_i"] = local_moran.Is
    work["local_moran_p_value"] = local_moran.p_sim
    work["local_moran_quadrant"] = local_moran.q
    significant = local_moran.p_sim < SPATIAL_SETTINGS["significance_level"]
    clusters = np.full(len(work), "Not significant", dtype=object)
    for code, label in QUADRANTS.items():
        clusters[significant & (local_moran.q == code)] = label
    work["local_moran_cluster"] = clusters

    write_table(global_result, args.output_dir / "global_moran.csv")
    write_table(work, args.output_dir / "local_moran.csv")


if __name__ == "__main__":
    main()
