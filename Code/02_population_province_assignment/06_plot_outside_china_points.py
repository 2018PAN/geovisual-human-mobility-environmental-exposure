from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    DIAGNOSTICS_DIR,
    add_date_arguments,
    validate_date_range,
)
from downstream_plotting import (  # noqa: E402
    NATIONAL_BOUNDARY_COLOR,
    apply_nature_style,
    projected_extent,
    target_crs,
)


def load_parameters(path: Path) -> dict[str, str]:
    metadata = pd.read_csv(path, dtype="string")
    parameters = metadata.loc[
        metadata["metadata_type"].eq("parameter"),
        ["label", "value"],
    ]
    return dict(parameters.itertuples(index=False, name=None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the formal-period outside-China unique-coordinate map "
            "from outside_china_records.parquet."
        )
    )
    parser.add_argument(
        "--outside-records",
        type=Path,
        default=DIAGNOSTICS_DIR / "outside_china_records.parquet",
    )
    parser.add_argument(
        "--outside-summary",
        type=Path,
        default=DIAGNOSTICS_DIR / "outside_china_summary.csv",
    )
    parser.add_argument(
        "--lookup-metadata",
        type=Path,
        default=DIAGNOSTICS_DIR / "coordinate_assignment_metadata.csv",
    )
    parser.add_argument(
        "--national-boundary",
        type=Path,
        default=DIAGNOSTICS_DIR / "national_boundary.geojson",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIAGNOSTICS_DIR / "outside_china_points_map.png",
    )
    parser.add_argument("--batch-size", type=int, default=5_000_000)
    parser.add_argument("--max-map-points", type=int, default=500_000)
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "local":
        raise ValueError(
            "outside_china_records.parquet is selected by Beijing local "
            "dates; use --date-basis local."
        )
    validate_date_range(args.start_date, args.end_date)
    for path in (
        args.outside_records,
        args.outside_summary,
        args.lookup_metadata,
        args.national_boundary,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Map exists; rerun with --overwrite: {args.output}"
        )

    parameters = load_parameters(args.lookup_metadata)
    lat_raw_min = int(parameters["lat_raw_min"])
    lat_raw_max = int(parameters["lat_raw_max"])
    lon_raw_min = int(parameters["lon_raw_min"])
    lon_raw_max = int(parameters["lon_raw_max"])
    nlon = int(parameters["nlon"])
    nlat = lat_raw_max - lat_raw_min + 1
    seen = np.zeros(nlat * nlon, dtype=bool)

    parquet_file = pq.ParquetFile(args.outside_records)
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            batch_size=args.batch_size,
            columns=["lat", "lon"],
        ),
        start=1,
    ):
        frame = batch.to_pandas()
        lat_raw = np.rint(
            pd.to_numeric(frame["lat"], errors="coerce").to_numpy() * 100
        )
        lon_raw = np.rint(
            pd.to_numeric(frame["lon"], errors="coerce").to_numpy() * 100
        )
        valid = (
            np.isfinite(lat_raw)
            & np.isfinite(lon_raw)
            & (lat_raw >= lat_raw_min)
            & (lat_raw <= lat_raw_max)
            & (lon_raw >= lon_raw_min)
            & (lon_raw <= lon_raw_max)
        )
        flat = (
            (lat_raw[valid].astype("int64") - lat_raw_min) * nlon
            + lon_raw[valid].astype("int64")
            - lon_raw_min
        )
        seen[flat] = True
        print(
            f"Outside-map scan batch {batch_number}: rows={len(frame):,}, "
            f"cumulative unique={int(seen.sum()):,}"
        )

    flat = np.flatnonzero(seen)
    lat = (flat // nlon + lat_raw_min).astype("float64") / 100.0
    lon = (flat % nlon + lon_raw_min).astype("float64") / 100.0
    expected_unique = int(
        pd.read_csv(args.outside_summary).iloc[0][
            "unique_coordinate_count"
        ]
    )
    if len(flat) != expected_unique:
        raise ValueError(
            "Formal outside unique-coordinate count does not match summary: "
            f"map={len(flat):,}, summary={expected_unique:,}"
        )

    if len(flat) > args.max_map_points:
        positions = np.linspace(
            0,
            len(flat) - 1,
            args.max_map_points,
            dtype="int64",
        )
        lat_plot = lat[positions]
        lon_plot = lon[positions]
    else:
        lat_plot = lat
        lon_plot = lon
    map_crs = target_crs()
    transformer = Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)
    x, y = transformer.transform(lon_plot, lat_plot)
    xlim, ylim = projected_extent((72.5, 135.5), (2.5, 54.5), transformer)
    national = gpd.read_file(args.national_boundary).to_crs(map_crs)
    apply_nature_style(300)
    fig, ax = plt.subplots(figsize=(7.09, 5.3))
    ax.scatter(
        x,
        y,
        s=0.25,
        c="#B24A50",
        alpha=0.30,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    national.boundary.plot(
        ax=ax,
        color=NATIONAL_BOUNDARY_COLOR,
        linewidth=0.68,
        zorder=2,
    )
    ax.set_title(
        "Coordinates excluded outside the national boundary",
        loc="left",
        fontweight="normal",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.01,
        0.01,
        f"shown {len(lat_plot):,}/{len(flat):,} unique coordinates",
        transform=ax.transAxes,
        fontsize=6.5,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Created: {args.output}")
    print(f"Formal-period outside unique coordinates: {len(flat):,}")


if __name__ == "__main__":
    main()
