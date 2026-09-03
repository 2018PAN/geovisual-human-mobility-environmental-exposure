from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_plotting import (  # noqa: E402
    NATIONAL_BOUNDARY_COLOR,
    PROVINCE_BOUNDARY_COLOR,
    apply_nature_style,
    projected_extent,
    target_crs,
)
from project_common import (  # noqa: E402
    BOUNDARY_ROOT,
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    read_province_mapping,
)
from province_boundary import (  # noqa: E402
    discover_province_boundary,
    load_dissolved_provinces,
)


MAP_LON_LIMITS = (72.5, 135.5)
MAP_LAT_LIMITS = (2.5, 54.5)
POINT_LIMIT = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replot the three coordinate-assignment diagnostic maps from "
            "persisted parquet outputs using the project China Albers CRS."
        )
    )
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--boundary-root", type=Path, default=BOUNDARY_ROOT)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--max-map-points", type=int, default=POINT_LIMIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sample(frame: pd.DataFrame, maximum: int, *, evenly: bool) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    if evenly:
        positions = np.linspace(0, len(frame) - 1, maximum, dtype="int64")
    else:
        positions = np.random.default_rng(2018).choice(
            len(frame), size=maximum, replace=False
        )
    return frame.iloc[positions]


def _save_map(
    *,
    points: pd.DataFrame,
    total: int,
    output: Path,
    title: str,
    color: str,
    label: str,
    national: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame | None,
) -> None:
    map_crs = target_crs()
    transformer = Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)
    x, y = transformer.transform(
        points["lon"].to_numpy(dtype="float64"),
        points["lat"].to_numpy(dtype="float64"),
    )
    xlim, ylim = projected_extent(MAP_LON_LIMITS, MAP_LAT_LIMITS, transformer)

    apply_nature_style(300)
    fig, ax = plt.subplots(figsize=(7.09, 5.3), dpi=300)
    ax.scatter(
        x,
        y,
        s=0.25,
        c=color,
        alpha=0.30,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    if provinces is not None:
        provinces.boundary.plot(
            ax=ax,
            color=PROVINCE_BOUNDARY_COLOR,
            linewidth=0.26,
            zorder=2,
        )
    national.boundary.plot(
        ax=ax,
        color=NATIONAL_BOUNDARY_COLOR,
        linewidth=0.68,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", fontweight="normal")
    ax.text(
        0.01,
        0.01,
        f"{label}: shown {len(points):,}/{total:,} unique coordinates",
        transform=ax.transAxes,
        fontsize=6.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        zorder=4,
    )
    ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.png")
    fig.savefig(
        temporary,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)
    temporary.replace(output)
    print(f"Created: {output}")


def main() -> None:
    args = parse_args()
    sources = {
        "outside": args.diagnostics_dir / "outside_china_unique_points.parquet",
        "unmatched": args.diagnostics_dir / "unmatched_assignment_results.parquet",
        "national": args.diagnostics_dir / "national_boundary.geojson",
    }
    outputs = {
        "outside": args.diagnostics_dir / "outside_china_points_map.png",
        "before": args.diagnostics_dir / "unmatched_points_before_map.png",
        "after": args.diagnostics_dir / "unmatched_points_after_map.png",
    }
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required persisted diagnostic inputs are missing: "
            + ", ".join(str(path) for path in missing)
        )
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Diagnostic maps exist; rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    mapping = read_province_mapping(args.mapping)
    province_names = mapping["province"].to_numpy(dtype=object)
    selection = discover_province_boundary(args.boundary_root, args.mapping)
    provinces, _selection, _source_crs, _effective_crs = load_dissolved_provinces(
        selection,
        boundary_root=args.boundary_root,
        mapping_path=args.mapping,
        missing_crs="EPSG:4326",
    )
    provinces = provinces.set_index("province").loc[province_names].reset_index()
    provinces["geometry"] = shapely.make_valid(provinces.geometry.array)
    provinces_projected = provinces.to_crs(target_crs())
    national_projected = gpd.read_file(sources["national"]).to_crs(target_crs())

    outside = pd.read_parquet(sources["outside"], columns=["lat", "lon"])
    outside_plot = _sample(outside, args.max_map_points, evenly=True)
    _save_map(
        points=outside_plot,
        total=len(outside),
        output=outputs["outside"],
        title="Coordinates excluded outside the national boundary",
        color="#B24A50",
        label="outside China",
        national=national_projected,
        provinces=None,
    )

    unmatched = pd.read_parquet(
        sources["unmatched"], columns=["lat", "lon", "final_status"]
    )
    before_plot = _sample(unmatched, args.max_map_points, evenly=False)
    _save_map(
        points=before_plot,
        total=len(unmatched),
        output=outputs["before"],
        title="Unmatched App population coordinates before grid fallback",
        color="#762a83",
        label="point-center unmatched",
        national=national_projected,
        provinces=provinces_projected,
    )

    after = unmatched.loc[unmatched["final_status"].eq("unmatched")]
    after_plot = _sample(after, args.max_map_points, evenly=False)
    _save_map(
        points=after_plot,
        total=len(after),
        output=outputs["after"],
        title="Unmatched App population coordinates after spatial QC",
        color="#D73027",
        label="final unmatched",
        national=national_projected,
        provinces=provinces_projected,
    )


if __name__ == "__main__":
    main()
