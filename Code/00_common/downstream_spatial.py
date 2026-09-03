from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from downstream_common import add_grid_center_lonlat
from project_common import CHINA_LCC


COMMON_DIR = Path(__file__).resolve().parent
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from province_boundary import load_dissolved_provinces  # noqa: E402


def grid_geodataframe(
    frame: pd.DataFrame, target_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    work = add_grid_center_lonlat(frame)
    points = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(
            work["grid_center_lon"], work["grid_center_lat"]
        ),
        crs="EPSG:4326",
    )
    return points.to_crs(target_crs)


def load_provinces(
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    provinces, _selection, _original, _effective = (
        load_dissolved_provinces()
    )
    return provinces.to_crs(target_crs)


def filter_grid_centers_to_china(frame: pd.DataFrame) -> pd.DataFrame:
    points = grid_geodataframe(frame)
    provinces = load_provinces()
    national = provinces.dissolve()
    boundary = national.geometry.iloc[0]
    keep = points.geometry.apply(boundary.covers)
    print(
        f"  national-boundary grid-centre filter: "
        f"{int(keep.sum()):,}/{len(points):,}"
    )
    return pd.DataFrame(points.loc[keep].drop(columns="geometry"))


def assign_grid_province(frame: pd.DataFrame) -> pd.DataFrame:
    # Some downstream inputs (notably grid_level_decomposition.parquet)
    # already contain a province column. Re-running sjoin with the same
    # column name would make GeoPandas emit province_left/province_right,
    # leaving no canonical "province" column for the summaries below.
    # Province assignment here is deliberately recomputed from the formal
    # 10 km grid centre, so remove any prior assignment before the join.
    work = frame.drop(
        columns=[
            "province",
            "province_left",
            "province_right",
            "index_right",
        ],
        errors="ignore",
    )
    points = grid_geodataframe(work)
    provinces = load_provinces()
    joined = gpd.sjoin(
        points,
        provinces[["province", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = (
        joined.sort_values(
            ["grid_id", "province"], na_position="last"
        )
        .loc[lambda item: ~item.index.duplicated(keep="first")]
        .drop(columns=["geometry", "index_right"], errors="ignore")
    )
    return pd.DataFrame(joined)


def spatial_weight_geodataframe(
    frame: pd.DataFrame, projected_crs: str
) -> gpd.GeoDataFrame:
    # The old Moran/LISA workflow evaluates its 50 km DistanceBand in
    # EPSG:3857. Grid centers are converted from the established China LCC to
    # WGS84 and then to the unchanged old analysis CRS.
    if {"grid_center_x", "grid_center_y"}.issubset(frame.columns):
        points = gpd.GeoDataFrame(
            frame.copy(),
            geometry=gpd.points_from_xy(
                frame["grid_center_x"], frame["grid_center_y"]
            ),
            crs=CHINA_LCC,
        )
        return points.to_crs(projected_crs)
    return grid_geodataframe(frame, target_crs=projected_crs)
