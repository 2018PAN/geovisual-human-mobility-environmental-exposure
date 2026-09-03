from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    BOUNDARY_ROOT,
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    POPULATION_INPUT_DIR,
    add_date_arguments,
    read_province_mapping,
    validate_date_range,
)
from province_boundary import (  # noqa: E402
    discover_province_boundary,
    load_dissolved_provinces,
)
from downstream_plotting import (  # noqa: E402
    NATIONAL_BOUNDARY_COLOR,
    PROVINCE_BOUNDARY_COLOR,
    apply_nature_style,
    projected_extent,
    target_crs,
)


AREA_CRS = CRS.from_proj4(
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+ellps=GRS80 +units=m +no_defs"
)

METHOD_LABELS = {
    0: "absent",
    1: "point_interior",
    2: "point_boundary_single",
    3: "point_boundary_multiple_deterministic",
    4: "grid_single_overlap_inside_china",
    5: "grid_largest_overlap_inside_china",
    6: "reserved_no_nearest_assignment",
    7: "unmatched_inside_national_boundary",
    8: "outside_national_boundary",
    9: "abnormal_coordinate",
}


def scan_present_coordinates(
    paths: list[Path],
    *,
    lat_raw_min: int,
    lat_raw_max: int,
    lon_raw_min: int,
    lon_raw_max: int,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    nlon = lon_raw_max - lon_raw_min + 1
    nlat = lat_raw_max - lat_raw_min + 1
    present = np.zeros(nlat * nlon, dtype=bool)
    abnormal_records = 0
    for file_index, path in enumerate(paths, start=1):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=["lat_raw", "lon_raw"]
        ):
            lat_raw = batch.column(0).to_numpy(zero_copy_only=False).astype(
                np.int32, copy=False
            )
            lon_raw = batch.column(1).to_numpy(zero_copy_only=False).astype(
                np.int32, copy=False
            )
            valid = (
                (lat_raw >= lat_raw_min)
                & (lat_raw <= lat_raw_max)
                & (lon_raw >= lon_raw_min)
                & (lon_raw <= lon_raw_max)
            )
            abnormal_records += int((~valid).sum())
            lat_index = lat_raw[valid] - lat_raw_min
            lon_index = lon_raw[valid] - lon_raw_min
            present[lat_index.astype(np.int64) * nlon + lon_index] = True
        print(
            f"Coordinate scan [{file_index}/{len(paths)}] {path.name}: "
            f"cumulative unique={int(present.sum()):,}"
        )
    return present, abnormal_records


def flat_to_coordinates(
    flat: np.ndarray,
    *,
    lat_raw_min: int,
    lon_raw_min: int,
    nlon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_index = flat // nlon
    lon_index = flat % nlon
    lat_raw = (lat_index + lat_raw_min).astype(np.int16)
    lon_raw = (lon_index + lon_raw_min).astype(np.int16)
    return (
        lat_raw,
        lon_raw,
        lat_raw.astype(np.float64) / 100.0,
        lon_raw.astype(np.float64) / 100.0,
    )


def build_rasters(
    provinces: gpd.GeoDataFrame,
    province_code: dict[str, int],
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    nlat = int(round((lat_max - lat_min) / resolution)) + 1
    nlon = int(round((lon_max - lon_min) / resolution)) + 1
    transform = from_origin(
        lon_min - resolution / 2,
        lat_max + resolution / 2,
        resolution,
        resolution,
    )
    shapes = [
        (geometry, int(province_code[province]))
        for province, geometry in provinces[["province", "geometry"]].itertuples(
            index=False
        )
    ]
    point_codes = rasterize(
        shapes,
        out_shape=(nlat, nlon),
        transform=transform,
        fill=-1,
        dtype="int16",
        all_touched=False,
    )
    boundary_shapes = [
        (geometry.boundary, 1)
        for geometry in provinces.geometry
        if geometry is not None and not geometry.is_empty
    ]
    boundary_mask = rasterize(
        boundary_shapes,
        out_shape=(nlat, nlon),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    return point_codes[::-1].reshape(-1), boundary_mask[::-1].reshape(-1)


def exact_point_check(
    candidate_flat: np.ndarray,
    *,
    province_lookup: np.ndarray,
    method_lookup: np.ndarray,
    provinces: gpd.GeoDataFrame,
    province_codes_array: np.ndarray,
    lat_raw_min: int,
    lon_raw_min: int,
    nlon: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = shapely.STRtree(provinces.geometry.array)
    fallback_parts: list[np.ndarray] = []
    boundary_resolved_parts: list[np.ndarray] = []
    for start in range(0, len(candidate_flat), chunk_size):
        flat = candidate_flat[start : start + chunk_size]
        _lat_raw, _lon_raw, lat, lon = flat_to_coordinates(
            flat,
            lat_raw_min=lat_raw_min,
            lon_raw_min=lon_raw_min,
            nlon=nlon,
        )
        points = shapely.points(lon, lat)
        pairs = tree.query(points, predicate="intersects")
        pair_counts = np.bincount(pairs[0], minlength=len(flat))
        first_pair = np.full(len(flat), -1, dtype=np.int64)
        if pairs.shape[1]:
            order = np.argsort(pairs[0], kind="stable")
            left_sorted = pairs[0, order]
            first_positions = np.r_[0, np.flatnonzero(np.diff(left_sorted)) + 1]
            first_pair[left_sorted[first_positions]] = order[first_positions]

        fallback = pair_counts == 0
        single = pair_counts == 1
        multiple = pair_counts > 1
        fallback |= multiple
        if single.any():
            local_positions = np.flatnonzero(single)
            pair_positions = first_pair[local_positions]
            right = pairs[1, pair_positions]
            point_geometries = points[local_positions]
            province_geometries = provinces.geometry.array[right]
            within = shapely.within(point_geometries, province_geometries)
            touches = shapely.touches(point_geometries, province_geometries)
            exact_codes = province_codes_array[right]
            within_positions = local_positions[within]
            boundary_positions = local_positions[~within & touches]
            other_positions = local_positions[~within & ~touches]
            if len(within_positions):
                province_lookup[flat[within_positions]] = exact_codes[within]
                method_lookup[flat[within_positions]] = 2
            if len(boundary_positions):
                boundary_codes = exact_codes[~within & touches]
                province_lookup[flat[boundary_positions]] = boundary_codes
                method_lookup[flat[boundary_positions]] = 3
                boundary_resolved_parts.append(flat[boundary_positions])
                fallback[boundary_positions] = True
            if len(other_positions):
                fallback[other_positions] = True
        province_lookup[flat[multiple]] = -1
        method_lookup[flat[multiple]] = 7
        fallback_parts.append(flat[fallback])
        print(
            f"  exact point chunk {start // chunk_size + 1}: "
            f"candidates={len(flat):,}, fallback={int(fallback.sum()):,}"
        )
    fallback_flat = (
        np.concatenate(fallback_parts) if fallback_parts else np.array([], dtype=np.int64)
    )
    boundary_resolved = (
        np.concatenate(boundary_resolved_parts)
        if boundary_resolved_parts
        else np.array([], dtype=np.int64)
    )
    return np.unique(fallback_flat), np.unique(boundary_resolved)


def resolve_grid_and_nearest(
    fallback_flat: np.ndarray,
    boundary_resolved: np.ndarray,
    *,
    province_lookup: np.ndarray,
    method_lookup: np.ndarray,
    provinces: gpd.GeoDataFrame,
    province_codes_array: np.ndarray,
    lat_raw_min: int,
    lon_raw_min: int,
    nlon: int,
    grid_size: float,
    nearest_threshold_m: float,
    nearshore_distance_m: float,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    n = len(fallback_flat)
    final_code = province_lookup[fallback_flat].astype(np.int16, copy=True)
    method_code = method_lookup[fallback_flat].astype(np.int8, copy=True)
    intersected_count = np.zeros(n, dtype=np.uint8)
    largest_area = np.full(n, np.nan, dtype=np.float64)
    largest_ratio = np.full(n, np.nan, dtype=np.float64)
    nearest_code = np.full(n, -1, dtype=np.int16)
    nearest_distance = np.full(n, np.nan, dtype=np.float64)
    classification_code = np.full(n, 6, dtype=np.int8)

    if len(boundary_resolved):
        boundary_positions = np.searchsorted(fallback_flat, boundary_resolved)
        in_bounds = boundary_positions < len(fallback_flat)
        boundary_positions = boundary_positions[in_bounds]
        boundary_values = boundary_resolved[in_bounds]
        boundary_positions = boundary_positions[
            fallback_flat[boundary_positions] == boundary_values
        ]
        classification_code[boundary_positions] = 1
        intersected_count[boundary_positions] = 1

    geographic_tree = shapely.STRtree(provinces.geometry.array)
    transformer = Transformer.from_crs(
        "EPSG:4326", AREA_CRS, always_xy=True
    )
    provinces_area = shapely.transform(
        provinces.geometry.array,
        transformer.transform,
        interleaved=False,
    )
    area_tree = shapely.STRtree(provinces_area)
    half = grid_size / 2.0

    unresolved_positions = np.flatnonzero(final_code < 0)
    for chunk_number, start in enumerate(
        range(0, len(unresolved_positions), chunk_size), start=1
    ):
        positions = unresolved_positions[start : start + chunk_size]
        flat = fallback_flat[positions]
        _lat_raw, _lon_raw, lat, lon = flat_to_coordinates(
            flat,
            lat_raw_min=lat_raw_min,
            lon_raw_min=lon_raw_min,
            nlon=nlon,
        )
        boxes = shapely.box(
            lon - half, lat - half, lon + half, lat + half
        )
        pairs = geographic_tree.query(boxes, predicate="intersects")
        pair_counts = np.bincount(pairs[0], minlength=len(boxes))
        intersected_count[positions] = np.minimum(pair_counts, 255).astype(
            np.uint8
        )

        assigned_local = np.zeros(len(boxes), dtype=bool)
        if pairs.shape[1]:
            boxes_area = shapely.transform(
                boxes, transformer.transform, interleaved=False
            )
            intersections = shapely.intersection(
                boxes_area[pairs[0]], provinces_area[pairs[1]]
            )
            areas = shapely.area(intersections)
            positive = areas > 0
            left = pairs[0, positive]
            right = pairs[1, positive]
            positive_areas = areas[positive]
            if len(left):
                order = np.lexsort((-positive_areas, left))
                left_sorted = left[order]
                first_positions = np.r_[
                    0, np.flatnonzero(np.diff(left_sorted)) + 1
                ]
                chosen_order = order[first_positions]
                chosen_left = left[chosen_order]
                chosen_right = right[chosen_order]
                chosen_area = positive_areas[chosen_order]
                grid_areas = shapely.area(boxes_area[chosen_left])
                global_positions = positions[chosen_left]
                final_code[global_positions] = province_codes_array[
                    chosen_right
                ]
                largest_area[global_positions] = chosen_area
                largest_ratio[global_positions] = np.divide(
                    chosen_area,
                    grid_areas,
                    out=np.zeros_like(chosen_area),
                    where=grid_areas > 0,
                )
                positive_counts = np.bincount(
                    left, minlength=len(boxes)
                )
                intersected_count[global_positions] = np.minimum(
                    positive_counts[chosen_left], 255
                ).astype(np.uint8)
                single = positive_counts[chosen_left] == 1
                method_code[global_positions[single]] = 4
                method_code[global_positions[~single]] = 5
                classification_code[global_positions[single]] = 2
                classification_code[global_positions[~single]] = 3
                assigned_local[chosen_left] = True

        no_grid_local = np.flatnonzero(~assigned_local)
        if len(no_grid_local):
            points = shapely.points(lon[no_grid_local], lat[no_grid_local])
            points_area = shapely.transform(
                points, transformer.transform, interleaved=False
            )
            nearest_indices = area_tree.nearest(points_area)
            distances = shapely.distance(
                points_area, provinces_area[nearest_indices]
            )
            global_positions = positions[no_grid_local]
            nearest_code[global_positions] = province_codes_array[
                nearest_indices
            ]
            nearest_distance[global_positions] = distances
            within_tolerance = distances <= nearest_threshold_m
            if within_tolerance.any():
                accepted = global_positions[within_tolerance]
                final_code[accepted] = province_codes_array[
                    nearest_indices[within_tolerance]
                ]
                method_code[accepted] = 6
                classification_code[accepted] = 3

            rejected_positions = global_positions[~within_tolerance]
            rejected_distances = distances[~within_tolerance]
            if len(rejected_positions):
                _lr, _lor, rejected_lat, rejected_lon = flat_to_coordinates(
                    fallback_flat[rejected_positions],
                    lat_raw_min=lat_raw_min,
                    lon_raw_min=lon_raw_min,
                    nlon=nlon,
                )
                bounds = provinces.total_bounds
                abnormal = (
                    ~np.isfinite(rejected_lat)
                    | ~np.isfinite(rejected_lon)
                    | (rejected_lat < 3)
                    | (rejected_lat > 54)
                    | (rejected_lon < 73)
                    | (rejected_lon > 135)
                )
                outside_extent = (
                    (rejected_lat < bounds[1] - half)
                    | (rejected_lat > bounds[3] + half)
                    | (rejected_lon < bounds[0] - half)
                    | (rejected_lon > bounds[2] + half)
                )
                nearshore = (
                    ~abnormal
                    & ~outside_extent
                    & (rejected_distances <= nearshore_distance_m)
                )
                method_code[rejected_positions[abnormal]] = 9
                classification_code[rejected_positions[abnormal]] = 5
                method_code[
                    rejected_positions[~abnormal & outside_extent]
                ] = 8
                classification_code[
                    rejected_positions[~abnormal & outside_extent]
                ] = 4
                method_code[rejected_positions[nearshore]] = 7
                classification_code[rejected_positions[nearshore]] = 3
                remaining = ~abnormal & ~outside_extent & ~nearshore
                method_code[rejected_positions[remaining]] = 7
                classification_code[rejected_positions[remaining]] = 6
        print(
            f"  grid chunk {chunk_number}: candidates={len(positions):,}, "
            f"grid-assigned={int(assigned_local.sum()):,}, "
            f"remaining={int((final_code[positions] < 0).sum()):,}"
        )

    province_lookup[fallback_flat] = final_code.astype(np.int16)
    method_lookup[fallback_flat] = method_code
    return {
        "flat": fallback_flat,
        "final_code": final_code,
        "method_code": method_code,
        "intersected_count": intersected_count,
        "largest_area": largest_area,
        "largest_ratio": largest_ratio,
        "nearest_code": nearest_code,
        "nearest_distance": nearest_distance,
        "classification_code": classification_code,
    }


def classification_label(code: np.ndarray) -> np.ndarray:
    labels = np.array(
        [
            "not_applicable",
            "A_boundary_point",
            "B_grid_intersects_one_province",
            "C_nearshore_or_complex_coast",
            "D_outside_34_region_extent",
            "E_abnormal_coordinate",
            "F_unresolved",
        ],
        dtype=object,
    )
    return labels[code]


def method_label(code: np.ndarray) -> np.ndarray:
    labels = np.array(
        [METHOD_LABELS[index] for index in range(max(METHOD_LABELS) + 1)],
        dtype=object,
    )
    return labels[code]


def write_lookup_parquet(
    path: Path,
    present_flat: np.ndarray,
    province_lookup: np.ndarray,
    method_lookup: np.ndarray,
    province_names: np.ndarray,
    *,
    lat_raw_min: int,
    lon_raw_min: int,
    nlon: int,
    chunk_size: int,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; rerun with --overwrite: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        temp.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(present_flat), chunk_size):
            flat = present_flat[start : start + chunk_size]
            lat_raw, lon_raw, lat, lon = flat_to_coordinates(
                flat,
                lat_raw_min=lat_raw_min,
                lon_raw_min=lon_raw_min,
                nlon=nlon,
            )
            codes = province_lookup[flat]
            provinces = np.full(len(flat), None, dtype=object)
            matched = codes >= 0
            provinces[matched] = province_names[codes[matched]]
            frame = pd.DataFrame(
                {
                    "coord_flat_index": flat,
                    "lat_raw": lat_raw,
                    "lon_raw": lon_raw,
                    "lat": lat,
                    "lon": lon,
                    "final_province_code": codes,
                    "final_province": provinces,
                    "assignment_method_code": method_lookup[flat],
                    "assignment_method": method_label(method_lookup[flat]),
                    "final_status": np.where(matched, "assigned", "unmatched"),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No coordinate lookup rows were written")
    temp.replace(path)


def write_fallback_outputs(
    before_path: Path,
    results_path: Path,
    result: dict[str, np.ndarray],
    province_names: np.ndarray,
    *,
    lat_raw_min: int,
    lon_raw_min: int,
    nlon: int,
    chunk_size: int,
    overwrite: bool,
) -> None:
    for path in (before_path, results_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists; rerun with --overwrite: {path}")
    before_temp = before_path.with_name(f".{before_path.name}.tmp")
    results_temp = results_path.with_name(f".{results_path.name}.tmp")
    for path in (before_temp, results_temp):
        if path.exists():
            path.unlink()
    before_writer: pq.ParquetWriter | None = None
    result_writer: pq.ParquetWriter | None = None
    flat_all = result["flat"]
    try:
        for start in range(0, len(flat_all), chunk_size):
            end = min(start + chunk_size, len(flat_all))
            flat = flat_all[start:end]
            lat_raw, lon_raw, lat, lon = flat_to_coordinates(
                flat,
                lat_raw_min=lat_raw_min,
                lon_raw_min=lon_raw_min,
                nlon=nlon,
            )
            before = pd.DataFrame(
                {
                    "coord_flat_index": flat,
                    "lat_raw": lat_raw,
                    "lon_raw": lon_raw,
                    "lat": lat,
                    "lon": lon,
                    "original_match_status": "point_unmatched",
                }
            )
            final_codes = result["final_code"][start:end]
            nearest_codes = result["nearest_code"][start:end]
            final_provinces = np.full(len(flat), None, dtype=object)
            nearest_provinces = np.full(len(flat), None, dtype=object)
            final_matched = final_codes >= 0
            nearest_matched = nearest_codes >= 0
            final_provinces[final_matched] = province_names[
                final_codes[final_matched]
            ]
            nearest_provinces[nearest_matched] = province_names[
                nearest_codes[nearest_matched]
            ]
            methods = method_label(result["method_code"][start:end])
            classes = classification_label(
                result["classification_code"][start:end]
            )
            reasons = np.where(
                methods == "point_boundary_inclusive",
                "point lies on an administrative boundary and is covered by one province",
                np.where(
                    methods == "grid_single_overlap",
                    "0.1 degree cell intersects exactly one province",
                    np.where(
                        methods == "grid_largest_overlap",
                        "0.1 degree cell intersects multiple provinces; largest equal-area overlap selected",
                        np.where(
                            methods == "nearest_boundary_tolerance",
                            "no cell overlap; nearest boundary is within explicit numerical-tolerance threshold",
                            "no permitted automatic assignment rule was satisfied",
                        ),
                    ),
                ),
            )
            results = pd.DataFrame(
                {
                    "coord_flat_index": flat,
                    "lat_raw": lat_raw,
                    "lon_raw": lon_raw,
                    "lat": lat,
                    "lon": lon,
                    "original_match_status": "point_unmatched",
                    "classification": classes,
                    "final_province": final_provinces,
                    "assignment_method": methods,
                    "intersected_province_count": result[
                        "intersected_count"
                    ][start:end],
                    "largest_overlap_area": result["largest_area"][start:end],
                    "largest_overlap_ratio": result["largest_ratio"][start:end],
                    "nearest_province": nearest_provinces,
                    "nearest_boundary_distance_m": result[
                        "nearest_distance"
                    ][start:end],
                    "final_status": np.where(
                        final_matched, "assigned", "unmatched"
                    ),
                    "decision_reason": reasons,
                }
            )
            before_table = pa.Table.from_pandas(before, preserve_index=False)
            result_table = pa.Table.from_pandas(results, preserve_index=False)
            if before_writer is None:
                before_writer = pq.ParquetWriter(
                    before_temp, before_table.schema, compression="zstd"
                )
                result_writer = pq.ParquetWriter(
                    results_temp, result_table.schema, compression="zstd"
                )
            before_writer.write_table(before_table)
            result_writer.write_table(result_table)
    finally:
        if before_writer is not None:
            before_writer.close()
        if result_writer is not None:
            result_writer.close()
    if before_writer is None or result_writer is None:
        raise ValueError("No fallback coordinate rows were written")
    before_temp.replace(before_path)
    results_temp.replace(results_path)


def plot_unmatched(
    provinces: gpd.GeoDataFrame,
    fallback_flat: np.ndarray,
    final_codes: np.ndarray,
    path: Path,
    *,
    lat_raw_min: int,
    lon_raw_min: int,
    nlon: int,
    after: bool,
    max_points: int = 500_000,
) -> None:
    apply_nature_style(300)
    rng = np.random.default_rng(2018)
    if after:
        selected = np.flatnonzero(final_codes < 0)
        flat = fallback_flat[selected]
        color = "#d73027"
        label = "final unmatched"
    else:
        flat = fallback_flat
        color = "#762a83"
        label = "point-center unmatched"
    total = len(flat)
    if total > max_points:
        flat = flat[rng.choice(total, size=max_points, replace=False)]
    _lat_raw, _lon_raw, lat, lon = flat_to_coordinates(
        flat,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
    )
    map_crs = target_crs()
    transformer = Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    xlim, ylim = projected_extent((72.5, 135.5), (2.5, 54.5), transformer)
    provinces_projected = provinces.to_crs(map_crs)

    fig, ax = plt.subplots(figsize=(7.09, 5.3), dpi=300)
    ax.scatter(
        x,
        y,
        s=0.25,
        alpha=0.30,
        c=color,
        rasterized=True,
        label=f"{label} (shown {len(flat):,}/{total:,})",
        zorder=1,
    )
    provinces_projected.boundary.plot(
        ax=ax,
        linewidth=0.26,
        color=PROVINCE_BOUNDARY_COLOR,
        zorder=2,
    )
    provinces_projected.dissolve().boundary.plot(
        ax=ax,
        linewidth=0.68,
        color=NATIONAL_BOUNDARY_COLOR,
        zorder=3,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(
        "Unmatched App population coordinates after spatial QC"
        if after
        else "Unmatched App population coordinates before grid fallback",
        loc="left",
        fontweight="normal",
    )
    ax.legend(loc="lower left", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable province assignment for all unique App coordinates "
            "using boundary-inclusive points, 0.1-degree cell overlap, and a "
            "bounded nearest-boundary tolerance."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument("--boundary-root", type=Path, default=BOUNDARY_ROOT)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    add_date_arguments(parser, default_basis="utc")
    parser.add_argument("--lat-min", type=float, default=3.0)
    parser.add_argument("--lat-max", type=float, default=54.0)
    parser.add_argument("--lon-min", type=float, default=73.0)
    parser.add_argument("--lon-max", type=float, default=135.0)
    parser.add_argument("--coordinate-resolution", type=float, default=0.01)
    parser.add_argument("--fallback-grid-size", type=float, default=0.1)
    parser.add_argument("--nearest-threshold-m", type=float, default=100.0)
    parser.add_argument("--nearshore-distance-m", type=float, default=50_000.0)
    parser.add_argument("--scan-batch-size", type=int, default=5_000_000)
    parser.add_argument("--spatial-chunk-size", type=int, default=250_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "utc":
        raise ValueError("Unique-coordinate source scan requires --date-basis utc")
    validate_date_range(args.start_date, args.end_date)
    paths = []
    for day in pd.date_range(args.start_date, args.end_date, freq="D"):
        path = args.input_dir / f"population_china_{day:%Y-%m-%d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Daily parquet missing: {path}")
        paths.append(path)

    mapping = read_province_mapping(args.mapping)
    province_names = mapping["province"].to_numpy(dtype=object)
    province_code = {
        province: index for index, province in enumerate(province_names)
    }
    selection = discover_province_boundary(args.boundary_root, args.mapping)
    provinces, _selection, original_crs, effective_crs = load_dissolved_provinces(
        selection,
        boundary_root=args.boundary_root,
        mapping_path=args.mapping,
        missing_crs="EPSG:4326",
    )
    provinces = provinces.set_index("province").loc[province_names].reset_index()
    provinces["geometry"] = shapely.make_valid(provinces.geometry.array)
    if int((~provinces.geometry.is_valid).sum()):
        raise ValueError("Dissolved province geometry remains invalid")
    province_codes_array = np.array(
        [province_code[value] for value in provinces["province"]],
        dtype=np.int16,
    )

    lat_raw_min = int(round(args.lat_min * 100))
    lat_raw_max = int(round(args.lat_max * 100))
    lon_raw_min = int(round(args.lon_min * 100))
    lon_raw_max = int(round(args.lon_max * 100))
    nlon = lon_raw_max - lon_raw_min + 1
    nlat = lat_raw_max - lat_raw_min + 1
    total_cells = nlat * nlon

    present, abnormal_records = scan_present_coordinates(
        paths,
        lat_raw_min=lat_raw_min,
        lat_raw_max=lat_raw_max,
        lon_raw_min=lon_raw_min,
        lon_raw_max=lon_raw_max,
        batch_size=args.scan_batch_size,
    )
    present_flat = np.flatnonzero(present)
    print(f"Full unique coordinate count: {len(present_flat):,}")
    print(f"Records outside configured coordinate domain: {abnormal_records:,}")

    point_raster, boundary_mask = build_rasters(
        provinces,
        province_code,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        resolution=args.coordinate_resolution,
    )
    if len(point_raster) != total_cells:
        raise ValueError("Raster lookup size does not match coordinate domain")
    province_lookup = np.full(total_cells, -2, dtype=np.int16)
    method_lookup = np.zeros(total_cells, dtype=np.int8)
    raster_codes = point_raster[present_flat]
    province_lookup[present_flat] = raster_codes
    method_lookup[present_flat[raster_codes >= 0]] = 1
    method_lookup[present_flat[raster_codes < 0]] = 7

    exact_candidates = present_flat[
        (raster_codes < 0) | boundary_mask[present_flat]
    ]
    print(f"Exact boundary/point candidates: {len(exact_candidates):,}")
    fallback_flat, boundary_resolved = exact_point_check(
        exact_candidates,
        province_lookup=province_lookup,
        method_lookup=method_lookup,
        provinces=provinces,
        province_codes_array=province_codes_array,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
        chunk_size=args.spatial_chunk_size,
    )
    print(f"Original point-unmatched unique coordinates: {len(fallback_flat):,}")
    print(f"Boundary-inclusive resolutions: {len(boundary_resolved):,}")

    result = resolve_grid_and_nearest(
        fallback_flat,
        boundary_resolved,
        province_lookup=province_lookup,
        method_lookup=method_lookup,
        provinces=provinces,
        province_codes_array=province_codes_array,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
        grid_size=args.fallback_grid_size,
        nearest_threshold_m=args.nearest_threshold_m,
        nearshore_distance_m=args.nearshore_distance_m,
        chunk_size=args.spatial_chunk_size,
    )

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    lookup_path = args.diagnostics_dir / "coordinate_province_assignment.parquet"
    province_npy = args.diagnostics_dir / "coordinate_province_code_lookup.npy"
    method_npy = args.diagnostics_dir / "coordinate_assignment_method_lookup.npy"
    metadata_path = args.diagnostics_dir / "coordinate_assignment_metadata.csv"
    before_path = args.diagnostics_dir / "unmatched_unique_points.parquet"
    results_path = args.diagnostics_dir / "unmatched_assignment_results.parquet"
    unique_summary_path = (
        args.diagnostics_dir / "unmatched_unique_summary_by_method.csv"
    )
    before_map = args.diagnostics_dir / "unmatched_points_before_map.png"
    after_map = args.diagnostics_dir / "unmatched_points_after_map.png"
    outputs = [
        lookup_path,
        province_npy,
        method_npy,
        metadata_path,
        before_path,
        results_path,
        unique_summary_path,
        before_map,
        after_map,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Coordinate assignment outputs exist; rerun with --overwrite: "
            + ", ".join(map(str, existing))
        )

    write_lookup_parquet(
        lookup_path,
        present_flat,
        province_lookup,
        method_lookup,
        province_names,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
        chunk_size=args.spatial_chunk_size,
        overwrite=args.overwrite,
    )
    write_fallback_outputs(
        before_path,
        results_path,
        result,
        province_names,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
        chunk_size=args.spatial_chunk_size,
        overwrite=args.overwrite,
    )
    np.save(province_npy, province_lookup)
    np.save(method_npy, method_lookup)
    metadata_rows = [
        {
            "metadata_type": "province_code",
            "code": index,
            "label": province,
            "value": "",
        }
        for index, province in enumerate(province_names)
    ] + [
        {
            "metadata_type": "assignment_method",
            "code": code,
            "label": label,
            "value": "",
        }
        for code, label in METHOD_LABELS.items()
    ] + [
        {
            "metadata_type": "parameter",
            "code": "",
            "label": name,
            "value": value,
        }
        for name, value in {
            "lat_raw_min": lat_raw_min,
            "lat_raw_max": lat_raw_max,
            "lon_raw_min": lon_raw_min,
            "lon_raw_max": lon_raw_max,
            "nlon": nlon,
            "coordinate_resolution_degree": args.coordinate_resolution,
            "fallback_grid_size_degree": args.fallback_grid_size,
            "nearest_threshold_m": args.nearest_threshold_m,
            "nearshore_distance_m": args.nearshore_distance_m,
            "area_crs": AREA_CRS.to_string(),
            "boundary_file": str(selection.path),
            "boundary_source_crs": original_crs or "MISSING",
            "boundary_effective_crs": effective_crs,
        }.items()
    ]
    pd.DataFrame(metadata_rows).to_csv(
        metadata_path, index=False, encoding="utf-8-sig"
    )

    method_counts = pd.Series(
        method_label(method_lookup[present_flat])
    ).value_counts()
    unique_summary = pd.DataFrame(
        {
            "assignment_method": method_counts.index,
            "unique_coordinate_count": method_counts.values,
            "unique_coordinate_share": method_counts.values
            / len(present_flat),
        }
    )
    unique_summary.to_csv(
        unique_summary_path, index=False, encoding="utf-8-sig"
    )
    plot_unmatched(
        provinces,
        result["flat"],
        result["final_code"],
        before_map,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
        after=False,
    )
    plot_unmatched(
        provinces,
        result["flat"],
        result["final_code"],
        after_map,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
        after=True,
    )

    final_unmatched = int((province_lookup[present_flat] < 0).sum())
    print(f"Created: {lookup_path}")
    print(f"Created: {before_path}")
    print(f"Created: {results_path}")
    print(f"Created: {before_map}")
    print(f"Created: {after_map}")
    print(f"Final unmatched unique coordinates: {final_unmatched:,}")
    print(f"Nearest automatic threshold: {args.nearest_threshold_m:.3f} m")
    print(f"Equal-area CRS: {AREA_CRS.to_string()}")


def _load_or_scan_coordinate_inventory(
    args: argparse.Namespace,
    paths: list[Path],
    *,
    lat_raw_min: int,
    lat_raw_max: int,
    lon_raw_min: int,
    lon_raw_max: int,
    nlon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inventory_path = (
        args.diagnostics_dir / "coordinate_province_assignment.parquet"
    )
    if inventory_path.exists():
        inventory = pd.read_parquet(
            inventory_path,
            columns=["coord_flat_index", "lat_raw", "lon_raw"],
        )
        inventory = inventory.drop_duplicates("coord_flat_index").sort_values(
            "coord_flat_index"
        )
        print(
            "Reusing existing full-range unique-coordinate inventory: "
            f"{inventory_path} ({len(inventory):,} coordinates)"
        )
        return (
            inventory["coord_flat_index"].to_numpy(dtype="int64"),
            inventory["lat_raw"].to_numpy(dtype="int16"),
            inventory["lon_raw"].to_numpy(dtype="int16"),
        )

    present, abnormal_records = scan_present_coordinates(
        paths,
        lat_raw_min=lat_raw_min,
        lat_raw_max=lat_raw_max,
        lon_raw_min=lon_raw_min,
        lon_raw_max=lon_raw_max,
        batch_size=args.scan_batch_size,
    )
    if abnormal_records:
        print(
            "Records outside configured coordinate domain during inventory "
            f"scan: {abnormal_records:,}"
        )
    flat = np.flatnonzero(present)
    lat_raw, lon_raw, _lat, _lon = flat_to_coordinates(
        flat,
        lat_raw_min=lat_raw_min,
        lon_raw_min=lon_raw_min,
        nlon=nlon,
    )
    return flat, lat_raw, lon_raw


def _national_exact_assignment(
    present_flat: np.ndarray,
    lat_raw: np.ndarray,
    lon_raw: np.ndarray,
    *,
    provinces: gpd.GeoDataFrame,
    province_codes_array: np.ndarray,
    total_cells: int,
    grid_size: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    province_lookup = np.full(total_cells, -2, dtype=np.int16)
    method_lookup = np.zeros(total_cells, dtype=np.int8)
    national_lookup = np.full(total_cells, -1, dtype=np.int8)
    province_lookup[present_flat] = -1

    national_boundary = shapely.union_all(provinces.geometry.array)
    national_boundary = shapely.make_valid(national_boundary)
    if not shapely.is_valid(national_boundary):
        raise ValueError("national_boundary remains invalid after make_valid")
    shapely.prepare(national_boundary)
    province_tree = shapely.STRtree(provinces.geometry.array)
    stats = {
        "inside_unique_coordinates": 0,
        "outside_unique_coordinates": 0,
        "point_interior_unique_coordinates": 0,
        "point_boundary_single_unique_coordinates": 0,
        "point_boundary_multiple_unique_coordinates": 0,
        "grid_single_unique_coordinates": 0,
        "grid_multiple_unique_coordinates": 0,
        "inside_unmatched_unique_coordinates": 0,
    }
    inside_unmatched_flat_parts: list[np.ndarray] = []

    for chunk_number, start in enumerate(
        range(0, len(present_flat), chunk_size),
        start=1,
    ):
        stop = min(start + chunk_size, len(present_flat))
        flat = present_flat[start:stop]
        lat = lat_raw[start:stop].astype("float64") / 100.0
        lon = lon_raw[start:stop].astype("float64") / 100.0
        points = shapely.points(lon, lat)
        inside = np.asarray(
            shapely.covers(national_boundary, points),
            dtype=bool,
        )
        national_lookup[flat] = inside.astype(np.int8)
        method_lookup[flat[~inside]] = 8
        stats["inside_unique_coordinates"] += int(inside.sum())
        stats["outside_unique_coordinates"] += int((~inside).sum())

        inside_positions = np.flatnonzero(inside)
        if len(inside_positions):
            inside_points = points[inside_positions]
            pairs = province_tree.query(
                inside_points,
                predicate="intersects",
            )
            pair_counts = np.bincount(
                pairs[0],
                minlength=len(inside_positions),
            )
            if pairs.shape[1]:
                pair_codes = province_codes_array[pairs[1]]
                order = np.lexsort((pair_codes, pairs[0]))
                left_sorted = pairs[0, order]
                first = np.r_[
                    0,
                    np.flatnonzero(np.diff(left_sorted)) + 1,
                ]
                selected_pair_positions = order[first]
                selected_left = pairs[0, selected_pair_positions]
                selected_right = pairs[1, selected_pair_positions]
                selected_global_positions = inside_positions[selected_left]
                selected_codes = province_codes_array[selected_right]
                province_lookup[flat[selected_global_positions]] = (
                    selected_codes
                )

                multiple = pair_counts[selected_left] > 1
                selected_points = inside_points[selected_left]
                selected_geometries = provinces.geometry.array[
                    selected_right
                ]
                within = np.asarray(
                    shapely.within(selected_points, selected_geometries),
                    dtype=bool,
                )
                single_boundary = ~multiple & ~within
                single_interior = ~multiple & within
                method_lookup[
                    flat[selected_global_positions[single_interior]]
                ] = 1
                method_lookup[
                    flat[selected_global_positions[single_boundary]]
                ] = 2
                method_lookup[
                    flat[selected_global_positions[multiple]]
                ] = 3
                stats["point_interior_unique_coordinates"] += int(
                    single_interior.sum()
                )
                stats["point_boundary_single_unique_coordinates"] += int(
                    single_boundary.sum()
                )
                stats["point_boundary_multiple_unique_coordinates"] += int(
                    multiple.sum()
                )

            unresolved_inside = inside_positions[pair_counts == 0]
            if len(unresolved_inside):
                inside_unmatched_flat_parts.append(flat[unresolved_inside])

        print(
            f"National/province exact chunk {chunk_number}: "
            f"coordinates={len(flat):,}, inside={int(inside.sum()):,}, "
            f"outside={int((~inside).sum()):,}, "
            f"inside exact-unmatched="
            f"{int((inside & (province_lookup[flat] < 0)).sum()):,}"
        )

    unresolved_flat = (
        np.concatenate(inside_unmatched_flat_parts)
        if inside_unmatched_flat_parts
        else np.array([], dtype="int64")
    )
    if len(unresolved_flat):
        transformer = Transformer.from_crs(
            "EPSG:4326",
            AREA_CRS,
            always_xy=True,
        )
        provinces_area = shapely.transform(
            provinces.geometry.array,
            transformer.transform,
            interleaved=False,
        )
        province_tree = shapely.STRtree(provinces.geometry.array)
        half = grid_size / 2.0
        for start in range(0, len(unresolved_flat), chunk_size):
            flat = unresolved_flat[start : start + chunk_size]
            # Coordinates are recovered through the already-aligned inventory
            # index, avoiding any filename/date inference.
            inventory_positions = np.searchsorted(present_flat, flat)
            lat = (
                lat_raw[inventory_positions].astype("float64") / 100.0
            )
            lon = (
                lon_raw[inventory_positions].astype("float64") / 100.0
            )
            boxes = shapely.box(
                lon - half,
                lat - half,
                lon + half,
                lat + half,
            )
            pairs = province_tree.query(boxes, predicate="intersects")
            if not pairs.shape[1]:
                continue
            boxes_area = shapely.transform(
                boxes,
                transformer.transform,
                interleaved=False,
            )
            intersections = shapely.intersection(
                boxes_area[pairs[0]],
                provinces_area[pairs[1]],
            )
            areas = shapely.area(intersections)
            positive = areas > 0
            left = pairs[0, positive]
            right = pairs[1, positive]
            areas = areas[positive]
            if not len(left):
                continue
            codes = province_codes_array[right]
            order = np.lexsort((codes, -areas, left))
            left_sorted = left[order]
            first = np.r_[0, np.flatnonzero(np.diff(left_sorted)) + 1]
            selected = order[first]
            selected_left = left[selected]
            selected_right = right[selected]
            positive_counts = np.bincount(left, minlength=len(boxes))
            selected_flat = flat[selected_left]
            province_lookup[selected_flat] = province_codes_array[
                selected_right
            ]
            single = positive_counts[selected_left] == 1
            method_lookup[selected_flat[single]] = 4
            method_lookup[selected_flat[~single]] = 5
            stats["grid_single_unique_coordinates"] += int(single.sum())
            stats["grid_multiple_unique_coordinates"] += int(
                (~single).sum()
            )

    final_inside_unmatched = present_flat[
        (national_lookup[present_flat] == 1)
        & (province_lookup[present_flat] < 0)
    ]
    method_lookup[final_inside_unmatched] = 7
    stats["inside_unmatched_unique_coordinates"] = int(
        len(final_inside_unmatched)
    )
    return (
        province_lookup,
        method_lookup,
        national_lookup,
        stats,
    )


def _write_national_coordinate_outputs(
    args: argparse.Namespace,
    *,
    present_flat: np.ndarray,
    lat_raw: np.ndarray,
    lon_raw: np.ndarray,
    province_lookup: np.ndarray,
    method_lookup: np.ndarray,
    national_lookup: np.ndarray,
    province_names: np.ndarray,
    metadata_rows: list[dict[str, object]],
    provinces: gpd.GeoDataFrame,
    national_boundary: object,
    stats: dict[str, int],
) -> None:
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    coordinate_path = (
        args.diagnostics_dir / "coordinate_province_assignment.parquet"
    )
    province_npy = (
        args.diagnostics_dir / "coordinate_province_code_lookup.npy"
    )
    method_npy = (
        args.diagnostics_dir / "coordinate_assignment_method_lookup.npy"
    )
    national_npy = (
        args.diagnostics_dir / "coordinate_national_inside_lookup.npy"
    )
    metadata_path = (
        args.diagnostics_dir / "coordinate_assignment_metadata.csv"
    )
    outside_unique_path = (
        args.diagnostics_dir / "outside_china_unique_points.parquet"
    )
    inside_unmatched_path = (
        args.diagnostics_dir / "inside_china_unmatched_unique_points.parquet"
    )
    unique_summary_path = (
        args.diagnostics_dir / "national_filtered_unique_summary.csv"
    )
    national_boundary_path = (
        args.diagnostics_dir / "national_boundary.geojson"
    )
    outside_map_path = (
        args.diagnostics_dir / "outside_china_points_map.png"
    )

    province_codes = province_lookup[present_flat]
    method_codes = method_lookup[present_flat]
    inside_values = national_lookup[present_flat] == 1
    province_values = np.full(len(present_flat), None, dtype=object)
    assigned = province_codes >= 0
    province_values[assigned] = province_names[province_codes[assigned]]
    method_values = method_label(method_codes)
    final_status = np.where(
        ~inside_values,
        "excluded_outside_china",
        np.where(assigned, "assigned", "unmatched_inside_china"),
    )
    coordinate = pd.DataFrame(
        {
            "coord_flat_index": present_flat,
            "lat_raw": lat_raw,
            "lon_raw": lon_raw,
            "lat": lat_raw.astype("float64") / 100.0,
            "lon": lon_raw.astype("float64") / 100.0,
            "inside_national_boundary": inside_values,
            "final_province_code": province_codes,
            "final_province": province_values,
            "assignment_method_code": method_codes,
            "assignment_method": method_values,
            "final_status": final_status,
        }
    )
    coordinate.to_parquet(
        coordinate_path,
        index=False,
        compression="zstd",
    )
    coordinate.loc[
        ~coordinate["inside_national_boundary"],
        [
            "coord_flat_index",
            "lat_raw",
            "lon_raw",
            "lat",
            "lon",
            "inside_national_boundary",
            "final_status",
        ],
    ].assign(
        exclusion_reason="outside_national_boundary"
    ).to_parquet(outside_unique_path, index=False, compression="zstd")
    coordinate.loc[
        coordinate["inside_national_boundary"]
        & coordinate["final_province"].isna()
    ].assign(
        decision_reason=(
            "inside national boundary but no province point/grid rule matched"
        )
    ).to_parquet(
        inside_unmatched_path,
        index=False,
        compression="zstd",
    )
    np.save(province_npy, province_lookup)
    np.save(method_npy, method_lookup)
    np.save(national_npy, national_lookup)
    pd.DataFrame(metadata_rows).to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )
    method_counts = coordinate.groupby(
        ["assignment_method_code", "assignment_method", "final_status"],
        as_index=False,
        dropna=False,
    ).size().rename(columns={"size": "unique_coordinate_count"})
    method_counts["unique_coordinate_share"] = (
        method_counts["unique_coordinate_count"] / len(coordinate)
    )
    method_counts.to_csv(
        unique_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    national_gdf = gpd.GeoDataFrame(
        {"name": ["national_boundary"], "region_count": [34]},
        geometry=[national_boundary],
        crs="EPSG:4326",
    )
    national_gdf.to_file(national_boundary_path, driver="GeoJSON")

    outside = coordinate.loc[
        ~coordinate["inside_national_boundary"], ["lon", "lat"]
    ]
    if len(outside) > 500_000:
        sample_positions = np.linspace(
            0,
            len(outside) - 1,
            500_000,
            dtype="int64",
        )
        outside_plot = outside.iloc[sample_positions]
    else:
        outside_plot = outside
    map_crs = target_crs()
    transformer = Transformer.from_crs("EPSG:4326", map_crs, always_xy=True)
    x, y = transformer.transform(
        outside_plot["lon"].to_numpy(),
        outside_plot["lat"].to_numpy(),
    )
    xlim, ylim = projected_extent((72.5, 135.5), (2.5, 54.5), transformer)
    provinces_projected = provinces.to_crs(map_crs)
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
    provinces_projected.boundary.plot(
        ax=ax,
        color=PROVINCE_BOUNDARY_COLOR,
        linewidth=0.26,
        zorder=2,
    )
    provinces_projected.dissolve().boundary.plot(
        ax=ax,
        color=NATIONAL_BOUNDARY_COLOR,
        linewidth=0.68,
        zorder=3,
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
        f"shown {len(outside_plot):,}/{len(outside):,} unique coordinates",
        transform=ax.transAxes,
        fontsize=6.5,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.set_axis_off()
    fig.savefig(
        outside_map_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)

    print(f"Created: {coordinate_path}")
    print(f"Created: {outside_unique_path}")
    print(f"Created: {inside_unmatched_path}")
    print(f"Created: {national_boundary_path}")
    print(f"Created: {outside_map_path}")
    for key, value in stats.items():
        print(f"{key}: {value:,}")


def national_filtered_main() -> None:
    args = parse_args()
    if args.date_basis != "utc":
        raise ValueError(
            "Unique-coordinate inventory uses UTC source partitions; use "
            "--date-basis utc."
        )
    validate_date_range(args.start_date, args.end_date)
    protected_outputs = [
        args.diagnostics_dir / "coordinate_province_assignment.parquet",
        args.diagnostics_dir / "coordinate_province_code_lookup.npy",
        args.diagnostics_dir / "coordinate_assignment_method_lookup.npy",
        args.diagnostics_dir / "coordinate_national_inside_lookup.npy",
        args.diagnostics_dir / "coordinate_assignment_metadata.csv",
        args.diagnostics_dir / "outside_china_unique_points.parquet",
        args.diagnostics_dir / "inside_china_unmatched_unique_points.parquet",
        args.diagnostics_dir / "national_filtered_unique_summary.csv",
        args.diagnostics_dir / "national_boundary.geojson",
        args.diagnostics_dir / "outside_china_points_map.png",
    ]
    existing_outputs = [
        path for path in protected_outputs if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "National-filtered coordinate outputs exist; rerun with "
            "--overwrite: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    paths: list[Path] = []
    for day in pd.date_range(args.start_date, args.end_date, freq="D"):
        path = (
            args.input_dir
            / f"population_china_{day:%Y-%m-%d}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Daily parquet missing: {path}")
        paths.append(path)

    mapping = read_province_mapping(args.mapping)
    province_names = mapping["province"].to_numpy(dtype=object)
    province_code = {
        province: index
        for index, province in enumerate(province_names)
    }
    selection = discover_province_boundary(
        args.boundary_root,
        args.mapping,
    )
    provinces, _selection, original_crs, effective_crs = (
        load_dissolved_provinces(
            selection,
            boundary_root=args.boundary_root,
            mapping_path=args.mapping,
            missing_crs="EPSG:4326",
        )
    )
    provinces = (
        provinces.set_index("province")
        .loc[province_names]
        .reset_index()
        .to_crs("EPSG:4326")
    )
    provinces["geometry"] = shapely.make_valid(
        provinces.geometry.array
    )
    if len(provinces) != 34:
        raise ValueError(
            f"Expected 34 dissolved provinces, found {len(provinces)}"
        )
    if int((~provinces.geometry.is_valid).sum()):
        raise ValueError("Dissolved province geometry remains invalid")
    national_boundary = shapely.make_valid(
        shapely.union_all(provinces.geometry.array)
    )
    if not shapely.is_valid(national_boundary):
        raise ValueError("national_boundary is invalid")
    province_codes_array = np.array(
        [province_code[value] for value in provinces["province"]],
        dtype="int16",
    )

    lat_raw_min = int(round(args.lat_min * 100))
    lat_raw_max = int(round(args.lat_max * 100))
    lon_raw_min = int(round(args.lon_min * 100))
    lon_raw_max = int(round(args.lon_max * 100))
    nlon = lon_raw_max - lon_raw_min + 1
    nlat = lat_raw_max - lat_raw_min + 1
    total_cells = nlat * nlon
    present_flat, lat_raw, lon_raw = (
        _load_or_scan_coordinate_inventory(
            args,
            paths,
            lat_raw_min=lat_raw_min,
            lat_raw_max=lat_raw_max,
            lon_raw_min=lon_raw_min,
            lon_raw_max=lon_raw_max,
            nlon=nlon,
        )
    )
    expected_flat = (
        (lat_raw.astype("int64") - lat_raw_min) * nlon
        + (lon_raw.astype("int64") - lon_raw_min)
    )
    if not np.array_equal(present_flat, expected_flat):
        raise ValueError(
            "Coordinate inventory does not match configured dense lookup "
            "domain"
        )

    (
        province_lookup,
        method_lookup,
        national_lookup,
        stats,
    ) = _national_exact_assignment(
        present_flat,
        lat_raw,
        lon_raw,
        provinces=provinces,
        province_codes_array=province_codes_array,
        total_cells=total_cells,
        grid_size=args.fallback_grid_size,
        chunk_size=args.spatial_chunk_size,
    )
    metadata_rows = [
        {
            "metadata_type": "province_code",
            "code": index,
            "label": province,
            "value": "",
        }
        for index, province in enumerate(province_names)
    ] + [
        {
            "metadata_type": "assignment_method",
            "code": code,
            "label": label,
            "value": "",
        }
        for code, label in METHOD_LABELS.items()
    ] + [
        {
            "metadata_type": "parameter",
            "code": "",
            "label": name,
            "value": value,
        }
        for name, value in {
            "lat_raw_min": lat_raw_min,
            "lat_raw_max": lat_raw_max,
            "lon_raw_min": lon_raw_min,
            "lon_raw_max": lon_raw_max,
            "nlon": nlon,
            "coordinate_resolution_degree": args.coordinate_resolution,
            "fallback_grid_size_degree": args.fallback_grid_size,
            "nearest_assignment_enabled": False,
            "area_crs": AREA_CRS.to_string(),
            "boundary_file": str(selection.path),
            "boundary_source_crs": original_crs or "MISSING",
            "boundary_effective_crs": effective_crs,
            "national_boundary_predicate": "covers",
            "national_boundary_region_count": 34,
            "multiple_province_rule": (
                "lowest canonical province code from province_name_mapping.csv"
            ),
        }.items()
    ]
    _write_national_coordinate_outputs(
        args,
        present_flat=present_flat,
        lat_raw=lat_raw,
        lon_raw=lon_raw,
        province_lookup=province_lookup,
        method_lookup=method_lookup,
        national_lookup=national_lookup,
        province_names=province_names,
        metadata_rows=metadata_rows,
        provinces=provinces,
        national_boundary=national_boundary,
        stats=stats,
    )
    print(f"Boundary used: {selection.path}")
    print(f"Source CRS: {original_crs or 'MISSING'}")
    print("Effective CRS: EPSG:4326")
    print("Dissolved province count: 34")
    print("National filter predicate: covers (boundary points retained)")
    print("Outside coordinates receive no grid or nearest assignment")


if __name__ == "__main__":
    national_filtered_main()
