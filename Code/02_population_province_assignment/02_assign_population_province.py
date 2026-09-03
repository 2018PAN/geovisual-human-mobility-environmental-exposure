from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    BOUNDARY_ROOT,
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    PROVINCE_ASSIGNED_DIR,
    TIME_CORRECTED_DIR,
    add_date_arguments,
    read_and_validate_official_population,
    selected_input_local_dates,
    validate_date_range,
)
from province_boundary import (  # noqa: E402
    discover_province_boundary,
    load_dissolved_provinces,
)


INPUT_PREFIX = "population_time_corrected_"
OUTPUT_PREFIX = "population_province_assigned_"


def pack_coordinates(lat_raw: np.ndarray, lon_raw: np.ndarray) -> np.ndarray:
    return (
        (lat_raw.astype(np.int64) + 32768) * 65536
        + lon_raw.astype(np.int64)
        + 32768
    )


def point_keys(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {"lat_raw", "lon_raw"}.issubset(frame.columns):
        lat_raw = pd.to_numeric(frame["lat_raw"], errors="coerce").to_numpy()
        lon_raw = pd.to_numeric(frame["lon_raw"], errors="coerce").to_numpy()
    else:
        lat_raw = np.rint(
            pd.to_numeric(frame["lat"], errors="coerce").to_numpy() * 100
        )
        lon_raw = np.rint(
            pd.to_numeric(frame["lon"], errors="coerce").to_numpy() * 100
        )
    if not np.isfinite(lat_raw).all() or not np.isfinite(lon_raw).all():
        raise ValueError("lat/lon contains missing or non-finite values")
    lat_raw = lat_raw.astype(np.int16)
    lon_raw = lon_raw.astype(np.int16)
    return pack_coordinates(lat_raw, lon_raw), lat_raw, lon_raw


def assign_new_coordinates(
    new_coordinates: pd.DataFrame,
    provinces: gpd.GeoDataFrame,
) -> tuple[dict[int, str | None], int]:
    if new_coordinates.empty:
        return {}, 0
    points = gpd.GeoDataFrame(
        new_coordinates[["coord_key"]].copy(),
        geometry=gpd.points_from_xy(
            new_coordinates["lon"], new_coordinates["lat"]
        ),
        crs="EPSG:4326",
    )
    if points.crs != provinces.crs:
        points = points.to_crs(provinces.crs)
    joined = gpd.sjoin(
        points,
        provinces[["province", "geometry"]],
        how="left",
        predicate="intersects",
    )
    province_counts = joined.groupby("coord_key", sort=False)["province"].nunique(
        dropna=True
    )
    ambiguous = int((province_counts > 1).sum())
    selected = (
        joined.sort_values(["coord_key", "province"], na_position="last")
        .drop_duplicates("coord_key", keep="first")
        [["coord_key", "province"]]
    )
    mapping: dict[int, str | None] = {}
    for key, province in selected.itertuples(index=False):
        mapping[int(key)] = None if pd.isna(province) else str(province)
    return mapping, ambiguous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign every time-corrected App population point to a province by "
            "point-in-polygon; unmatched rows are retained."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=TIME_CORRECTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROVINCE_ASSIGNED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--boundary-root", type=Path, default=BOUNDARY_ROOT)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    parser.add_argument("--missing-crs", default="EPSG:4326")
    add_date_arguments(parser, default_basis="local")
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    official, official_total, official_path = read_and_validate_official_population(
        args.official_population, args.mapping
    )
    official_lookup = official.set_index("province")[
        "official_population_2018"
    ].to_dict()

    selection = discover_province_boundary(args.boundary_root, args.mapping)
    provinces, _selection, original_crs, effective_crs = load_dissolved_provinces(
        selection,
        boundary_root=args.boundary_root,
        mapping_path=args.mapping,
        missing_crs=args.missing_crs,
    )
    if set(provinces["province"]) != set(official["province"]):
        raise ValueError("Boundary and official population province sets differ")

    local_dates = selected_input_local_dates(
        args.start_date, args.end_date, args.date_basis
    )
    input_paths = [
        args.input_dir / f"{INPUT_PREFIX}{local_date.isoformat()}.parquet"
        for local_date in local_dates
    ]
    missing_inputs = [path for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Selected time-corrected inputs are missing: "
            + ", ".join(map(str, missing_inputs))
        )

    utc_start = pd.Timestamp(datetime.combine(args.start_date, time.min))
    utc_end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    local_start = utc_start
    local_end = utc_end

    print("Population province assignment")
    print(f"  inputs: {args.input_dir.resolve()}")
    print(f"  outputs: {args.output_dir.resolve()}")
    print(f"  date basis: {args.date_basis}")
    print(f"  selected dates: {args.start_date} to {args.end_date}")
    print(f"  official population: {official_path}")
    print(f"  official total: {official_total:,}")
    print(f"  boundary: {selection.path}")
    print(f"  province identifier: {selection.province_field}")
    print(f"  source CRS: {original_crs or 'MISSING'}")
    print(f"  effective CRS: {effective_crs}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    coordinate_cache: dict[int, str | None] = {}
    unmatched_keys: set[int] = set()
    ambiguous_unique_coordinates = 0
    province_records: dict[str, int] = defaultdict(int)
    province_counts: dict[str, int] = defaultdict(int)
    total_records = 0
    total_app_count = 0
    unmatched_records = 0
    unmatched_app_count = 0
    unmatched_lat_min = np.inf
    unmatched_lat_max = -np.inf
    unmatched_lon_min = np.inf
    unmatched_lon_max = -np.inf
    created_paths: list[Path] = []

    for input_path in input_paths:
        local_date_text = input_path.stem.removeprefix(INPUT_PREFIX)
        final_path = args.output_dir / f"{OUTPUT_PREFIX}{local_date_text}.parquet"
        if final_path.exists() and not args.overwrite:
            print(f"Skipping existing output: {final_path}")
            continue
        temp_path = final_path.with_name(f".{final_path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        writer: pq.ParquetWriter | None = None
        parquet_file = pq.ParquetFile(input_path)
        required = {
            "utc_time",
            "local_time",
            "local_date",
            "lat",
            "lon",
            "count",
        }
        missing = sorted(required.difference(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"{input_path} is missing columns: {missing}")

        print(f"Reading: {input_path}")
        try:
            for batch_number, batch in enumerate(
                parquet_file.iter_batches(batch_size=args.batch_size), start=1
            ):
                frame = batch.to_pandas()
                utc_time_values = pd.to_datetime(frame["utc_time"], errors="coerce")
                local_time_values = pd.to_datetime(
                    frame["local_time"], errors="coerce"
                )
                if utc_time_values.isna().any() or local_time_values.isna().any():
                    raise ValueError(f"{input_path} contains invalid converted timestamps")
                selected_rows = (
                    (utc_time_values >= utc_start) & (utc_time_values < utc_end)
                    if args.date_basis == "utc"
                    else (local_time_values >= local_start)
                    & (local_time_values < local_end)
                )
                frame = frame.loc[selected_rows].copy()
                if frame.empty:
                    continue

                keys, lat_raw, lon_raw = point_keys(frame)
                unique_coordinates = pd.DataFrame(
                    {
                        "coord_key": keys,
                        "lat": lat_raw.astype("float64") / 100.0,
                        "lon": lon_raw.astype("float64") / 100.0,
                    }
                ).drop_duplicates("coord_key")
                is_new = [
                    int(key) not in coordinate_cache
                    for key in unique_coordinates["coord_key"].to_numpy()
                ]
                new_coordinates = unique_coordinates.loc[is_new].copy()
                new_mapping, ambiguous = assign_new_coordinates(
                    new_coordinates, provinces
                )
                coordinate_cache.update(new_mapping)
                ambiguous_unique_coordinates += ambiguous

                province_values = pd.Series(keys).map(coordinate_cache)
                frame["province"] = province_values.to_numpy()
                frame["official_population_2018"] = (
                    frame["province"].map(official_lookup).astype("Int64")
                )
                count_values = pd.to_numeric(frame["count"], errors="coerce")
                if count_values.isna().any():
                    raise ValueError(f"{input_path} contains nonnumeric count values")

                unmatched = frame["province"].isna()
                total_records += len(frame)
                total_app_count += int(count_values.sum())
                unmatched_records += int(unmatched.sum())
                unmatched_app_count += int(count_values.loc[unmatched].sum())
                if unmatched.any():
                    unmatched_keys.update(int(key) for key in keys[unmatched.to_numpy()])
                    unmatched_lat = pd.to_numeric(
                        frame.loc[unmatched, "lat"], errors="coerce"
                    )
                    unmatched_lon = pd.to_numeric(
                        frame.loc[unmatched, "lon"], errors="coerce"
                    )
                    unmatched_lat_min = min(unmatched_lat_min, float(unmatched_lat.min()))
                    unmatched_lat_max = max(unmatched_lat_max, float(unmatched_lat.max()))
                    unmatched_lon_min = min(unmatched_lon_min, float(unmatched_lon.min()))
                    unmatched_lon_max = max(unmatched_lon_max, float(unmatched_lon.max()))

                matched = frame.loc[~unmatched].copy()
                if not matched.empty:
                    batch_stats = matched.groupby("province", observed=True)["count"].agg(
                        ["size", "sum"]
                    )
                    for province, row in batch_stats.iterrows():
                        province_records[str(province)] += int(row["size"])
                        province_counts[str(province)] += int(row["sum"])

                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path, table.schema, compression=args.compression
                    )
                writer.write_table(table)
                print(
                    f"  batch {batch_number}: records={len(frame):,}, "
                    f"unique coords={len(unique_coordinates):,}, "
                    f"new coords={len(new_coordinates):,}, "
                    f"unmatched={int(unmatched.sum()):,}"
                )
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            raise ValueError(f"No selected rows were written from {input_path}")
        temp_path.replace(final_path)
        created_paths.append(final_path)
        print(f"Created: {final_path}")

    unmatched_share = (
        unmatched_app_count / total_app_count if total_app_count else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "date_basis": args.date_basis,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "total_records": total_records,
                "total_app_count": total_app_count,
                "unmatched_records": unmatched_records,
                "unmatched_unique_coordinates": len(unmatched_keys),
                "unmatched_app_count": unmatched_app_count,
                "unmatched_count_share": unmatched_share,
                "unmatched_lat_min": (
                    unmatched_lat_min if np.isfinite(unmatched_lat_min) else np.nan
                ),
                "unmatched_lat_max": (
                    unmatched_lat_max if np.isfinite(unmatched_lat_max) else np.nan
                ),
                "unmatched_lon_min": (
                    unmatched_lon_min if np.isfinite(unmatched_lon_min) else np.nan
                ),
                "unmatched_lon_max": (
                    unmatched_lon_max if np.isfinite(unmatched_lon_max) else np.nan
                ),
                "ambiguous_unique_coordinates": ambiguous_unique_coordinates,
                "boundary_file": str(selection.path),
                "province_identifier_field": selection.province_field,
                "effective_crs": effective_crs,
            }
        ]
    )
    province_totals = pd.DataFrame(
        [
            {
                "province": province,
                "matched_records": province_records.get(province, 0),
                "app_count_sum": province_counts.get(province, 0),
            }
            for province in official["province"]
        ]
    ).merge(
        official[["province", "official_population_2018"]],
        on="province",
        how="left",
        validate="one_to_one",
    )

    summary_path = args.diagnostics_dir / "province_assignment_summary.csv"
    totals_path = args.diagnostics_dir / "province_assignment_totals.csv"
    for path in (summary_path, totals_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Diagnostic output exists; rerun with --overwrite: {path}"
            )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    province_totals.to_csv(totals_path, index=False, encoding="utf-8-sig")
    print(f"Created: {summary_path}")
    print(f"Created: {totals_path}")
    print(
        f"Unmatched records={unmatched_records:,}; "
        f"unique coordinates={len(unmatched_keys):,}; "
        f"app_count={unmatched_app_count:,}; share={unmatched_share:.8%}"
    )


if __name__ == "__main__":
    main()
