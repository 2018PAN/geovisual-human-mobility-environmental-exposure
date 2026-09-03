from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely

from project_common import BOUNDARY_ROOT, MAPPING_PATH, read_province_mapping


@dataclass(frozen=True)
class BoundarySelection:
    path: Path
    province_field: str
    field_kind: str
    coverage_count: int
    feature_count: int


def _attribute_values(path: Path, field: str) -> set[str]:
    frame = pyogrio.read_dataframe(
        path,
        columns=[field],
        read_geometry=False,
        encoding="GBK",
    )
    return set(frame[field].dropna().astype("string").str.strip().tolist())


def discover_province_boundary(
    boundary_root: Path = BOUNDARY_ROOT,
    mapping_path: Path = MAPPING_PATH,
) -> BoundarySelection:
    mapping = read_province_mapping(mapping_path)
    expected_codes = set(mapping["boundary_code"].astype(str))
    expected_names = set(mapping["boundary_name"].astype(str))
    candidates: list[BoundarySelection] = []

    for path in sorted(boundary_root.rglob("*.shp")):
        info = pyogrio.read_info(path, encoding="GBK")
        geometry_type = str(info.get("geometry_type", ""))
        if "Polygon" not in geometry_type:
            continue
        fields = set(map(str, info.get("fields", [])))
        for field in ("SH2", "NAME", "NAME99", "ADCODE93", "ADCODE99"):
            if field not in fields:
                continue
            try:
                values = _attribute_values(path, field)
            except (UnicodeDecodeError, ValueError):
                continue
            as_codes = len(values.intersection(expected_codes))
            as_names = len(values.intersection(expected_names))
            if as_codes:
                candidates.append(
                    BoundarySelection(
                        path=path.resolve(),
                        province_field=field,
                        field_kind="code",
                        coverage_count=as_codes,
                        feature_count=int(info["features"]),
                    )
                )
            if as_names:
                candidates.append(
                    BoundarySelection(
                        path=path.resolve(),
                        province_field=field,
                        field_kind="name",
                        coverage_count=as_names,
                        feature_count=int(info["features"]),
                    )
                )

    if not candidates:
        raise FileNotFoundError(
            f"No polygon boundary with recognizable province identifiers under {boundary_root}"
        )
    candidates.sort(
        key=lambda item: (
            -item.coverage_count,
            item.feature_count,
            str(item.path).casefold(),
            item.province_field,
        )
    )
    selected = candidates[0]
    print("Boundary auto-discovery:")
    print(f"  selected file: {selected.path}")
    print(f"  province identifier field: {selected.province_field}")
    print(f"  identifier kind: {selected.field_kind}")
    print(f"  mapped coverage: {selected.coverage_count}/34")
    return selected


def load_dissolved_provinces(
    selection: BoundarySelection | None = None,
    *,
    boundary_root: Path = BOUNDARY_ROOT,
    mapping_path: Path = MAPPING_PATH,
    missing_crs: str = "EPSG:4326",
) -> tuple[gpd.GeoDataFrame, BoundarySelection, str | None, str]:
    selection = selection or discover_province_boundary(boundary_root, mapping_path)
    mapping = read_province_mapping(mapping_path)
    boundary = gpd.read_file(selection.path, encoding="GBK")
    original_crs = boundary.crs.to_string() if boundary.crs is not None else None

    values = boundary[selection.province_field].astype("string").str.strip()
    if selection.field_kind == "code":
        lookup = mapping.set_index("boundary_code")["province"]
    else:
        lookup = mapping.set_index("boundary_name")["province"]
    boundary["province"] = values.map(lookup)
    boundary = boundary.loc[boundary["province"].notna(), ["province", "geometry"]].copy()
    if boundary.empty:
        raise ValueError("Selected boundary produced no mapped province geometries")

    if boundary.crs is None:
        bounds = boundary.total_bounds
        looks_geographic = (
            -180 <= bounds[0] <= 180
            and -90 <= bounds[1] <= 90
            and -180 <= bounds[2] <= 180
            and -90 <= bounds[3] <= 90
        )
        if not looks_geographic:
            raise ValueError(
                f"Boundary CRS is missing and coordinate bounds are not geographic: {bounds}"
            )
        boundary = boundary.set_crs(missing_crs, allow_override=True)
        print(
            f"  warning: source CRS metadata is missing; using explicit "
            f"{missing_crs} because bounds are geographic"
        )

    invalid_count = int((~boundary.geometry.is_valid).sum())
    if invalid_count:
        print(
            f"  repairing {invalid_count} invalid source geometries with "
            "shapely.make_valid"
        )
        boundary["geometry"] = shapely.make_valid(boundary.geometry.array)
        still_invalid = int((~boundary.geometry.is_valid).sum())
        if still_invalid:
            raise ValueError(
                f"Boundary still contains {still_invalid} invalid geometries "
                "after make_valid"
            )
    dissolved = boundary.dissolve(by="province", as_index=False)
    dissolved["geometry"] = shapely.make_valid(dissolved.geometry.array)
    effective_crs = dissolved.crs.to_string()

    expected = set(mapping["province"])
    actual = set(dissolved["province"])
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))

    print(f"  source CRS: {original_crs or 'MISSING'}")
    print(f"  effective CRS: {effective_crs}")
    print(f"  unique mapped provinces: {len(actual)}")
    print(f"  provinces: {', '.join(sorted(actual))}")
    print(f"  missing expected provinces: {missing or 'none'}")
    print(f"  unexpected provinces: {extra or 'none'}")
    return dissolved, selection, original_crs, effective_crs
