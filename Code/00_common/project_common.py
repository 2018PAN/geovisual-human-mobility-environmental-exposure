from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


NEW_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = NEW_ROOT.parent
RAW_DATA_ROOT = WORKSPACE_ROOT / "RawData"

POPULATION_INPUT_DIR = NEW_ROOT / "Output" / "Population" / "daily_parquet"
TIME_CORRECTED_DIR = NEW_ROOT / "Output" / "Population" / "time_corrected_daily_parquet"
PROVINCE_ASSIGNED_DIR = (
    NEW_ROOT / "Output" / "Population" / "province_assigned_daily_parquet"
)
CALIBRATED_DIR = NEW_ROOT / "Output" / "Population" / "calibrated_daily_parquet"
DIAGNOSTICS_DIR = NEW_ROOT / "Output" / "Population" / "diagnostics"
BOUNDARY_ROOT = NEW_ROOT / "Boundary"
MAPPING_PATH = NEW_ROOT / "Config" / "province_name_mapping.csv"

CHUNYUN_START = date(2018, 2, 1)
CHUNYUN_END = date(2018, 3, 12)
BEIJING_OFFSET_HOURS = 8

# Reuse the established projection and 10 km grid definition from the legacy
# population-redistribution workflow.  Keep these values centralized so later
# population, exposure, and visualization stages cannot silently diverge.
CHINA_LCC = (
    "+proj=lcc +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=110 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)
GRID_SIZE_M = 10_000


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}; expected YYYY-MM-DD."
        ) from exc


def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def validate_date_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("--start-date must not be after --end-date")


def add_date_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_basis: str,
    default_start: date = CHUNYUN_START,
    default_end: date = CHUNYUN_END,
) -> None:
    parser.add_argument("--start-date", type=parse_iso_date, default=default_start)
    parser.add_argument("--end-date", type=parse_iso_date, default=default_end)
    parser.add_argument(
        "--date-basis",
        choices=("utc", "local"),
        default=default_basis,
        help=(
            "Meaning of --start-date/--end-date. 'utc' selects source UTC dates; "
            "'local' selects Beijing local output dates."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace selected outputs. Existing outputs are skipped by default.",
    )


def discover_official_population_file() -> Path:
    filename = "official_population_2018.csv"
    new_matches = sorted(NEW_ROOT.rglob(filename))
    print(f"Official population exact-name search under {NEW_ROOT}:")
    for match in new_matches:
        print(f"  {match.resolve()}")
    print(f"  matches: {len(new_matches)}")
    if len(new_matches) > 1:
        raise RuntimeError(
            f"Multiple {filename} files found under {NEW_ROOT}; refusing to choose."
        )
    if len(new_matches) == 1:
        return new_matches[0].resolve()

    raw_matches = sorted(RAW_DATA_ROOT.rglob(filename))
    print(
        f"No match under New; searching confirmed raw-data root {RAW_DATA_ROOT}:"
    )
    for match in raw_matches:
        print(f"  {match.resolve()}")
    print(f"  matches: {len(raw_matches)}")
    if len(raw_matches) > 1:
        raise RuntimeError(
            f"Multiple {filename} files found under {RAW_DATA_ROOT}; refusing to choose."
        )
    if not raw_matches:
        raise FileNotFoundError(
            f"{filename} was not found under either {NEW_ROOT} or {RAW_DATA_ROOT}."
        )
    return raw_matches[0].resolve()


def read_province_mapping(path: Path = MAPPING_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Province mapping file not found: {path}")
    mapping = pd.read_csv(path, dtype={"boundary_code": "string"})
    required = {"boundary_code", "boundary_name", "province", "official_name"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"Province mapping is missing columns: {sorted(missing)}")
    mapping["boundary_code"] = mapping["boundary_code"].str.strip()
    for column in ("boundary_name", "province", "official_name"):
        mapping[column] = mapping[column].astype("string").str.strip()
    if len(mapping) != 34:
        raise ValueError(f"Province mapping must contain 34 rows, found {len(mapping)}")
    for column in required:
        if mapping[column].isna().any() or mapping[column].duplicated().any():
            raise ValueError(
                f"Province mapping column {column!r} contains missing or duplicate values"
            )
    return mapping


def _parse_population_values(series: pd.Series) -> pd.Series:
    values = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    parsed = pd.to_numeric(values, errors="coerce")
    return parsed


def read_and_validate_official_population(
    population_path: Path | None = None,
    mapping_path: Path = MAPPING_PATH,
) -> tuple[pd.DataFrame, int, Path]:
    actual_path = (
        discover_official_population_file()
        if population_path is None
        else population_path.resolve()
    )
    print(f"Official population file used: {actual_path}")
    source = pd.read_csv(actual_path, dtype="string")
    required = {"province", "official_population_2018"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(
            f"Official population file is missing columns: {sorted(missing)}"
        )

    source["province"] = source["province"].str.strip()
    source["population_value"] = _parse_population_values(
        source["official_population_2018"]
    )
    total_rows = source[source["province"].str.casefold() == "total"].copy()
    regions = source[source["province"].str.casefold() != "total"].copy()

    if len(regions) != 34:
        raise ValueError(
            f"Official population file must contain 34 regional rows; found {len(regions)}"
        )
    if regions["province"].duplicated().any():
        duplicates = regions.loc[
            regions["province"].duplicated(keep=False), "province"
        ].tolist()
        raise ValueError(f"Duplicate official population names: {duplicates}")
    if regions["population_value"].isna().any():
        bad = regions.loc[regions["population_value"].isna(), "province"].tolist()
        raise ValueError(f"Unparseable official population values for: {bad}")
    if (regions["population_value"] <= 0).any():
        bad = regions.loc[regions["population_value"] <= 0, "province"].tolist()
        raise ValueError(f"Non-positive official population values for: {bad}")
    if not (regions["population_value"] % 1 == 0).all():
        raise ValueError("Official population values must be whole persons")

    mapping = read_province_mapping(mapping_path)
    merged = mapping.merge(
        regions[["province", "population_value"]],
        left_on="official_name",
        right_on="province",
        how="left",
        validate="one_to_one",
        suffixes=("", "_official_source"),
    )
    if merged["population_value"].isna().any():
        missing_names = merged.loc[
            merged["population_value"].isna(), "official_name"
        ].tolist()
        raise ValueError(
            f"Mapping has official names absent from population file: {missing_names}"
        )

    official_total = int(merged["population_value"].sum())
    if len(total_rows) > 1:
        raise ValueError("Official population file contains multiple Total rows")
    if len(total_rows) == 1:
        declared_total = total_rows["population_value"].iloc[0]
        if pd.isna(declared_total) or int(declared_total) != official_total:
            raise ValueError(
                f"Declared Total does not equal direct 34-region sum: "
                f"declared={declared_total}, sum={official_total}"
            )

    result = merged[
        ["province", "official_name", "boundary_code", "boundary_name", "population_value"]
    ].rename(columns={"population_value": "official_population_2018"})
    result["official_population_2018"] = result[
        "official_population_2018"
    ].astype("int64")

    print("Official population validation:")
    print(f"  regional rows: {len(result)}")
    print("  duplicate region names: 0")
    print("  non-positive populations: 0")
    print("  unit: persons (absolute whole-person counts)")
    print(f"  direct 34-region total: {official_total:,}")
    return result, official_total, actual_path


def selected_input_local_dates(
    start_date: date, end_date: date, date_basis: str
) -> list[date]:
    validate_date_range(start_date, end_date)
    if date_basis == "local":
        return list(iter_dates(start_date, end_date))
    return list(iter_dates(start_date, end_date + timedelta(days=1)))


def selected_input_utc_dates(
    start_date: date, end_date: date, date_basis: str
) -> list[date]:
    validate_date_range(start_date, end_date)
    if date_basis == "utc":
        # Raw daily filenames are normally UTC dates, but validation found at
        # least one timestamp stored in an adjacent filename partition. Read
        # both neighbors and filter by the actual utc_time in the caller.
        return list(
            iter_dates(
                start_date - timedelta(days=1),
                end_date + timedelta(days=1),
            )
        )
    return list(iter_dates(start_date - timedelta(days=1), end_date))
