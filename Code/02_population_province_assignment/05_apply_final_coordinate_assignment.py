from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    PROVINCE_ASSIGNED_DIR,
    TIME_CORRECTED_DIR,
    add_date_arguments,
    read_and_validate_official_population,
    selected_input_local_dates,
    validate_date_range,
)


INPUT_PREFIX = "population_time_corrected_"
OUTPUT_PREFIX = "population_province_assigned_"


def parse_boolean(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def require_passing_gate(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(
            f"Spatial assignment quality gate not found: {path}"
        )
    gate = pd.read_csv(path)
    required = {
        "quality_gate_pass",
        "final_unmatched_app_count_share",
        "in_china_unmatched_share",
        "maximum_allowed_final_unmatched_share",
    }
    missing = sorted(required.difference(gate.columns))
    if len(gate) != 1 or missing:
        raise ValueError(
            f"Invalid spatial quality gate {path}; missing={missing}, "
            f"rows={len(gate)}"
        )
    row = gate.iloc[0]
    if not parse_boolean(row["quality_gate_pass"]):
        raise RuntimeError(
            "Province-assigned formal outputs blocked by spatial quality "
            "gate: final unmatched App count share="
            f"{float(row['final_unmatched_app_count_share']):.10%}, "
            "maximum="
            f"{float(row['maximum_allowed_final_unmatched_share']):.2%}."
        )
    if not np.isclose(
        float(row["final_unmatched_app_count_share"]),
        float(row["in_china_unmatched_share"]),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "Spatial gate final share is not the China-only unmatched share"
        )
    return row


def load_lookup_metadata(
    path: Path,
) -> tuple[dict[int, str], dict[int, str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Coordinate lookup metadata not found: {path}")
    metadata = pd.read_csv(path, dtype="string")
    provinces = metadata.loc[
        metadata["metadata_type"].eq("province_code"), ["code", "label"]
    ].dropna()
    methods = metadata.loc[
        metadata["metadata_type"].eq("assignment_method"),
        ["code", "label"],
    ].dropna()
    parameters = metadata.loc[
        metadata["metadata_type"].eq("parameter"), ["label", "value"]
    ].dropna(subset=["label"])
    province_names = {
        int(code): str(label)
        for code, label in provinces.itertuples(index=False, name=None)
    }
    method_names = {
        int(code): str(label)
        for code, label in methods.itertuples(index=False, name=None)
    }
    parameter_values = {
        str(label): str(value)
        for label, value in parameters.itertuples(index=False, name=None)
    }
    if len(province_names) != 34:
        raise ValueError(
            f"Coordinate lookup must contain 34 provinces, found "
            f"{len(province_names)}"
        )
    return province_names, method_names, parameter_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the validated point/boundary/0.1-degree-grid coordinate "
            "assignment lookup to time-corrected records. Formal output is "
            "refused unless the full-range unmatched quality gate passes."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=TIME_CORRECTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROVINCE_ASSIGNED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    parser.add_argument(
        "--province-lookup",
        type=Path,
        default=DIAGNOSTICS_DIR / "coordinate_province_code_lookup.npy",
    )
    parser.add_argument(
        "--method-lookup",
        type=Path,
        default=DIAGNOSTICS_DIR / "coordinate_assignment_method_lookup.npy",
    )
    parser.add_argument(
        "--national-inside-lookup",
        type=Path,
        default=DIAGNOSTICS_DIR / "coordinate_national_inside_lookup.npy",
    )
    parser.add_argument(
        "--lookup-metadata",
        type=Path,
        default=DIAGNOSTICS_DIR / "coordinate_assignment_metadata.csv",
    )
    parser.add_argument(
        "--spatial-quality-gate",
        type=Path,
        default=DIAGNOSTICS_DIR / "spatial_assignment_quality_gate.csv",
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--compression", default="zstd")
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    gate = require_passing_gate(args.spatial_quality_gate)
    for path in (
        args.province_lookup,
        args.method_lookup,
        args.national_inside_lookup,
        args.lookup_metadata,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required coordinate lookup missing: {path}")

    official, _official_total, _official_path = (
        read_and_validate_official_population(
            args.official_population,
            args.mapping,
        )
    )
    official_lookup = official.set_index("province")[
        "official_population_2018"
    ].to_dict()
    province_names, method_names, parameters = load_lookup_metadata(
        args.lookup_metadata
    )
    if set(province_names.values()) != set(official["province"]):
        raise ValueError(
            "Coordinate lookup provinces and official population provinces "
            "differ"
        )

    lat_raw_min = int(parameters["lat_raw_min"])
    lat_raw_max = int(parameters["lat_raw_max"])
    lon_raw_min = int(parameters["lon_raw_min"])
    lon_raw_max = int(parameters["lon_raw_max"])
    nlon = int(parameters["nlon"])
    province_lookup = np.load(args.province_lookup, mmap_mode="r")
    method_lookup = np.load(args.method_lookup, mmap_mode="r")
    national_lookup = np.load(
        args.national_inside_lookup,
        mmap_mode="r",
    )
    if not (
        province_lookup.shape
        == method_lookup.shape
        == national_lookup.shape
    ):
        raise ValueError(
            "Province, assignment-method, and national-boundary lookups "
            "differ in size"
        )

    local_dates = selected_input_local_dates(
        args.start_date,
        args.end_date,
        args.date_basis,
    )
    input_paths = [
        args.input_dir / f"{INPUT_PREFIX}{day.isoformat()}.parquet"
        for day in local_dates
    ]
    missing_inputs = [path for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Selected time-corrected inputs are missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )

    selection_start = pd.Timestamp(
        datetime.combine(args.start_date, time.min)
    )
    selection_end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    method_records: dict[int, int] = defaultdict(int)
    method_app_count: dict[int, int] = defaultdict(int)
    province_records: dict[str, int] = defaultdict(int)
    province_app_count: dict[str, int] = defaultdict(int)
    total_records = 0
    total_app_count = 0
    unmatched_records = 0
    unmatched_app_count = 0
    excluded_outside_records = 0
    excluded_outside_app_count = 0
    created: list[Path] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    province_object = np.empty(max(province_names) + 1, dtype=object)
    for code, name in province_names.items():
        province_object[code] = name
    method_object = np.empty(max(method_names) + 1, dtype=object)
    for code, name in method_names.items():
        method_object[code] = name

    for input_path in input_paths:
        local_date_text = input_path.stem.removeprefix(INPUT_PREFIX)
        final_path = (
            args.output_dir
            / f"{OUTPUT_PREFIX}{local_date_text}.parquet"
        )
        if final_path.exists() and not args.overwrite:
            print(f"Skipped existing output: {final_path}")
            continue
        temp_path = final_path.with_name(f".{final_path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        parquet_file = pq.ParquetFile(input_path)
        required = {
            "utc_time",
            "local_time",
            "lat",
            "lon",
            "count",
        }
        missing = sorted(
            required.difference(parquet_file.schema_arrow.names)
        )
        if missing:
            raise ValueError(f"{input_path} is missing columns: {missing}")

        writer: pq.ParquetWriter | None = None
        try:
            for batch_number, batch in enumerate(
                parquet_file.iter_batches(batch_size=args.batch_size),
                start=1,
            ):
                frame = batch.to_pandas()
                utc_time = pd.to_datetime(
                    frame["utc_time"], errors="coerce"
                )
                local_time = pd.to_datetime(
                    frame["local_time"], errors="coerce"
                )
                if utc_time.isna().any() or local_time.isna().any():
                    raise ValueError(
                        f"{input_path} contains invalid converted time"
                    )
                selected = (
                    (utc_time >= selection_start)
                    & (utc_time < selection_end)
                    if args.date_basis == "utc"
                    else (local_time >= selection_start)
                    & (local_time < selection_end)
                )
                frame = frame.loc[selected].copy()
                if frame.empty:
                    continue

                if {"lat_raw", "lon_raw"}.issubset(frame.columns):
                    lat_raw = pd.to_numeric(
                        frame["lat_raw"], errors="coerce"
                    ).to_numpy()
                    lon_raw = pd.to_numeric(
                        frame["lon_raw"], errors="coerce"
                    ).to_numpy()
                else:
                    lat_raw = np.rint(
                        pd.to_numeric(frame["lat"], errors="coerce").to_numpy()
                        * 100
                    )
                    lon_raw = np.rint(
                        pd.to_numeric(frame["lon"], errors="coerce").to_numpy()
                        * 100
                    )
                finite = np.isfinite(lat_raw) & np.isfinite(lon_raw)
                lat_int = np.where(finite, lat_raw, lat_raw_min - 1).astype(
                    "int64"
                )
                lon_int = np.where(finite, lon_raw, lon_raw_min - 1).astype(
                    "int64"
                )
                lookup_range = (
                    finite
                    & (lat_int >= lat_raw_min)
                    & (lat_int <= lat_raw_max)
                    & (lon_int >= lon_raw_min)
                    & (lon_int <= lon_raw_max)
                )
                flat = np.zeros(len(frame), dtype="int64")
                flat[lookup_range] = (
                    (lat_int[lookup_range] - lat_raw_min) * nlon
                    + (lon_int[lookup_range] - lon_raw_min)
                )
                inside_national = np.zeros(len(frame), dtype=bool)
                inside_national[lookup_range] = (
                    national_lookup[flat[lookup_range]] == 1
                )
                raw_count = pd.to_numeric(
                    frame["count"],
                    errors="coerce",
                )
                if raw_count.isna().any() or (raw_count < 0).any():
                    raise ValueError(f"Invalid count values in {input_path}")
                outside = ~inside_national
                excluded_outside_records += int(outside.sum())
                excluded_outside_app_count += int(
                    raw_count.iloc[np.flatnonzero(outside)].sum()
                )
                if not inside_national.any():
                    continue

                inside_positions = np.flatnonzero(inside_national)
                frame = frame.iloc[inside_positions].copy()
                flat = flat[inside_positions]
                province_codes = province_lookup[flat].astype(
                    "int16",
                    copy=False,
                )
                method_codes = method_lookup[flat].astype(
                    "int8",
                    copy=False,
                )

                province_values = np.full(len(frame), None, dtype=object)
                assigned = province_codes >= 0
                province_values[assigned] = province_object[
                    province_codes[assigned]
                ]
                method_values = method_object[method_codes]
                frame["province"] = province_values
                frame["official_population_2018"] = (
                    pd.Series(province_values, index=frame.index)
                    .map(official_lookup)
                    .astype("Int64")
                )
                frame["province_assignment_method"] = method_values
                frame["province_assignment_method_code"] = method_codes
                frame["province_assignment_status"] = np.where(
                    assigned, "assigned", "unmatched"
                )
                frame["province_assignment_reason"] = np.where(
                    assigned,
                    "validated reusable coordinate assignment",
                    "no permitted point, grid-overlap, or distance rule",
                )

                count = pd.to_numeric(frame["count"], errors="coerce")
                if count.isna().any() or (count < 0).any():
                    raise ValueError(f"Invalid count values in {input_path}")
                total_records += len(frame)
                total_app_count += int(count.sum())
                unmatched = ~assigned
                unmatched_records += int(unmatched.sum())
                unmatched_app_count += int(
                    count.iloc[np.flatnonzero(unmatched)].sum()
                )
                for method_code in np.unique(method_codes):
                    mask = method_codes == method_code
                    method_records[int(method_code)] += int(mask.sum())
                    method_app_count[int(method_code)] += int(
                        count.iloc[np.flatnonzero(mask)].sum()
                    )
                matched_frame = pd.DataFrame(
                    {
                        "province": province_values[assigned],
                        "count": count.iloc[
                            np.flatnonzero(assigned)
                        ].to_numpy(),
                    }
                )
                if not matched_frame.empty:
                    stats = matched_frame.groupby(
                        "province", observed=True
                    )["count"].agg(["size", "sum"])
                    for province, row in stats.iterrows():
                        province_records[str(province)] += int(row["size"])
                        province_app_count[str(province)] += int(row["sum"])

                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path,
                        table.schema,
                        compression=args.compression,
                    )
                writer.write_table(table)
                print(
                    f"  batch {batch_number}: selected={len(frame):,}, "
                    f"unmatched={int(unmatched.sum()):,}"
                )
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"No selected rows written from {input_path}")
        temp_path.replace(final_path)
        created.append(final_path)
        print(f"Created: {final_path}")

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    method_path = (
        args.diagnostics_dir / "province_assignment_applied_by_method.csv"
    )
    province_path = (
        args.diagnostics_dir / "province_assignment_applied_by_province.csv"
    )
    summary_path = (
        args.diagnostics_dir / "province_assignment_applied_summary.csv"
    )
    diagnostics = [method_path, province_path, summary_path]
    existing = [path for path in diagnostics if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Assignment diagnostics exist; rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    pd.DataFrame(
        [
            {
                "assignment_method_code": code,
                "assignment_method": method_names.get(code, "unknown"),
                "record_count": method_records.get(code, 0),
                "app_count": method_app_count.get(code, 0),
                "app_count_share": (
                    method_app_count.get(code, 0) / total_app_count
                    if total_app_count
                    else np.nan
                ),
            }
            for code in sorted(method_names)
        ]
    ).to_csv(method_path, index=False, encoding="utf-8-sig")
    official.assign(
        matched_records=official["province"].map(province_records).fillna(0),
        matched_app_count=official["province"].map(
            province_app_count
        ).fillna(0),
    ).to_csv(province_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "date_basis": args.date_basis,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "records": total_records,
                "app_count": total_app_count,
                "unmatched_records": unmatched_records,
                "unmatched_app_count": unmatched_app_count,
                "unmatched_app_count_share": (
                    unmatched_app_count / total_app_count
                    if total_app_count
                    else np.nan
                ),
                "validated_gate_share": float(
                    gate["final_unmatched_app_count_share"]
                ),
                "coordinate_lookup": str(args.province_lookup.resolve()),
                "national_inside_lookup": str(
                    args.national_inside_lookup.resolve()
                ),
                "excluded_outside_records": excluded_outside_records,
                "excluded_outside_app_count": excluded_outside_app_count,
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Created outputs: {len(created)}")
    print(
        "Outside-national-boundary rows excluded from formal province "
        f"outputs: {excluded_outside_records:,}"
    )
    print(
        "Outside-national-boundary App count excluded from formal province "
        f"outputs: {excluded_outside_app_count:,}"
    )
    for path in diagnostics:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
