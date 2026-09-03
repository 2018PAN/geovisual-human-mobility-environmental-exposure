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
    CHUNYUN_END,
    CHUNYUN_START,
    DIAGNOSTICS_DIR,
    POPULATION_INPUT_DIR,
    add_date_arguments,
    validate_date_range,
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


def load_lookup_parameters(metadata_path: Path) -> dict[str, str]:
    metadata = pd.read_csv(metadata_path, dtype="string")
    parameters = metadata.loc[
        metadata["metadata_type"] == "parameter",
        ["label", "value"],
    ]
    return dict(parameters.itertuples(index=False, name=None))


def load_recovery_lookup(
    audit_path: Path,
) -> dict[tuple[str, str, int], pd.Timestamp]:
    audit = pd.read_csv(
        audit_path,
        dtype={
            "source_date": "string",
            "source_file": "string",
            "line_no": "int64",
            "is_recoverable": "string",
        },
    )
    audit["recovered_utc_time"] = pd.to_datetime(
        audit["recovered_utc_time"],
        errors="coerce",
    )
    return {
        (
            str(row.source_date),
            str(row.source_file),
            int(row.line_no),
        ): row.recovered_utc_time
        for row in audit.itertuples(index=False)
        if str(row.is_recoverable).strip().casefold()
        in {"true", "1", "yes"}
        and pd.notna(row.recovered_utc_time)
    }


def resolve_utc_time(
    frame: pd.DataFrame,
    recovery_lookup: dict[tuple[str, str, int], pd.Timestamp],
) -> tuple[pd.Series, pd.Series]:
    utc_time = pd.to_datetime(
        frame["time"],
        errors="coerce",
        cache=True,
    )
    invalid = utc_time.isna()
    if invalid.any() and recovery_lookup:
        invalid_positions = np.flatnonzero(invalid.to_numpy())
        recovered_values = [
            recovery_lookup.get(
                (
                    str(frame["date"].iat[position]),
                    str(frame["source_file"].iat[position]),
                    int(frame["line_no"].iat[position]),
                ),
                pd.NaT,
            )
            for position in invalid_positions
        ]
        recovered = pd.to_datetime(
            pd.Series(recovered_values),
            errors="coerce",
        )
        available = recovered.notna().to_numpy()
        if available.any():
            utc_time.iloc[invalid_positions[available]] = recovered.loc[
                available
            ].to_numpy(dtype="datetime64[ns]")
    return utc_time, utc_time.isna()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply blank-time exclusion and Beijing-date filtering, exclude "
            "points outside the inclusive 34-region national boundary, then "
            "evaluate province assignment only within China."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DIAGNOSTICS_DIR,
    )
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
        "--blank-time-audit",
        type=Path,
        default=DIAGNOSTICS_DIR / "blank_time_audit.csv",
    )
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    parser.add_argument(
        "--max-final-unmatched-share",
        type=float,
        default=0.01,
    )
    parser.add_argument("--compression", default="zstd")
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def empty_record_table(*, outside: bool) -> pa.Table:
    fields = [
        ("utc_time", pa.timestamp("ns")),
        ("local_time", pa.timestamp("ns")),
        ("lat", pa.float32()),
        ("lon", pa.float32()),
        ("count", pa.int32()),
        ("source_file", pa.string()),
        ("line_no", pa.int32()),
    ]
    if outside:
        fields.append(("exclusion_reason", pa.string()))
    else:
        fields.extend(
            [
                ("assignment_method_code", pa.int8()),
                ("assignment_method", pa.string()),
                ("unmatched_reason", pa.string()),
            ]
        )
    return pa.Table.from_pylist([], schema=pa.schema(fields))


def main() -> None:
    args = parse_args()
    if args.date_basis != "local":
        raise ValueError(
            "The national-filtered quality gate is defined on Beijing local "
            "dates; use --date-basis local."
        )
    validate_date_range(args.start_date, args.end_date)
    required_inputs = (
        args.province_lookup,
        args.method_lookup,
        args.national_inside_lookup,
        args.lookup_metadata,
        args.blank_time_audit,
    )
    for path in required_inputs:
        if not path.exists():
            raise FileNotFoundError(f"Required QC input not found: {path}")

    parameters = load_lookup_parameters(args.lookup_metadata)
    lat_raw_min = int(parameters["lat_raw_min"])
    lon_raw_min = int(parameters["lon_raw_min"])
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
        raise ValueError("Coordinate lookup arrays have different shapes")
    recovery_lookup = load_recovery_lookup(args.blank_time_audit)

    local_start = pd.Timestamp(
        datetime.combine(args.start_date, time.min)
    )
    local_end = pd.Timestamp(
        datetime.combine(
            args.end_date + timedelta(days=1),
            time.min,
        )
    )

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    outside_path = args.diagnostics_dir / "outside_china_records.parquet"
    outside_summary_path = (
        args.diagnostics_dir / "outside_china_summary.csv"
    )
    inside_unmatched_path = (
        args.diagnostics_dir / "inside_china_unmatched_records.parquet"
    )
    method_summary_path = (
        args.diagnostics_dir / "unmatched_summary_by_method.csv"
    )
    gate_path = (
        args.diagnostics_dir / "spatial_assignment_quality_gate.csv"
    )
    inside_summary_path = (
        args.diagnostics_dir
        / "inside_china_province_assignment_summary.csv"
    )
    outputs = (
        outside_path,
        outside_summary_path,
        inside_unmatched_path,
        method_summary_path,
        gate_path,
        inside_summary_path,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "National-filtered spatial QC outputs exist; rerun with "
            "--overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    outside_temp = outside_path.with_name(f".{outside_path.name}.tmp")
    unmatched_temp = inside_unmatched_path.with_name(
        f".{inside_unmatched_path.name}.tmp"
    )
    for temp_path in (outside_temp, unmatched_temp):
        if temp_path.exists():
            temp_path.unlink()
    outside_writer: pq.ParquetWriter | None = None
    unmatched_writer: pq.ParquetWriter | None = None

    method_records: dict[int, int] = defaultdict(int)
    method_app_count: dict[int, int] = defaultdict(int)
    valid_records = 0
    valid_app_count = 0
    invalid_time_records = 0
    invalid_time_app_count = 0
    outside_records = 0
    outside_app_count = 0
    inside_records = 0
    inside_app_count = 0
    inside_matched_records = 0
    inside_matched_app_count = 0
    inside_unmatched_records = 0
    inside_unmatched_app_count = 0
    lookup_missing_records = 0
    lookup_missing_app_count = 0
    outside_seen = np.zeros(len(national_lookup), dtype=bool)
    outside_abnormal_coordinates: set[tuple[int, int]] = set()
    outside_lon_min = np.inf
    outside_lon_max = -np.inf
    outside_lat_min = np.inf
    outside_lat_max = -np.inf

    paths: list[Path] = []
    for day in pd.date_range(CHUNYUN_START, CHUNYUN_END, freq="D"):
        path = (
            args.input_dir
            / f"population_china_{day:%Y-%m-%d}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Daily parquet missing: {path}")
        paths.append(path)

    try:
        for file_index, path in enumerate(paths, start=1):
            parquet_file = pq.ParquetFile(path)
            required_columns = {
                "date",
                "source_file",
                "line_no",
                "time",
                "lat_raw",
                "lon_raw",
                "lat",
                "lon",
                "count",
            }
            missing = sorted(
                required_columns.difference(
                    parquet_file.schema_arrow.names
                )
            )
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")

            file_valid_records = 0
            file_outside_records = 0
            file_inside_unmatched = 0
            for batch in parquet_file.iter_batches(
                batch_size=args.batch_size
            ):
                frame = batch.to_pandas()
                count = pd.to_numeric(frame["count"], errors="coerce")
                if count.isna().any() or (count < 0).any():
                    raise ValueError(f"Invalid count values in {path}")
                utc_time, invalid_time = resolve_utc_time(
                    frame,
                    recovery_lookup,
                )
                invalid_time_records += int(invalid_time.sum())
                invalid_time_app_count += int(
                    count.loc[invalid_time].sum()
                )
                local_time = utc_time + pd.Timedelta(hours=8)
                selected = (
                    ~invalid_time
                    & local_time.ge(local_start)
                    & local_time.lt(local_end)
                )
                positions = np.flatnonzero(selected.to_numpy())
                if not len(positions):
                    continue
                frame = frame.iloc[positions].copy()
                count_values = count.iloc[positions].to_numpy(
                    dtype="int64"
                )
                utc_selected = utc_time.iloc[positions]
                local_selected = local_time.iloc[positions]

                lat_raw = pd.to_numeric(
                    frame["lat_raw"],
                    errors="coerce",
                ).to_numpy()
                lon_raw = pd.to_numeric(
                    frame["lon_raw"],
                    errors="coerce",
                ).to_numpy()
                finite = np.isfinite(lat_raw) & np.isfinite(lon_raw)
                lat_int = np.where(
                    finite,
                    lat_raw,
                    lat_raw_min - 1,
                ).astype("int64")
                lon_int = np.where(
                    finite,
                    lon_raw,
                    lon_raw_min - 1,
                ).astype("int64")
                flat = (
                    (lat_int - lat_raw_min) * nlon
                    + (lon_int - lon_raw_min)
                )
                in_lookup_range = (
                    finite
                    & (flat >= 0)
                    & (flat < len(national_lookup))
                )
                present_in_lookup = np.zeros(len(frame), dtype=bool)
                present_in_lookup[in_lookup_range] = (
                    national_lookup[flat[in_lookup_range]] >= 0
                )
                usable_lookup = in_lookup_range & present_in_lookup
                lookup_missing = ~usable_lookup
                lookup_missing_records += int(lookup_missing.sum())
                lookup_missing_app_count += int(
                    count_values[lookup_missing].sum()
                )

                inside = np.zeros(len(frame), dtype=bool)
                inside[usable_lookup] = (
                    national_lookup[flat[usable_lookup]] == 1
                )
                outside = ~inside
                province_codes = np.full(
                    len(frame),
                    -1,
                    dtype="int16",
                )
                methods = np.full(len(frame), 9, dtype="int8")
                province_codes[usable_lookup] = province_lookup[
                    flat[usable_lookup]
                ]
                methods[usable_lookup] = method_lookup[
                    flat[usable_lookup]
                ]
                methods[outside & usable_lookup] = 8

                matched = inside & (province_codes >= 0)
                unmatched = inside & (province_codes < 0)
                methods[unmatched] = 7

                batch_total_count = int(count_values.sum())
                valid_records += len(frame)
                valid_app_count += batch_total_count
                file_valid_records += len(frame)
                outside_records += int(outside.sum())
                outside_app_count += int(count_values[outside].sum())
                file_outside_records += int(outside.sum())
                inside_records += int(inside.sum())
                inside_app_count += int(count_values[inside].sum())
                inside_matched_records += int(matched.sum())
                inside_matched_app_count += int(
                    count_values[matched].sum()
                )
                inside_unmatched_records += int(unmatched.sum())
                inside_unmatched_app_count += int(
                    count_values[unmatched].sum()
                )
                file_inside_unmatched += int(unmatched.sum())
                for method in np.unique(methods):
                    mask = methods == method
                    method_records[int(method)] += int(mask.sum())
                    method_app_count[int(method)] += int(
                        count_values[mask].sum()
                    )

                if outside.any():
                    outside_positions = np.flatnonzero(outside)
                    valid_outside_lookup = outside & usable_lookup
                    outside_seen[
                        flat[valid_outside_lookup]
                    ] = True
                    abnormal_positions = np.flatnonzero(
                        outside & ~usable_lookup
                    )
                    outside_abnormal_coordinates.update(
                        (
                            int(lat_int[position]),
                            int(lon_int[position]),
                        )
                        for position in abnormal_positions
                    )
                    outside_lat = pd.to_numeric(
                        frame.iloc[outside_positions]["lat"],
                        errors="coerce",
                    )
                    outside_lon = pd.to_numeric(
                        frame.iloc[outside_positions]["lon"],
                        errors="coerce",
                    )
                    outside_lat_min = min(
                        outside_lat_min,
                        float(outside_lat.min()),
                    )
                    outside_lat_max = max(
                        outside_lat_max,
                        float(outside_lat.max()),
                    )
                    outside_lon_min = min(
                        outside_lon_min,
                        float(outside_lon.min()),
                    )
                    outside_lon_max = max(
                        outside_lon_max,
                        float(outside_lon.max()),
                    )
                    outside_part = pd.DataFrame(
                        {
                            "utc_time": utc_selected.iloc[
                                outside_positions
                            ].to_numpy(dtype="datetime64[ns]"),
                            "local_time": local_selected.iloc[
                                outside_positions
                            ].to_numpy(dtype="datetime64[ns]"),
                            "lat": frame.iloc[outside_positions][
                                "lat"
                            ].to_numpy(),
                            "lon": frame.iloc[outside_positions][
                                "lon"
                            ].to_numpy(),
                            "count": frame.iloc[outside_positions][
                                "count"
                            ].to_numpy(),
                            "source_file": frame.iloc[outside_positions][
                                "source_file"
                            ].astype("string").to_numpy(),
                            "line_no": frame.iloc[outside_positions][
                                "line_no"
                            ].to_numpy(),
                            "exclusion_reason": (
                                "outside_national_boundary"
                            ),
                        }
                    )
                    table = pa.Table.from_pandas(
                        outside_part,
                        preserve_index=False,
                    )
                    if outside_writer is None:
                        outside_writer = pq.ParquetWriter(
                            outside_temp,
                            table.schema,
                            compression=args.compression,
                        )
                    outside_writer.write_table(table)

                if unmatched.any():
                    unmatched_positions = np.flatnonzero(unmatched)
                    unmatched_part = pd.DataFrame(
                        {
                            "utc_time": utc_selected.iloc[
                                unmatched_positions
                            ].to_numpy(dtype="datetime64[ns]"),
                            "local_time": local_selected.iloc[
                                unmatched_positions
                            ].to_numpy(dtype="datetime64[ns]"),
                            "lat": frame.iloc[unmatched_positions][
                                "lat"
                            ].to_numpy(),
                            "lon": frame.iloc[unmatched_positions][
                                "lon"
                            ].to_numpy(),
                            "count": frame.iloc[unmatched_positions][
                                "count"
                            ].to_numpy(),
                            "source_file": frame.iloc[
                                unmatched_positions
                            ]["source_file"].astype("string").to_numpy(),
                            "line_no": frame.iloc[unmatched_positions][
                                "line_no"
                            ].to_numpy(),
                            "assignment_method_code": methods[
                                unmatched
                            ],
                            "assignment_method": [
                                METHOD_LABELS[int(value)]
                                for value in methods[unmatched]
                            ],
                            "unmatched_reason": (
                                "inside national boundary but no province "
                                "point or 0.1-degree grid rule matched"
                            ),
                        }
                    )
                    table = pa.Table.from_pandas(
                        unmatched_part,
                        preserve_index=False,
                    )
                    if unmatched_writer is None:
                        unmatched_writer = pq.ParquetWriter(
                            unmatched_temp,
                            table.schema,
                            compression=args.compression,
                        )
                    unmatched_writer.write_table(table)

            print(
                f"National-filtered scan [{file_index}/{len(paths)}] "
                f"{path.name}: valid={file_valid_records:,}, "
                f"outside={file_outside_records:,}, "
                f"inside_unmatched={file_inside_unmatched:,}"
            )
    finally:
        if outside_writer is not None:
            outside_writer.close()
        if unmatched_writer is not None:
            unmatched_writer.close()

    if outside_writer is None:
        pq.write_table(
            empty_record_table(outside=True),
            outside_temp,
            compression=args.compression,
        )
    if unmatched_writer is None:
        pq.write_table(
            empty_record_table(outside=False),
            unmatched_temp,
            compression=args.compression,
        )
    outside_temp.replace(outside_path)
    unmatched_temp.replace(inside_unmatched_path)

    outside_unique_coordinate_count = int(outside_seen.sum()) + len(
        outside_abnormal_coordinates
    )
    outside_share = (
        outside_app_count / valid_app_count
        if valid_app_count
        else np.nan
    )
    inside_unmatched_share = (
        inside_unmatched_app_count / inside_app_count
        if inside_app_count
        else np.nan
    )
    gate_pass = bool(
        np.isfinite(inside_unmatched_share)
        and inside_unmatched_share
        <= args.max_final_unmatched_share
    )

    outside_summary = pd.DataFrame(
        [
            {
                "record_count": outside_records,
                "unique_coordinate_count": (
                    outside_unique_coordinate_count
                ),
                "app_count": outside_app_count,
                "share_of_valid_app_count": outside_share,
                "longitude_min": (
                    outside_lon_min
                    if np.isfinite(outside_lon_min)
                    else np.nan
                ),
                "longitude_max": (
                    outside_lon_max
                    if np.isfinite(outside_lon_max)
                    else np.nan
                ),
                "latitude_min": (
                    outside_lat_min
                    if np.isfinite(outside_lat_min)
                    else np.nan
                ),
                "latitude_max": (
                    outside_lat_max
                    if np.isfinite(outside_lat_max)
                    else np.nan
                ),
                "exclusion_reason": "outside_national_boundary",
            }
        ]
    )
    outside_summary.to_csv(
        outside_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    method_rows = []
    for code, label in METHOD_LABELS.items():
        records = method_records.get(code, 0)
        app_count = method_app_count.get(code, 0)
        method_rows.append(
            {
                "assignment_method_code": code,
                "assignment_method": label,
                "scope": (
                    "outside_china"
                    if code in {8, 9}
                    else "inside_china"
                ),
                "record_count": records,
                "app_count": app_count,
                "app_count_share_of_all_valid": (
                    app_count / valid_app_count
                    if valid_app_count
                    else np.nan
                ),
                "app_count_share_of_inside_china": (
                    app_count / inside_app_count
                    if inside_app_count and code not in {8, 9}
                    else np.nan
                ),
            }
        )
    pd.DataFrame(method_rows).to_csv(
        method_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    inside_summary = pd.DataFrame(
        [
            {
                "inside_china_records": inside_records,
                "inside_china_app_count": inside_app_count,
                "matched_province_records": inside_matched_records,
                "matched_province_app_count": inside_matched_app_count,
                "unmatched_inside_china_records": (
                    inside_unmatched_records
                ),
                "unmatched_inside_china_app_count": (
                    inside_unmatched_app_count
                ),
                "in_china_unmatched_share": inside_unmatched_share,
                "point_boundary_single_records": method_records.get(2, 0),
                "point_boundary_multiple_records": method_records.get(3, 0),
                "grid_single_overlap_records": method_records.get(4, 0),
                "grid_largest_overlap_records": method_records.get(5, 0),
                "multiple_province_rule": (
                    "lowest canonical province code from "
                    "province_name_mapping.csv"
                ),
                "nearest_assignment_used": False,
            }
        ]
    )
    inside_summary.to_csv(
        inside_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    gate = pd.DataFrame(
        [
            {
                "date_basis": args.date_basis,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "all_valid_records": valid_records,
                "all_valid_app_count": valid_app_count,
                "excluded_unrecoverable_time_records": (
                    invalid_time_records
                ),
                "excluded_unrecoverable_time_app_count": (
                    invalid_time_app_count
                ),
                "outside_china_records": outside_records,
                "outside_china_app_count": outside_app_count,
                "outside_china_app_count_share_of_valid": outside_share,
                "inside_china_records": inside_records,
                "inside_china_app_count": inside_app_count,
                "inside_china_matched_records": inside_matched_records,
                "inside_china_matched_app_count": (
                    inside_matched_app_count
                ),
                "inside_china_unmatched_records": (
                    inside_unmatched_records
                ),
                "inside_china_unmatched_app_count": (
                    inside_unmatched_app_count
                ),
                "in_china_unmatched_share": inside_unmatched_share,
                "final_unmatched_app_count_share": (
                    inside_unmatched_share
                ),
                "maximum_allowed_final_unmatched_share": (
                    args.max_final_unmatched_share
                ),
                "quality_gate_pass": gate_pass,
                "lookup_missing_records": lookup_missing_records,
                "lookup_missing_app_count": lookup_missing_app_count,
                "national_boundary_predicate": "covers",
                "outside_grid_fallback_used": False,
                "nearest_assignment_used": False,
            }
        ]
    )
    gate.to_csv(gate_path, index=False, encoding="utf-8-sig")

    print(f"Created: {outside_path}")
    print(f"Created: {outside_summary_path}")
    print(f"Created: {inside_unmatched_path}")
    print(f"Created: {inside_summary_path}")
    print(f"Created: {method_summary_path}")
    print(f"Created: {gate_path}")
    print(f"All valid records: {valid_records:,}")
    print(f"All valid App count: {valid_app_count:,}")
    print(f"Outside-China records excluded: {outside_records:,}")
    print(f"Outside-China App count: {outside_app_count:,}")
    print(f"Outside-China App count share: {outside_share:.10%}")
    print(f"Inside-China records: {inside_records:,}")
    print(f"Inside-China App count: {inside_app_count:,}")
    print(
        "Inside-China unmatched records: "
        f"{inside_unmatched_records:,}"
    )
    print(
        "Inside-China unmatched App count: "
        f"{inside_unmatched_app_count:,}"
    )
    print(
        "Inside-China unmatched App count share: "
        f"{inside_unmatched_share:.10%}"
    )
    print(
        "National-filtered spatial quality gate: "
        f"{'PASS' if gate_pass else 'FAIL'} "
        f"(threshold={args.max_final_unmatched_share:.2%})"
    )


if __name__ == "__main__":
    main()
