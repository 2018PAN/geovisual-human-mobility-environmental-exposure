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
    POPULATION_INPUT_DIR,
    PROVINCE_ASSIGNED_DIR,
    add_date_arguments,
    iter_dates,
    read_and_validate_official_population,
    validate_date_range,
)


OUTPUT_PREFIX = "population_province_assigned_"


def parse_boolean(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def require_passing_gate(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Spatial quality gate not found: {path}")
    gate = pd.read_csv(path)
    required = {
        "date_basis",
        "start_date",
        "end_date",
        "all_valid_records",
        "all_valid_app_count",
        "outside_china_records",
        "outside_china_app_count",
        "inside_china_records",
        "inside_china_app_count",
        "inside_china_unmatched_records",
        "inside_china_unmatched_app_count",
        "in_china_unmatched_share",
        "quality_gate_pass",
    }
    missing = sorted(required.difference(gate.columns))
    if len(gate) != 1 or missing:
        raise ValueError(
            f"Invalid spatial quality gate {path}; rows={len(gate)}, "
            f"missing={missing}"
        )
    row = gate.iloc[0]
    if not parse_boolean(row["quality_gate_pass"]):
        raise RuntimeError("Formal province output blocked by spatial quality gate")
    if float(row["in_china_unmatched_share"]) > 0.01:
        raise RuntimeError("China-only unmatched share exceeds 1%")
    return row


def load_lookup_metadata(
    path: Path,
) -> tuple[dict[int, str], dict[int, str], dict[str, str]]:
    metadata = pd.read_csv(path, dtype="string")
    provinces = metadata.loc[
        metadata["metadata_type"].eq("province_code"), ["code", "label"]
    ].dropna()
    methods = metadata.loc[
        metadata["metadata_type"].eq("assignment_method"), ["code", "label"]
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
        raise ValueError(f"Expected 34 lookup provinces, found {len(province_names)}")
    return province_names, method_names, parameter_values


def load_blank_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Blank-time audit not found: {path}")
    audit = pd.read_csv(
        path,
        dtype={
            "source_date": "string",
            "source_file": "string",
            "line_no": "int64",
            "is_recoverable": "string",
            "recovered_utc_time": "string",
        },
    )
    required = {
        "source_date",
        "source_file",
        "line_no",
        "is_recoverable",
        "recovered_utc_time",
    }
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"Blank-time audit is missing columns: {missing}")
    audit["is_recoverable"] = (
        audit["is_recoverable"]
        .fillna("")
        .str.strip()
        .str.casefold()
        .isin({"true", "1", "yes"})
    )
    audit["recovered_utc_time"] = pd.to_datetime(
        audit["recovered_utc_time"], errors="coerce"
    )
    if (
        audit["is_recoverable"] & audit["recovered_utc_time"].isna()
    ).any():
        raise ValueError("Recoverable audit entry lacks recovered UTC time")
    if audit.duplicated(["source_date", "source_file", "line_no"]).any():
        raise ValueError("Blank-time audit contains duplicate source keys")
    return audit[
        [
            "source_date",
            "source_file",
            "line_no",
            "is_recoverable",
            "recovered_utc_time",
        ]
    ].copy()


def resolve_audited_times(
    frame: pd.DataFrame, audit: pd.DataFrame
) -> tuple[pd.Series, pd.Series, int]:
    utc_time = pd.to_datetime(frame["time"], errors="coerce", cache=True)
    invalid = utc_time.isna()
    if not invalid.any():
        return utc_time, invalid, 0

    invalid_rows = frame.loc[
        invalid, ["date", "source_file", "line_no"]
    ].copy()
    invalid_rows.columns = ["source_date", "source_file", "line_no"]
    invalid_rows["source_date"] = invalid_rows["source_date"].astype("string")
    invalid_rows["source_file"] = invalid_rows["source_file"].astype("string")
    invalid_rows["line_no"] = pd.to_numeric(
        invalid_rows["line_no"], errors="raise"
    ).astype("int64")
    invalid_rows["_frame_index"] = invalid_rows.index
    resolved = invalid_rows.merge(
        audit,
        on=["source_date", "source_file", "line_no"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if resolved["_merge"].ne("both").any():
        examples = resolved.loc[
            resolved["_merge"].ne("both"),
            ["source_date", "source_file", "line_no"],
        ].drop_duplicates().head(10)
        raise ValueError(
            "Invalid timestamps absent from the completed audit:\n"
            + examples.to_string(index=False)
        )
    recoverable = (
        resolved["is_recoverable"].fillna(False)
        & resolved["recovered_utc_time"].notna()
    )
    if recoverable.any():
        positions = resolved.loc[recoverable, "_frame_index"]
        utc_time.loc[positions] = resolved.loc[
            recoverable, "recovered_utc_time"
        ].to_numpy(dtype="datetime64[ns]")
    return utc_time, utc_time.isna(), int(recoverable.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build formal Beijing-local-date province partitions directly from "
            "audited UTC daily parquet, applying the national-boundary filter "
            "before the validated province lookup."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROVINCE_ASSIGNED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    parser.add_argument(
        "--blank-time-audit",
        type=Path,
        default=DIAGNOSTICS_DIR / "blank_time_audit.csv",
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
    if args.date_basis != "local":
        raise ValueError("Formal outputs in this script require --date-basis local")
    gate = require_passing_gate(args.spatial_quality_gate)
    audit = load_blank_audit(args.blank_time_audit)
    official, official_total, official_path = read_and_validate_official_population(
        args.official_population, args.mapping
    )
    official_lookup = official.set_index("province")[
        "official_population_2018"
    ].to_dict()
    province_names, method_names, parameters = load_lookup_metadata(
        args.lookup_metadata
    )
    if set(province_names.values()) != set(official["province"]):
        raise ValueError("Lookup and official-population province sets differ")

    lat_raw_min = int(parameters["lat_raw_min"])
    lat_raw_max = int(parameters["lat_raw_max"])
    lon_raw_min = int(parameters["lon_raw_min"])
    lon_raw_max = int(parameters["lon_raw_max"])
    nlon = int(parameters["nlon"])
    province_lookup = np.load(args.province_lookup, mmap_mode="r")
    method_lookup = np.load(args.method_lookup, mmap_mode="r")
    national_lookup = np.load(args.national_inside_lookup, mmap_mode="r")
    if not (
        province_lookup.shape == method_lookup.shape == national_lookup.shape
    ):
        raise ValueError("Coordinate lookup array shapes differ")

    province_object = np.empty(max(province_names) + 1, dtype=object)
    for code, name in province_names.items():
        province_object[code] = name
    method_object = np.empty(max(method_names) + 1, dtype=object)
    for code, name in method_names.items():
        method_object[code] = name

    local_start = pd.Timestamp(datetime.combine(args.start_date, time.min))
    local_end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    # The confirmed source archive begins on UTC 2018-02-01. Its absent
    # 2018-01-31 neighbor is deliberately not fabricated, so the first Beijing
    # day starts at 08:00.
    candidate_start = args.start_date - timedelta(days=1)
    candidate_dates = list(iter_dates(candidate_start, args.end_date))
    input_dates = [
        day
        for day in candidate_dates
        if (
            args.input_dir
            / f"population_china_{day.isoformat()}.parquet"
        ).exists()
    ]
    if (
        args.start_date != datetime(2018, 2, 1).date()
        and candidate_start not in input_dates
    ):
        raise FileNotFoundError(
            "The preceding UTC partition is required for a complete local-date "
            f"selection: {candidate_start}"
        )
    input_paths = [
        args.input_dir / f"population_china_{day.isoformat()}.parquet"
        for day in input_dates
    ]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required UTC source partitions: "
            + ", ".join(map(str, missing))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        args.output_dir / f"{OUTPUT_PREFIX}{day.isoformat()}.parquet"
        for day in iter_dates(args.start_date, args.end_date)
    ]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Formal province outputs already exist; use --overwrite: "
            + ", ".join(map(str, existing_outputs[:5]))
        )

    print("Formal audited national-filtered province assignment")
    print(f"  local range: {args.start_date} to {args.end_date}")
    print("  order: audited time -> local-date filter -> national covers -> province")
    print(f"  official file: {official_path}")
    print(f"  official total: {official_total:,}")
    print(f"  national predicate: {parameters['national_boundary_predicate']}")
    print("  obsolete pre-national unmatched outputs: not used")

    writers: dict[str, pq.ParquetWriter] = {}
    temp_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    date_records: dict[str, int] = defaultdict(int)
    date_app_count: dict[str, int] = defaultdict(int)
    date_timestamps: dict[str, set[pd.Timestamp]] = defaultdict(set)
    total_valid_records = 0
    total_valid_count = 0
    outside_records = 0
    outside_count = 0
    inside_records = 0
    inside_count = 0
    excluded_blank_records = 0
    excluded_blank_count = 0
    recovered_records = 0
    unmatched_inside_records = 0
    unmatched_inside_count = 0

    try:
        for input_number, input_path in enumerate(input_paths, start=1):
            parquet_file = pq.ParquetFile(input_path)
            required = {
                "date",
                "source_file",
                "line_no",
                "time",
                "lat_raw",
                "lon_raw",
                "count",
                "lat",
                "lon",
            }
            missing_columns = sorted(
                required.difference(parquet_file.schema_arrow.names)
            )
            if missing_columns:
                raise ValueError(
                    f"{input_path} is missing columns: {missing_columns}"
                )
            print(
                f"[{input_number:02d}/{len(input_paths):02d}] reading "
                f"{input_path.name}"
            )
            for batch_number, batch in enumerate(
                parquet_file.iter_batches(batch_size=args.batch_size), start=1
            ):
                frame = batch.to_pandas()
                count = pd.to_numeric(frame["count"], errors="coerce")
                if count.isna().any() or (count < 0).any():
                    raise ValueError(f"Invalid count in {input_path}")
                utc_time, invalid_time, recovered = resolve_audited_times(
                    frame, audit
                )
                recovered_records += recovered
                if invalid_time.any():
                    excluded_blank_records += int(invalid_time.sum())
                    excluded_blank_count += int(count.loc[invalid_time].sum())

                local_time = utc_time + pd.Timedelta(hours=8)
                selected = (
                    ~invalid_time
                    & (local_time >= local_start)
                    & (local_time < local_end)
                )
                if not selected.any():
                    continue
                frame = frame.loc[selected].copy()
                count = count.loc[selected]
                utc_time = utc_time.loc[selected]
                local_time = local_time.loc[selected]
                total_valid_records += len(frame)
                total_valid_count += int(count.sum())

                lat_raw = pd.to_numeric(
                    frame["lat_raw"], errors="coerce"
                ).to_numpy()
                lon_raw = pd.to_numeric(
                    frame["lon_raw"], errors="coerce"
                ).to_numpy()
                finite = np.isfinite(lat_raw) & np.isfinite(lon_raw)
                lat_int = np.where(
                    finite, lat_raw, lat_raw_min - 1
                ).astype("int64")
                lon_int = np.where(
                    finite, lon_raw, lon_raw_min - 1
                ).astype("int64")
                in_lookup_range = (
                    finite
                    & (lat_int >= lat_raw_min)
                    & (lat_int <= lat_raw_max)
                    & (lon_int >= lon_raw_min)
                    & (lon_int <= lon_raw_max)
                )
                flat = np.zeros(len(frame), dtype="int64")
                flat[in_lookup_range] = (
                    (lat_int[in_lookup_range] - lat_raw_min) * nlon
                    + (lon_int[in_lookup_range] - lon_raw_min)
                )
                inside = np.zeros(len(frame), dtype=bool)
                inside[in_lookup_range] = (
                    national_lookup[flat[in_lookup_range]] == 1
                )
                outside = ~inside
                outside_records += int(outside.sum())
                if outside.any():
                    outside_count += int(
                        count.iloc[np.flatnonzero(outside)].sum()
                    )
                if not inside.any():
                    continue

                positions = np.flatnonzero(inside)
                frame = frame.iloc[positions].copy()
                count = count.iloc[positions]
                utc_time = utc_time.iloc[positions]
                local_time = local_time.iloc[positions]
                flat = flat[positions]
                province_codes = province_lookup[flat].astype(
                    "int16", copy=False
                )
                method_codes = method_lookup[flat].astype("int8", copy=False)
                unmatched = province_codes < 0
                if unmatched.any():
                    unmatched_inside_records += int(unmatched.sum())
                    unmatched_inside_count += int(
                        count.iloc[np.flatnonzero(unmatched)].sum()
                    )
                    raise RuntimeError(
                        "China-inside record lacks a valid province assignment"
                    )

                province_values = province_object[province_codes]
                method_values = method_object[method_codes]
                frame["utc_date"] = utc_time.dt.strftime("%Y-%m-%d").to_numpy()
                frame["utc_time"] = utc_time.to_numpy(dtype="datetime64[ns]")
                frame["utc_hour"] = utc_time.dt.hour.astype("int8").to_numpy()
                frame["local_date"] = local_time.dt.strftime(
                    "%Y-%m-%d"
                ).to_numpy()
                frame["local_time"] = local_time.to_numpy(dtype="datetime64[ns]")
                frame["local_hour"] = local_time.dt.hour.astype("int8").to_numpy()
                frame["province"] = province_values
                frame["official_population_2018"] = (
                    pd.Series(province_values, index=frame.index)
                    .map(official_lookup)
                    .astype("int64")
                )
                frame["province_assignment_method"] = method_values
                frame["province_assignment_method_code"] = method_codes
                frame["province_assignment_status"] = "assigned"
                frame["province_assignment_reason"] = (
                    "validated national-first coordinate assignment"
                )
                inside_records += len(frame)
                inside_count += int(count.sum())

                required_order = [
                    "date",
                    "time",
                    "utc_date",
                    "utc_time",
                    "utc_hour",
                    "local_date",
                    "local_time",
                    "local_hour",
                    "lat_raw",
                    "lon_raw",
                    "lat",
                    "lon",
                    "count",
                    "source_file",
                    "line_no",
                    "province",
                    "official_population_2018",
                    "province_assignment_method",
                    "province_assignment_method_code",
                    "province_assignment_status",
                    "province_assignment_reason",
                ]
                frame = frame[required_order]
                for local_date_text, part in frame.groupby(
                    "local_date", sort=True
                ):
                    local_date_text = str(local_date_text)
                    final_path = (
                        args.output_dir
                        / f"{OUTPUT_PREFIX}{local_date_text}.parquet"
                    )
                    if local_date_text not in writers:
                        temp_path = final_path.with_name(
                            f".{final_path.name}.tmp"
                        )
                        if temp_path.exists():
                            temp_path.unlink()
                        table = pa.Table.from_pandas(
                            part, preserve_index=False
                        )
                        writers[local_date_text] = pq.ParquetWriter(
                            temp_path,
                            table.schema,
                            compression=args.compression,
                            use_dictionary=True,
                        )
                        temp_paths[local_date_text] = temp_path
                        final_paths[local_date_text] = final_path
                    else:
                        table = pa.Table.from_pandas(
                            part,
                            preserve_index=False,
                            schema=writers[local_date_text].schema,
                        )
                    writers[local_date_text].write_table(table)
                    date_records[local_date_text] += len(part)
                    date_app_count[local_date_text] += int(part["count"].sum())
                    date_timestamps[local_date_text].update(
                        pd.to_datetime(part["local_time"]).unique()
                    )
                    del table
                if batch_number % 20 == 0:
                    print(
                        f"  batch {batch_number}: formal rows "
                        f"{inside_records:,}"
                    )
    except Exception:
        for writer in writers.values():
            writer.close()
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink()
        raise
    else:
        for writer in writers.values():
            writer.close()

    actual_dates = sorted(final_paths)
    expected_date_texts = [
        day.isoformat() for day in iter_dates(args.start_date, args.end_date)
    ]
    if actual_dates != expected_date_texts:
        raise RuntimeError(
            f"Output local dates differ: actual={actual_dates}, "
            f"expected={expected_date_texts}"
        )

    full_gate_range = (
        str(gate["date_basis"]) == "local"
        and str(gate["start_date"]) == args.start_date.isoformat()
        and str(gate["end_date"]) == args.end_date.isoformat()
    )
    if full_gate_range:
        checks = {
            "all_valid_records": total_valid_records,
            "all_valid_app_count": total_valid_count,
            "outside_china_records": outside_records,
            "outside_china_app_count": outside_count,
            "inside_china_records": inside_records,
            "inside_china_app_count": inside_count,
            "inside_china_unmatched_records": unmatched_inside_records,
            "inside_china_unmatched_app_count": unmatched_inside_count,
        }
        disagreements = {
            key: (int(gate[key]), value)
            for key, value in checks.items()
            if int(gate[key]) != value
        }
        if disagreements:
            raise RuntimeError(
                f"Formal streaming counts disagree with passing gate: "
                f"{disagreements}"
            )

    for local_date_text, temp_path in temp_paths.items():
        final_path = final_paths[local_date_text]
        temp_path.replace(final_path)
        print(f"Created: {final_path}")

    manifest_rows = []
    for local_date_text in expected_date_texts:
        day_start = pd.Timestamp(local_date_text)
        expected_index = pd.date_range(
            day_start,
            day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=5),
            freq="5min",
        )
        observed_index = pd.DatetimeIndex(
            date_timestamps[local_date_text]
        ).drop_duplicates().sort_values()
        missing_index = expected_index.difference(observed_index)
        groups: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if len(missing_index):
            group_start = missing_index[0]
            previous = missing_index[0]
            for value in missing_index[1:]:
                if value - previous != pd.Timedelta(minutes=5):
                    groups.append((group_start, previous))
                    group_start = value
                previous = value
            groups.append((group_start, previous))
        missing_intervals = " | ".join(
            f"{left:%H:%M}-{right:%H:%M}" for left, right in groups
        )
        available = len(observed_index)
        expected = len(expected_index)
        missing_count = len(missing_index)
        manifest_rows.append(
            {
                "local_date": local_date_text,
                "record_count": date_records[local_date_text],
                "app_count": date_app_count[local_date_text],
                "expected_five_minute_timestamps": expected,
                "available_five_minute_timestamps": available,
                "missing_five_minute_timestamps": missing_count,
                "is_complete_local_date": missing_count == 0,
                "missing_intervals": missing_intervals,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = (
        args.diagnostics_dir / "formal_province_assignment_manifest.csv"
    )
    summary_path = (
        args.diagnostics_dir / "formal_province_assignment_summary.csv"
    )
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "date_basis": "local",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "input_valid_records": total_valid_records,
                "input_valid_app_count": total_valid_count,
                "excluded_unrecoverable_time_records": excluded_blank_records,
                "excluded_unrecoverable_time_app_count": excluded_blank_count,
                "recovered_time_records": recovered_records,
                "excluded_outside_records": outside_records,
                "excluded_outside_app_count": outside_count,
                "formal_inside_assigned_records": inside_records,
                "formal_inside_assigned_app_count": inside_count,
                "inside_unmatched_records": unmatched_inside_records,
                "inside_unmatched_app_count": unmatched_inside_count,
                "national_boundary_predicate": parameters[
                    "national_boundary_predicate"
                ],
                "official_population_total": official_total,
                "obsolete_pre_national_unmatched_used": False,
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Created: {manifest_path}")
    print(f"Created: {summary_path}")
    print(f"Formal province rows: {inside_records:,}")
    print(f"Formal province App count: {inside_count:,}")


if __name__ == "__main__":
    main()
