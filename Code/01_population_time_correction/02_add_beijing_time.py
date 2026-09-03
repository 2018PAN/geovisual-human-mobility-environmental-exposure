from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    BEIJING_OFFSET_HOURS,
    CHUNYUN_END,
    CHUNYUN_START,
    DIAGNOSTICS_DIR,
    POPULATION_INPUT_DIR,
    TIME_CORRECTED_DIR,
    add_date_arguments,
    selected_input_utc_dates,
    validate_date_range,
)


OUTPUT_PREFIX = "population_time_corrected_"


def output_path(output_dir: Path, local_date: date) -> Path:
    return output_dir / f"{OUTPUT_PREFIX}{local_date.isoformat()}.parquet"


def ordered_columns(source_columns: list[str]) -> list[str]:
    required_order = [
        "date",
        "time",
        "utc_date",
        "utc_time",
        "utc_hour",
        "local_date",
        "local_time",
        "local_hour",
        "lat",
        "lon",
        "count",
    ]
    extras = [column for column in source_columns if column not in required_order]
    return required_order + extras


def load_blank_time_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Blank-time audit is required before formal conversion: "
            f"{path}"
        )
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
        "decision",
        "reason",
    }
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"Blank-time audit is missing columns: {missing}")
    audit["is_recoverable"] = (
        audit["is_recoverable"]
        .fillna("")
        .str.strip()
        .str.casefold()
        .isin(["true", "1", "yes"])
    )
    audit["recovered_utc_time"] = pd.to_datetime(
        audit["recovered_utc_time"], errors="coerce"
    )
    invalid_recovery = (
        audit["is_recoverable"] & audit["recovered_utc_time"].isna()
    )
    if invalid_recovery.any():
        raise ValueError(
            "Audit marks rows recoverable but has no valid recovered UTC "
            "timestamp"
        )
    if audit.duplicated(
        ["source_date", "source_file", "line_no"]
    ).any():
        raise ValueError("Blank-time audit contains duplicate source keys")
    return audit[
        [
            "source_date",
            "source_file",
            "line_no",
            "is_recoverable",
            "recovered_utc_time",
            "decision",
            "reason",
        ]
    ].copy()


def resolve_audited_utc_time(
    frame: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.Series, int, pd.Series]:
    utc_time = pd.to_datetime(frame["time"], errors="coerce", cache=True)
    invalid = utc_time.isna()
    if not invalid.any():
        return utc_time, 0, invalid

    positions = np.flatnonzero(invalid.to_numpy())
    keys = frame.iloc[positions][
        ["date", "source_file", "line_no"]
    ].copy()
    keys.columns = ["source_date", "source_file", "line_no"]
    keys["source_date"] = keys["source_date"].astype("string")
    keys["source_file"] = keys["source_file"].astype("string")
    keys["line_no"] = pd.to_numeric(
        keys["line_no"], errors="raise"
    ).astype("int64")
    keys["_frame_position"] = positions
    resolved = keys.merge(
        audit,
        on=["source_date", "source_file", "line_no"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    not_audited = resolved["_merge"].ne("both")
    if not_audited.any():
        examples = resolved.loc[
            not_audited,
            ["source_date", "source_file", "line_no"],
        ].drop_duplicates().head(10)
        raise ValueError(
            "Unparseable time records were not covered by the targeted "
            "blank-time audit:\n"
            + examples.to_string(index=False)
        )

    recoverable = (
        resolved["is_recoverable"].fillna(False)
        & resolved["recovered_utc_time"].notna()
    )
    if recoverable.any():
        recovered_positions = resolved.loc[
            recoverable, "_frame_position"
        ].to_numpy(dtype="int64")
        utc_time.iloc[recovered_positions] = resolved.loc[
            recoverable, "recovered_utc_time"
        ].to_numpy(dtype="datetime64[ns]")
    remaining_invalid = utc_time.isna()
    return utc_time, int(recoverable.sum()), remaining_invalid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve the original App population time as UTC and add Beijing "
            "local time (+8 hours), writing partitions by local_date."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=TIME_CORRECTED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument(
        "--blank-time-audit",
        type=Path,
        default=DIAGNOSTICS_DIR / "blank_time_audit.csv",
    )
    add_date_arguments(parser, default_basis="local")
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Population input directory not found: {args.input_dir}")
    blank_time_audit = load_blank_time_audit(args.blank_time_audit)

    input_dates = selected_input_utc_dates(
        args.start_date, args.end_date, args.date_basis
    )
    input_paths = [
        args.input_dir / f"population_china_{utc_date.isoformat()}.parquet"
        for utc_date in input_dates
    ]
    existing_inputs = [path for path in input_paths if path.exists()]
    missing_inputs = [path for path in input_paths if not path.exists()]
    required_utc_paths = {
        args.input_dir / f"population_china_{utc_date.isoformat()}.parquet"
        for utc_date in pd.date_range(
            args.start_date, args.end_date, freq="D"
        ).date
    }
    missing_required_inputs = sorted(
        required_utc_paths.difference(existing_inputs)
    )
    if missing_required_inputs:
        raise FileNotFoundError(
            "Required source UTC filename partitions are missing: "
            + ", ".join(map(str, missing_required_inputs))
        )
    if not existing_inputs:
        raise FileNotFoundError("No selected UTC source parquet files exist")

    if args.date_basis == "local":
        local_start = datetime.combine(args.start_date, time.min)
        local_end_exclusive = datetime.combine(
            args.end_date + timedelta(days=1), time.min
        )
    else:
        local_start = datetime.combine(CHUNYUN_START, time.min)
        local_end_exclusive = datetime.combine(
            CHUNYUN_END + timedelta(days=1), time.min
        )
    utc_start = datetime.combine(args.start_date, time.min)
    utc_end_exclusive = datetime.combine(args.end_date + timedelta(days=1), time.min)

    print("UTC to Beijing-time population conversion")
    print(f"  input: {args.input_dir.resolve()}")
    print(f"  output: {args.output_dir.resolve()}")
    print(f"  date basis: {args.date_basis}")
    print(f"  selected dates: {args.start_date} to {args.end_date}")
    print("  transformation: utc_time = original time")
    print(f"  transformation: local_time = utc_time + {BEIJING_OFFSET_HOURS} hours")
    print("  original date/time columns are preserved")
    if missing_inputs:
        print("  adjacent/edge UTC filename partitions absent (not filled):")
        for path in missing_inputs:
            print(f"    {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    temp_paths: dict[str, Path] = {}
    final_paths: dict[str, Path] = {}
    skipped_dates: set[str] = set()
    coverage_counts: dict[tuple[str, int], int] = defaultdict(int)
    utc_time_min: pd.Timestamp | None = None
    utc_time_max: pd.Timestamp | None = None
    local_time_min: pd.Timestamp | None = None
    local_time_max: pd.Timestamp | None = None
    input_rows = 0
    input_app_count = 0
    output_rows = 0
    filtered_rows = 0
    recovered_time_rows = 0
    excluded_unrecoverable_time_rows = 0
    excluded_unrecoverable_time_app_count = 0

    try:
        for input_path in existing_inputs:
            print(f"Reading UTC source: {input_path}")
            parquet_file = pq.ParquetFile(input_path)
            source_columns = parquet_file.schema_arrow.names
            required = {
                "date",
                "source_file",
                "line_no",
                "time",
                "lat",
                "lon",
                "count",
            }
            missing = sorted(required.difference(source_columns))
            if missing:
                raise ValueError(f"{input_path} is missing columns: {missing}")

            for batch_number, batch in enumerate(
                parquet_file.iter_batches(batch_size=args.batch_size), start=1
            ):
                frame = batch.to_pandas()
                input_rows += len(frame)
                input_app_count += int(
                    pd.to_numeric(frame["count"], errors="raise").sum()
                )
                utc_time, recovered_count, invalid_time = (
                    resolve_audited_utc_time(frame, blank_time_audit)
                )
                recovered_time_rows += recovered_count
                invalid_count = int(invalid_time.sum())
                if invalid_count:
                    excluded_unrecoverable_time_rows += invalid_count
                    excluded_unrecoverable_time_app_count += int(
                        pd.to_numeric(
                            frame.loc[invalid_time, "count"],
                            errors="raise",
                        ).sum()
                    )
                    print(
                        f"  batch {batch_number}: explicitly excluding "
                        f"{invalid_count:,} audited unrecoverable-time "
                        "records"
                    )
                local_time = utc_time + pd.Timedelta(hours=BEIJING_OFFSET_HOURS)

                if args.date_basis == "utc":
                    selected = (utc_time >= utc_start) & (utc_time < utc_end_exclusive)
                else:
                    selected = (local_time >= local_start) & (
                        local_time < local_end_exclusive
                    )
                selected &= (local_time >= datetime.combine(CHUNYUN_START, time.min)) & (
                    local_time
                    < datetime.combine(CHUNYUN_END + timedelta(days=1), time.min)
                )
                selected &= ~invalid_time
                filtered_rows += int((~selected & ~invalid_time).sum())
                if not selected.any():
                    continue

                frame = frame.loc[selected].copy()
                utc_time = utc_time.loc[selected]
                local_time = local_time.loc[selected]
                frame["utc_time"] = utc_time.to_numpy(dtype="datetime64[ns]")
                frame["utc_date"] = utc_time.dt.strftime("%Y-%m-%d").to_numpy()
                frame["utc_hour"] = utc_time.dt.hour.astype("int8").to_numpy()
                frame["local_time"] = local_time.to_numpy(dtype="datetime64[ns]")
                frame["local_date"] = local_time.dt.strftime("%Y-%m-%d").to_numpy()
                frame["local_hour"] = local_time.dt.hour.astype("int8").to_numpy()
                frame = frame[ordered_columns(source_columns)]

                batch_utc_min = utc_time.min()
                batch_utc_max = utc_time.max()
                batch_local_min = local_time.min()
                batch_local_max = local_time.max()
                utc_time_min = (
                    batch_utc_min
                    if utc_time_min is None
                    else min(utc_time_min, batch_utc_min)
                )
                utc_time_max = (
                    batch_utc_max
                    if utc_time_max is None
                    else max(utc_time_max, batch_utc_max)
                )
                local_time_min = (
                    batch_local_min
                    if local_time_min is None
                    else min(local_time_min, batch_local_min)
                )
                local_time_max = (
                    batch_local_max
                    if local_time_max is None
                    else max(local_time_max, batch_local_max)
                )

                for (local_date_text, local_hour), part in frame.groupby(
                    ["local_date", "local_hour"], sort=True
                ):
                    coverage_counts[(str(local_date_text), int(local_hour))] += len(part)

                for local_date_text, part in frame.groupby("local_date", sort=True):
                    local_date_text = str(local_date_text)
                    local_date_value = datetime.strptime(
                        local_date_text, "%Y-%m-%d"
                    ).date()
                    final_path = output_path(args.output_dir, local_date_value)
                    if final_path.exists() and not args.overwrite:
                        if local_date_text not in skipped_dates:
                            print(f"Skipping existing local partition: {final_path}")
                            skipped_dates.add(local_date_text)
                        continue
                    if local_date_text not in writers:
                        temp_path = final_path.with_name(f".{final_path.name}.tmp")
                        if temp_path.exists():
                            temp_path.unlink()
                        table = pa.Table.from_pandas(part, preserve_index=False)
                        writers[local_date_text] = pq.ParquetWriter(
                            temp_path, table.schema, compression=args.compression
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
                    output_rows += len(part)
                    del table

                print(
                    f"  batch {batch_number}: input={len(batch):,}, "
                    f"kept={len(frame):,}, cumulative output={output_rows:,}"
                )
    finally:
        for writer in writers.values():
            writer.close()

    for local_date_text, temp_path in temp_paths.items():
        final_path = final_paths[local_date_text]
        temp_path.replace(final_path)
        print(f"Created local partition: {final_path}")

    coverage_rows = []
    local_dates = sorted({key[0] for key in coverage_counts})
    for local_date_text in local_dates:
        available_hours = sorted(
            hour
            for (date_text, hour), count in coverage_counts.items()
            if date_text == local_date_text and count > 0
        )
        missing_hours = sorted(set(range(24)).difference(available_hours))
        coverage_rows.append(
            {
                "local_date": local_date_text,
                "expected_hours": 24,
                "available_hours": len(available_hours),
                "missing_hours": ",".join(
                    f"{hour:02d}" for hour in missing_hours
                ),
                "is_complete_local_day": len(available_hours) == 24,
                "records": sum(
                    count
                    for (date_text, _hour), count in coverage_counts.items()
                    if date_text == local_date_text
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    manifest_path = args.diagnostics_dir / "population_time_correction_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Diagnostic manifest exists; rerun with --overwrite: {manifest_path}"
        )
    coverage.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"Created diagnostic manifest: {manifest_path}")
    exclusions_path = (
        args.diagnostics_dir / "population_time_correction_exclusions.csv"
    )
    if exclusions_path.exists() and not args.overwrite:
        raise FileExistsError(
            "Diagnostic exclusions file exists; rerun with --overwrite: "
            f"{exclusions_path}"
        )
    exclusions = pd.DataFrame(
        [
            {
                "date_basis": args.date_basis,
                "start_date": args.start_date.isoformat(),
                "end_date": args.end_date.isoformat(),
                "recovered_blank_time_records": recovered_time_rows,
                "excluded_unrecoverable_time_records": (
                    excluded_unrecoverable_time_rows
                ),
                "excluded_unrecoverable_time_app_count": (
                    excluded_unrecoverable_time_app_count
                ),
                "scanned_input_app_count_denominator": input_app_count,
                "excluded_unrecoverable_time_app_count_share": (
                    excluded_unrecoverable_time_app_count / input_app_count
                    if input_app_count
                    else np.nan
                ),
                "decision": (
                    "use audited recovered UTC time; exclude audited "
                    "unrecoverable records without imputation"
                ),
            }
        ]
    )
    exclusions.to_csv(
        exclusions_path, index=False, encoding="utf-8-sig"
    )
    print(f"Created diagnostic exclusions: {exclusions_path}")
    print(f"Input rows: {input_rows:,}")
    print(f"Rows excluded by selected/formal local range: {filtered_rows:,}")
    print(f"Blank-time rows recovered from raw TXT: {recovered_time_rows:,}")
    print(
        "Audited unrecoverable-time rows explicitly excluded: "
        f"{excluded_unrecoverable_time_rows:,}"
    )
    print(
        "Audited unrecoverable-time App count explicitly excluded: "
        f"{excluded_unrecoverable_time_app_count:,}"
    )
    if input_app_count:
        print(
            "Audited unrecoverable-time App count share of scanned input: "
            f"{excluded_unrecoverable_time_app_count / input_app_count:.10%}"
        )
    print(f"Output rows written: {output_rows:,}")
    print(f"Actual UTC range: {utc_time_min} to {utc_time_max}")
    print(f"Actual Beijing range: {local_time_min} to {local_time_max}")


if __name__ == "__main__":
    main()
