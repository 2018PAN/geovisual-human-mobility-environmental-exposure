from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    DIAGNOSTICS_DIR,
    POPULATION_INPUT_DIR,
    RAW_DATA_ROOT,
    add_date_arguments,
    validate_date_range,
)


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def raw_population_root() -> Path:
    return RAW_DATA_ROOT / "Population" / "2018"


def row_group_may_have_blank_time(
    parquet_file: pq.ParquetFile, row_group_index: int
) -> bool:
    time_index = parquet_file.schema_arrow.names.index("time")
    statistics = (
        parquet_file.metadata.row_group(row_group_index)
        .column(time_index)
        .statistics
    )
    if statistics is None or not statistics.has_min_max:
        return True
    return str(statistics.min).strip() == ""


def read_exact_line(path: Path, line_no: int) -> tuple[str | None, str]:
    if not path.exists():
        return None, "raw_file_missing"
    with path.open("r", encoding="utf-8") as handle:
        for current_line_no, line in enumerate(handle, start=1):
            if current_line_no == line_no:
                return line.rstrip("\r\n"), "raw_line_found"
    return None, "raw_line_missing"


def inspect_raw_line(path: Path, line_no: int) -> dict[str, object]:
    line, status = read_exact_line(path, line_no)
    if line is None:
        return {
            "raw_line_status": status,
            "raw_time_text": "",
            "is_recoverable": False,
            "recovered_utc_time": pd.NaT,
            "decision": "exclude_unrecoverable",
            "reason": status,
        }
    if ";" not in line:
        return {
            "raw_line_status": "raw_line_has_no_semicolon",
            "raw_time_text": "",
            "is_recoverable": False,
            "recovered_utc_time": pd.NaT,
            "decision": "exclude_unrecoverable",
            "reason": "raw line has no time/data separator",
        }
    raw_time_text = line.split(";", 1)[0].strip()
    if not raw_time_text:
        return {
            "raw_line_status": "raw_time_blank",
            "raw_time_text": "",
            "is_recoverable": False,
            "recovered_utc_time": pd.NaT,
            "decision": "exclude_unrecoverable",
            "reason": "raw TXT line itself has no time; inference is prohibited",
        }
    try:
        recovered = datetime.strptime(raw_time_text, TIME_FORMAT)
    except ValueError:
        return {
            "raw_line_status": "raw_time_unparseable",
            "raw_time_text": raw_time_text,
            "is_recoverable": False,
            "recovered_utc_time": pd.NaT,
            "decision": "exclude_unrecoverable",
            "reason": f"raw time does not match {TIME_FORMAT}",
        }
    return {
        "raw_line_status": "raw_time_valid",
        "raw_time_text": raw_time_text,
        "is_recoverable": True,
        "recovered_utc_time": recovered,
        "decision": "use_recovered_utc_time",
        "reason": "valid UTC time recovered from the exact raw TXT line",
    }


def scan_blank_groups(
    parquet_paths: list[Path],
) -> tuple[pd.DataFrame, list[tuple[Path, int]]]:
    partials: list[pd.DataFrame] = []
    suspect_row_groups: list[tuple[Path, int]] = []
    columns = [
        "date",
        "source_file",
        "line_no",
        "time",
        "lat",
        "lon",
        "count",
    ]
    for path in parquet_paths:
        parquet_file = pq.ParquetFile(path)
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            if not row_group_may_have_blank_time(parquet_file, row_group_index):
                continue
            table = parquet_file.read_row_group(
                row_group_index, columns=columns
            )
            frame = table.to_pandas()
            blank = frame["time"].astype("string").fillna("").str.strip().eq("")
            frame = frame.loc[blank].copy()
            if frame.empty:
                continue
            suspect_row_groups.append((path, row_group_index))
            grouped = (
                frame.groupby(
                    ["date", "source_file", "line_no"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    blank_record_count=("count", "size"),
                    app_count_sum=("count", "sum"),
                    lat_min=("lat", "min"),
                    lat_max=("lat", "max"),
                    lon_min=("lon", "min"),
                    lon_max=("lon", "max"),
                )
            )
            partials.append(grouped)
            print(
                f"Blank scan {path.name} row_group={row_group_index}: "
                f"records={len(frame):,}"
            )
    if not partials:
        return pd.DataFrame(), suspect_row_groups
    groups = (
        pd.concat(partials, ignore_index=True)
        .groupby(
            ["date", "source_file", "line_no"],
            as_index=False,
            dropna=False,
        )
        .agg(
            blank_record_count=("blank_record_count", "sum"),
            app_count_sum=("app_count_sum", "sum"),
            lat_min=("lat_min", "min"),
            lat_max=("lat_max", "max"),
            lon_min=("lon_min", "min"),
            lon_max=("lon_max", "max"),
        )
    )
    return groups, suspect_row_groups


def total_app_count(parquet_paths: list[Path], batch_size: int) -> int:
    total = 0
    for index, path in enumerate(parquet_paths, start=1):
        parquet_file = pq.ParquetFile(path)
        file_total = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=["count"]
        ):
            value = pc.sum(batch.column(0)).as_py()
            file_total += int(value or 0)
        total += file_total
        print(
            f"App-count denominator [{index}/{len(parquet_paths)}] "
            f"{path.name}: {file_total:,}"
        )
    return total


def write_unrecoverable_records(
    suspect_row_groups: list[tuple[Path, int]],
    audit: pd.DataFrame,
    output_path: Path,
    *,
    overwrite: bool,
    compression: str,
) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Unrecoverable output exists; rerun with --overwrite: {output_path}"
        )
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    lookup = audit.set_index(["source_date", "source_file", "line_no"])[
        ["is_recoverable", "decision", "reason"]
    ]
    writer: pq.ParquetWriter | None = None
    record_count = 0
    app_count_sum = 0
    try:
        for path, row_group_index in suspect_row_groups:
            parquet_file = pq.ParquetFile(path)
            frame = parquet_file.read_row_group(row_group_index).to_pandas()
            blank = frame["time"].astype("string").fillna("").str.strip().eq("")
            frame = frame.loc[blank].copy()
            if frame.empty:
                continue
            keys = pd.MultiIndex.from_arrays(
                [
                    frame["date"].astype(str),
                    frame["source_file"].astype(str),
                    frame["line_no"].astype(int),
                ],
                names=["source_date", "source_file", "line_no"],
            )
            decisions = lookup.reindex(keys)
            if decisions["is_recoverable"].isna().any():
                raise ValueError("Blank-time audit lookup is incomplete")
            unrecoverable = ~decisions["is_recoverable"].astype(bool).to_numpy()
            part = frame.loc[unrecoverable].copy()
            if part.empty:
                continue
            part["blank_time_decision"] = decisions.loc[
                unrecoverable, "decision"
            ].to_numpy()
            part["blank_time_reason"] = decisions.loc[
                unrecoverable, "reason"
            ].to_numpy()
            record_count += len(part)
            app_count_sum += int(pd.to_numeric(part["count"]).sum())
            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path, table.schema, compression=compression
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        schema = pa.schema(
            [
                ("date", pa.string()),
                ("source_file", pa.string()),
                ("line_no", pa.int32()),
                ("time", pa.string()),
                ("lat_raw", pa.int16()),
                ("lon_raw", pa.int16()),
                ("count", pa.int32()),
                ("lat", pa.float32()),
                ("lon", pa.float32()),
                ("blank_time_decision", pa.string()),
                ("blank_time_reason", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist([], schema=schema), temp_path)
    temp_path.replace(output_path)
    return record_count, app_count_sum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit blank population timestamps by reading only the exact raw TXT "
            "source_file and line_no retained in parquet."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=POPULATION_INPUT_DIR)
    parser.add_argument("--raw-root", type=Path, default=raw_population_root())
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    add_date_arguments(parser, default_basis="utc")
    parser.add_argument("--batch-size", type=int, default=5_000_000)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date_basis != "utc":
        raise ValueError("Raw parquet audit requires --date-basis utc")
    validate_date_range(args.start_date, args.end_date)
    parquet_paths = []
    for day in pd.date_range(args.start_date, args.end_date, freq="D"):
        path = args.input_dir / f"population_china_{day:%Y-%m-%d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Daily parquet not found: {path}")
        parquet_paths.append(path)

    print("Blank-time targeted raw TXT audit")
    print(f"  parquet input: {args.input_dir.resolve()}")
    print(f"  raw TXT root: {args.raw_root.resolve()}")
    groups, suspect_row_groups = scan_blank_groups(parquet_paths)
    if groups.empty:
        print("No blank-time records found")
        return

    audit_rows = []
    for row in groups.itertuples(index=False):
        raw_path = args.raw_root / str(row.date) / str(row.source_file)
        result = inspect_raw_line(raw_path, int(row.line_no))
        audit_rows.append(
            {
                "source_date": str(row.date),
                "source_file": str(row.source_file),
                "line_no": int(row.line_no),
                "raw_file_path": str(raw_path.resolve()),
                "raw_line_status": result["raw_line_status"],
                "raw_time_text": result["raw_time_text"],
                "is_recoverable": result["is_recoverable"],
                "recovered_utc_time": result["recovered_utc_time"],
                "decision": result["decision"],
                "reason": result["reason"],
                "blank_record_count": int(row.blank_record_count),
                "app_count_sum": int(row.app_count_sum),
                "lat_min": row.lat_min,
                "lat_max": row.lat_max,
                "lon_min": row.lon_min,
                "lon_max": row.lon_max,
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit["recovered_utc_time"] = pd.to_datetime(
        audit["recovered_utc_time"], errors="coerce"
    )

    print("Computing full App-count denominator without reading raw TXT...")
    all_app_count = total_app_count(parquet_paths, args.batch_size)
    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.diagnostics_dir / "blank_time_audit.csv"
    unrecoverable_path = (
        args.diagnostics_dir / "blank_time_unrecoverable.parquet"
    )
    summary_path = args.diagnostics_dir / "blank_time_summary.csv"
    for path in (audit_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Diagnostic output exists; rerun with --overwrite: {path}"
            )

    unrecoverable_records, unrecoverable_app_count = write_unrecoverable_records(
        suspect_row_groups,
        audit,
        unrecoverable_path,
        overwrite=args.overwrite,
        compression=args.compression,
    )

    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    summary = audit.copy()
    summary["recoverable_record_count"] = np.where(
        summary["is_recoverable"], summary["blank_record_count"], 0
    )
    summary["unrecoverable_record_count"] = np.where(
        summary["is_recoverable"], 0, summary["blank_record_count"]
    )
    summary["recoverable_app_count"] = np.where(
        summary["is_recoverable"], summary["app_count_sum"], 0
    )
    summary["unrecoverable_app_count"] = np.where(
        summary["is_recoverable"], 0, summary["app_count_sum"]
    )
    blank_record_total = int(summary["blank_record_count"].sum())
    blank_app_count_total = int(summary["app_count_sum"].sum())
    summary["line_no_text"] = summary["line_no"].astype("string")

    def join_unique(values: pd.Series) -> str:
        return " | ".join(
            sorted({str(value) for value in values if pd.notna(value)})
        )

    file_summary = (
        summary.groupby(
            ["source_date", "source_file", "raw_file_path"],
            as_index=False,
            dropna=False,
        )
        .agg(
            affected_source_lines=("line_no", "nunique"),
            line_numbers=("line_no_text", join_unique),
            blank_record_count=("blank_record_count", "sum"),
            recoverable_record_count=("recoverable_record_count", "sum"),
            unrecoverable_record_count=(
                "unrecoverable_record_count",
                "sum",
            ),
            app_count_sum=("app_count_sum", "sum"),
            recoverable_app_count=("recoverable_app_count", "sum"),
            unrecoverable_app_count=("unrecoverable_app_count", "sum"),
            lat_min=("lat_min", "min"),
            lat_max=("lat_max", "max"),
            lon_min=("lon_min", "min"),
            lon_max=("lon_max", "max"),
            raw_line_status=("raw_line_status", join_unique),
            decision=("decision", join_unique),
            reason=("reason", join_unique),
        )
    )
    file_summary["blank_record_share_of_all_blank_records"] = (
        file_summary["blank_record_count"] / blank_record_total
    )
    file_summary["app_count_share_of_all_blank_time_app_count"] = (
        file_summary["app_count_sum"] / blank_app_count_total
    )
    file_summary["app_count_share_of_all_population_app_count"] = (
        file_summary["app_count_sum"] / all_app_count
    )
    file_summary[
        "unrecoverable_app_count_share_of_all_population_app_count"
    ] = file_summary["unrecoverable_app_count"] / all_app_count
    file_summary["all_blank_time_records"] = blank_record_total
    file_summary["all_blank_time_app_count"] = blank_app_count_total
    file_summary["all_population_app_count"] = all_app_count
    file_summary["rank_by_unrecoverable_app_count"] = (
        file_summary["unrecoverable_app_count"]
        .rank(method="dense", ascending=False)
        .astype("int32")
    )
    file_summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    print(f"Created: {audit_path}")
    print(f"Created: {unrecoverable_path}")
    print(f"Created: {summary_path}")
    print(f"Blank-time records: {int(audit['blank_record_count'].sum()):,}")
    print(
        f"Recoverable records: "
        f"{int(audit.loc[audit['is_recoverable'], 'blank_record_count'].sum()):,}"
    )
    print(f"Unrecoverable records: {unrecoverable_records:,}")
    print(f"Unrecoverable App count: {unrecoverable_app_count:,}")
    print(
        f"Unrecoverable App count share: "
        f"{unrecoverable_app_count / all_app_count:.10%}"
    )


if __name__ == "__main__":
    main()
