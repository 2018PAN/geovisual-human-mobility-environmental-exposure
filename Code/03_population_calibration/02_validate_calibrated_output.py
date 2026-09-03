from __future__ import annotations

import argparse
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    CALIBRATED_DIR,
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    add_date_arguments,
    read_and_validate_official_population,
    selected_input_local_dates,
    validate_date_range,
)


INPUT_PREFIX = "population_calibrated_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently read back calibrated parquet outputs and verify time "
            "semantics, schema, row preservation, and national totals."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=CALIBRATED_DIR)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    add_date_arguments(parser, default_basis="local")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--relative-tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_date_range(args.start_date, args.end_date)
    official, official_total, official_path = read_and_validate_official_population(
        args.official_population, args.mapping
    )
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
            "Calibrated files missing: " + ", ".join(map(str, missing_inputs))
        )

    start = pd.Timestamp(datetime.combine(args.start_date, time.min))
    end = pd.Timestamp(
        datetime.combine(args.end_date + timedelta(days=1), time.min)
    )
    required = {
        "time",
        "utc_date",
        "utc_time",
        "utc_hour",
        "local_date",
        "local_time",
        "local_hour",
        "lat",
        "lon",
        "province",
        "count",
        "app_count",
        "official_population_2018",
        "province_reference_app_count",
        "province_expansion_weight",
        "preliminary_population",
        "national_timestamp_scaling_factor",
        "estimated_population",
    }
    partials: list[pd.DataFrame] = []
    partition_rows = []
    total_rows = 0
    invalid_original_utc_relation = 0
    invalid_local_offset = 0
    invalid_local_partition = 0
    invalid_count_preservation = 0
    negative_estimated = 0
    unmatched_records = 0
    matched_provinces: set[str] = set()
    schema_texts: list[str] = []

    for input_path in input_paths:
        partition_date = input_path.stem.removeprefix(INPUT_PREFIX)
        parquet_file = pq.ParquetFile(input_path)
        missing = sorted(required.difference(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"{input_path} is missing columns: {missing}")
        schema_texts.append(
            "|".join(
                f"{field.name}:{field.type}" for field in parquet_file.schema_arrow
            )
        )
        partition_count = 0
        print(f"Read-back validation: {input_path}")
        columns = sorted(required)
        for batch_number, batch in enumerate(
            parquet_file.iter_batches(batch_size=args.batch_size, columns=columns),
            start=1,
        ):
            frame = batch.to_pandas()
            utc_time = pd.to_datetime(frame["utc_time"], errors="coerce")
            local_time = pd.to_datetime(frame["local_time"], errors="coerce")
            original_time = pd.to_datetime(frame["time"], errors="coerce")
            if utc_time.isna().any() or local_time.isna().any():
                raise ValueError(f"{input_path} contains invalid output timestamps")
            if args.date_basis == "utc":
                selected = (utc_time >= start) & (utc_time < end)
            else:
                selected = (local_time >= start) & (local_time < end)
            frame = frame.loc[selected].copy()
            utc_time = utc_time.loc[selected]
            local_time = local_time.loc[selected]
            original_time = original_time.loc[selected]
            if frame.empty:
                continue

            total_rows += len(frame)
            partition_count += len(frame)
            invalid_original_utc_relation += int(
                (original_time != utc_time).sum() + original_time.isna().sum()
            )
            invalid_local_offset += int(
                (local_time - utc_time != pd.Timedelta(hours=8)).sum()
            )
            invalid_local_partition += int(
                (frame["local_date"].astype(str) != partition_date).sum()
            )
            invalid_count_preservation += int(
                (
                    pd.to_numeric(frame["count"], errors="coerce")
                    != pd.to_numeric(frame["app_count"], errors="coerce")
                ).sum()
            )
            estimated = pd.to_numeric(
                frame["estimated_population"], errors="coerce"
            )
            negative_estimated += int((estimated < 0).sum())
            unmatched = frame["province"].isna()
            unmatched_records += int(unmatched.sum())
            matched_provinces.update(
                frame.loc[~unmatched, "province"].dropna().astype(str).unique()
            )
            partials.append(
                pd.DataFrame(
                    {
                        "utc_time": utc_time.to_numpy(),
                        "estimated_population": estimated.to_numpy(),
                        "records": 1,
                        "unmatched_records": unmatched.astype("int8").to_numpy(),
                    }
                )
                .groupby("utc_time", as_index=False)
                .agg(
                    final_estimated_population_total=(
                        "estimated_population",
                        "sum",
                    ),
                    records=("records", "sum"),
                    unmatched_records=("unmatched_records", "sum"),
                )
            )
            print(
                f"  batch {batch_number}: selected rows={len(frame):,}; "
                f"cumulative={total_rows:,}"
            )
        partition_rows.append(
            {
                "local_partition": partition_date,
                "rows": partition_count,
                "file": str(input_path.resolve()),
            }
        )

    totals = (
        pd.concat(partials, ignore_index=True)
        .groupby("utc_time", as_index=False)
        .agg(
            final_estimated_population_total=(
                "final_estimated_population_total",
                "sum",
            ),
            records=("records", "sum"),
            unmatched_records=("unmatched_records", "sum"),
        )
        .sort_values("utc_time")
    )
    totals["target_population_total"] = official_total
    totals["relative_error"] = (
        totals["final_estimated_population_total"]
        - totals["target_population_total"]
    ).abs() / totals["target_population_total"]
    max_error = float(totals["relative_error"].max())
    mean_error = float(totals["relative_error"].mean())
    max_absolute_error = float(
        (
            totals["final_estimated_population_total"]
            - totals["target_population_total"]
        ).abs().max()
    )
    mean_absolute_error = float(
        (
            totals["final_estimated_population_total"]
            - totals["target_population_total"]
        ).abs().mean()
    )
    provinces_missing = sorted(set(official["province"]).difference(matched_provinces))
    checks = {
        "original_time_equals_utc_time": invalid_original_utc_relation == 0,
        "local_time_equals_utc_plus_8h": invalid_local_offset == 0,
        "local_partition_matches_local_date": invalid_local_partition == 0,
        "count_preserved_as_app_count": invalid_count_preservation == 0,
        "estimated_population_nonnegative": negative_estimated == 0,
        "all_34_provinces_present": not provinces_missing,
        "national_total_within_tolerance": max_error <= args.relative_tolerance,
        "schemas_identical": len(set(schema_texts)) == 1,
    }
    passed = all(checks.values())

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    totals_path = args.diagnostics_dir / "calibrated_output_readback_totals.csv"
    partitions_path = (
        args.diagnostics_dir / "calibrated_output_partition_inventory.csv"
    )
    summary_path = args.diagnostics_dir / "calibrated_output_validation.md"
    existing = [
        path for path in (totals_path, partitions_path, summary_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Read-back diagnostics exist; rerun with --overwrite: "
            + ", ".join(map(str, existing))
        )
    totals.to_csv(totals_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(partition_rows).to_csv(
        partitions_path, index=False, encoding="utf-8-sig"
    )
    summary = f"""# Calibrated output read-back validation

- Result: {"PASS" if passed else "FAIL"}
- Date selection: {args.start_date} to {args.end_date} (`{args.date_basis}` basis).
- Official population: `{official_path}`.
- Direct 34-region target: {official_total:,}.
- Calibrated rows read back: {total_rows:,}.
- UTC timestamps read back: {len(totals)}.
- Unmatched records retained with null estimated population: {unmatched_records:,}.
- Maximum row-output national relative error: {max_error:.12g}.
- Mean row-output national relative error: {mean_error:.12g}.
- Maximum row-output national absolute error: {max_absolute_error:.12g}.
- Mean row-output national absolute error: {mean_absolute_error:.12g}.
- Missing matched provinces: {", ".join(provinces_missing) or "none"}.
- Required output fields: {", ".join(sorted(required))}.

Checks:

{chr(10).join(f"- {name}: {value}" for name, value in checks.items())}
"""
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Created: {totals_path}")
    print(f"Created: {partitions_path}")
    print(f"Created: {summary_path}")
    print(f"Read-back validation result: {'PASS' if passed else 'FAIL'}")
    print(f"Maximum row-output national relative error: {max_error:.12g}")
    print(f"Mean row-output national relative error: {mean_error:.12g}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
