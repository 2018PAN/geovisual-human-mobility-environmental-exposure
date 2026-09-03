from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from project_common import (  # noqa: E402
    BOUNDARY_ROOT,
    DIAGNOSTICS_DIR,
    MAPPING_PATH,
    add_date_arguments,
    read_and_validate_official_population,
)
from province_boundary import (  # noqa: E402
    discover_province_boundary,
    load_dissolved_provinces,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate official 2018 population and auto-discovered province boundaries."
    )
    parser.add_argument("--boundary-root", type=Path, default=BOUNDARY_ROOT)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--official-population", type=Path, default=None)
    parser.add_argument("--diagnostics-dir", type=Path, default=DIAGNOSTICS_DIR)
    parser.add_argument(
        "--missing-crs",
        default="EPSG:4326",
        help="Explicit CRS used only when selected source metadata is missing and bounds are geographic.",
    )
    add_date_arguments(parser, default_basis="local")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Date arguments recorded for workflow consistency: "
        f"{args.start_date} to {args.end_date}, basis={args.date_basis}"
    )
    population, official_total, official_path = (
        read_and_validate_official_population(
            args.official_population, args.mapping
        )
    )
    selection = discover_province_boundary(args.boundary_root, args.mapping)
    provinces, selection, original_crs, effective_crs = load_dissolved_provinces(
        selection,
        boundary_root=args.boundary_root,
        mapping_path=args.mapping,
        missing_crs=args.missing_crs,
    )

    mapping_names = set(population["province"])
    boundary_names = set(provinces["province"])
    coverage = sorted(mapping_names.intersection(boundary_names))
    missing = sorted(mapping_names.difference(boundary_names))
    extra = sorted(boundary_names.difference(mapping_names))
    if len(coverage) != 34 or missing or extra:
        raise ValueError(
            f"Boundary does not cover the required 34 regions: "
            f"coverage={len(coverage)}, missing={missing}, extra={extra}"
        )

    boundary_validation = population[
        ["boundary_code", "boundary_name", "province", "official_name"]
    ].copy()
    boundary_validation["boundary_present"] = boundary_validation[
        "province"
    ].isin(boundary_names)
    boundary_validation["boundary_file"] = str(selection.path)
    boundary_validation["province_identifier_field"] = selection.province_field
    boundary_validation["source_crs"] = original_crs or "MISSING"
    boundary_validation["effective_crs"] = effective_crs

    args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    population_output = (
        args.diagnostics_dir / "official_population_2018_validation.csv"
    )
    boundary_output = args.diagnostics_dir / "boundary_province_validation.csv"
    summary_output = args.diagnostics_dir / "boundary_population_validation_summary.md"
    existing = [
        path
        for path in (population_output, boundary_output, summary_output)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Validation outputs exist; rerun with --overwrite: "
            + ", ".join(map(str, existing))
        )

    population.to_csv(population_output, index=False, encoding="utf-8-sig")
    boundary_validation.to_csv(
        boundary_output, index=False, encoding="utf-8-sig"
    )
    summary = f"""# Boundary and official population validation

- Official population file: `{official_path}`
- Official regional rows: {len(population)}
- Official population unit: persons
- Official population direct 34-region sum: {official_total:,}
- Province mapping: `{args.mapping.resolve()}`
- Selected boundary file: `{selection.path}`
- Province identifier field: `{selection.province_field}` ({selection.field_kind})
- Source CRS: {original_crs or "MISSING"}
- Effective CRS for point-in-polygon matching: {effective_crs}
- CRS handling: source coordinates have geographic bounds; missing metadata is explicitly assigned `{args.missing_crs}`.
- Boundary coverage: {len(coverage)}/34
- Covered regions: {", ".join(coverage)}
- Missing regions: {", ".join(missing) or "none"}
- Extra regions: {", ".join(extra) or "none"}
"""
    summary_output.write_text(summary, encoding="utf-8")

    print(f"Created: {population_output}")
    print(f"Created: {boundary_output}")
    print(f"Created: {summary_output}")
    print(f"Official population total: {official_total:,}")
    print(f"Boundary coverage: {len(coverage)}/34")


if __name__ == "__main__":
    main()
