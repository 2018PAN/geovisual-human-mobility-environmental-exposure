from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    DECOMPOSITION_DIR,
    atomic_parquet,
    require_columns,
    require_file,
    stage_log,
)
from downstream_spatial import assign_grid_province  # noqa: E402


INPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
OUTPUT_PATH = DECOMPOSITION_DIR / "decomposition_input.parquet"

REQUIRED = [
    "grid_id",
    "grid_x",
    "grid_y",
    "grid_center_x",
    "grid_center_y",
    "grid_center_lon",
    "grid_center_lat",
    "pre_mean_population",
    "festival_mean_population",
    "post_mean_population",
    "pre_mean_pm25",
    "festival_mean_pm25",
    "post_mean_pm25",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare formal calibrated-population input for the unchanged "
            "legacy Chunyun accounting decomposition."
        )
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, REQUIRED, INPUT_PATH)
    work = frame.replace([np.inf, -np.inf], np.nan).copy()
    numeric = [
        "pre_mean_population",
        "festival_mean_population",
        "post_mean_population",
        "pre_mean_pm25",
        "festival_mean_pm25",
        "post_mean_pm25",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=REQUIRED).copy()
    populations = [
        "pre_mean_population",
        "festival_mean_population",
        "post_mean_population",
    ]
    concentrations = [
        "pre_mean_pm25",
        "festival_mean_pm25",
        "post_mean_pm25",
    ]
    valid = ~(
        (work[populations] < 0).any(axis=1)
        | (work[concentrations] <= 0).any(axis=1)
    )
    work = work.loc[valid].copy()
    if work.empty:
        raise ValueError("No valid decomposition rows remain")

    # Old decomposition variable names are retained as explicit calibrated
    # population aliases.
    for phase in ["pre", "festival", "post"]:
        work[f"{phase}_mean_count"] = work[
            f"{phase}_mean_population"
        ]
    if "province" not in work:
        work = assign_grid_province(work)
    return work.sort_values("grid_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    with stage_log("07_01_prepare_decomposition_input"):
        require_file(INPUT_PATH)
        if OUTPUT_PATH.exists() and not args.overwrite:
            print(f"SKIP existing: {OUTPUT_PATH}")
            return
        print(f"READ {INPUT_PATH}")
        result = prepare(pd.read_parquet(INPUT_PATH))
        atomic_parquet(result, OUTPUT_PATH)
        print(f"WRITE {OUTPUT_PATH} rows={len(result):,}")
        print(
            "Compatibility note: *_mean_count fields are calibrated "
            "estimated population."
        )


if __name__ == "__main__":
    main()
