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
    atomic_csv,
    atomic_parquet,
    compatibility_alias_note,
    ensure_outputs,
    require_columns,
    require_file,
    stage_log,
)


INPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_summary.parquet"
OUTPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
FIELDS_PATH = AGGREGATED_DIR / "chunyun_period_change_fields.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the exact legacy Chunyun period contrasts "
            "(comparison minus baseline)."
        )
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_changes(frame: pd.DataFrame) -> pd.DataFrame:
    required = []
    for phase in ["pre", "festival", "post"]:
        required.extend(
            [
                f"{phase}_mean_population",
                f"{phase}_mean_exposure",
                f"{phase}_weighted_pm25",
            ]
        )
    require_columns(frame, required, INPUT_PATH)
    result = frame.copy()
    contrasts = {
        "festival_pre": ("festival", "pre"),
        "post_festival": ("post", "festival"),
    }
    measures = {
        "population": "mean_population",
        "exposure": "mean_exposure",
        "pm25": "mean_pm25",
        "weighted_pm25": "weighted_pm25",
    }
    for prefix, (comparison, baseline) in contrasts.items():
        for name, suffix in measures.items():
            result[f"{prefix}_{name}_change"] = (
                result[f"{comparison}_{suffix}"]
                - result[f"{baseline}_{suffix}"]
            )

    # Exact old names retained for downstream compatibility.
    result["festival_pre_count_change"] = result[
        "festival_pre_population_change"
    ]
    result["post_festival_count_change"] = result[
        "post_festival_population_change"
    ]
    for phase in ["pre", "festival", "post"]:
        result[f"{phase}_mean_count"] = result[
            f"{phase}_mean_population"
        ]
    return result


def main() -> None:
    args = parse_args()
    with stage_log("05_03_calculate_period_changes"):
        require_file(INPUT_PATH)
        if not ensure_outputs(
            [OUTPUT_PATH, FIELDS_PATH], args.overwrite
        ):
            return
        print(f"READ {INPUT_PATH}")
        result = add_changes(pd.read_parquet(INPUT_PATH))
        change_columns = [c for c in result if c.endswith("_change")]
        if np.isinf(result[change_columns].to_numpy(dtype="float64")).any():
            raise ValueError("Infinite values found in period changes")
        atomic_parquet(result, OUTPUT_PATH)
        descriptions = []
        for column in change_columns:
            alias = column.endswith("_count_change")
            descriptions.append(
                {
                    "field": column,
                    "definition": (
                        "compatibility alias for calibrated population "
                        "change"
                        if alias
                        else "comparison period minus baseline period"
                    ),
                    "formal_field": not alias,
                    "note": compatibility_alias_note() if alias else "",
                }
            )
        atomic_csv(pd.DataFrame(descriptions), FIELDS_PATH)
        print(f"WRITE {OUTPUT_PATH} rows={len(result):,}")
        print(f"WRITE {FIELDS_PATH}")


if __name__ == "__main__":
    main()
