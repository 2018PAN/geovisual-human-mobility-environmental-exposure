from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    EXPOSURE_COLUMN,
    FIGURES_DIR,
    HOURLY_ANALYSIS_DIR,
    POPULATION_COLUMN,
    all_chunyun_dates,
    atomic_csv,
    require_file,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    NATURE_DOUBLE_COLUMN_IN,
    apply_nature_style,
)


SUMMARY_PATH = FIGURES_DIR / "Timeseries" / "national_hourly_timeseries.csv"
OUTPUT_PATH = FIGURES_DIR / "Timeseries" / "national_hourly_timeseries.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot national Beijing-hour estimated population and "
            "population-weighted PM2.5 time series."
        )
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_summary() -> pd.DataFrame:
    rows = []
    for day in all_chunyun_dates():
        path = HOURLY_ANALYSIS_DIR / f"hourly_analysis_{day}.parquet"
        require_file(path)
        frame = pd.read_parquet(
            path,
            columns=[
                "local_datetime",
                "local_date",
                "period",
                POPULATION_COLUMN,
                EXPOSURE_COLUMN,
                "coverage_ratio",
                "is_complete_hour",
            ],
        )
        group = frame.groupby(
            ["local_datetime", "local_date", "period"], as_index=False
        ).agg(
            estimated_population=(POPULATION_COLUMN, "sum"),
            calibrated_population_exposure=(EXPOSURE_COLUMN, "sum"),
            mean_grid_coverage_ratio=("coverage_ratio", "mean"),
            all_grid_hours_complete=("is_complete_hour", "all"),
            grid_count=(POPULATION_COLUMN, "count"),
        )
        group["population_weighted_pm25"] = np.divide(
            group["calibrated_population_exposure"],
            group["estimated_population"],
            out=np.full(len(group), np.nan),
            where=group["estimated_population"].to_numpy(dtype="float64") > 0,
        )
        rows.append(group)
    return pd.concat(rows, ignore_index=True).sort_values(
        "local_datetime"
    )


def main() -> None:
    args = parse_args()
    with stage_log("09_01_plot_hourly_timeseries"):
        expected = [
            SUMMARY_PATH,
            OUTPUT_PATH,
            OUTPUT_PATH.with_suffix(".pdf"),
        ]
        existing = [path for path in expected if path.exists()]
        if len(existing) == len(expected) and not args.overwrite:
            print(f"SKIP existing: {SUMMARY_PATH}")
            print(f"SKIP existing: {OUTPUT_PATH}")
            return
        if existing and not args.overwrite:
            raise FileExistsError(
                f"Partial time-series output set exists: {existing}"
            )
        summary = build_summary()
        atomic_csv(summary, SUMMARY_PATH)
        summary["local_datetime"] = pd.to_datetime(
            summary["local_datetime"]
        )
        apply_nature_style(300)
        fig, population_axis = plt.subplots(
            figsize=(NATURE_DOUBLE_COLUMN_IN, 3.8)
        )
        pm_axis = population_axis.twinx()
        population_axis.plot(
            summary["local_datetime"],
            summary["estimated_population"],
            color="#4C78A8",
            linewidth=0.8,
            label="Estimated population",
        )
        pm_axis.plot(
            summary["local_datetime"],
            summary["population_weighted_pm25"],
            color="#E45756",
            linewidth=0.8,
            label="Population-weighted PM2.5",
        )
        population_axis.set_ylabel("Estimated population")
        pm_axis.set_ylabel("Population-weighted PM2.5 (µg/m³)")
        population_axis.set_xlabel("Beijing local time")
        population_axis.set_title(
            "National hourly estimated population and PM2.5"
        )
        handles1, labels1 = population_axis.get_legend_handles_labels()
        handles2, labels2 = pm_axis.get_legend_handles_labels()
        population_axis.legend(
            handles1 + handles2,
            labels1 + labels2,
            frameon=False,
            loc="upper right",
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
        fig.savefig(OUTPUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"WRITE {SUMMARY_PATH}")
        print(f"WRITE {OUTPUT_PATH} and PDF")


if __name__ == "__main__":
    main()
