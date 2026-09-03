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
    AGGREGATED_DIR,
    FIGURES_DIR,
    atomic_csv,
    require_file,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    NATURE_DOUBLE_COLUMN_IN,
    apply_nature_style,
)
from downstream_spatial import assign_grid_province  # noqa: E402


INPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"
OUTPUT_DIR = FIGURES_DIR / "Summary"
PROVINCE_PATH = OUTPUT_DIR / "province_period_summary.csv"
MAPPING_PATH = Path(__file__).resolve().parents[2] / "Config" / "province_name_mapping.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create province and national Chunyun summary figures."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--population-only",
        action="store_true",
        help=(
            "Regenerate only the province population chart from the existing "
            "province summary without rebuilding analytical tables."
        ),
    )
    return parser.parse_args()


def build_province(frame: pd.DataFrame) -> pd.DataFrame:
    assigned = assign_grid_province(frame)
    assigned = assigned.loc[assigned["province"].notna()].copy()
    sums = [
        f"{phase}_mean_{measure}"
        for phase in ["pre", "festival", "post"]
        for measure in ["population", "exposure"]
    ]
    weighted_rows = []
    for province, group in assigned.groupby("province"):
        row = {"province": province, "grid_count": len(group)}
        for column in sums:
            row[column] = group[column].sum(min_count=1)
        for phase in ["pre", "festival", "post"]:
            row[f"{phase}_weighted_pm25"] = np.divide(
                row[f"{phase}_mean_exposure"],
                row[f"{phase}_mean_population"],
            )
        weighted_rows.append(row)
    return pd.DataFrame(weighted_rows).sort_values("province")


def main() -> None:
    args = parse_args()
    with stage_log("09_04_plot_summary_figures"):
        frame = None
        if args.population_only:
            require_file(PROVINCE_PATH)
            province = pd.read_csv(PROVINCE_PATH)
            print(f"REUSE existing province summary: {PROVINCE_PATH}")
        else:
            require_file(INPUT_PATH)
            frame = pd.read_parquet(INPUT_PATH)
            if PROVINCE_PATH.exists() and not args.overwrite:
                province = pd.read_csv(PROVINCE_PATH)
                print(f"SKIP existing and reuse: {PROVINCE_PATH}")
            else:
                province = build_province(frame)
                atomic_csv(province, PROVINCE_PATH)
                print(f"WRITE {PROVINCE_PATH}")

        population_path = OUTPUT_DIR / "province_population_by_period.png"
        if args.overwrite or not population_path.exists():
            mapping = pd.read_csv(MAPPING_PATH, dtype="string")[[
                "province",
                "official_name",
            ]]
            ordered = province.merge(
                mapping,
                on="province",
                how="left",
                validate="one_to_one",
            ).sort_values(
                "pre_mean_population", ascending=True
            )
            if ordered["official_name"].isna().any():
                missing = ordered.loc[
                    ordered["official_name"].isna(), "province"
                ].tolist()
                raise ValueError(f"Missing English province labels: {missing}")
            y = np.arange(len(ordered))
            apply_nature_style(300)
            fig, axis = plt.subplots(
                figsize=(7.09, 6.90), dpi=300
            )
            width = 0.235
            for offset, phase, color in [
                (-width, "pre", "#4C78A8"),
                (0, "festival", "#F58518"),
                (width, "post", "#54A24B"),
            ]:
                axis.barh(
                    y + offset,
                    ordered[f"{phase}_mean_population"] / 1_000_000_000,
                    height=width,
                    label={
                        "pre": "Pre-festival",
                        "festival": "Festival",
                        "post": "Post-festival",
                    }[phase],
                    color=color,
                )
            axis.set_yticks(y)
            axis.set_yticklabels(ordered["official_name"], fontsize=6.8)
            axis.set_xlabel("Calibrated population (billions)")
            axis.tick_params(axis="x", labelsize=7.2)
            axis.grid(axis="x", color="#D9DEE3", linewidth=0.45, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.legend(
                frameon=False,
                ncol=3,
                loc="lower right",
                fontsize=7.2,
                handlelength=1.4,
                columnspacing=1.2,
            )
            fig.subplots_adjust(
                left=0.175,
                right=0.985,
                top=0.985,
                bottom=0.075,
            )
            population_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(population_path, dpi=300, facecolor="white")
            fig.savefig(
                population_path.with_suffix(".pdf"),
                facecolor="white",
            )
            plt.close(fig)
            print(f"WRITE {population_path} and PDF")

        scatter_path = OUTPUT_DIR / "population_vs_exposure_change.png"
        if not args.population_only and (
            args.overwrite or not scatter_path.exists()
        ):
            assert frame is not None
            apply_nature_style(300)
            fig, axes = plt.subplots(
                1, 2, figsize=(NATURE_DOUBLE_COLUMN_IN, 3.1)
            )
            for axis, comparison in zip(
                axes, ["festival_pre", "post_festival"]
            ):
                axis.scatter(
                    frame[f"{comparison}_population_change"],
                    frame[f"{comparison}_exposure_change"],
                    s=3,
                    alpha=0.28,
                    linewidths=0,
                    color="#4C78A8",
                    rasterized=True,
                )
                axis.axhline(0, color="#777777", linewidth=0.4)
                axis.axvline(0, color="#777777", linewidth=0.4)
                axis.set_xlabel("Estimated population change")
                axis.set_ylabel("Calibrated exposure change")
                axis.set_title(comparison.replace("_", " ").title())
            fig.tight_layout()
            scatter_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
            fig.savefig(
                scatter_path.with_suffix(".pdf"), bbox_inches="tight"
            )
            plt.close(fig)
            print(f"WRITE {scatter_path} and PDF")


if __name__ == "__main__":
    main()
