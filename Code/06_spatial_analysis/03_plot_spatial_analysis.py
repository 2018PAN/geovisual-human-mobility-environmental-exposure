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
    FIGURES_DIR,
    SPATIAL_ANALYSIS_DIR,
    load_analysis_config,
    require_file,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    NATURE_DOUBLE_COLUMN_IN,
    apply_nature_style,
    plot_lisa_map,
)


GLOBAL_PATH = (
    SPATIAL_ANALYSIS_DIR / "GlobalMoran" / "global_moran_results.csv"
)
LISA_DIR = SPATIAL_ANALYSIS_DIR / "LISA"
OUTPUT_DIR = FIGURES_DIR / "SpatialAnalysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot legacy-style Global Moran and LISA results."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        help="Defaults to variables in downstream_analysis.json.",
    )
    parser.add_argument(
        "--global-results",
        type=Path,
        default=GLOBAL_PATH,
    )
    parser.add_argument(
        "--global-figure-name",
        default="global_moran_results.png",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Regenerate LISA maps without rewriting the Moran bar chart.",
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Regenerate the Moran bar chart without rewriting LISA maps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_global(
    source_path: Path, path: Path, overwrite: bool, dpi: int
) -> None:
    if path.exists() and not overwrite:
        print(f"SKIP existing: {path}")
        return
    require_file(source_path)
    frame = pd.read_csv(source_path)
    apply_nature_style(dpi)
    fig, axis = plt.subplots(figsize=(NATURE_DOUBLE_COLUMN_IN, 2.9))
    x = np.arange(len(frame))
    bars = axis.bar(
        x,
        frame["moran_i"],
        color="#4C78A8",
        width=0.62,
        edgecolor="black",
        linewidth=0.35,
    )
    axis.axhline(0, color="#555555", linewidth=0.5)
    axis.set_xticks(x)
    axis.set_xticklabels(
        frame["variable"].str.replace("_", " ").str.title(),
        rotation=30,
        ha="right",
    )
    axis.set_ylabel("Global Moran's I")
    title = (
        "Spatial autocorrelation of decomposition components"
        if "component" in path.stem
        else "Spatial autocorrelation of exposure change"
    )
    axis.set_title(title)
    for bar, pvalue in zip(bars, frame["p_value"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"p={pvalue:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    save_figure(fig, path, dpi)
    print(f"WRITE {path} and PDF")


def plot_lisa(
    variable: str,
    path: Path,
    order: list[str],
    overwrite: bool,
    dpi: int,
) -> None:
    if path.exists() and not overwrite:
        print(f"SKIP existing: {path}")
        return
    input_path = LISA_DIR / f"lisa_{variable}.parquet"
    require_file(input_path)
    frame = pd.read_parquet(input_path)
    plot_lisa_map(
        frame,
        "lisa_cluster",
        path,
        title=lisa_title(variable),
        order=order,
        dpi=dpi,
    )
    print(f"WRITE {path} and PDF")


def available_lisa_variables() -> list[str]:
    return sorted(
        path.stem.removeprefix("lisa_")
        for path in LISA_DIR.glob("lisa_*.parquet")
    )


def lisa_title(variable: str) -> str:
    if variable.startswith("festival_pre_"):
        comparison = "festival − pre-festival"
        measure = variable.removeprefix("festival_pre_")
    elif variable.startswith("post_festival_"):
        comparison = "post-festival − festival"
        measure = variable.removeprefix("post_festival_")
    else:
        comparison = ""
        measure = variable
    labels = {
        "exposure_change": "exposure change",
        "mobility_component": "mobility component",
        "pollution_component": "pollution component",
    }
    label = labels.get(measure, measure.replace("_", " "))
    return (
        f"LISA clusters of {label}: {comparison}"
        if comparison
        else f"LISA clusters of {label}"
    )


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    spatial = config["spatial_analysis"]
    variables = args.variables or available_lisa_variables()
    if not variables:
        variables = spatial["variables"]
    dpi = int(config["plotting"]["dpi_maps"])
    with stage_log("06_03_plot_spatial_analysis"):
        if (
            Path(args.global_figure_name).name
            != args.global_figure_name
        ):
            raise ValueError(
                "--global-figure-name must be a filename"
            )
        if not args.maps_only:
            plot_global(
                args.global_results,
                OUTPUT_DIR / args.global_figure_name,
                args.overwrite,
                dpi,
            )
        if not args.global_only:
            for variable in variables:
                plot_lisa(
                    variable,
                    OUTPUT_DIR / f"lisa_{variable}.png",
                    spatial["cluster_order"],
                    args.overwrite,
                    dpi,
                )


if __name__ == "__main__":
    main()
