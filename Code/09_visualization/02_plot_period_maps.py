from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    AGGREGATED_DIR,
    FIGURES_DIR,
    load_analysis_config,
    require_file,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    plot_grid_map,
    pooled_normalization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Pre/Festival/Post formal 10 km period maps."
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    dpi = int(config["plotting"]["dpi_maps"])
    output_dir = FIGURES_DIR / "Periods"
    measures = {
        "mean_population": (
            "Estimated population",
            "Estimated population",
            False,
        ),
        "mean_exposure": (
            "Calibrated population exposure",
            "Calibrated population exposure",
            False,
        ),
        "weighted_pm25": (
            "Population-weighted PM2.5",
            "PM2.5 (µg/m³)",
            False,
        ),
    }
    with stage_log("09_02_plot_period_maps"):
        frames = {}
        for phase in ["pre", "festival", "post"]:
            path = AGGREGATED_DIR / f"{phase}_period_grid.parquet"
            require_file(path)
            frames[phase] = pd.read_parquet(path)

        norms = {}
        for suffix in measures:
            norms[suffix] = pooled_normalization(
                [
                    frames[phase][f"{phase}_{suffix}"]
                    for phase in ["pre", "festival", "post"]
                ],
                diverging=False,
            )

        phase_titles = {
            "pre": "Pre-festival",
            "festival": "Festival holiday",
            "post": "Post-festival",
        }
        for phase in ["pre", "festival", "post"]:
            frame = frames[phase]
            for suffix, (title, label, diverging) in measures.items():
                column = f"{phase}_{suffix}"
                output = output_dir / f"{phase}_{suffix}.png"
                if output.exists() and not args.overwrite:
                    print(f"SKIP existing: {output}")
                    continue
                plot_grid_map(
                    frame,
                    column,
                    output,
                    title=f"{phase_titles[phase]}: {title}",
                    colorbar_label=label,
                    diverging=diverging,
                    norm=norms[suffix],
                    dpi=dpi,
                )
                print(f"WRITE {output} and PDF")


if __name__ == "__main__":
    main()
