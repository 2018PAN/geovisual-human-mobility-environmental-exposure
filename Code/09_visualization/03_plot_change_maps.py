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


INPUT_PATH = AGGREGATED_DIR / "chunyun_period_grid_changes.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot legacy-defined formal period-change maps."
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    dpi = int(config["plotting"]["dpi_maps"])
    output_dir = FIGURES_DIR / "Changes"
    labels = {
        "population": "Estimated population change",
        "exposure": "Calibrated population exposure change",
        "pm25": "PM2.5 change (µg/m³)",
    }
    with stage_log("09_03_plot_change_maps"):
        require_file(INPUT_PATH)
        frame = pd.read_parquet(INPUT_PATH)
        comparisons = ["festival_pre", "post_festival"]
        norms = {
            measure: pooled_normalization(
                [
                    frame[f"{comparison}_{measure}_change"]
                    for comparison in comparisons
                ],
                diverging=True,
            )
            for measure in labels
        }
        comparison_titles = {
            "festival_pre": "Festival minus pre-festival",
            "post_festival": "Post-festival minus festival",
        }
        for comparison in comparisons:
            for measure, label in labels.items():
                column = f"{comparison}_{measure}_change"
                output = output_dir / f"{column}.png"
                if output.exists() and not args.overwrite:
                    print(f"SKIP existing: {output}")
                    continue
                plot_grid_map(
                    frame,
                    column,
                    output,
                    title=(
                        f"{comparison_titles[comparison]}: "
                        f"{measure} change"
                    ),
                    colorbar_label=label,
                    diverging=True,
                    norm=norms[measure],
                    dpi=dpi,
                )
                print(f"WRITE {output} and PDF")


if __name__ == "__main__":
    main()
