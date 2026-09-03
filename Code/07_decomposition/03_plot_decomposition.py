from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COMMON_DIR = Path(__file__).resolve().parents[1] / "00_common"
sys.path.insert(0, str(COMMON_DIR))

from downstream_common import (  # noqa: E402
    DECOMPOSITION_DIR,
    FIGURES_DIR,
    load_analysis_config,
    require_file,
    stage_log,
)
from downstream_plotting import (  # noqa: E402
    NATURE_DOUBLE_COLUMN_IN,
    apply_nature_style,
    plot_grid_map,
    pooled_normalization,
)


INPUT_PATH = DECOMPOSITION_DIR / "grid_level_decomposition.parquet"
NATIONAL_PATH = DECOMPOSITION_DIR / "national_summary.csv"
OUTPUT_DIR = FIGURES_DIR / "Decomposition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot legacy-style Chunyun decomposition maps."
    )
    parser.add_argument(
        "--comparison",
        choices=["festival_pre", "post_festival", "all"],
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Regenerate spatial decomposition maps without the national bar chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_analysis_config()
    contrasts = config["decomposition"]["contrasts"]
    names = (
        list(contrasts)
        if args.comparison == "all"
        else [args.comparison]
    )
    dpi = int(config["plotting"]["dpi_maps"])
    with stage_log("07_03_plot_decomposition"):
        require_file(INPUT_PATH)
        frame = pd.read_parquet(INPUT_PATH)
        comparison_titles = {
            "festival_pre": "Festival minus pre-festival",
            "post_festival": "Post-festival minus festival",
        }
        components = [
            "mobility_component",
            "pollution_component",
            "interaction_component",
            "decomposed_total_change",
            "actual_total_change",
        ]
        scale_groups = {
            "mobility_component": ["mobility_component"],
            "pollution_component": ["pollution_component"],
            "interaction_component": ["interaction_component"],
            "total": [
                "decomposed_total_change",
                "actual_total_change",
            ],
        }
        norms = {}
        for scale_name, members in scale_groups.items():
            norms[scale_name] = pooled_normalization(
                [
                    frame[f"{name}_{component}"]
                    for name in names
                    for component in members
                ],
                diverging=True,
            )
        for name in names:
            for component in components:
                column = f"{name}_{component}"
                output = OUTPUT_DIR / f"{column}.png"
                if output.exists() and not args.overwrite:
                    print(f"SKIP existing: {output}")
                    continue
                plot_grid_map(
                    frame,
                    column,
                    output,
                    title=(
                        f"{comparison_titles[name]}: "
                        f"{component.replace('_', ' ')}"
                    ),
                    colorbar_label="Calibrated exposure change",
                    diverging=True,
                    norm=norms[
                        "total" if "total_change" in component else component
                    ],
                    dpi=dpi,
                )
                print(f"WRITE {output} and PDF")

        if args.maps_only:
            return
        require_file(NATIONAL_PATH)
        bar_path = OUTPUT_DIR / "national_component_contributions.png"
        if bar_path.exists() and not args.overwrite:
            print(f"SKIP existing: {bar_path}")
            return
        national = pd.read_csv(NATIONAL_PATH).iloc[0]
        rows = []
        for name in names:
            for component in [
                "mobility_component",
                "pollution_component",
                "interaction_component",
            ]:
                rows.append(
                    {
                        "comparison": name,
                        "component": component.replace("_component", ""),
                        "value": national[f"{name}_{component}"],
                    }
                )
        plot = pd.DataFrame(rows)
        pivot = plot.pivot(
            index="comparison", columns="component", values="value"
        )
        apply_nature_style(300)
        fig, axis = plt.subplots(
            figsize=(NATURE_DOUBLE_COLUMN_IN, 3.2)
        )
        pivot.plot(
            kind="bar",
            ax=axis,
            color=["#4C78A8", "#F58518", "#54A24B"],
            edgecolor="black",
            linewidth=0.25,
        )
        axis.axhline(0, color="#555555", linewidth=0.5)
        axis.set_xlabel("")
        axis.set_ylabel("National calibrated exposure change")
        axis.set_xticklabels(
            [item.replace("_", " ").title() for item in pivot.index],
            rotation=0,
        )
        axis.legend(title="Component", frameon=False)
        bar_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(bar_path, dpi=300, bbox_inches="tight")
        fig.savefig(bar_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"WRITE {bar_path} and PDF")


if __name__ == "__main__":
    main()
