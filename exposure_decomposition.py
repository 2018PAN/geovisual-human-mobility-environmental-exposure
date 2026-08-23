"""Three-term accounting decomposition of population-weighted exposure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from io_utils import read_table, require_columns, write_table


CONTRASTS = {
    "festival_pre": ("pre", "festival"),
    "post_festival": ("festival", "post"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast", choices=CONTRASTS, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline, comparison = CONTRASTS[args.contrast]
    required = [
        f"{baseline}_mean_population",
        f"{baseline}_mean_pm25",
        f"{comparison}_mean_population",
        f"{comparison}_mean_pm25",
    ]
    frame = read_table(args.input).replace([np.inf, -np.inf], np.nan)
    require_columns(frame, required)

    p0 = frame[f"{baseline}_mean_population"]
    c0 = frame[f"{baseline}_mean_pm25"]
    p1 = frame[f"{comparison}_mean_population"]
    c1 = frame[f"{comparison}_mean_pm25"]
    delta_p = p1 - p0
    delta_c = c1 - c0
    prefix = args.contrast

    frame[f"{prefix}_mobility_component"] = c0 * delta_p
    frame[f"{prefix}_pollution_component"] = p0 * delta_c
    frame[f"{prefix}_interaction_component"] = delta_p * delta_c
    component_columns = [
        f"{prefix}_mobility_component",
        f"{prefix}_pollution_component",
        f"{prefix}_interaction_component",
    ]
    frame[f"{prefix}_decomposed_change"] = frame[component_columns].sum(
        axis=1, min_count=3
    )
    frame[f"{prefix}_observed_change"] = p1 * c1 - p0 * c0
    error = (
        frame[f"{prefix}_decomposed_change"]
        - frame[f"{prefix}_observed_change"]
    )
    valid_error = error.dropna().abs()
    scale = frame[f"{prefix}_observed_change"].dropna().abs().max()
    tolerance = max(1e-8, float(scale or 0.0) * 1e-10)
    if not valid_error.empty and valid_error.max() > tolerance:
        raise ValueError("The decomposition identity check failed")

    write_table(frame, args.output)


if __name__ == "__main__":
    main()
