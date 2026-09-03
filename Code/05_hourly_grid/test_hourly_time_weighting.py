from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "01_aggregate_population_hourly_10km.py"
)
SPEC = importlib.util.spec_from_file_location("hourly_10km", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class HourlyTimeWeightingTests(unittest.TestCase):
    def test_complete_regular_hour_equals_arithmetic_mean(self) -> None:
        times = pd.date_range(
            "2018-02-02 00:00:00", periods=12, freq="5min"
        )
        weights, quality = MODULE.build_hourly_time_weights(times)
        snapshot = pd.DataFrame(
            {
                "local_time": times,
                "grid_x": 0,
                "grid_y": 0,
                "estimated_population": np.arange(12, dtype="float64"),
            }
        )
        hourly = MODULE.aggregate_snapshot_grids_to_hourly(
            snapshot,
            weights,
            quality,
            [
                MODULE.ValueSpec(
                    "estimated_population", "hourly_population"
                )
            ],
        )
        self.assertAlmostEqual(hourly["hourly_population"].iat[0], 5.5)
        self.assertEqual(hourly["sample_count"].iat[0], 12)
        self.assertTrue(bool(hourly["is_complete_hour"].iat[0]))
        self.assertEqual(
            hourly["aggregation_method"].iat[0],
            "arithmetic_mean_regular_complete",
        )
        self.assertTrue(
            np.allclose(weights["weight_seconds"].to_numpy(), 300.0)
        )

    def test_missing_first_snapshot_is_not_zero_filled(self) -> None:
        times = pd.date_range(
            "2018-02-02 00:05:00", periods=11, freq="5min"
        )
        weights, quality = MODULE.build_hourly_time_weights(times)
        snapshot = pd.DataFrame(
            {
                "local_time": times,
                "grid_x": 0,
                "grid_y": 0,
                "estimated_population": 100.0,
            }
        )
        hourly = MODULE.aggregate_snapshot_grids_to_hourly(
            snapshot,
            weights,
            quality,
            [
                MODULE.ValueSpec(
                    "estimated_population", "hourly_population"
                )
            ],
        )
        self.assertAlmostEqual(hourly["hourly_population"].iat[0], 100.0)
        self.assertEqual(hourly["sample_count"].iat[0], 11)
        self.assertAlmostEqual(hourly["coverage_ratio"].iat[0], 11 / 12)
        self.assertAlmostEqual(
            hourly["weighted_duration_minutes"].iat[0], 55.0
        )
        self.assertFalse(bool(hourly["is_complete_hour"].iat[0]))

    def test_long_gap_is_capped_and_left_uncovered(self) -> None:
        times = pd.to_datetime(
            [
                "2018-02-02 00:00:00",
                "2018-02-02 00:30:00",
                "2018-02-02 00:55:00",
            ]
        )
        weights, quality = MODULE.build_hourly_time_weights(
            times, max_snapshot_duration_minutes=10.0
        )
        self.assertTrue(
            np.allclose(weights["weight_seconds"], [600.0, 600.0, 300.0])
        )
        self.assertAlmostEqual(
            quality["weighted_duration_minutes"].iat[0], 25.0
        )
        self.assertAlmostEqual(
            quality["duration_coverage_ratio"].iat[0], 25 / 60
        )

    def test_absent_grid_at_available_snapshot_uses_global_denominator(
        self,
    ) -> None:
        times = pd.date_range(
            "2018-02-02 00:00:00", periods=12, freq="5min"
        )
        weights, quality = MODULE.build_hourly_time_weights(times)
        snapshot = pd.concat(
            [
                pd.DataFrame(
                    {
                        "local_time": times,
                        "grid_x": 0,
                        "grid_y": 0,
                        "estimated_population": 1.0,
                    }
                ),
                pd.DataFrame(
                    {
                        "local_time": times[:6],
                        "grid_x": 10_000,
                        "grid_y": 0,
                        "estimated_population": 1.0,
                    }
                ),
            ],
            ignore_index=True,
        )
        hourly = MODULE.aggregate_snapshot_grids_to_hourly(
            snapshot,
            weights,
            quality,
            [
                MODULE.ValueSpec(
                    "estimated_population", "hourly_population"
                )
            ],
        ).sort_values("grid_x")
        self.assertTrue(
            np.allclose(hourly["hourly_population"], [1.0, 0.5])
        )


if __name__ == "__main__":
    unittest.main()
