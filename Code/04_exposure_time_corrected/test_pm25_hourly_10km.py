from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "01_match_pm25_and_aggregate_hourly_10km.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pm25_hourly_10km", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NearestIndexTests(unittest.TestCase):
    def test_ascending_descending_and_tie_rule(self) -> None:
        query = np.array([0.2, 1.5, 2.9])
        ascending = np.array([0.0, 1.0, 2.0, 3.0])
        descending = ascending[::-1]
        ascending_indices = MODULE.PM25Matcher.nearest_indices(
            query, ascending
        )
        descending_indices = MODULE.PM25Matcher.nearest_indices(
            query, descending
        )
        self.assertTrue(
            np.array_equal(ascending_indices, np.array([0, 1, 3]))
        )
        self.assertTrue(
            np.allclose(
                descending[descending_indices],
                ascending[ascending_indices],
            )
        )


class HourlyAggregationTests(unittest.TestCase):
    def test_complete_hour_is_arithmetic_snapshot_mean(self) -> None:
        times = pd.date_range(
            "2018-02-02 00:00:00", periods=12, freq="5min"
        )
        weights, quality = MODULE.build_hourly_time_weights(times)
        snapshots = pd.DataFrame(
            {
                "local_time": times,
                "grid_x": 0,
                "grid_y": 0,
                "snapshot_population": np.arange(12, dtype=float),
                "snapshot_app_count": np.arange(12, dtype=float),
                "snapshot_exposure": np.arange(12, dtype=float) * 2,
                "snapshot_app_exposure": (
                    np.arange(12, dtype=float) * 2
                ),
            }
        )
        hourly = MODULE.aggregate_snapshot_grids_to_hourly(
            snapshots, weights, quality
        )
        self.assertAlmostEqual(hourly["hourly_population"].iat[0], 5.5)
        self.assertAlmostEqual(hourly["hourly_exposure"].iat[0], 11.0)
        self.assertEqual(hourly["sample_count"].iat[0], 12)
        self.assertTrue(bool(hourly["is_complete_hour"].iat[0]))
        self.assertEqual(
            hourly["aggregation_method"].iat[0],
            "arithmetic_mean_regular_complete",
        )

    def test_incomplete_hour_caps_long_gaps(self) -> None:
        times = pd.to_datetime(
            [
                "2018-02-02 00:00:00",
                "2018-02-02 00:30:00",
                "2018-02-02 00:55:00",
            ]
        )
        weights, quality = MODULE.build_hourly_time_weights(times)
        self.assertTrue(
            np.allclose(
                weights["weight_seconds"].to_numpy(),
                [600.0, 600.0, 300.0],
            )
        )
        self.assertAlmostEqual(quality["covered_minutes"].iat[0], 25.0)
        self.assertFalse(bool(quality["is_complete_hour"].iat[0]))
        self.assertEqual(
            quality["aggregation_method"].iat[0],
            "time_weighted_forward_duration_capped",
        )

    def test_spooling_aggregator_accepts_unsorted_chunks(self) -> None:
        first_time = pd.Timestamp("2018-02-15 00:00:00")
        second_time = pd.Timestamp("2018-02-15 00:05:00")

        def snapshot(
            timestamp: pd.Timestamp, population: float
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "local_time": [timestamp],
                    "grid_x": [0],
                    "grid_y": [0],
                    "snapshot_population": [population],
                    "snapshot_app_count": [population],
                    "snapshot_exposure": [population * 10],
                    "snapshot_app_exposure": [population * 10],
                }
            )

        with tempfile.TemporaryDirectory() as temporary_root:
            parts = Path(temporary_root) / "parts"
            aggregator = MODULE.SpoolingHourlyAggregator(
                parts,
                expected_sample_count=12,
                expected_interval_minutes=5.0,
                max_snapshot_duration_minutes=10.0,
            )
            # Deliberately submit 00:05 before 00:00. The earlier timestamp is
            # also split across chunks and must be summed before weighting.
            aggregator.consume_batch(
                [(second_time, "2018-02-14")],
                snapshot(second_time, 3.0),
            )
            aggregator.consume_batch(
                [(first_time, "2018-02-14")],
                snapshot(first_time, 1.0),
            )
            aggregator.consume_batch(
                [(first_time, "2018-02-14")],
                snapshot(first_time, 2.0),
            )
            hourly, quality = aggregator.finish()
            self.assertEqual(len(hourly), 1)
            self.assertAlmostEqual(
                hourly["hourly_population"].iat[0], 3.0
            )
            self.assertEqual(quality["sample_count"].iat[0], 2)
            self.assertEqual(
                quality["pm25_utc_dates_used"].iat[0], "2018-02-14"
            )
            aggregator.cleanup()
            self.assertFalse(parts.exists())


if __name__ == "__main__":
    unittest.main()
