from __future__ import annotations

import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from pyproj import Transformer

from project_common import CHINA_LCC, NEW_ROOT


WORKSPACE_ROOT = NEW_ROOT.parent
CONFIG_DIR = NEW_ROOT / "Config"
PERIOD_CONFIG_PATH = CONFIG_DIR / "chunyun_periods.yaml"
ANALYSIS_CONFIG_PATH = CONFIG_DIR / "downstream_analysis.json"

HOURLY_EXPOSURE_DIR = NEW_ROOT / "Output" / "Exposure" / "hourly_grid"
AGGREGATED_DIR = NEW_ROOT / "Output" / "Aggregated"
HOURLY_ANALYSIS_DIR = AGGREGATED_DIR / "hourly_analysis"
DAILY_ANALYSIS_DIR = AGGREGATED_DIR / "daily_grid"
SPATIAL_ANALYSIS_DIR = NEW_ROOT / "Output" / "SpatialAnalysis"
DECOMPOSITION_DIR = NEW_ROOT / "Output" / "Decomposition"
MODELING_DIR = NEW_ROOT / "Output" / "Modeling" / "XGBoost"
FIGURES_DIR = NEW_ROOT / "Output" / "Figures"
LOG_DIR = NEW_ROOT / "Logs" / "downstream"

LOCAL_COMPLETENESS_PATH = (
    NEW_ROOT
    / "Output"
    / "Population"
    / "diagnostics"
    / "calibrated_local_date_completeness.csv"
)

POPULATION_COLUMN = "hourly_population"
EXPOSURE_COLUMN = "hourly_exposure"
PW_PM25_COLUMN = "population_weighted_pm25"

GRID_COLUMNS = [
    "grid_id",
    "grid_x",
    "grid_y",
    "grid_center_x",
    "grid_center_y",
    "grid_center_lon",
    "grid_center_lat",
]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_period_config() -> dict:
    # The .yaml file deliberately uses the JSON subset of YAML 1.2 so the
    # workflow has no PyYAML dependency.
    return load_json(PERIOD_CONFIG_PATH)


def load_analysis_config() -> dict:
    config = load_json(ANALYSIS_CONFIG_PATH)
    configured = config["formal_fields"]
    expected = {
        "population": POPULATION_COLUMN,
        "exposure": EXPOSURE_COLUMN,
        "population_weighted_pm25": PW_PM25_COLUMN,
    }
    if configured != expected:
        raise ValueError(
            f"Formal field configuration changed unexpectedly: "
            f"{configured}; expected {expected}"
        )
    return config


def periods() -> dict[str, dict]:
    return load_period_config()["periods"]


def all_chunyun_dates() -> list[str]:
    configured = periods()
    start = min(item["start"] for item in configured.values())
    end = max(item["end"] for item in configured.values())
    return [
        item.strftime("%Y-%m-%d")
        for item in pd.date_range(start, end, freq="D")
    ]


def period_dates(period_name: str) -> list[str]:
    item = periods()[period_name]
    return [
        day.strftime("%Y-%m-%d")
        for day in pd.date_range(item["start"], item["end"], freq="D")
    ]


def assign_period(local_date: str) -> str:
    for name, item in periods().items():
        if item["start"] <= local_date <= item["end"]:
            return name
    raise ValueError(
        f"Date {local_date} is outside configured Chunyun periods"
    )


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], path: Path | str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}; "
            f"available={list(frame.columns)}"
        )


def ensure_outputs(paths: Iterable[Path], overwrite: bool) -> bool:
    paths = list(paths)
    existing = [path for path in paths if path.exists()]
    if not existing:
        return True
    if overwrite:
        print("Overwrite enabled for existing outputs:")
        for path in existing:
            print(f"  {path}")
        return True
    if len(existing) == len(paths):
        print("All requested outputs already exist; skipped:")
        for path in existing:
            print(f"  {path}")
        return False
    raise FileExistsError(
        "A mixed old/new output set exists. Review these paths and use "
        "--overwrite only if replacement is intended:\n"
        + "\n".join(f"  {path}" for path in existing)
    )


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def atomic_parquet(
    frame: pd.DataFrame, path: Path, *, compression: str = "zstd"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(
        temporary, index=False, compression=compression
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    if temporary.exists():
        temporary.unlink()
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    if temporary.exists():
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def stage_log(stage_name: str) -> Iterator[Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{stage_name}_{timestamp}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with path.open("w", encoding="utf-8") as stream:
        sys.stdout = _Tee(original_stdout, stream)
        sys.stderr = _Tee(original_stderr, stream)
        try:
            print(f"Stage log: {path}")
            yield path
        except Exception:
            import traceback

            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def add_grid_center_lonlat(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if {
        "grid_center_lon",
        "grid_center_lat",
    }.issubset(result.columns):
        return result
    require_columns(
        result,
        ["grid_center_x", "grid_center_y"],
        "<grid frame>",
    )
    inverse = Transformer.from_crs(
        CHINA_LCC, "EPSG:4326", always_xy=True
    )
    longitude, latitude = inverse.transform(
        result["grid_center_x"].to_numpy(dtype="float64"),
        result["grid_center_y"].to_numpy(dtype="float64"),
    )
    result["grid_center_lon"] = np.asarray(longitude)
    result["grid_center_lat"] = np.asarray(latitude)
    return result


def add_legacy_01_degree_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = add_grid_center_lonlat(frame)
    result["legacy_grid_lon"] = (
        np.floor(
            result["grid_center_lon"].to_numpy(dtype="float64") * 10
            + 1e-9
        )
        / 10
    ).round(4)
    result["legacy_grid_lat"] = (
        np.floor(
            result["grid_center_lat"].to_numpy(dtype="float64") * 10
            + 1e-9
        )
        / 10
    ).round(4)
    return result


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    require_file(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, encoding="utf-8-sig")
        return frame if columns is None else frame[columns]
    raise ValueError(f"Unsupported table type: {path}")


def read_completeness() -> pd.DataFrame:
    require_file(LOCAL_COMPLETENESS_PATH)
    frame = pd.read_csv(LOCAL_COMPLETENESS_PATH)
    require_columns(
        frame,
        [
            "local_date",
            "coverage_ratio",
            "is_complete_local_date",
        ],
        LOCAL_COMPLETENESS_PATH,
    )
    frame["local_date"] = frame["local_date"].astype(str)
    return frame


def select_comparisons(value: str) -> list[str]:
    names = list(load_analysis_config()["modeling"]["comparisons"])
    return names if value == "all" else [value]


def analysis_target_choices() -> list[str]:
    return ["total_exposure", "mobility", "pollution"]


def modeling_spec(
    config: dict, comparison: str, analysis_target: str
) -> dict:
    if analysis_target == "total_exposure":
        result = dict(config["modeling"]["comparisons"][comparison])
        result.setdefault("accounting_features", [])
        result.setdefault(
            "prefix", f"{comparison}_total_exposure"
        )
        return result
    try:
        return dict(
            config["component_modeling"][analysis_target][comparison]
        )
    except KeyError as exc:
        raise ValueError(
            f"No modeling specification for {comparison}/"
            f"{analysis_target}"
        ) from exc


def numeric_finite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if np.isinf(values).any():
            raise ValueError(f"Infinite values found in {column}")


def compatibility_alias_note() -> str:
    return (
        "Compatibility fields containing 'count' are aliases for formal "
        "calibrated estimated population, never raw App count."
    )
