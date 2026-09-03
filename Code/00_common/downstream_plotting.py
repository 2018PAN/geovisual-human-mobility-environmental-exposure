from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, box

from downstream_common import NEW_ROOT
from downstream_spatial import load_provinces
from project_common import CHINA_LCC, GRID_SIZE_M


CHINA_PRJ = NEW_ROOT / "Boundary" / "China.prj"
MAIN_XLIM = (73.0, 135.0)
MAIN_YLIM = (18.0, 54.0)
SCS_XLIM = (105.0, 125.0)
SCS_YLIM = (3.0, 26.0)

NATURE_DOUBLE_COLUMN_IN = 180.0 / 25.4

OUTSIDE_COLOR = "#FFFFFF"
VALID_ZERO_COLOR = "#F8F6F2"
TRUE_NODATA_COLOR = "#E3E3E3"
LAND_NODATA_COLOR = TRUE_NODATA_COLOR
GRID_NODATA_COLOR = TRUE_NODATA_COLOR
OUTSIDE_MODEL_SUPPORT_COLOR = "#F0F0F0"
NOT_SIGNIFICANT_COLOR = "#DEDEDE"
NATIONAL_BOUNDARY_COLOR = "#626262"
PROVINCE_BOUNDARY_COLOR = "#B9B9B9"
INSET_FRAME_COLOR = "#808080"

LISA_COLORS = {
    "High-High": "#C35A63",
    "Low-Low": "#477BAA",
    "High-Low": "#DC916B",
    "Low-High": "#82AAB9",
    "Not significant": NOT_SIGNIFICANT_COLOR,
    "NoData": "#F1F1F1",
    "No data": "#F1F1F1",
    "Outside model support": "#F1F1F1",
}
LISA_ORDER = [
    "High-High",
    "Low-Low",
    "High-Low",
    "Low-High",
    "Not significant",
    "NoData",
]

SOFT_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "thesis_soft_diverging",
    ["#315F8C", "#8FB8D3", VALID_ZERO_COLOR, "#E7A17E", "#A83F45"],
    N=256,
)


@dataclass(frozen=True)
class FigureLayoutSpec:
    """Fixed-canvas layout parameters for one publication map family."""

    name: str
    figsize: tuple[float, float]
    left: float
    right: float
    top: float
    bottom: float
    title_y: float
    legend_y: float
    inset_bounds: tuple[float, float, float, float]
    footer_height_ratio: float
    colorbar_bounds: tuple[float, float, float, float]
    support_anchor: tuple[float, float]
    wspace: float = 0.0
    hspace: float = 0.04
    title_fontsize: float = 8.0
    panel_letter_fontsize: float = 8.0
    panel_title_fontsize: float = 6.8
    inset_label_fontsize: float = 6.0


@dataclass(frozen=True)
class PublicationSafetySpec:
    top_clearance_in: float = 0.12
    bottom_clearance_in: float = 0.12
    side_clearance_in: float = 0.10
    title_map_clearance_in: float = 0.10
    legend_map_clearance_in: float = 0.08
    title_center_tolerance_in: float = 0.04
    minimum_font_size_pt: float = 5.0


SINGLE_CONTINUOUS_LAYOUT = FigureLayoutSpec(
    name="single_continuous",
    figsize=(NATURE_DOUBLE_COLUMN_IN, 4.75),
    left=0.035,
    right=0.965,
    top=0.885,
    bottom=0.045,
    title_y=0.955,
    legend_y=0.56,
    inset_bounds=(0.785, 0.070, 0.155, 0.225),
    footer_height_ratio=0.115,
    colorbar_bounds=(0.33, 0.43, 0.42, 0.20),
    support_anchor=(0.305, 0.54),
    hspace=0.035,
)

LISA_LAYOUT = FigureLayoutSpec(
    name="lisa_categorical",
    figsize=(NATURE_DOUBLE_COLUMN_IN, 4.80),
    left=0.035,
    right=0.965,
    top=0.885,
    bottom=0.045,
    title_y=0.955,
    legend_y=0.54,
    inset_bounds=(0.785, 0.070, 0.155, 0.225),
    footer_height_ratio=0.125,
    colorbar_bounds=(0.0, 0.0, 0.0, 0.0),
    support_anchor=(0.5, 0.54),
    hspace=0.03,
)

SHAP_SHARED_LAYOUT = FigureLayoutSpec(
    name="shap_shared_scale",
    figsize=(NATURE_DOUBLE_COLUMN_IN, 5.15),
    left=0.025,
    right=0.975,
    top=0.895,
    bottom=0.045,
    title_y=0.955,
    legend_y=0.56,
    inset_bounds=(0.805, 0.055, 0.135, 0.200),
    footer_height_ratio=0.13,
    colorbar_bounds=(0.33, 0.43, 0.42, 0.20),
    support_anchor=(0.305, 0.54),
    wspace=0.035,
    hspace=0.105,
)

SHAP_INDIVIDUAL_LAYOUT = FigureLayoutSpec(
    name="shap_individual_scale",
    figsize=(NATURE_DOUBLE_COLUMN_IN, 6.75),
    left=0.025,
    right=0.975,
    top=0.900,
    bottom=0.045,
    title_y=0.958,
    legend_y=0.53,
    inset_bounds=(0.805, 0.055, 0.135, 0.200),
    footer_height_ratio=0.10,
    colorbar_bounds=(0.0, 0.0, 0.0, 0.0),
    support_anchor=(0.5, 0.53),
    wspace=0.080,
    hspace=0.200,
)

PUBLICATION_SAFETY = PublicationSafetySpec()

_BASE_CACHE: dict[
    str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
] = {}
_GRID_GEOMETRY_CACHE: dict[
    str, tuple[np.ndarray, gpd.GeoSeries]
] = {}


@dataclass(frozen=True)
class MapDiagnostics:
    count: int
    finite_count: int
    minimum: float
    maximum: float
    median: float
    p95: float
    p98: float
    p99: float
    zero_percentage: float


def _select_font() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "Liberation Sans"):
        if candidate in available:
            return candidate
    warnings.warn(
        "Arial, Helvetica and Liberation Sans are unavailable; using "
        "Matplotlib's sans-serif fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "sans-serif"


def apply_nature_style(dpi: int = 600) -> None:
    """Apply one restrained journal-style typography and export system."""
    font = _select_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.7,
            "legend.title_fontsize": 6.8,
            "axes.linewidth": 0.45,
            "lines.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": dpi,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def target_crs() -> CRS:
    if not CHINA_PRJ.exists():
        raise FileNotFoundError(f"Map projection not found: {CHINA_PRJ}")
    return CRS.from_wkt(CHINA_PRJ.read_text(encoding="utf-8"))


def _densified_geographic_box(
    bounds: tuple[float, float, float, float],
    samples_per_edge: int = 161,
) -> Polygon:
    xmin, ymin, xmax, ymax = bounds
    bottom = [(x, ymin) for x in np.linspace(xmin, xmax, samples_per_edge)]
    right = [
        (xmax, y)
        for y in np.linspace(ymin, ymax, samples_per_edge)[1:]
    ]
    top = [
        (x, ymax)
        for x in np.linspace(xmax, xmin, samples_per_edge)[1:]
    ]
    left = [
        (xmin, y)
        for y in np.linspace(ymax, ymin, samples_per_edge)[1:]
    ]
    return Polygon([*bottom, *right, *top, *left])


def projected_selection(
    bounds: tuple[float, float, float, float],
    crs,
) -> gpd.GeoDataFrame:
    selection = gpd.GeoDataFrame(
        {"selection": [1]},
        geometry=[_densified_geographic_box(bounds)],
        crs="EPSG:4326",
    )
    return selection.to_crs(crs)


def projected_extent(
    lon_limits: tuple[float, float],
    lat_limits: tuple[float, float],
    transformer: Transformer,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Backward-compatible extent helper using a densified geographic frame."""
    polygon = _densified_geographic_box(
        (lon_limits[0], lat_limits[0], lon_limits[1], lat_limits[1])
    )
    x, y = transformer.transform(*polygon.exterior.xy)
    xpad = (np.nanmax(x) - np.nanmin(x)) * 0.012
    ypad = (np.nanmax(y) - np.nanmin(y)) * 0.012
    return (
        (float(np.nanmin(x) - xpad), float(np.nanmax(x) + xpad)),
        (float(np.nanmin(y) - ypad), float(np.nanmax(y) + ypad)),
    )


def _numeric(values: Iterable) -> pd.Series:
    return pd.to_numeric(pd.Series(values, copy=False), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def distribution_diagnostics(
    values: Iterable, label: str = "value"
) -> MapDiagnostics:
    numeric = _numeric(values)
    finite = numeric.dropna()
    if finite.empty:
        raise ValueError(f"No finite values available for {label}")
    p95, p98, p99 = finite.quantile([0.95, 0.98, 0.99])
    result = MapDiagnostics(
        count=len(numeric),
        finite_count=len(finite),
        minimum=float(finite.min()),
        maximum=float(finite.max()),
        median=float(finite.median()),
        p95=float(p95),
        p98=float(p98),
        p99=float(p99),
        zero_percentage=float(finite.eq(0).mean() * 100.0),
    )
    print(
        f"[map diagnostics] {label}: n={result.count:,}; "
        f"finite={result.finite_count:,}; min={result.minimum:.6g}; "
        f"max={result.maximum:.6g}; median={result.median:.6g}; "
        f"p95={result.p95:.6g}; p98={result.p98:.6g}; "
        f"p99={result.p99:.6g}; zero={result.zero_percentage:.3f}%"
    )
    return result


def normalization(
    values: Iterable,
    diverging: bool,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> Normalize:
    valid = _numeric(values).dropna()
    if valid.empty:
        raise ValueError("No finite values are available for normalization")
    lower, upper = np.nanpercentile(valid, percentiles)
    if diverging:
        limit = max(abs(float(lower)), abs(float(upper)))
        if not np.isfinite(limit) or limit <= 0:
            limit = max(
                abs(float(valid.min())),
                abs(float(valid.max())),
                1e-12,
            )
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = float(valid.min()), float(valid.max())
    if lower == upper:
        upper = lower + max(abs(lower) * 1e-9, 1e-12)
    return Normalize(vmin=float(lower), vmax=float(upper))


def pooled_normalization(
    value_groups: Sequence[Iterable],
    diverging: bool,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> Normalize:
    pooled = pd.concat(
        [_numeric(values) for values in value_groups],
        ignore_index=True,
    )
    return normalization(
        pooled,
        diverging=diverging,
        percentiles=percentiles,
    )


def _base_layers(crs) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    key = CRS.from_user_input(crs).to_wkt()
    if key not in _BASE_CACHE:
        provinces = load_provinces(key)
        national = provinces.dissolve()
        _BASE_CACHE[key] = (provinces, national)
    provinces, national = _BASE_CACHE[key]
    return provinces.copy(), national.copy()


def prepare_map_data(
    frame: pd.DataFrame,
    column: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, CRS]:
    """Build the established 10 km cells without changing mapped values."""
    required = {"grid_center_x", "grid_center_y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Formal map data require established China-LCC grid centres; "
            f"missing {sorted(missing)}"
        )
    crs = target_crs()
    provinces, national = _base_layers(crs)
    work = frame.reset_index(drop=True).copy()
    if column is not None:
        if column not in work:
            raise KeyError(f"Mapped column not found: {column}")
        work[column] = _numeric(work[column]).to_numpy()

    coordinates = work[
        ["grid_center_x", "grid_center_y"]
    ].to_numpy(dtype="float64", copy=True)
    digest = hashlib.blake2b(
        coordinates.tobytes(),
        digest_size=16,
    ).hexdigest()
    cache_key = (
        f"{len(coordinates)}:{digest}:"
        f"{CRS.from_user_input(crs).to_wkt()}"
    )
    if cache_key in _GRID_GEOMETRY_CACHE:
        keep_array, cached_geometry = _GRID_GEOMETRY_CACHE[cache_key]
        kept = work.loc[keep_array].reset_index(drop=True)
        cells = gpd.GeoDataFrame(
            kept,
            geometry=cached_geometry.copy(),
            crs=crs,
        )
        print("  reused cached 10 km plotting geometry")
    else:
        centers = gpd.GeoDataFrame(
            work,
            geometry=gpd.points_from_xy(
                work["grid_center_x"], work["grid_center_y"]
            ),
            crs=CHINA_LCC,
        ).to_crs(crs)
        boundary = national.geometry.iloc[0]
        keep_array = centers.geometry.apply(boundary.covers).to_numpy()
        kept = work.loc[keep_array].reset_index(drop=True)
        half = GRID_SIZE_M / 2.0
        cell_geometry = [
            box(x - half, y - half, x + half, y + half)
            for x, y in zip(
                kept["grid_center_x"].to_numpy(dtype="float64"),
                kept["grid_center_y"].to_numpy(dtype="float64"),
            )
        ]
        cells = gpd.GeoDataFrame(
            kept,
            geometry=cell_geometry,
            crs=CHINA_LCC,
        ).to_crs(crs)
        cached_geometry = cells.geometry.reset_index(drop=True).copy()
        _GRID_GEOMETRY_CACHE[cache_key] = (
            keep_array.copy(),
            cached_geometry,
        )
    print(
        f"  national-boundary grid-centre filter: "
        f"{len(cells):,}/{len(frame):,}"
    )
    return cells, provinces, national, crs


def prepare_supported_map_data(
    frame: pd.DataFrame,
    columns: Sequence[str],
    national_grid: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, CRS]:
    """Attach a complete result surface to the complete national grid.

    Rows in ``frame`` define model support. Missing mapped values within that
    support are scientific errors. National-grid rows absent from ``frame``
    are retained explicitly as outside model support.
    """
    required_grid = {"grid_id", "grid_center_x", "grid_center_y"}
    grid_missing = required_grid.difference(national_grid.columns)
    if grid_missing:
        raise ValueError(
            "Complete national grid is missing "
            f"{sorted(grid_missing)}"
        )
    result_missing = {"grid_id", *columns}.difference(frame.columns)
    if result_missing:
        raise ValueError(
            f"Model-support result is missing {sorted(result_missing)}"
        )
    if national_grid["grid_id"].duplicated().any():
        raise ValueError("Complete national grid contains duplicate grid_id")
    if frame["grid_id"].duplicated().any():
        raise ValueError("Model-support result contains duplicate grid_id")
    unknown_ids = pd.Index(frame["grid_id"]).difference(
        pd.Index(national_grid["grid_id"])
    )
    if len(unknown_ids):
        raise ValueError(
            f"{len(unknown_ids):,} model-support grid IDs are absent from "
            "the complete national grid"
        )

    values = frame[["grid_id", *columns]].copy()
    for column in columns:
        values[column] = _numeric(values[column]).to_numpy()
        missing_count = int(values[column].isna().sum())
        if missing_count:
            raise ValueError(
                f"True missing values inside model support for {column}: "
                f"{missing_count:,}"
            )
    base_columns = list(
        dict.fromkeys(
            [
                "grid_id",
                "grid_x",
                "grid_y",
                "grid_center_x",
                "grid_center_y",
                "grid_center_lon",
                "grid_center_lat",
            ]
        )
    )
    base_columns = [
        column for column in base_columns if column in national_grid.columns
    ]
    merged = national_grid[base_columns].merge(
        values,
        on="grid_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    merged["inside_model_support"] = merged["_merge"].eq("both")
    merged["spatial_status"] = np.where(
        merged["inside_model_support"],
        "Inside model support",
        "Outside model support",
    )
    merged = merged.drop(columns="_merge")
    support_rows = int(merged["inside_model_support"].sum())
    if support_rows != len(frame):
        raise ValueError(
            "Model-support merge changed row coverage: "
            f"{support_rows:,}/{len(frame):,}"
        )
    cells, provinces, national, crs = prepare_map_data(merged)
    print(
        f"  model support inside plotted national grid: "
        f"{int(cells['inside_model_support'].sum()):,}/"
        f"{len(cells):,}"
    )
    return cells, provinces, national, crs


def _draw_land(
    axis,
    national: gpd.GeoDataFrame,
    color: str = LAND_NODATA_COLOR,
) -> None:
    national.plot(
        ax=axis,
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        zorder=0,
    )


def _draw_flat_cells(
    axis,
    cells: gpd.GeoDataFrame,
    color: str,
    *,
    zorder: int = 1,
) -> None:
    if cells.empty:
        return
    start = len(axis.collections)
    cells.plot(
        ax=axis,
        facecolor=color,
        edgecolor="none",
        linewidth=0,
        antialiased=False,
        rasterized=True,
        zorder=zorder,
    )
    _finish_dense_collections(axis, start)


def _draw_boundaries(
    axis,
    provinces: gpd.GeoDataFrame,
    national: gpd.GeoDataFrame,
    *,
    inset: bool = False,
) -> None:
    provinces.boundary.plot(
        ax=axis,
        color=PROVINCE_BOUNDARY_COLOR,
        linewidth=0.15 if inset else 0.20,
        zorder=7,
    )
    national.boundary.plot(
        ax=axis,
        color=NATIONAL_BOUNDARY_COLOR,
        linewidth=0.40 if inset else 0.63,
        zorder=8,
    )


def _finish_dense_collections(axis, start: int) -> None:
    for collection in axis.collections[start:]:
        collection.set_rasterized(True)
        collection.set_linewidth(0)
        collection.set_edgecolor("none")
        try:
            collection.set_antialiased(False)
        except AttributeError:
            pass


def _draw_continuous_cells(
    axis,
    cells: gpd.GeoDataFrame,
    column: str,
    cmap: Colormap | str,
    norm: Normalize,
) -> ScalarMappable | None:
    numeric = _numeric(cells[column])
    valid = cells.loc[numeric.notna()].copy()
    if valid.empty:
        return None
    valid[column] = numeric.loc[valid.index]
    start = len(axis.collections)
    valid.plot(
        ax=axis,
        column=column,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        rasterized=True,
        zorder=2,
    )
    _finish_dense_collections(axis, start)
    return ScalarMappable(norm=norm, cmap=cmap)


def _draw_categorical_cells(
    axis,
    cells: gpd.GeoDataFrame,
    column: str,
    colors: Mapping[str, str],
    order: Sequence[str],
) -> None:
    for category in order:
        subset = cells.loc[cells[column].astype(str).eq(category)]
        if subset.empty:
            continue
        start = len(axis.collections)
        subset.plot(
            ax=axis,
            facecolor=colors[category],
            edgecolor="none",
            linewidth=0,
            antialiased=False,
            rasterized=True,
            zorder=2,
        )
        _finish_dense_collections(axis, start)


def _sentence_title(title: str) -> str:
    text = " ".join(str(title).replace("_", " ").split())
    protected = {
        "PM2.5": "__pm25__",
        "SHAP": "__shap__",
        "LISA": "__lisa__",
        "XGBoost": "__xgboost__",
    }
    for token, placeholder in protected.items():
        text = text.replace(token, placeholder)
    text = text.lower()
    for token, placeholder in protected.items():
        text = text.replace(placeholder, token)
    return text[:1].upper() + text[1:] if text else text


def _style_main_axis(axis, crs) -> None:
    selection = projected_selection(
        (MAIN_XLIM[0], MAIN_YLIM[0], MAIN_XLIM[1], MAIN_YLIM[1]),
        crs,
    )
    xmin, ymin, xmax, ymax = selection.total_bounds
    xpad = (xmax - xmin) * 0.012
    ypad = (ymax - ymin) * 0.012
    axis.set_xlim(xmin - xpad, xmax + xpad)
    axis.set_ylim(ymin - ypad, ymax + ypad)
    axis.set_aspect("equal", adjustable="box")
    axis.set_anchor("N")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _scale_exponent(norm: Normalize) -> int:
    values = [abs(float(norm.vmin)), abs(float(norm.vmax))]
    maximum = max(values)
    if maximum == 0 or not np.isfinite(maximum):
        return 0
    exponent = int(np.floor(np.log10(maximum)))
    return exponent if abs(exponent) >= 4 else 0


def _display_colorbar_label(
    label: str,
    norm: Normalize,
) -> tuple[str, float]:
    exponent = _scale_exponent(norm)
    scale = 10.0**exponent
    display_label = (
        f"{label} (×10$^{{{exponent}}}$)" if exponent else label
    )
    return display_label, scale


def _add_horizontal_colorbar(
    fig,
    footer_axis,
    mappable: ScalarMappable,
    label: str,
    extend: str,
    layout: FigureLayoutSpec,
):
    display_label, scale = _display_colorbar_label(
        label,
        mappable.norm,
    )
    formatter = FuncFormatter(lambda value, _: f"{value / scale:g}")
    colorbar_axis = footer_axis.inset_axes(layout.colorbar_bounds)
    colorbar = fig.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation="horizontal",
        extend=extend,
        format=formatter,
    )
    colorbar.locator = MaxNLocator(nbins=5)
    colorbar.update_ticks()
    colorbar.ax.tick_params(
        length=2.0,
        width=0.4,
        pad=1.4,
        labelsize=6.2,
    )
    colorbar.outline.set_linewidth(0.35)
    colorbar.set_label(display_label, fontsize=6.7, labelpad=2.0)
    colorbar.ax.xaxis.get_offset_text().set_visible(False)
    return colorbar, colorbar_axis


def _continuous_extend(values: Iterable, norm: Normalize) -> str:
    valid = _numeric(values).dropna()
    below = int((valid < norm.vmin).sum())
    above = int((valid > norm.vmax).sum())
    print(
        f"[display clipping] below={below:,}; above={above:,}; "
        "source values unchanged"
    )
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _add_support_key(
    footer_axis,
    layout: FigureLayoutSpec,
    *,
    label: str = "No data",
    color: str = GRID_NODATA_COLOR,
    centered: bool = False,
):
    return footer_axis.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor="none",
                label=label,
            )
        ],
        loc="center" if centered else "center right",
        bbox_to_anchor=layout.support_anchor,
        frameon=False,
        borderaxespad=0,
        borderpad=0,
        handlelength=0.90,
        handleheight=0.75,
        handletextpad=0.40,
        fontsize=6.7,
    )


def _inset_view(
    cells: gpd.GeoDataFrame,
    crs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    selection = projected_selection(
        (SCS_XLIM[0], SCS_YLIM[0], SCS_XLIM[1], SCS_YLIM[1]),
        crs,
    )
    xmin, ymin, xmax, ymax = selection.total_bounds
    display_window = box(xmin, ymin, xmax, ymax)
    inset_cells = cells.loc[
        cells.geometry.intersects(display_window)
    ].copy()
    return inset_cells, selection


def _style_inset(
    axis,
    selection: gpd.GeoDataFrame,
    *,
    label: str | None = "South China Sea",
    label_fontsize: float = 6.0,
):
    xmin, ymin, xmax, ymax = selection.total_bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor(OUTSIDE_COLOR)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(INSET_FRAME_COLOR)
        spine.set_linewidth(0.38)
    title_artist = None
    if label:
        title_artist = axis.set_title(
            label,
            loc="center",
            fontsize=label_fontsize,
            fontweight="normal",
            pad=2.0,
            color="#4D4D4D",
        )
    return title_artist


def _validate_inset(
    cells: gpd.GeoDataFrame,
    inset_cells: gpd.GeoDataFrame,
    selection: gpd.GeoDataFrame,
    column: str,
    norm=None,
) -> None:
    xmin, ymin, xmax, ymax = selection.total_bounds
    display_window = box(xmin, ymin, xmax, ymax)
    expected_cells = cells.loc[
        cells.geometry.intersects(display_window)
    ]
    if "grid_id" in cells.columns and "grid_id" in inset_cells.columns:
        expected_ids = pd.Index(expected_cells["grid_id"])
        actual_ids = pd.Index(inset_cells["grid_id"])
        missing_ids = expected_ids.difference(actual_ids)
        unexpected_ids = actual_ids.difference(expected_ids)
        if len(missing_ids):
            raise ValueError(
                "Inset omits cells visible in its displayed extent: "
                f"{len(missing_ids):,}"
            )
        if len(unexpected_ids):
            raise ValueError(
                "Inset contains cells outside its displayed extent: "
                f"{len(unexpected_ids):,}"
            )
        if len(expected_ids) != len(actual_ids):
            raise ValueError(
                "Inset displayed-extent grid count differs despite matching "
                "unique ID sets"
            )
    if inset_cells.empty:
        warnings.warn(
            f"South China Sea inset has no intersecting cells for {column}",
            RuntimeWarning,
            stacklevel=2,
        )
    if "grid_id" in cells.columns and "grid_id" in inset_cells.columns:
        main_indexed = cells.set_index("grid_id")
        inset_indexed = inset_cells.set_index("grid_id")
        main = main_indexed[column]
        inset = inset_indexed[column]
        common = inset.index.intersection(main.index)
        if len(common) != len(inset):
            raise ValueError("Inset contains grid IDs absent from the main map")
        left = main.loc[common]
        right = inset.loc[common]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                _numeric(left).to_numpy(),
                _numeric(right).to_numpy(),
                equal_nan=True,
            ):
                raise ValueError("Main/inset values differ for shared grid IDs")
        elif not left.astype("string").equals(right.astype("string")):
            raise ValueError("Main/inset categories differ for shared grid IDs")
        for status_column in [
            "inside_model_support",
            "spatial_status",
        ]:
            if (
                status_column in main_indexed
                and status_column in inset_indexed
                and not main_indexed.loc[common, status_column]
                .astype("string")
                .equals(
                    inset_indexed.loc[common, status_column].astype("string")
                )
            ):
                raise ValueError(
                    "Main/inset support status differs for shared grid IDs: "
                    f"{status_column}"
                )
    numeric_values = (
        _numeric(inset_cells[column])
        if column in inset_cells
        else pd.Series(dtype="float64")
    )
    is_numeric = (
        column in inset_cells
        and pd.api.types.is_numeric_dtype(cells[column])
    )
    valid = (
        int(numeric_values.notna().sum())
        if is_numeric
        else int(inset_cells[column].notna().sum())
        if column in inset_cells
        else 0
    )
    print("[South China Sea inset]")
    print(f"  data CRS: {cells.crs}")
    print(f"  main cells: {len(cells):,}")
    print(f"  expected displayed-window cells: {len(expected_cells):,}")
    print(f"  actual inset cells: {len(inset_cells):,}")
    print(
        f"  {'finite values' if is_numeric else 'classified values'}: "
        f"{valid:,}"
    )
    if is_numeric and valid:
        finite = numeric_values.dropna()
        print(
            f"  inset value range: "
            f"{float(finite.min()):.6g} to "
            f"{float(finite.max()):.6g}"
        )
    print(f"  shared normalization object: {norm is not None}")
    if "inside_model_support" in inset_cells:
        print(
            "  supported inset cells: "
            f"{int(inset_cells['inside_model_support'].sum()):,}"
        )
    return {
        "expected_grid_count": int(len(expected_cells)),
        "actual_grid_count": int(len(inset_cells)),
        "valid_value_or_category_count": valid,
        "grid_id_sets_match": True,
        "overlapping_values_or_categories_match": True,
        "support_status_matches": True,
    }


def _single_map_canvas(layout: FigureLayoutSpec):
    fig = plt.figure(figsize=layout.figsize, facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.0, layout.footer_height_ratio),
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
        hspace=layout.hspace,
    )
    map_axis = fig.add_subplot(grid[0, 0])
    footer_axis = fig.add_subplot(grid[1, 0])
    footer_axis.set_axis_off()
    return fig, map_axis, footer_axis


def _shared_multipanel_canvas(layout: FigureLayoutSpec):
    fig = plt.figure(figsize=layout.figsize, facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, layout.footer_height_ratio),
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(2)
        ]
    )
    footer_axis = fig.add_subplot(grid[2, :])
    footer_axis.set_axis_off()
    return fig, axes, footer_axis


def _individual_multipanel_canvas(layout: FigureLayoutSpec):
    fig = plt.figure(figsize=layout.figsize, facecolor="white")
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=(1.0, 1.0, 1.0, layout.footer_height_ratio),
        left=layout.left,
        right=layout.right,
        top=layout.top,
        bottom=layout.bottom,
        wspace=layout.wspace,
        hspace=layout.hspace,
    )
    title_axes = []
    map_axes = []
    colorbar_axes = []
    for row in range(3):
        title_row = []
        map_row = []
        colorbar_row = []
        for column in range(2):
            panel = outer[row, column].subgridspec(
                3,
                1,
                height_ratios=(0.14, 1.0, 0.10),
                hspace=0.015,
            )
            title_axis = fig.add_subplot(panel[0, 0])
            title_axis.set_axis_off()
            map_axis = fig.add_subplot(panel[1, 0])
            colorbar_slot = panel[2, 0].subgridspec(
                1,
                3,
                width_ratios=(0.18, 0.64, 0.18),
                wspace=0,
            )
            colorbar_axis = fig.add_subplot(colorbar_slot[0, 1])
            title_row.append(title_axis)
            map_row.append(map_axis)
            colorbar_row.append(colorbar_axis)
        title_axes.append(title_row)
        map_axes.append(map_row)
        colorbar_axes.append(colorbar_row)
    footer_axis = fig.add_subplot(outer[3, :])
    footer_axis.set_axis_off()
    return (
        fig,
        np.asarray(title_axes),
        np.asarray(map_axes),
        np.asarray(colorbar_axes),
        footer_axis,
    )


def _bbox_in_inches(fig, artist, renderer) -> Bbox:
    return artist.get_window_extent(renderer).transformed(
        fig.dpi_scale_trans.inverted()
    )


def _axes_bbox_in_inches(fig, axis, renderer, *, tight: bool) -> Bbox:
    bbox = axis.get_tightbbox(renderer) if tight else axis.get_window_extent(
        renderer
    )
    return bbox.transformed(fig.dpi_scale_trans.inverted())


def _bbox_dict(bbox: Bbox | None) -> dict | None:
    if bbox is None:
        return None
    return {
        "x0_in": float(bbox.x0),
        "y0_in": float(bbox.y0),
        "x1_in": float(bbox.x1),
        "y1_in": float(bbox.y1),
        "width_in": float(bbox.width),
        "height_in": float(bbox.height),
    }


def validate_publication_layout(
    fig,
    *,
    title_artist=None,
    legend_artist=None,
    colorbar_axes: Sequence | None = None,
    inset_axes: Sequence | None = None,
    main_axes: Sequence | None = None,
    panel_text_artists: Sequence | None = None,
    normalization_limits: Sequence[tuple[float, float]] | None = None,
    category_counts: Mapping[str, int] | None = None,
    inset_validation: Sequence[Mapping] | None = None,
    safety: PublicationSafetySpec = PUBLICATION_SAFETY,
) -> dict:
    """Validate fixed-canvas publication clearances and artist overlaps."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = (float(value) for value in fig.get_size_inches())
    main_axes = list(main_axes or [])
    colorbar_axes = list(colorbar_axes or [])
    inset_axes = list(inset_axes or [])
    panel_text_artists = [
        artist
        for artist in (panel_text_artists or [])
        if artist is not None
    ]

    title_bbox = (
        _bbox_in_inches(fig, title_artist, renderer)
        if title_artist is not None
        else None
    )
    legend_bbox = (
        _bbox_in_inches(fig, legend_artist, renderer)
        if legend_artist is not None
        else None
    )
    main_bboxes = [
        _axes_bbox_in_inches(fig, axis, renderer, tight=False)
        for axis in main_axes
    ]
    colorbar_bboxes = [
        _axes_bbox_in_inches(fig, axis, renderer, tight=True)
        for axis in colorbar_axes
    ]
    inset_bboxes = [
        _axes_bbox_in_inches(fig, axis, renderer, tight=True)
        for axis in inset_axes
    ]
    panel_bboxes = [
        _bbox_in_inches(fig, artist, renderer)
        for artist in panel_text_artists
    ]

    tracked_bboxes = [
        bbox
        for bbox in [
            title_bbox,
            legend_bbox,
            *main_bboxes,
            *colorbar_bboxes,
            *inset_bboxes,
            *panel_bboxes,
        ]
        if bbox is not None
    ]
    outside_canvas = [
        index
        for index, bbox in enumerate(tracked_bboxes)
        if (
            bbox.x0 < -1e-4
            or bbox.y0 < -1e-4
            or bbox.x1 > width + 1e-4
            or bbox.y1 > height + 1e-4
        )
    ]
    side_clearance = min(
        min(bbox.x0, width - bbox.x1) for bbox in tracked_bboxes
    )
    title_top_clearance = (
        height - title_bbox.y1 if title_bbox is not None else None
    )
    title_center_offset = (
        abs((title_bbox.x0 + title_bbox.x1) / 2.0 - width / 2.0)
        if title_bbox is not None
        else None
    )
    legend_bottom_clearance = (
        legend_bbox.y0 if legend_bbox is not None else None
    )
    title_map_clearance = (
        min(title_bbox.y0 - bbox.y1 for bbox in main_bboxes)
        if title_bbox is not None and main_bboxes
        else None
    )
    legend_map_clearance = (
        min(bbox.y0 - legend_bbox.y1 for bbox in main_bboxes)
        if legend_bbox is not None and main_bboxes
        else None
    )

    overlap_failures: list[str] = []
    if title_bbox is not None:
        for bbox in main_bboxes:
            if title_bbox.overlaps(bbox):
                overlap_failures.append("title-map")
                break
    if legend_bbox is not None:
        for bbox in main_bboxes:
            if legend_bbox.overlaps(bbox):
                overlap_failures.append("legend-map")
                break
        for bbox in colorbar_bboxes:
            if legend_bbox.overlaps(bbox):
                overlap_failures.append("colorbar-legend")
                break
        for bbox in inset_bboxes:
            if legend_bbox.overlaps(bbox):
                overlap_failures.append("inset-legend")
                break
    overlap_details: list[str] = []
    for colorbar_index, colorbar_bbox in enumerate(colorbar_bboxes):
        for map_index, map_bbox in enumerate(main_bboxes):
            if colorbar_bbox.overlaps(map_bbox):
                overlap_failures.append("colorbar-map")
                overlap_details.append(
                    f"colorbar[{colorbar_index}]-map[{map_index}]"
                )
                break
        for panel_index, panel_bbox in enumerate(panel_bboxes):
            if colorbar_bbox.overlaps(panel_bbox):
                overlap_failures.append("colorbar-panel-title")
                overlap_details.append(
                    f"colorbar[{colorbar_index}]-"
                    f"panel_text[{panel_index}]"
                )
                break
        if (
            "colorbar-map" in overlap_failures
            or "colorbar-panel-title" in overlap_failures
        ):
            break

    text_sizes = [
        float(text.get_fontsize())
        for text in fig.findobj(
            match=lambda artist: isinstance(artist, Text)
        )
        if text.get_visible() and str(text.get_text()).strip()
    ]
    too_small_text = [
        size
        for size in text_sizes
        if size + 1e-9 < safety.minimum_font_size_pt
    ]

    failures: list[str] = []
    if (
        title_top_clearance is not None
        and title_top_clearance < safety.top_clearance_in
    ):
        failures.append("title top clearance")
    if (
        title_center_offset is not None
        and title_center_offset > safety.title_center_tolerance_in
    ):
        failures.append("title horizontal centring")
    if (
        legend_bottom_clearance is not None
        and legend_bottom_clearance < safety.bottom_clearance_in
    ):
        failures.append("legend bottom clearance")
    if side_clearance < safety.side_clearance_in:
        failures.append("minimum side clearance")
    if (
        title_map_clearance is not None
        and title_map_clearance < safety.title_map_clearance_in
    ):
        failures.append("title-map clearance")
    if (
        legend_map_clearance is not None
        and legend_map_clearance < safety.legend_map_clearance_in
    ):
        failures.append("legend-map clearance")
    if overlap_failures:
        failures.extend(overlap_failures)
    if outside_canvas:
        failures.append("artist outside canvas")
    if too_small_text:
        failures.append("text below minimum size")

    report = {
        "figure_dimensions_in": [width, height],
        "title_bbox": _bbox_dict(title_bbox),
        "legend_bbox": _bbox_dict(legend_bbox),
        "colorbar_bboxes": [
            _bbox_dict(bbox) for bbox in colorbar_bboxes
        ],
        "main_axes_bboxes": [_bbox_dict(bbox) for bbox in main_bboxes],
        "inset_bboxes": [_bbox_dict(bbox) for bbox in inset_bboxes],
        "title_top_clearance_in": title_top_clearance,
        "title_center_offset_in": title_center_offset,
        "legend_bottom_clearance_in": legend_bottom_clearance,
        "minimum_side_clearance_in": float(side_clearance),
        "title_map_clearance_in": title_map_clearance,
        "legend_map_clearance_in": legend_map_clearance,
        "overlap_failures": sorted(set(overlap_failures)),
        "overlap_details": overlap_details,
        "outside_canvas_artist_indices": outside_canvas,
        "minimum_text_size_pt": min(text_sizes) if text_sizes else None,
        "normalization_limits": [
            [float(lower), float(upper)]
            for lower, upper in (normalization_limits or [])
        ],
        "category_counts": dict(category_counts or {}),
        "inset_validation": list(inset_validation or []),
        "layout_validation_status": "passed" if not failures else "failed",
        "layout_failures": sorted(set(failures)),
    }
    print("[publication layout]")
    print(
        "  title top clearance: "
        f"{title_top_clearance:.3f} in"
        if title_top_clearance is not None
        else "  title top clearance: not applicable"
    )
    print(
        "  legend bottom clearance: "
        f"{legend_bottom_clearance:.3f} in"
        if legend_bottom_clearance is not None
        else "  legend bottom clearance: not applicable"
    )
    print(f"  minimum side clearance: {side_clearance:.3f} in")
    print(f"  overlaps detected: {len(set(overlap_failures))}")
    if overlap_details:
        print("  overlap details: " + ", ".join(overlap_details))
    if failures:
        raise ValueError(
            "Publication layout validation failed: "
            + ", ".join(sorted(set(failures)))
        )
    return report


def save_figure(
    fig,
    output_path: Path,
    dpi: int,
    *,
    layout_report: dict | None = None,
    diagnostic_path: Path | None = None,
    thumbnail_path: Path | None = None,
) -> dict | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=None,
        facecolor="white",
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches=None,
        facecolor="white",
    )
    if thumbnail_path is not None:
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            thumbnail_path,
            dpi=120,
            bbox_inches=None,
            facecolor="white",
        )
    if layout_report is not None:
        layout_report["output_file_size_bytes"] = int(
            output_path.stat().st_size
        )
        layout_report["pdf_file_size_bytes"] = int(
            output_path.with_suffix(".pdf").stat().st_size
        )
        if thumbnail_path is not None:
            layout_report["thumbnail_file_size_bytes"] = int(
                thumbnail_path.stat().st_size
            )
    if diagnostic_path is not None:
        if layout_report is None:
            raise ValueError(
                "diagnostic_path requires a publication layout report"
            )
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_path.write_text(
            json.dumps(layout_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    plt.close(fig)
    return layout_report


def plot_grid_map(
    frame: pd.DataFrame,
    column: str,
    output_path: Path,
    title: str,
    colorbar_label: str,
    *,
    diverging: bool,
    cmap: Colormap | str | None = None,
    norm: Normalize | None = None,
    national_grid: pd.DataFrame | None = None,
    support_label: str = "Outside model support",
    dpi: int = 600,
    diagnostic_path: Path | None = None,
    thumbnail_path: Path | None = None,
) -> dict:
    """Render a formal 10 km continuous map and overwrite PNG/PDF."""
    apply_nature_style(dpi)
    support_aware = national_grid is not None
    if support_aware:
        cells, provinces, national, crs = prepare_supported_map_data(
            frame,
            [column],
            national_grid,
        )
    else:
        cells, provinces, national, crs = prepare_map_data(frame, column)
    valid_values = cells.loc[
        cells["inside_model_support"]
        if support_aware
        else cells[column].notna(),
        column,
    ]
    diagnostics = distribution_diagnostics(valid_values, column)
    selected_norm = norm or normalization(
        valid_values,
        diverging=diverging,
    )
    selected_cmap = cmap or (
        SOFT_DIVERGING_CMAP if diverging else "cividis"
    )

    layout = SINGLE_CONTINUOUS_LAYOUT
    fig, axis, footer_axis = _single_map_canvas(layout)
    land_color = (
        OUTSIDE_MODEL_SUPPORT_COLOR
        if support_aware
        else LAND_NODATA_COLOR
    )
    _draw_land(axis, national, land_color)
    if support_aware:
        _draw_flat_cells(
            axis,
            cells.loc[~cells["inside_model_support"]],
            OUTSIDE_MODEL_SUPPORT_COLOR,
        )
    artist = _draw_continuous_cells(
        axis,
        cells,
        column,
        selected_cmap,
        selected_norm,
    )
    _draw_boundaries(axis, provinces, national)
    _style_main_axis(axis, crs)
    title_artist = fig.suptitle(
        _sentence_title(title),
        x=0.5,
        y=layout.title_y,
        ha="center",
        va="top",
        fontsize=layout.title_fontsize,
        fontweight="normal",
    )
    legend_artist = _add_support_key(
        footer_axis,
        layout,
        label=support_label if support_aware else "No data",
        color=(
            OUTSIDE_MODEL_SUPPORT_COLOR
            if support_aware
            else GRID_NODATA_COLOR
        ),
    )
    if artist is None:
        raise ValueError(f"No finite cells available for {column}")
    _, colorbar_axis = _add_horizontal_colorbar(
        fig,
        footer_axis,
        artist,
        colorbar_label,
        _continuous_extend(valid_values, selected_norm),
        layout,
    )

    inset = axis.inset_axes(layout.inset_bounds)
    inset_cells, selection = _inset_view(cells, crs)
    _draw_land(inset, national, land_color)
    if support_aware:
        _draw_flat_cells(
            inset,
            inset_cells.loc[~inset_cells["inside_model_support"]],
            OUTSIDE_MODEL_SUPPORT_COLOR,
        )
    _draw_continuous_cells(
        inset,
        inset_cells,
        column,
        selected_cmap,
        selected_norm,
    )
    _draw_boundaries(inset, provinces, national, inset=True)
    inset_title = _style_inset(
        inset,
        selection,
        label_fontsize=layout.inset_label_fontsize,
    )
    inset_report = _validate_inset(
        cells,
        inset_cells,
        selection,
        column,
        selected_norm,
    )
    if diagnostics.finite_count == 0:
        raise ValueError(f"No finite map cells for {column}")
    layout_report = validate_publication_layout(
        fig,
        title_artist=title_artist,
        legend_artist=legend_artist,
        colorbar_axes=[colorbar_axis],
        inset_axes=[inset],
        main_axes=[axis],
        panel_text_artists=[inset_title],
        normalization_limits=[
            (float(selected_norm.vmin), float(selected_norm.vmax))
        ],
        inset_validation=[inset_report],
    )
    layout_report.update(
        {
            "layout_spec": layout.name,
            "map_type": "continuous_diverging"
            if diverging
            else "continuous_sequential",
            "mapped_column": column,
        }
    )
    return save_figure(
        fig,
        output_path,
        dpi,
        layout_report=layout_report,
        diagnostic_path=diagnostic_path,
        thumbnail_path=thumbnail_path,
    )


def plot_shap_multipanel(
    frame: pd.DataFrame,
    columns: Sequence[str],
    panel_titles: Sequence[str],
    output_path: Path,
    *,
    figure_title: str,
    national_grid: pd.DataFrame,
    norm: Normalize | None = None,
    dpi: int = 600,
    diagnostic_path: Path | None = None,
    thumbnail_path: Path | None = None,
) -> dict:
    """Render six full-support SHAP surfaces in a 2 x 3 figure."""
    if len(columns) != 6 or len(panel_titles) != 6:
        raise ValueError("SHAP multi-panel figures require exactly 6 panels")
    apply_nature_style(dpi)
    cells, provinces, national, crs = prepare_supported_map_data(
        frame,
        columns,
        national_grid,
    )
    supported = cells["inside_model_support"]
    pooled = pd.concat(
        [_numeric(cells.loc[supported, column]) for column in columns],
        ignore_index=True,
    )
    selected_norm = norm or normalization(pooled, diverging=True)
    selected_cmap = SOFT_DIVERGING_CMAP
    layout = SHAP_SHARED_LAYOUT
    fig, axes, footer_axis = _shared_multipanel_canvas(layout)
    mappable = ScalarMappable(norm=selected_norm, cmap=selected_cmap)
    inset_cells, selection = _inset_view(cells, crs)
    panel_text_artists = []
    inset_axes = []
    inset_reports = []
    for panel_index, (axis, column, panel_title) in enumerate(
        zip(axes.ravel(), columns, panel_titles)
    ):
        _draw_land(axis, national, OUTSIDE_MODEL_SUPPORT_COLOR)
        _draw_flat_cells(
            axis,
            cells.loc[~supported],
            OUTSIDE_MODEL_SUPPORT_COLOR,
        )
        _draw_continuous_cells(
            axis,
            cells,
            column,
            selected_cmap,
            selected_norm,
        )
        _draw_boundaries(axis, provinces, national)
        panel_letter = chr(ord("a") + panel_index)
        _style_main_axis(axis, crs)
        letter_artist = axis.text(
            0.00,
            1.015,
            panel_letter,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=layout.panel_letter_fontsize,
            fontweight="bold",
            clip_on=False,
        )
        title_artist = axis.text(
            0.055,
            1.015,
            str(panel_title).replace("PM2.5", r"PM$_{2.5}$"),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=layout.panel_title_fontsize,
            fontweight="normal",
            clip_on=False,
        )
        panel_text_artists.extend([letter_artist, title_artist])
        inset = axis.inset_axes(layout.inset_bounds)
        inset_axes.append(inset)
        _draw_land(inset, national, OUTSIDE_MODEL_SUPPORT_COLOR)
        _draw_flat_cells(
            inset,
            inset_cells.loc[
                ~inset_cells["inside_model_support"]
            ],
            OUTSIDE_MODEL_SUPPORT_COLOR,
        )
        _draw_continuous_cells(
            inset,
            inset_cells,
            column,
            selected_cmap,
            selected_norm,
        )
        _draw_boundaries(inset, provinces, national, inset=True)
        inset_title = _style_inset(
            inset,
            selection,
            label="SCS" if panel_index == 0 else None,
            label_fontsize=5.5,
        )
        panel_text_artists.append(inset_title)
        inset_reports.append(_validate_inset(
            cells,
            inset_cells,
            selection,
            column,
            selected_norm,
        ))

    _, colorbar_axis = _add_horizontal_colorbar(
        fig,
        footer_axis,
        mappable,
        "SHAP contribution",
        _continuous_extend(pooled, selected_norm),
        layout,
    )
    legend_artist = _add_support_key(
        footer_axis,
        layout,
        label="Outside model support",
        color=OUTSIDE_MODEL_SUPPORT_COLOR,
    )
    overall_title = fig.suptitle(
        _sentence_title(figure_title),
        x=0.5,
        y=layout.title_y,
        ha="center",
        va="top",
        fontsize=layout.title_fontsize,
        fontweight="normal",
    )
    layout_report = validate_publication_layout(
        fig,
        title_artist=overall_title,
        legend_artist=legend_artist,
        colorbar_axes=[colorbar_axis],
        inset_axes=inset_axes,
        main_axes=list(axes.ravel()),
        panel_text_artists=panel_text_artists,
        normalization_limits=[
            (float(selected_norm.vmin), float(selected_norm.vmax))
        ],
        inset_validation=inset_reports,
    )
    layout_report.update(
        {
            "layout_spec": layout.name,
            "map_type": "shap_shared_scale",
            "mapped_columns": list(columns),
        }
    )
    return save_figure(
        fig,
        output_path,
        dpi,
        layout_report=layout_report,
        diagnostic_path=diagnostic_path,
        thumbnail_path=thumbnail_path,
    )


def plot_shap_multipanel_individual_scale(
    frame: pd.DataFrame,
    columns: Sequence[str],
    panel_titles: Sequence[str],
    output_path: Path,
    *,
    figure_title: str,
    national_grid: pd.DataFrame,
    dpi: int = 600,
    diagnostic_path: Path | None = None,
    thumbnail_path: Path | None = None,
) -> dict:
    """Render six full-support SHAP surfaces with per-feature scales."""
    if len(columns) != 6 or len(panel_titles) != 6:
        raise ValueError(
            "Individual-scale SHAP figures require exactly 6 panels"
        )
    apply_nature_style(dpi)
    cells, provinces, national, crs = prepare_supported_map_data(
        frame,
        columns,
        national_grid,
    )
    supported = cells["inside_model_support"]
    layout = SHAP_INDIVIDUAL_LAYOUT
    (
        fig,
        title_axes,
        axes,
        colorbar_axes,
        footer_axis,
    ) = _individual_multipanel_canvas(layout)
    inset_cells, selection = _inset_view(cells, crs)
    panel_text_artists = []
    inset_axes = []
    inset_reports = []
    normalization_limits = []

    for panel_index, (
        title_axis,
        axis,
        colorbar_axis,
        column,
        panel_title,
    ) in enumerate(
        zip(
            title_axes.ravel(),
            axes.ravel(),
            colorbar_axes.ravel(),
            columns,
            panel_titles,
        )
    ):
        values = _numeric(cells.loc[supported, column])
        selected_norm = normalization(values, diverging=True)
        normalization_limits.append(
            (float(selected_norm.vmin), float(selected_norm.vmax))
        )
        mean_absolute = float(values.abs().mean())
        exponent = _scale_exponent(selected_norm)
        scale = 10.0**exponent
        mean_display = mean_absolute / scale
        scale_note = (
            f"; scale ×10$^{{{exponent}}}$" if exponent else ""
        )
        _draw_land(axis, national, OUTSIDE_MODEL_SUPPORT_COLOR)
        _draw_flat_cells(
            axis,
            cells.loc[~supported],
            OUTSIDE_MODEL_SUPPORT_COLOR,
        )
        _draw_continuous_cells(
            axis,
            cells,
            column,
            SOFT_DIVERGING_CMAP,
            selected_norm,
        )
        _draw_boundaries(axis, provinces, national)
        _style_main_axis(axis, crs)
        panel_letter = chr(ord("a") + panel_index)
        letter_artist = title_axis.text(
            0.00,
            0.58,
            panel_letter,
            transform=title_axis.transAxes,
            ha="left",
            va="center",
            fontsize=layout.panel_letter_fontsize,
            fontweight="bold",
        )
        title_artist = title_axis.text(
            0.055,
            0.58,
            (
                str(panel_title).replace(
                    "PM2.5",
                    r"PM$_{2.5}$",
                )
                + f"\nmean |SHAP| = {mean_display:.3g}{scale_note}"
            ),
            transform=title_axis.transAxes,
            ha="left",
            va="center",
            fontsize=layout.panel_title_fontsize,
            fontweight="normal",
            linespacing=0.95,
        )
        panel_text_artists.extend([letter_artist, title_artist])

        inset = axis.inset_axes(layout.inset_bounds)
        inset_axes.append(inset)
        _draw_land(inset, national, OUTSIDE_MODEL_SUPPORT_COLOR)
        _draw_flat_cells(
            inset,
            inset_cells.loc[
                ~inset_cells["inside_model_support"]
            ],
            OUTSIDE_MODEL_SUPPORT_COLOR,
        )
        _draw_continuous_cells(
            inset,
            inset_cells,
            column,
            SOFT_DIVERGING_CMAP,
            selected_norm,
        )
        _draw_boundaries(inset, provinces, national, inset=True)
        inset_title = _style_inset(
            inset,
            selection,
            label="SCS" if panel_index == 0 else None,
            label_fontsize=5.5,
        )
        panel_text_artists.append(inset_title)
        inset_reports.append(
            _validate_inset(
                cells,
                inset_cells,
                selection,
                column,
                selected_norm,
            )
        )

        colorbar = fig.colorbar(
            ScalarMappable(
                norm=selected_norm,
                cmap=SOFT_DIVERGING_CMAP,
            ),
            cax=colorbar_axis,
            orientation="horizontal",
            extend=_continuous_extend(values, selected_norm),
            format=FuncFormatter(
                lambda value, _, factor=scale: f"{value / factor:g}"
            ),
        )
        colorbar.locator = MaxNLocator(nbins=3)
        colorbar.update_ticks()
        colorbar.ax.tick_params(
            length=1.8,
            width=0.35,
            pad=1.0,
            labelsize=5.5,
        )
        colorbar.outline.set_linewidth(0.35)
        colorbar.ax.xaxis.get_offset_text().set_visible(False)

    legend_artist = _add_support_key(
        footer_axis,
        layout,
        label="Outside model support",
        color=OUTSIDE_MODEL_SUPPORT_COLOR,
        centered=True,
    )
    overall_title = fig.suptitle(
        _sentence_title(figure_title),
        x=0.5,
        y=layout.title_y,
        ha="center",
        va="top",
        fontsize=layout.title_fontsize,
        fontweight="normal",
    )
    layout_report = validate_publication_layout(
        fig,
        title_artist=overall_title,
        legend_artist=legend_artist,
        colorbar_axes=list(colorbar_axes.ravel()),
        inset_axes=inset_axes,
        main_axes=list(axes.ravel()),
        panel_text_artists=panel_text_artists,
        normalization_limits=normalization_limits,
        inset_validation=inset_reports,
    )
    layout_report.update(
        {
            "layout_spec": layout.name,
            "map_type": "shap_individual_scale",
            "mapped_columns": list(columns),
        }
    )
    return save_figure(
        fig,
        output_path,
        dpi,
        layout_report=layout_report,
        diagnostic_path=diagnostic_path,
        thumbnail_path=thumbnail_path,
    )


def plot_lisa_map(
    frame: pd.DataFrame,
    category_column: str,
    output_path: Path,
    title: str,
    *,
    colors: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    dpi: int = 600,
    diagnostic_path: Path | None = None,
    thumbnail_path: Path | None = None,
) -> dict:
    """Render an unsmoothed categorical LISA map and overwrite PNG/PDF."""
    apply_nature_style(dpi)
    cells, provinces, national, crs = prepare_map_data(frame)
    if category_column not in cells:
        raise KeyError(f"LISA category column not found: {category_column}")
    palette = dict(LISA_COLORS)
    if colors:
        palette.update(colors)
    category_order = list(order or LISA_ORDER)
    category_order = [
        "NoData" if item == "No data" else item
        for item in category_order
    ]
    cells[category_column] = (
        cells[category_column]
        .astype("string")
        .replace({"No data": "NoData", "<NA>": "NoData"})
        .fillna("NoData")
    )
    unknown = set(cells[category_column].dropna()) - set(category_order)
    if unknown:
        raise ValueError(f"Unknown LISA categories: {sorted(unknown)}")

    layout = LISA_LAYOUT
    fig, axis, footer_axis = _single_map_canvas(layout)
    _draw_land(axis, national)
    _draw_categorical_cells(
        axis,
        cells,
        category_column,
        palette,
        category_order,
    )
    _draw_boundaries(axis, provinces, national)
    _style_main_axis(axis, crs)
    title_artist = fig.suptitle(
        _sentence_title(title),
        x=0.5,
        y=layout.title_y,
        ha="center",
        va="top",
        fontsize=layout.title_fontsize,
        fontweight="normal",
    )
    present = [
        item
        for item in category_order
        if cells[category_column].eq(item).any()
    ]
    labels = {
        "NoData": "No data",
        "Not significant": "Not significant",
        "High-High": "High\u2013High",
        "Low-Low": "Low\u2013Low",
        "High-Low": "High\u2013Low",
        "Low-High": "Low\u2013High",
    }
    legend_ncols = 3
    legend_nrows = int(np.ceil(len(present) / legend_ncols))
    legend_items = [
        present[row * legend_ncols + column]
        for column in range(legend_ncols)
        for row in range(legend_nrows)
        if row * legend_ncols + column < len(present)
    ]
    legend_artist = footer_axis.legend(
        handles=[
            Patch(
                facecolor=palette[item],
                edgecolor="none",
                label=labels.get(item, item),
            )
            for item in legend_items
        ],
        loc="center",
        bbox_to_anchor=layout.support_anchor,
        frameon=False,
        ncol=legend_ncols,
        borderaxespad=0,
        columnspacing=1.2,
        labelspacing=0.35,
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.4,
        fontsize=6.7,
    )

    inset = axis.inset_axes(layout.inset_bounds)
    inset_cells, selection = _inset_view(cells, crs)
    _draw_land(inset, national)
    _draw_categorical_cells(
        inset,
        inset_cells,
        category_column,
        palette,
        category_order,
    )
    _draw_boundaries(inset, provinces, national, inset=True)
    inset_title = _style_inset(
        inset,
        selection,
        label_fontsize=layout.inset_label_fontsize,
    )
    inset_report = _validate_inset(
        cells,
        inset_cells,
        selection,
        category_column,
    )
    print(
        "[LISA categories] "
        + ", ".join(
            f"{item}={int(cells[category_column].eq(item).sum()):,}"
            for item in present
        )
    )
    category_counts = {
        labels.get(item, item): int(
            cells[category_column].eq(item).sum()
        )
        for item in present
    }
    layout_report = validate_publication_layout(
        fig,
        title_artist=title_artist,
        legend_artist=legend_artist,
        inset_axes=[inset],
        main_axes=[axis],
        panel_text_artists=[inset_title],
        category_counts=category_counts,
        inset_validation=[inset_report],
    )
    layout_report.update(
        {
            "layout_spec": layout.name,
            "map_type": "lisa_categorical",
            "mapped_column": category_column,
        }
    )
    return save_figure(
        fig,
        output_path,
        dpi,
        layout_report=layout_report,
        diagnostic_path=diagnostic_path,
        thumbnail_path=thumbnail_path,
    )
