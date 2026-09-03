# Publication cartography standard

This specification governs thesis map figures produced through
`downstream_plotting.py`. It changes presentation only. Analytical values,
classifications, grid geometry, masks and coordinate reference systems remain
authoritative upstream products.

## 1. Figure families and dimensions

All dimensions are final exported dimensions. The fixed canvas is preserved
for PNG and PDF; artist-dependent tight cropping is prohibited.

| Figure family | Dimensions | Map arrangement |
|---|---:|---|
| Single continuous map | 7.09 × 4.75 in | one map row + footer |
| LISA categorical map | 7.09 × 4.80 in | one map row + legend footer |
| Shared-scale SHAP | 7.09 × 5.15 in | 2 × 3 maps + shared footer |
| Individual-scale SHAP | 7.09 × 6.55 in | 3 × 2 maps with local colour bars + support footer |

The 7.09-inch width is the thesis 180-mm double-column standard. Map axes,
footers and local colour-bar rows use `GridSpec`; no figure mixes constrained
layout, `subplots_adjust`, floating `Figure.add_axes` and tight export.

## 2. Typography

- Font order: Arial, Helvetica, Liberation Sans, Matplotlib sans-serif.
- Overall title: 8 pt, normal weight, centred on the complete canvas.
- Panel letter: bold lowercase, 8 pt, no parentheses.
- Panel title: 6.8 pt, normal weight.
- Legend: 6.7 pt.
- Colour-bar label: 6.7 pt; tick labels 6.2 pt.
- Multi-panel local colour-bar text: at least 5.2 pt.
- Inset label: 6.0 pt for single maps and 5.5 pt for multi-panels.
- Minimum permitted rendered text: 5 pt.
- PDF and PostScript fonts use Type 42; text remains editable.
- PM2.5 is displayed as PM$_{2.5}$ in panel titles.

## 3. Safe margins

Central layout specifications reserve dedicated title, map and footer
regions. Minimum measured clearances are:

- title to top edge: 0.12 in;
- legend to bottom edge: 0.12 in;
- any tracked content to either side edge: 0.10 in;
- title to map: 0.10 in;
- legend to map: 0.08 in.

The renderer measures the final artists before export. A violation stops the
plot rather than creating a clipped publication file.

## 4. Title and panel placement

- Overall titles use `Figure.suptitle`, `x=0.5`.
- Single-map title y-coordinate: 0.955.
- Shared-scale SHAP title y-coordinate: 0.955.
- Individual-scale SHAP title y-coordinate: 0.958.
- Panel letters and titles occupy the title region immediately above each
  panel and are included in automated clipping checks.
- Long methodological explanations belong in the thesis caption, not the
  figure title or colour-bar label.

## 5. Cartographic hierarchy

| Layer | Colour | Main linewidth | Inset linewidth |
|---|---|---:|---:|
| National boundary | `#626262` | 0.63 pt | 0.40 pt |
| Provincial boundary | `#B9B9B9` | 0.20 pt | 0.15 pt |
| Inset frame | `#808080` | — | 0.38 pt |

Dense 10-km cells have no outlines, are not smoothed or interpolated, and are
rasterized. Text, administrative boundaries, legends and colour bars remain
vector objects in PDF.

## 6. Continuous colour rules

Signed values use the shared soft blue–off-white–red palette:

- negative extreme `#315F8C`;
- negative intermediate `#8FB8D3`;
- valid zero `#F8F6F2`;
- positive intermediate `#E7A17E`;
- positive extreme `#A83F45`.

Signed normalization is symmetric and centred exactly at zero. Robust display
limits use the existing 2nd/98th-percentile rule; underlying values are never
clipped. Comparable shared-scale panels pool values before deriving one
normalization.

Non-negative variables use `cividis` unless an existing scientifically
justified sequential palette is explicitly passed.

## 7. Categorical colours and semantics

LISA uses:

- High–High `#C35A63`;
- Low–Low `#477BAA`;
- High–Low `#DC916B`;
- Low–High `#82AAB9`;
- Not significant `#DEDEDE`;
- No data `#F1F1F1`.

The figure-level key has three columns and two rows in visual reading order:

1. High–High, Low–Low, High–Low;
2. Low–High, Not significant, No data.

No category is interpolated, reclassified, made transparent or outlined.

Spatial blanks retain distinct meanings:

- ocean/outside China: `#FFFFFF`;
- valid zero: `#F8F6F2`;
- outside model support: `#F0F0F0`;
- true continuous NoData: `#E3E3E3`;
- LISA not significant: `#DEDEDE`;
- LISA No data: `#F1F1F1`.

## 8. Footer systems

Single continuous maps use one dedicated footer row:

- solid support/NoData key immediately left of the colour bar;
- horizontal colour bar centred in the footer;
- no isolated lower-left legend.

LISA uses a dedicated footer row containing a centred figure-level
classification key.

Shared-scale SHAP uses one shared horizontal colour bar and one model-support
key. Individual-scale SHAP uses one local symmetric horizontal colour bar per
panel plus one footer support key.

## 9. South China Sea inset

The validated extent remains `105–125°E, 3–26°N`.

The prepared main-map cell GeoDataFrame is reused directly. The geographic
selection polygon is projected, its rectangular `total_bounds` defines the
displayed window, and all main-map cells intersecting that rectangle are
selected. The inset never independently merges, clips, classifies,
rasterizes, interpolates, masks or normalizes data.

Single-map inset bounds are `(0.785, 0.070, 0.155, 0.225)`. Multi-panel inset
bounds are `(0.805, 0.055, 0.135, 0.200)`.

Single maps show the centred label “South China Sea”. Multi-panels show “SCS”
only in panel a. Every render validates expected and actual inset grid IDs,
mapped values/categories and support statuses.

## 10. Export

- PNG: 600 dpi, RGB, white background.
- PDF: fixed canvas, editable text and vector boundaries.
- `bbox_inches=None`; no artist-dependent tight cropping.
- No `pad_inches` dependence.
- Preview thumbnails render the same fixed canvas at 120 dpi.

## 11. Automated quality control

`validate_publication_layout()` draws the canvas and measures:

- title centring and top clearance;
- legend bottom clearance;
- minimum side clearance;
- title–map and legend–map clearances;
- title–map, legend–map, colourbar–legend and inset–legend overlaps;
- tracked artists outside the canvas;
- minimum visible font size;
- main/inset grid, value/category and support-state equality.

Preview diagnostics are saved as JSON and include figure dimensions, artist
bounding boxes, measured clearances, normalization limits, category counts,
inset grid counts, overlap results and output file sizes.

## 12. Reference principles

The standard follows the project requirements and the current guidance from:

- Nature Research Figure Guide: final-size panels, standard sans-serif fonts,
  editable text, compact panel arrangement and vector artwork;
- Matplotlib: a single controlled layout system, figure-level titles and
  legends, explicit colour-bar axes and deterministic save behaviour;
- GeoPandas: common-CRS main/inset mapping with explicit displayed bounds;
- ColorBrewer and perceptually uniform colour principles: sequential,
  diverging and qualitative palettes chosen according to data semantics.
