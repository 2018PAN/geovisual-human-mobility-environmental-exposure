// =====================================================
// GHSL built-up ratio raster
// Dataset: JRC/GHSL/P2023A/GHS_BUILT_S
// Resolution: 100 m product, exported as 1 km raster
// Year used: 2015, pre-2018 built environment background
// Region: China and surrounding region
// =====================================================

// 1. Study area
var geometry = ee.Geometry.Rectangle([73, 3, 135, 54], null, false);

// 2. Load GHSL 100m built-up surface collection
var ghsl = ee.ImageCollection('JRC/GHSL/P2023A/GHS_BUILT_S');

print('GHSL image IDs:', ghsl.aggregate_array('system:index'));

// 3. Select 2015 image
var built2015 = ee.Image(
  ghsl.filter(ee.Filter.eq('system:index', '2015')).first()
);

print('Built 2015 image:', built2015);
print('Band names:', built2015.bandNames());

// 4. Select built-up surface band
var builtSurface = built2015
  .select('built_surface')
  .clip(geometry);

// 5. Convert built-up surface to built-up fraction
// 100m pixel area = 10,000 m²
// built_up_ratio pixel value = built_surface / 10000
var builtFraction = builtSurface
  .divide(10000)
  .clamp(0, 1)
  .rename('built_up_ratio');

// 6. Aggregate to about 1 km to reduce file size
var builtFraction1km = builtFraction
  .reduceResolution({
    reducer: ee.Reducer.mean(),
    maxPixels: 1024
  })
  .reproject({
    crs: 'EPSG:4326',
    scale: 1000
  });

// 7. Display
Map.centerObject(geometry, 4);

Map.addLayer(
  builtFraction1km,
  {min: 0, max: 1},
  'GHSL built-up ratio 2015 1km'
);

// 8. Export GeoTIFF to Google Drive
Export.image.toDrive({
  image: builtFraction1km,
  description: 'GHSL_2015_built_up_ratio_China_region_1km',
  folder: 'GEE_exports',
  fileNamePrefix: 'GHSL_2015_built_up_ratio_China_region_1km',
  region: geometry,
  scale: 1000,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});