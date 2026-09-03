// ===============================
// 2018 Spring Festival Nighttime Lights
// NASA Black Marble VNP46A2
// Study area: China and surrounding region
// ===============================

// 1. Study area
var geometry = ee.Geometry.Rectangle([73, 3, 135, 54]);

// 2. Load NASA Black Marble VNP46A2 daily data
// 2018 Spring Festival was around 2018-02-16,
// so we use February 2018 mean as the nighttime-lights background.
var ntlCollection = ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
  .filterDate('2018-02-01', '2018-03-01')
  .filterBounds(geometry)
  .select('Gap_Filled_DNB_BRDF_Corrected_NTL');

// 3. Check number of daily images
print('Number of images:', ntlCollection.size());

// 4. Monthly mean nighttime lights
var ntl201802 = ntlCollection
  .mean()
  .clip(geometry);

// 5. Remove invalid negative values, keep zero as valid dark areas
ntl201802 = ntl201802.updateMask(ntl201802.gte(0));

// 6. Display
Map.centerObject(geometry, 4);

Map.addLayer(
  ntl201802,
  {min: 0, max: 60},
  'Nighttime Lights Mean - Feb 2018'
);

// 7. Export GeoTIFF to Google Drive
Export.image.toDrive({
  image: ntl201802,
  description: 'VNP46A2_201802_mean_NTL_China_region',
  folder: 'GEE_exports',
  fileNamePrefix: 'VNP46A2_201802_mean_NTL_China_region',
  region: geometry,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});