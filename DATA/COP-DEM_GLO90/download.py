from dem_stitcher import stitch_dem

# as xmin, ymin, xmax, ymax in epsg:4326
bounds = [-15, 30, 25, 65]

X, p = stitch_dem(bounds,
                  dem_name='glo_90',  # Global Copernicus 90 meter resolution DEM
                  dst_ellipsoidal_height=False,
                  dst_area_or_point='Point')


import rasterio

with rasterio.open('GLO90.tif', 'w', **p) as ds:
   ds.write(X, 1)
   ds.update_tags(AREA_OR_POINT='Point')