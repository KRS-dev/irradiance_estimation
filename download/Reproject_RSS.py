import xarray as xr
import rioxarray
import cartopy.crs as ccrs
import cartopy
from pyproj import Transformer
import matplotlib.pyplot as plt
import ocf_blosc2
import numpy as np
import time
import gc
import sys


zarr_path = sys.argv[1]
path = sys.argv[2]
# zarr_path = "/scratch/snx3000/acarpent/EumetsatData/SEVIRI_RSS.zarr"
seviri_proj = ccrs.Geostationary(satellite_height=35785831, central_longitude=9.5)
# heliomont_proj = ccrs.Geostationary(satellite_height=35785831, central_longitude=0.0)
# path = "/scratch/snx3000/acarpent/EumetsatData/SEVIRI_RSS_WGS.zarr"


variables = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
             'VIS006', 'VIS008', 'WV_062', 'WV_073']

def reproject_to_wgs(dataset, grid_dataset):
    # dataset = dataset.persist()
    new_dataset = xr.Dataset()
    for var in variables:
        dataarray = dataset['data'].sel(variable=var)
        rss_da = xr.DataArray(
            data=dataarray.values.astype(np.float32),
            dims=["time", "y", "x"],
            coords=dict(
                x=("x", dataarray.x.values),
                y=("y", dataarray.y.values),
                time = dataarray.time.values),
            ).rio.write_crs(seviri_proj)
        rss_da = rss_da.rio.reproject_match(grid_dataset, 
                                            nodata=np.nan)
        new_dataset[var] = rss_da
        del new_dataset[var].attrs["_FillValue"]
    return new_dataset


def main():
    dataset = xr.open_dataset(
        zarr_path,
        engine="zarr", 
        chunks={},  # Load the data as a Dask array.
    )
    # my_dataset = xr.open_dataset(path,
    #     engine="zarr", 
    #     chunks="auto",  # Load the data as a Dask array.
    # )
    # timestamps = set(my_dataset.time.values)
    idx_start = 0
    dataset_len = len(dataset['data'])
    step = 192
    lon = np.arange(-7.775,28.975+0.05,0.05)
    lat = np.arange(28.975,61.825+0.05,0.05)
    grid_x, grid_y = np.mgrid[-7.775:28.975+0.05:0.05, 28.975:61.825+0.05:0.05]
    grid_dataset = xr.Dataset(coords=dict(
            x=("x", grid_x[:,0]),
            y=("y", grid_y[0,:]))).rio.write_crs('wgs84')
    

    for j in range(idx_start, dataset_len, step):
        start = time.time()
        dataset_sliced = (
            dataset
            .isel(time=slice(j, j+step))
            ).persist().drop_duplicates('time')
        
        wgs_dataset = reproject_to_wgs(dataset_sliced, grid_dataset).chunk({'time':24, 'x':len(lon), 'y':len(lat)})
        if j == 0:
            wgs_dataset.to_zarr(path, mode='w')
        else:
            wgs_dataset.to_zarr(path, append_dim='time')
        end = time.time()
        print(j/dataset_len, end-start, wgs_dataset.chunks)
        gc.collect()
        del wgs_dataset
        del dataset_sliced

if __name__ == '__main__':
    main()