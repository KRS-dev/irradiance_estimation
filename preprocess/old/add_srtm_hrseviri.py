import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from datetime import datetime

if __name__ == '__main__':
    zarr_store = '/scratch/snx3000/kschuurm/DATA/train.zarr'

    hres = xarray.open_zarr(zarr_store)

    DEM = xarray.open_dataset('SRTM.nc')
    roi= (65,35,-15,28)
    SRTM = DEM.SRTM.isel(lat=(DEM.lat >= roi[1])&(DEM.lat<= roi[0]), lon=(DEM.lon >= roi[2])&(DEM.lon <= roi[3]))

    weight_lat = int(np.ceil(len(SRTM.lat) / len(hres.lon)))
    weight_lon = int(np.ceil(len(SRTM.lon)/ len(hres.lon)))
    SRTM_coarse = SRTM.coarsen(lon=weight_lon, boundary='trim').mean().coarsen(lat=weight_lat, boundary='trim').mean()
    SRTM_coarse = SRTM_coarse.reindex(lat=hres.lat, lon=hres.lon, method='nearest')


    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(3,1, figsize=(10,6),subplot_kw={'projection': proj})

    for axi in ax.flatten():
        axi.add_feature(cf.BORDERS)

    SRTM.plot.imshow(
        ax=ax[0],
        add_colorbar=True,
        # transform=proj,
    )
    SRTM_coarse.plot.imshow(
        ax=ax[1],
        # transform=proj,
        add_colorbar=True,
    )
    hres['SIS'].isel(time=2).plot.imshow(
        ax=ax[2],
        # transform=proj,
        add_colorbar=True,
    )

    

    fig.savefig('DEM_test.png')

    SRTM.to_zarr('train.zarr', mode='a')