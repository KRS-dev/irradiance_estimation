import xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import rioxarray

if __name__ == '__main__':
    zarr_store = '/scratch/snx3000/kschuurm/ZARR/SEVIRI_FULLDISK.zarr'

    seviri = xarray.open_zarr(zarr_store)
    seviri = seviri.rename({
        'x':'lon',
        'y':'lat',
    })
    print(seviri)

    DEM = rioxarray.open_rasterio('GLO90.tif')
    DEM = DEM.reindex(y=DEM.y[::-1]).to_dataset(dim='band')
    DEM = DEM.rename({1:'DEM', 'x':'lon', 'y':'lat'})
    roi= (seviri.lat.max().item(),
          seviri.lat.min().item(),
          seviri.lon.min().item(),
          seviri.lon.max().item())
    print(roi)
    DEM = DEM.isel(lat=(DEM.lat >= roi[1])&(DEM.lat<= roi[0]), lon=(DEM.lon >= roi[2])&(DEM.lon <= roi[3]))

    weight_lat = int(np.ceil(len(DEM.lat) / len(seviri.lat)))
    weight_lon = int(np.ceil(len(DEM.lon)/ len(seviri.lon)))
    DEM_coarse = DEM.coarsen(lon=weight_lon, boundary='trim').mean().coarsen(lat=weight_lat, boundary='trim').mean()
    DEM_coarse = DEM_coarse.reindex(lat=seviri.lat, lon=seviri.lon, method='nearest')


    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(3,1, sharex=True,sharey=True, figsize=(10,6),subplot_kw={'projection': proj})

    for axi in ax.flatten():
        axi.add_feature(cf.BORDERS)

    DEM.DEM.plot.imshow(
        ax=ax[0],
        add_colorbar=True,
        # transform=proj,
    )
    DEM_coarse.DEM.plot.imshow(
        ax=ax[1],
        # transform=proj,
        add_colorbar=True,
    )
    seviri['VIS006'].isel(time=2).plot.imshow(
        ax=ax[2],
        # transform=proj,
        add_colorbar=True,
    )

    fig.savefig('DEM_test.png')

    DEM_coarse.to_zarr('/scratch/snx3000/kschuurm/ZARR/DEM.zarr', mode='w')