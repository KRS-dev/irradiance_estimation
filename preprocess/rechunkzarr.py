import xarray
from dask.distributed import Client
import zarr

if __name__ == '__main__':  
    with Client(n_workers=10) as client:
        print(client)

        hres = xarray.open_zarr('HRSEVIRI.zarr')
        for var in hres:
            del hres[var].encoding['chunks']
        print('all good before write')
        zarray = hres.chunk({'time':60, 'lat':-1, 'lon':-1})
        zarray.to_zarr('HRSEVIRI_30.zarr', mode = 'w')
        print('Done')
