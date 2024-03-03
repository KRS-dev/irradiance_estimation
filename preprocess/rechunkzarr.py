import xarray
from dask.distributed import Client
import zarr

from dask.diagnostics import ProgressBar

if __name__ == '__main__':  
    hres = xarray.open_zarr('/scratch/snx3000/acarpent/EumetsatData/SEVIRI_WGS_2016-2022_RSS.zarr',
                            overwrite_encoded_chunks=True,
                            chunks={'time':1, 'x':-1, 'y':-1})
    
    hres = hres.chunk({'time':1, 'x':-1, 'y':-1})
    delayed = hres.to_zarr('/scratch/snx3000/kschuurm/ZARR/SEVIRI.zarr', mode = 'w', compute=False)


    with ProgressBar():
        out = delayed.compute()
    print('Done')
