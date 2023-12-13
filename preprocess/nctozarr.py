import xarray
from dask.distributed import Client
from glob import glob

if __name__ == '__main__':

    with Client(n_workers = 32) as client:    
        print(client)

        #ds = xr = xarray.open_mfdataset(
        #            "customized/HRSEVIRI_2015*FPC.nc",
        #            parallel=True,
        #            # chunks={'time':1, 'lat':-1, 'lon':-1},
        #            concat_dim="time",
        #            combine="nested",
        #            data_vars="minimal",
        #            coords="minimal",
        #            compat="override",
        #            engine="h5netcdf",
        #            ).chunk({'time':1, 'lat':-1, 'lon':-1})
        
        phts = [pth for pth in glob('customized/HRSEVIRI_201*')]
        dss = [ xarray.open_dataset(pth, engine='h5netcdf').chunk({'time':1, 'lat':-1, 'lon':-1}) for pth in phts]
         
        ds_big = xarray.concat(dss, dim='time', data_vars='minimal', compat='equals', coords= 'minimal')

        a = ds_big.to_zarr('HRSEVIRI.zarr', mode='w', consolidated=True) # consolidate means storing metadata in one file instead of seperate. Needed because of file limit.

         #   zarrstore =  ds.to_zarr(
        #      '/capstor/scratch/cscs/kschuurm/DATA/HRSEVIRI_2015.zarr',
         #       append_dim='time')
        print(a)

