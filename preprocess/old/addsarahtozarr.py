import xarray

import xarray
from dask.distributed import Client
from glob import glob

if __name__ == '__main__':

    with Client(n_workers = 16) as client:
        print(client)

        #phts = [pth for pth in glob('customized/HRSEVIRI_201*')]
        #dss = [ xarray.open_dataset(pth, engine='h5netcdf') for pth in phts]
        hres = xarray.open_mfdataset(
                'customized/HRSEVIRI_201*',
                parallel=True,
                #chunks={'time':60, 'lat':-1, 'lon':-1},
                concat_dim="time",
                combine="nested",
                data_vars="minimal",
                coords="minimal",
                compat="override",
                engine="h5netcdf",
            )
        #hres = xarray.concat(dss, dim='time', data_vars='minimal', compat='equals', coords= 'minimal')
        #hres = hres.chunk({'time':60, 'lat':-1, 'lon':-1})
        sis = xarray.open_mfdataset('SIS_*.nc', concat_dim='time', combine='nested', data_vars='minimal', coords='minimal', compat='override', engine='h5netcdf')

        intersec_tidx = sorted(
                list(
                    set(hres.time.dt.round("30min").values).intersection(
                        set(sis.time.dt.round("30min").values)
                    )
                )
            )

        hres_tidx_sort = hres.time.copy().sortby(hres.time)
        hres_intersec_tidx = hres_tidx_sort.sel(time=intersec_tidx, method='nearest')

        sis_reindex = sis.reindex(time=hres_intersec_tidx, lat=hres.lat, lon=hres.lon, method='nearest')
        shape = hres.channel_1.shape
        combined = xarray.merge([hres, sis_reindex]).drop('crs')
        #combined['SIS'] = combined.SIS.chunk({'time':60, 'lat':-1, 'lon':-1})
        #combined['record_status'] = combined['record_status'].chunk({'time':60})
        encoding = {}
        for key in combined.variables:
            if key == 'record_status':
                encoding[key] = {'chunks': 60}
            else:
                encoding[key] = {'chunks':(60, shape[1], shape[2])}
        combined.to_zarr('HRSEVIRI_60.zarr', mode='w', encoding=encoding, safe_chunks=False)

