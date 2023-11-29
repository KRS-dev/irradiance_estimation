import argparse
import xarray
from distributed import Client
from etc import benchmark
from dask.diagnostics import ResourceProfiler, ProgressBar, CacheProfiler, Profiler, visualize
from bokeh.io import export_png

parser = argparse.ArgumentParser(
    prog='reindex',
    description='Customizes the native encoded MSG level 1.5 data to a time aggregated time filtered .nc file. Aggregation normally happens per day.',
)
parser.add_argument('files', nargs='+')
parser.add_argument('reindexlike', default=None)
# parser.add_argument('-d', '--dimensions', option) # todo add option for dimensions of reindex
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-o', '--outputfile', default='combined.nc')


if __name__ ==  '__main__':
    
    args = parser.parse_args()
    client = Client(n_workers=12, processes=True, threads_per_worker=1)  # Connect to distributed cluster and override default
    print(client)
    
    input = xarray.open_dataset(args.reindexlike) 

    def reindex(ds):
        return ds.reindex(lat=input.lat, lon=input.lon, method='nearest')


    with benchmark('working'), Profiler() as prof, ResourceProfiler(dt=.5) as rprof, CacheProfiler() as cprof:
        if args.reindexlike is not None:
            combined = xarray.open_mfdataset(
                args.files,  
                parallel=True, 
                preprocess=reindex,
                decode_cf=False,
                chunks={'time':100, 'lat':-1, 'lon':-1},
                concat_dim="time", 
                combine="nested",
                data_vars='minimal', 
                coords='minimal', 
                compat='override')
        else:
            combined = xarray.open_mfdataset(
                args.files,  
                parallel=True, 
                decode_cf=False,
                chunks={'time':100, 'lat':-1, 'lon':-1},
                concat_dim="time", 
                combine="nested",
                data_vars='minimal', 
                coords='minimal', 
                compat='override')
        
        combined.to_netcdf(args.outputfile)
    plot = visualize([prof, rprof, cprof])
    export_png(plot, filename="profilers.png")

