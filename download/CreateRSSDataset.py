import xarray as xr
import cartopy.crs as ccrs  # CRS stands for "coordinate reference system"
import ocf_blosc2
import sys 
import psutil

start_year = int(sys.argv[1])
end_year = int(sys.argv[2])
year_range = range(start_year, end_year+1)
save_path = "/scratch/snx3000/kschuurm/ZARR/SEVIRI_RSS.zarr"
def main():
    for year in year_range:
        zarr_path = "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/{}_nonhrv.zarr".format(year)
    
        dataset = xr.open_dataset(
            zarr_path,
            engine="zarr", 
            chunks="auto",  # Load the data as a Dask array.
        ).rename({'x_geostationary':'x', 'y_geostationary':'y'})
    
        dataset_sliced = (
            dataset
            .isel(y=slice(450,1250), x=slice(1200,2400)))
            # .where(dataset["time.minute"].isin([0,15,30,45]), drop=True))
        
        for var in dataset_sliced:
            del dataset_sliced[var].encoding['chunks']

        step = 192
        data_len = len(dataset['data'])
        for i in range(0, len(dataset['data']), step):
            if i==0 and year==start_year:
                dataset_sliced.isel(time=slice(0, step)).chunk({'y':800, 'x':1200, 'time':4}).to_zarr(save_path, mode='w')
            else:
                dataset_sliced.isel(time=slice(i, i+step)).chunk({'y':800, 'x':1200, 'time':4}).to_zarr(save_path, append_dim='time')
            print(100*(i+1)/data_len, '%', '  ', "total_cpu_usage", psutil.cpu_percent(interval=1))
        print()
        print("################ FINISHED YEAR {} ######################".format(year))
        print()
if __name__ == '__main__':
    main()