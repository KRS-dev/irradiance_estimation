import xarray
import zarr
from dask.distributed import Client
if __name__ == '__main__':
    with Client() as client:
        print(client)

        hres = xarray.open_zarr('HRSEVIRI_30chunked.zarr')

        store = zarr.ZipStore('HRSEVIRI_30chunked.zip', mode='w')
        hres.to_zarr(chunk_store=store)
