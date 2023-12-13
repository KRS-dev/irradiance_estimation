import pandas as pd
from datetime import datetime, timedelta
import xarray
from dask.distributed import Client

import random

random.seed(0)


def train_test_split(ds):
    def last_day_of_month(any_day):
        # The day 28 exists in every month. 4 days later, it's always next month
        next_month = any_day.replace(day=28) + timedelta(days=4)
        # subtracting the number of the current day brings us back one month
        return next_month - timedelta(days=next_month.day)

    timeindex = ds.indexes["time"]
    dt_start = ds.time.min().values
    dt_end = ds.time.max().values
    month_dr = pd.date_range(start=dt_start, end=dt_end, freq="M")  # monthly daterange

    train_ls = []
    test_ls = []

    for month in month_dr:
        start = datetime(month.year, month.month, 1)
        end = last_day_of_month(start) + timedelta(hours=23, minutes=59, seconds=59)
        test = end - timedelta(days=7)

        idxstart = timeindex.get_slice_bound(start, "left")
        idxtest = timeindex.get_slice_bound(test, "left")
        idxend = timeindex.get_slice_bound(end, "left")

        train_ls.extend([i for i in range(idxstart, idxtest)])
        test_ls.extend([i for i in range(idxtest, idxend)])

        # train_ls.append(slice(start, test)) # first ~3 weeks
        # test_ls.append(slice(test, end)) # last week of the month

    ds_train = ds.isel(time=train_ls)
    ds_test = ds.isel(time=test_ls)

    return ds_train, ds_test


def shuffle_timeindex(ds):
    num = len(ds.time)
    samples = random.sample(range(num), num)

    return ds.isel(time=samples)


def chunk_zarr(zarr_store, patch_size, batch_size):
    full_ds = xarray.open_zarr(zarr_store)

    for var in full_ds:
        del full_ds[var].encoding["chunks"]

    i = len(full_ds.time)
    full_ds = full_ds.dropna(dim="time", how="all", subset=["SIS"])
    x_vars = set(full_ds.variables) - {
        "lat_bnds",
        "lon_bnds",
        "lat",
        "lon",
        "time",
        "SIS",
        "record_status",
    }
    for var in x_vars:
        full_ds = full_ds.dropna(dim="time", how="all", subset=[var])

    print("dropped", len(full_ds.time) / i)

    half_ds = full_ds.sel(
        time=slice(datetime(2015, 1, 1, 0, 0), None)
    )  # 2014 is used as test
    test_ds = full_ds.sel(
        time=slice(datetime(2014, 1, 1, 0, 0, 0), datetime(2015, 1, 1, 0, 0))
    )

    train_ds, valid_ds = train_test_split(half_ds)

    print("split")
    train_ds = shuffle_timeindex(train_ds)
    print(train_ds.time.values)
    print("shuffle")

    train_ds.chunk(
        {"time": batch_size, "lat": patch_size[1], "lon": patch_size[0]}
    ).to_zarr("/scratch/snx3000/kschuurm/DATA/train.zarr", mode="w", safe_chunks=True)
    valid_ds.chunk(
        {"time": batch_size, "lat": patch_size[1], "lon": patch_size[0]}
    ).to_zarr("/scratch/snx3000/kschuurm/DATA/valid.zarr", mode="w", safe_chunks=True)
    test_ds.chunk(
        {"time": batch_size, "lat": patch_size[1], "lon": patch_size[0]}
    ).to_zarr("/scratch/snx3000/kschuurm/DATA/test.zarr", mode="w", safe_chunks=True)
    print("done")


if __name__ == "__main__":
    with Client() as client:
        print(client)

        zarr_store = "/scratch/snx3000/kschuurm/DATA/HRSEVIRI.zarr"

        chunk_zarr(zarr_store, (64, 64), 254)
