import os
import torch
import pandas as pd
from datetime import datetime
from dask_jobqueue import SLURMCluster
from torch.utils.data import DataLoader, Dataset, Subset
import xarray
from datetime import timedelta
import dask

import lightning as L

dask.config.set(**{"array.slicing.split_large_chunks": False})


def train_test_split(msgdataset, timeindex):
    def last_day_of_month(any_day):
        # The day 28 exists in every month. 4 days later, it's always next month
        next_month = any_day.replace(day=28) + timedelta(days=4)
        # subtracting the number of the current day brings us back one month
        return next_month - timedelta(days=next_month.day)

    dt_start = timeindex.min()
    dt_end = timeindex.max()
    month_dr = pd.date_range(start=dt_start, end=dt_end, freq="M")

    train_ls = []
    test_ls = []

    for month in month_dr:
        slice_start = datetime(month.year, month.month, 1)
        slice_end = last_day_of_month(slice_start) + timedelta(
            hours=23, minutes=59, seconds=59
        )
        slice_test = slice_end - timedelta(days=7)

        idxstart = timeindex.get_slice_bound(slice_start, "left")
        idxtest = timeindex.get_slice_bound(slice_test, "left")
        idxend = timeindex.get_slice_bound(slice_end, "left")

        train_ls.extend(range(idxstart, idxtest))
        test_ls.extend(range(idxtest, idxend))

    return Subset(msgdataset, train_ls), Subset(msgdataset, test_ls)


class MSGDataset(Dataset):
    def __init__(self, train_paths, sarah_paths):
        # self.cluster = SLURMCluster(
        #     cores=12,
        #     memory='61GB',
        #     account='go41',
        #     walltime='0:10:00',
        #     job_extra_directives=['-C gpu', '--reservation=interact_gpu']
        #     )

        # print(self.cluster)

        # self.cluster.scale(4)

        self.x = xarray.open_mfdataset(
            train_path,
            parallel=True,
            # chunks={'time':100, 'lat':-1, 'lon':-1},
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            engine="h5netcdf",
        )

        self.sarah = xarray.open_mfdataset(
            sarah_path,
            parallel=True,
            # chunks={'time':48, 'lat':-1, 'lon':-1},
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            engine="h5netcdf",
        )

        a = sorted(
            list(
                set(self.x.time.dt.round("30min").values).intersection(
                    set(self.sarah.time.dt.round("30min").values)
                )
            )
        )

        self.sarah = self.sarah.reindex(
            time=a, lat=self.x.lat, lon=self.x.lon, method="nearest"
        )
        self.sarah = self.sarah.drop(["lon_bnds", "lat_bnds", "record_status"])
        self.x = self.x.reindex(time=a, method="nearest")
        self.x = self.x.drop(["crs"])

        self.attributes = self.x.attrs

        self.output = self.sarah
        # self.cluster.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        trainarr = self.train.isel(time=idx).to_array()
        outputarr = self.output.isel(time=idx).to_array()
        x = torch.as_tensor(trainarr.values)
        y = torch.as_tensor(outputarr.values)
        return x, y


class MSGDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        # self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.msg_full = MSGDataset(
            "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_2015*",
            "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2015.nc",
        )
        self.msg_test = MSGDataset(
            "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_2014*",
            "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2014.nc",
        )

        self.msg_train, self.msg_validation = train_test_split(
            self.msg_full, self.msg_full.output.indexes["time"]
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


if __name__ == "__main__":
    train_path = "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_2014*"
    sarah_path = "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2014.nc"

    dataloader = MSGDataModule()
    dataloader.setup("fit")
    print(len(dataloader.msg_train))
    print(len(dataloader.msg_validation))
