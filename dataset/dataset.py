import itertools
import torch
import pandas as pd
from datetime import datetime
# from dask_jobqueue import SLURMCluster
from torch.utils.data import DataLoader, Dataset, Subset
import xarray
from datetime import timedelta
import dask
from xrpatcher import XRDAPatcher

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
    def __init__(self, train_paths, sarah_paths, rechunk=None, patch_size = (32,32)):
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
            train_paths,
            parallel=True,
            chunks=rechunk,
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            engine="h5netcdf",
        )
        self.attributes = self.x.attrs

        self._sarah = xarray.open_mfdataset(
            sarah_paths,
            parallel=True,
            chunks=rechunk,
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
                    set(self._sarah.time.dt.round("30min").values)
                )
            )
        )

        self._sarah = self._sarah.reindex(
            time=a, lat=self.x.lat, lon=self.x.lon, method="nearest"
        )
        self._sarah = self._sarah.drop(["lon_bnds", "lat_bnds", "record_status"])
        self.x = self.x.reindex(time=a, method="nearest")
        self.x = self.x.drop(["crs"])
        self.y = self._sarah

        self.patch_size = patch_size
        if self.patch_size is not None:

            xy = xarray.merge([self.x, self.y], 'equals').to_array()
            patches = dict(time=1, lon=self.patch_size[0], lat=self.patch_size[1])
            self.patcher = XRDAPatcher(
                da = xy,
                patches=patches,
                strides=patches
            )
        # self.cluster.close()

    def __len__(self):
        if self.patch_size is None:
            return len(self.y)
        else:
            return len(self.patcher)
    
    def reconstruct_from_batches(self, batches, **rec_kws):
        if self.patch_size is not None:
            return self.batcher.reconstruct([*itertools.chain(*batches)], **rec_kws)

    def __getitem__(self, idx):
        if self.patch_size is None:
            print('nopatch')
            xarr = self.x.isel(time=idx).to_array()
            yarr = self.y.isel(time=idx).to_array()
            x = torch.as_tensor(xarr.values)
            y = torch.as_tensor(yarr.values)
        else:
            print('patch')
            xy = self.patcher[idx].isel(time=0).load()
            xarr, yarr = xy[:-1], xy[-1]
            x = torch.as_tensor(xarr.values)
            y = torch.as_tensor(yarr.values)

        return x, y


class MSGDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, patch_size=None):
        super().__init__()
        # self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size

    def setup(self, stage: str):
        self.msg_full = MSGDataset(
            "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_201501*",
            "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2015.nc",
            patch_size = self.patch_size
        )
        self.msg_test = MSGDataset(
            "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_201401*",
            "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2014.nc",
            patch_size = self.patch_size
        )

        self.msg_train, self.msg_validation = train_test_split(
            self.msg_full, self.msg_full.y.indexes["time"]
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

    dm = MSGDataModule(patch_size=(64, 64))
    dm.setup("fit")
    print(len(dm.msg_train))
    print(len(dm.msg_validation))

    for x,y in dm.msg_train:
        print('Feature data shape:', x.shape)
        print('Output data shape:', y.shape)
        break
