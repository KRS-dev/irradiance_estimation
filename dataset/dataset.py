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

import pytorch_lightning as L

dask.config.set(**{"array.slicing.split_large_chunks": False})

class MSGDataset(Dataset):
    def __init__(self, train_paths, sarah_paths, rechunk=None):

        if train_paths is not None and sarah_paths is not None:
        # self.cluster = SLURMCluster(
        #     cores=12,
        #     memory='61GB',
        #     account='go41',
        #     walltime='0:10:00',
        #     job_extra_directives=['-C gpu', '--reservation=interact_gpu']
        #     )


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

        # self.cluster.close()

    def __len__(self):
        return len(self.y.time)

    def __getitem__(self, idx):
        xarr = self.x.isel(time=idx).to_array()
        yarr = self.y.isel(time=idx).to_array()
        x = torch.as_tensor(xarr.values).reshape(-1, xarr.shape[-2], xarr.shape[-1])
        y = torch.as_tensor(yarr.values).reshape(-1, xarr.shape[-2], xarr.shape[-1])         
        return x, y
    
    @classmethod
    def _from_xy(cls, x,y):
        msgdataset = cls(None, None)
        msgdataset.x = x
        msgdataset.y = y
        return msgdataset


    
class MSGDatasetPatched(MSGDataset):
    def __init__(self, patch_size = (32,32), msgdataset=None, *args, **kwargs):
        if msgdataset is not None:
            self.x = msgdataset.x
            self.y = msgdataset.y
        else:
            super().__init__(*args, **kwargs)

        self.patch_size = patch_size
        xy = xarray.merge([self.x, self.y], 'equals').to_array()
        patches = dict(time=1, lon=self.patch_size[0], lat=self.patch_size[1])
        self.patcher = XRDAPatcher(
            da = xy,
            patches=patches,
            strides=patches
        )

    def reconstruct_from_batches(self, batches, **rec_kws):
        if self.patch_size is not None:
            return self.batcher.reconstruct([*itertools.chain(*batches)], **rec_kws)
    
    def __len__(self):
        return len(self.patcher)
    
    def __getitem__(self, idx):
        xy = self.patcher[idx].isel(time=0).load()
        xarr, yarr = xy[:-1], xy[-1]
        x = torch.as_tensor(xarr.values).reshape(-1, self.patch_size[0], self.patch_size[1])
        y = torch.as_tensor(yarr.values).reshape(-1, self.patch_size[0], self.patch_size[1])
        return x, y
    
class SubsetAttr(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

        for attribute_key in dataset.__dict__.keys():
            self.__dict__[attribute_key] = dataset.__dict__[attribute_key]
    

class MSGDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, patch_size=None, num_workers=12):
        super().__init__()
        # self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers

    def setup(self, stage: str):

        self.msg_full = MSGDataset(
            "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_201501*",
            "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2015.nc",
        )
        self.msg_test = MSGDataset(
            "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_201401*",
            "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2014.nc",
        )
        self.msg_train, self.msg_validation = train_test_split(
            self.msg_full, self.msg_full.y.indexes["time"]
        )
        if self.patch_size is not None:
            self.msg_train = MSGDatasetPatched(msgdataset=self.msg_train, patch_size=self.patch_size)
            self.msg_validation = MSGDatasetPatched(msgdataset=self.msg_validation, patch_size=self.patch_size)

    def train_dataloader(self):
        return DataLoader(self.msg_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.msg_validation, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.msg_test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

def train_test_split( msgdataset, timeindex):
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

            train_ls.extend([i for i in range(idxstart, idxtest)])
            test_ls.extend([i for i in range(idxtest, idxend)])

        xtrain, ytrain = msgdataset.x.isel(time=train_ls), msgdataset.y.isel(time=train_ls)
        xval, yval = msgdataset.x.isel(time=test_ls), msgdataset.y.isel(time=test_ls)

        return MSGDataset._from_xy(xtrain, ytrain), MSGDataset._from_xy(xval, yval)
    

if __name__ == "__main__":
    train_path = "/scratch/snx3000/kschuurm/DATA/customized/HRSEVIRI_2014*"
    sarah_path = "/scratch/snx3000/kschuurm/DATA/SARAH3/SIS_2014.nc"

    dm = MSGDataModule(patch_size=(64, 64))
    dm.setup("fit")

    for x,y in dm.msg_train:
        print('Feature data shape:', x.shape)
        print('Output data shape:', y.shape)
        break
