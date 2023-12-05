import itertools
from typing import Any, Tuple
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, Subset
import xarray
from datetime import timedelta
# import dask
# from xrpatcher import XRDAPatcher
from xbatcher import BatchGenerator
from glob import glob
from preprocess.etc import benchmark
import pytorch_lightning as L
import torchvision



import xbatcher
from xbatcher.loaders.torch import MapDataset as xbMapDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def train_test_split( ds):
        timeindex = ds.time.indexes['time'] # returns a pandas time index useful for the manipulation

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
            start = datetime(month.year, month.month, 1)
            end = last_day_of_month(start) + timedelta(
                hours=23, minutes=59, seconds=59
            )
            test = end - timedelta(days=7)


            train_ls.append(slice(start, test))
            test_ls.append(slice(test, end))

        ds_train = ds.sel(time=train_ls)
        ds_test = ds.sel(time=test_ls)

        return ds_train, ds_test

class MSGDataset(Dataset):
    def __init__(self, dataset, y_var:str, x_vars=None, rechunk=None):

        if isinstance(dataset, str):
            self.ds = xarray.open_zarr(dataset)
        elif isinstance(dataset, xarray.Dataset):
            self.ds = dataset
        else:
            raise ValueError(f'{dataset} is not a zarr store or a xarray.Dataset')
            
        self.attributes = self.ds.attrs

        vars = set(self.ds.keys())
        ignore_vars = set(['lon_bnds', 'lat_bnds', 'record_status', 'crs'])
        assert y_var in vars, 'y_var not available in the dataset'
        self.y_var = y_var
        self.x_vars = vars - ignore_vars - set([y_var,])

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        if self.ds is not None:
            xarr = self.ds[self.x_vars].isel(time=idx)[['time', 'lon', 'lat']].to_array().squeeze(dim='variable') # select time slice, reshape to t lon lat
            yarr = self.ds[self.y_var].isel(time=idx)[['time', 'lon', 'lat']].to_array().squeeze(dim='variable')
            x = torch.tensor(data=xarr.data, names=tuple(xarr.sizes))
            y = torch.tensor(data=yarr.data, names=tuple(xarr.sizes))      
            return x, y
        else:
            raise ValueError('There is no zarr dataset to get images from.')
        
class MSGDatasetBatched(Dataset):
    def __init__(self, generator, y_var:str, x_vars:[str]=None, transform=None, target_transform=None):

        self.generator = generator
        self.transform = transform
        self.target_transform = target_transform
        self.ds = generator.ds
        self.attributes = self.ds.attrs
        assert y_var in vars, 'y_var not available in the dataset'
        self.y_var = y_var

        vars = set(self.ds.keys())
        ignore_vars = set(['lon_bnds', 'lat_bnds', 'record_status', 'crs'])
        other_vars = vars - ignore_vars - set([y_var,])
        if x_vars is not None:
            assert set(x_vars) in other_vars, f'{set(x_vars)^other_vars} not in the dataset'
            self.x_vars = x_vars
        else:
            self.x_vars = other_vars

    def __len__(self) -> int:
        return len(self.generator)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        X_batch = self.generator[idx][self.x_vars].torch.to_named_tensor()
        y_batch = self.y_generator[idx][self.y_var].torch.to_named_tensor()

        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch



class MSGDataModule(L.LightningDataModule):
    def __init__(self, zarr_store:str, batch_size: int = 32, patch_size=None, num_workers=12):
        super().__init__()
        # self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.zarr_store = zarr_store

        ## Do not rechunk the dataset anymore
        # if self.patch_size:
        #     self.rechunk = {'time':self.batch_size, 'lat':self.patch_size[1], 'lon':self.patch_size[0]}
        # else:
        #     self.rechunk = None

    def setup(self, stage: str ):
        if stage == 'fit':
            with benchmark('initialize MSGdataset full test'):
                self.msg_full = MSGDataset(
                    dataset=self.zarr_store,
                    y_var='SIS',
                )


            with benchmark('train_test_split'):
                train_ds = self.msg_full.ds.isel(time=slice(datetime(2015,1,1,0,0), None))
                self.msg_train_ds, self.msg_validation_ds = train_test_split(self.msg_full.ds)
            
            train_ds, valid_ds = train_test_split(self.msg_full.ds)
            if self.patch_size is not None:
                traingenerator = BatchGenerator(ds=train_ds, 
                                           input_dims={'lat':self.patch_size[1], 'lon':self.patch_size[0]}
                            )
                validgenerator = BatchGenerator(ds=valid_ds, 
                                           input_dims={'lat':self.patch_size[1], 'lon':self.patch_size[0]}
                            )
                 
                self.msg_train = MSGDatasetBatched(generator=traingenerator, y_var='SIS')
                self.msg_test = MSGDatasetBatched(generator=validgenerator, y_var='SIS')
            else:
                with benchmark('patch train'):
                    self.msg_train = MSGDataset(self.msg_train_ds)
                with benchmark('patch validation'):
                    self.msg_validation = MSGDataset(self.msg_validation_ds)

        if stage == 'test':
            test_ds = self.msg_full.ds.sel(time=slice(datetime(2014,1,1,0,0), datetime(2014,12,31,23,59)))
            if self.patch_size is not None:
                testgenerator = BatchGenerator(ds=test_ds, 
                                               input_dims={'lat':self.patch_size[1], 'lon':self.patch_size[0]}
                                               )
                self.msg_test = MSGDatasetBatched(generator=testgenerator, y_var='SIS')

    def prepare_data(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.msg_train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, prefetch_factor=3)

    def val_dataloader(self):
        return DataLoader(self.msg_validation, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, prefetch_factor=2)

    def test_dataloader(self):
        return DataLoader(self.msg_test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)




if __name__ == '__main__':
    filenames = ['/scratch/snx3000/kschuurm/DATA/HRSEVIRI.zarr']

    patch_size = dict(lat=128, lon=128)

    dm = MSGDataModule(batch_size=32, patch_size=(128,128), num_workers=7)
    dm.setup('fit')
    dl = dm.train_dataloader()
    
    samples = []
    for i, sample in enumerate(dl):
        
        print(sample.shape)

        if i >= 4:
            # Take only five samples for illustration purposes
            break

    samples = torch.hstack(samples)

    plt.imshow(samples.to_numpy())
    plt.savefig('testsamples.png')
