from typing import Tuple
import numpy as np
import pandas as pd
from preprocess.sza import solarzenithangle
import torch
import xarray
from xarray.core.dataset import Dataset
from xbatcher import BatchGenerator

from dataset.dataset import MSGDatasetPoint, MSGDatasetBatched
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from xrpatcher import XRDAPatcher
from tqdm import tqdm


class RollingGenerator:
    def __init__(self, ds, input_dims):
        keys = input_dims.keys()
        self.input_dims = input_dims
        self.ds = ds
        self.batches = (self.ds
            .rolling(self.input_dims, min_periods=None)
            .construct({key: key + '_batch' for key in keys})
            .stack({'batch': list(keys)})
            .dropna(dim='batch', subset=['channel_1'])
        )
    def __getitem__(self, idx):
        return self.batches.isel(batch=idx)
    
    def __len__(self):
        return len(self.batches) ## fix needed

class MSGSingleImageDataset(MSGDatasetPoint):
    def __init__(
        self,
        ds,
        input_dims={'lat':15, 'lon':15},
        batch_dims={'lat':400},
        y_vars=['SIS'],
        x_vars=None,
        x_features='all',
        transform=None,
        target_transform=None
    ):
        self.input_dims = input_dims
        self.batch_dims = {'lon':10}
        self.input_overlap = {'lat':self.input_dims['lat']-1, 'lon':self.input_dims['lon']-1}
        self.ds = ds


        # self.generator = BatchGenerator(
        #     ds=self.ds, 
        #     input_dims=self.input_dims, 
        #     input_overlap=self.input_overlap,
        #     # batch_dims=self.batch_dims,
        #     preload_batch=True,
        #     )

        self.generator = RollingGenerator(self.ds, self.input_dims)
        
        self.transform = transform
        self.target_transform = target_transform
        self.filter = None
        self.x_features = x_features

        self.y_vars = y_vars
        self.x_vars = x_vars

        self.check_vars()

    
    def __getitem__(self, idx):
        X, x_attributes, y = super().__getitem__(idx)
        X = X.squeeze(dim=0)
        x_attributes = x_attributes.squeeze(dim=0)
        y = y.squeeze(dim=0)
        return X, x_attributes, y

    def get_dataset_batch(self, idx) -> Tuple[xarray.Dataset, xarray.Dataset]:
        """returns a batch in xarray.DataArray form"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.get_dataset_batch currently requires a single integer key"
                )

        ds = self.generator[idx]
        if self.filter:
            ds = self.apply_filter(ds)
        X_batch = ds[self.x_vars]
        y_batch = ds[self.y_vars] 
        x_attributes = self.get_point_features(X_batch)
        if self.x_features != 'all':
            x_attributes = x_attributes[self.x_features]

        if self.input_dims['lat'] % 2 == 0:
            idx = (int(self.input_dims['lat'] / 2 - 1), int(self.input_dims['lat'] / 2))
            y_batch_point = (
                y_batch.isel(
                    lat=slice(idx[0], idx[1] + 1), lon=slice(idx[0], idx[1] + 1)
                )
                .mean(dim=("lat_batch", "lon_batch"))
            )
        else:
            idx = int(self.input_dims['lat'] / 2)
            y_batch_point = y_batch.isel(lat_batch=idx, lon_batch=idx)

        if self.transform:
            X_batch = self.transform(X_batch)
            x_attributes = self.transform(x_attributes)

        if self.target_transform:
            y_batch_point = self.target_transform(y_batch_point)

        return X_batch, x_attributes, y_batch_point
    
    def get_point_features(self, X_batch_ds):

        features = X_batch_ds.mean(['lat_batch', 'lon_batch'])
        lat = X_batch_ds.lat.item()
        lon = X_batch_ds.lon.item()

        # If SRTM is not in the x_var take it from the nearest pixel in the original dataset
        if "SRTM" in set(X_batch_ds.keys()):
            srtm = features.SRTM.item()
        else:
            srtm = self.ds.SRTM.sel(lat=lat, lon=lon, method="nearest").values.item()

        datetimes = pd.to_datetime(features.time)
        batch_size = len(datetimes) # batchsize could have changed when applying the filter
        dayofyear = features.time.dt.dayofyear.values
        sza, azi = solarzenithangle(
            datetimes, lat, lon, srtm if not np.isnan(srtm) else 0
        )  # Assume altitude is zero at nans for sza calc

        lat = np.repeat(lat, batch_size)
        lon = np.repeat(lon, batch_size)
        srtm = np.repeat(srtm, batch_size)

        features = xarray.Dataset(
            data_vars={
                "dayofyear": (('time'), dayofyear),
                "lat": (('time'), lat),
                "lon": (('time'), lon),
                "SRTM": (('time'), srtm),
                "SZA": (('time'), sza),
                "AZI": (('time'), azi),
            },
            coords={
                'sample': (('time'), features.time.data)
            }
        )

        return features  # (B, F) , F columns: [dayofyear, lat, lon, srtm, sza]



def point_prediction_to_image(model, y_hat, lat, lon):

    y_hat = model.dm.target_transform.inverse(y_hat, model.dm.y_vars)
    lat = model.dm.transform.inverse(lat, ['lat'])
    lon = model.dm.transform.inverse(lon, ['lon'])


    output_arr = xarray.DataArray(data=(('lat', 'lon'), ys), coords={'lat':lat, 'lon':lon})


    return output_arr



if __name__ == '__main__':

    ds = xarray.open_zarr('/scratch/snx3000/kschuurm/DATA/valid.zarr')
    
    ds_slice = ds.sel(time=[datetime(2015,6,1,13,1)], method='nearest').load()

    dataset = MSGSingleImageDataset(
        ds_slice, 
        input_dims={'lat':15, 'lon':15}, 
        x_vars=['channel_1'],
    )

    # X, x ,y = dataset[0]

    # dataloader= DataLoader(dataset, batch_size=800, num_workers=12)

    # for i, batch in enumerate(tqdm(dataloader)):
    #     X, x, y = batch
    #     pass
    #     # print(X.shape, x.shape, y.shape)
    #     # if i > 30:
    #     #     break

    # patcher = XRDAPatcher(ds_slice.to_dataarray(), patches={'lat':15, 'lon':15}, strides={'lat':1, 'lon':1}, check_full_scan=True)
    # xrdataset = XrTorchDataset(patcher)

    # dataloader2 = DataLoader(xrdataset, batch_size=800, num_workers=12)
    # for i, batch in enumerate(tqdm(dataloader2)):
    #     pass
    #     # print(batch.shape)