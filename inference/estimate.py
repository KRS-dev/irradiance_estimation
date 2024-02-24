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
        self.ds_rolled = (self.ds
            .rolling(self.input_dims, min_periods=None, center=True)
            .construct({key: key + '_batch' for key in keys})
            .stack({'batch': list(keys)}, create_index=False)
            .dropna(dim='batch', subset=['channel_1'])
        )
        
    def __getitem__(self, idx):
        return self.ds_rolled.isel(batch=idx)
    
    def __len__(self):
        return self.ds_rolled.sizes['batch'] ## fix needed

class MSGSingleImageDatasetOld(MSGDatasetPoint):
    def __init__(
        self,
        ds,
        input_dims={'lat':15, 'lon':15},
        y_vars=['SIS'],
        x_vars=None,
        x_features=['dayofyear', 'lat', 'lon','SZA', 'AZI', 'SRTM'],
        transform=None,
        target_transform=None
    ):
        self.input_dims = input_dims
        self.batch_dims = {'lon':10}
        self.input_overlap = {'lat':self.input_dims['lat']-1, 'lon':self.input_dims['lon']-1}
        self.ds = ds


        self.generator = RollingGenerator(self.ds, self.input_dims)
        
        self.transform = transform
        self.target_transform = target_transform
        self.filter = None
        self.x_features = x_features
        self.x_dim_features = ['dayofyear', 'lat', 'lon']

        self.y_vars = y_vars
        self.x_vars = x_vars

        # self.check_vars()

    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns a batch in torch tensors form"""


        X_batch, x_attributes, y = self.get_xarray_batch(idx)
        X_batch = X_batch.fillna(-99)  # for training we do not want any nans. 
        x_attributes = x_attributes.fillna(-99) #Very negative values should cancel with ReLU
        # if y.isnull().any():
        #     print('nan in y')

        # y = y.fillna(-99)

        X_batch = torch.tensor(X_batch.values).squeeze()  # , names=('C', 'B', 'X', 'Y'))
        x_attributes = torch.tensor(x_attributes.values).squeeze()  # ('C', 'B')
        y = torch.tensor(y.values).squeeze()  #  ('C','B')

        return X_batch, x_attributes, y

    def get_xarray_batch(self, idx) -> Tuple[xarray.DataArray, xarray.DataArray]:
        """returns a batch in xarray.DataArray form"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.get_xarray_batch currently requires a single integer key"
                )

        X_batch, x_attributes, y_batch_point = self.get_dataset_batch(idx)
        
        X_batch = X_batch.to_dataarray(dim="channels")
        x_attributes = x_attributes.to_dataarray(dim="channels")
        y_batch_point = y_batch_point.to_dataarray(dim="channels")

        return X_batch, x_attributes, y_batch_point

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

        # if self.filter:
        #     ds = self.apply_filter(ds)

        X_batch = ds[self.x_vars]
        y_batch = ds[self.y_vars] 

        x_attributes = self.get_point_features(ds)

        y_batch_point = y_batch.isel(
            lat_batch = (self.input_dims['lat'] + 1)//2,
            lon_batch = (self.input_dims['lon'] + 1)//2,
        )

        if self.transform:
            X_batch = self.transform(X_batch)
            x_attributes = self.transform(x_attributes)

        if self.target_transform:
            y_batch_point = self.target_transform(y_batch_point)

        return X_batch, x_attributes, y_batch_point
    
    def get_point_features(self, X_batch_ds):
        
        features_ = set(self.x_features) - set(['dayofyear', 'lat', 'lon'])
        features = X_batch_ds[features_].isel(
            lat_batch = (self.input_dims['lat'] + 1)//2,
            lon_batch = (self.input_dims['lon'] + 1)//2,
        )
        
        if 'SRTM' in features:
            features['SRTM'] = features['SRTM'].fillna(0)
        

        # batch_size = len(datetimes) # batchsize could have changed when applying the filter
        dayofyear = features.time.dt.dayofyear.values

        if 'dayofyear' in self.x_features:
            features = features.assign({'dayofyear':  dayofyear})
        if 'lat' in self.x_features:
            features = features.rename(name_dict={'lat':'lat_', 'lon':'lon_'})
            features = features.assign({'lat': features['lat_'],
                                        'lon': features['lon_']})
        

        return features[self.x_features]  # (B, F) , F columns: [dayofyear, lat, lon, srtm, sza




def point_prediction_to_image(model, y_hat, lat, lon):

    y_hat = model.dm.target_transform.inverse(y_hat, model.dm.y_vars)
    lat = model.dm.transform.inverse(lat, ['lat'])
    lon = model.dm.transform.inverse(lon, ['lon'])


    output_arr = xarray.DataArray(data=(('lat', 'lon'), y_hat), coords={'lat':lat, 'lon':lon})


    return output_arr





if __name__ == '__main__':

    ds = xarray.open_zarr('/scratch/snx3000/kschuurm/DATA/valid.zarr')
    
    ds_slice = ds.sel(time=[datetime(2015,6,1,13,1)], method='nearest').load()

    dataset = MSGSingleImageDataset(
        ds_slice, 
        input_dims={'lat':15, 'lon':15}, 
        x_vars=["SRTM", "channel_1", "channel_2", "channel_3", "channel_4", "channel_5", "channel_6", "channel_7", "channel_8", "channel_9", "channel_10", "channel_11"],
        y_vars=['SIS', 'CAL'],
    )

    X, x ,y = dataset[0]

    dataloader= DataLoader(dataset, batch_size=100, num_workers=1)

    for i, batch in enumerate(tqdm(dataloader)):
        X, x, y = batch
        print(X.shape, x.shape, y.shape)
        if i > 30:
            break

    # patcher = XRDAPatcher(ds_slice.to_dataarray(), patches={'lat':15, 'lon':15}, strides={'lat':1, 'lon':1}, check_full_scan=True)
    # xrdataset = XrTorchDataset(patcher)

    # dataloader2 = DataLoader(xrdataset, batch_size=800, num_workers=12)
    # for i, batch in enumerate(tqdm(dataloader2)):
    #     pass
    #     # print(batch.shape)