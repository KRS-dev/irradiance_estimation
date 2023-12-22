import itertools
from typing import Any, Callable, Dict, Hashable, Tuple
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from tqdm import tqdm
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import xarray
from datetime import timedelta
from xbatcher import BatchGenerator
from preprocess.etc import benchmark
import lightning.pytorch as L
import numpy as np
import pandas as pd
from preprocess.sza import solarzenithangle

import xbatcher
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




class MSGDataset(Dataset):
    def __init__(
        self, zarr_store, y_vars, x_vars=None, transform:Callable=None, target_transform:Callable=None, filter:Callable[[xarray.Dataset],xarray.Dataset]=None
    ):
        super().__init__()
        if isinstance(zarr_store, str):
            self.ds = xarray.open_zarr(zarr_store)
        elif isinstance(zarr_store, xarray.Dataset):
            self.ds = zarr_store
        else:
            raise ValueError(f"{zarr_store} is not a zarr store or a xarray.Dataset.")

        self.attributes = self.ds.attrs

        self.transform = transform
        self.target_transform = target_transform
        self.filter = filter

        self.y_vars = y_vars
        self.x_vars = x_vars

        self.check_vars()

        
    def check_vars(self):
        vars = set(self.ds.keys())
        assert set(self.y_vars).issubset(vars), f"y_vars: {set(self.y_vars) - vars} not available in the dataset"
        ignore_vars = set(["lon_bnds", "lat_bnds", "record_status", "crs"])
        other_vars = (vars - ignore_vars - set(self.y_vars))
        if self.x_vars is not None:
            assert set(self.x_vars).issubset(vars), f"{set(self.x_vars) - vars} not in the dataset"
        else:
            self.x_vars = other_vars

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        X, y = self.get_xarray(idx)
        X = torch.tensor(data=X.values)  # (C, X, Y)
        y = torch.tensor(data=y.values)  # (X, Y)
        return X, y

    def get_xarray(self, idx):
        X, y = self.get_dataset(idx)
        X = X.to_dataarray(dim='channels') 
        y = y.to_dataarray(dim='channels')
        return X, y

    def get_dataset(self, idx):
        
        ds = self.ds
        if self.filter:
            ds = self.apply_filter(ds)

        X = ds[self.x_vars].isel(time=idx)  # select time slice, reshape to t lon lat
        y = ds[self.y_vars].isel(time=idx)
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y
    
    def apply_filter(self, ds):
        return self.filter(ds)

class MSGDatasetBatched(MSGDataset):
    def __init__(
        self,
        patch_size=(64, 64),
        input_overlap={},
        batch_dims={'time':10},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.input_dims = {'lat':self.patch_size[1], 'lon':self.patch_size[0]}
        self.batch_dims = batch_dims
        self.input_overlap = input_overlap
        self.generator = BatchGenerator(
            self.ds,
            input_dims=self.input_dims,
            batch_dims=self.batch_dims,
            input_overlap=self.input_overlap,
            # concat_input_dims=True,
        )

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

        X_batch_ds, y_batch_ds = self.get_dataset_batch(idx)
        X_batch_ds = X_batch_ds.fillna(
            -99999
        )  # for training we do not want any nans. Very negative values should cancel with ReLU
        X_batch = torch.tensor(
            X_batch_ds.to_dataarray(dim="channels").values
        )  # , names=('C','B', 'X', 'Y'))
        y_batch = torch.tensor(
            y_batch_ds.to_dataarray(dim="channels").values
        )  # , names=('B', 'X', 'Y'))

        X_batch = X_batch.permute(1, 0, 2, 3) 
        y_batch = y_batch.permute(1, 0, 2, 3)

        return X_batch, y_batch  # ('B', 'C', 'X', 'Y') &  ('B', 'X', 'Y')

    def get_xarray_batch(self, idx) -> Tuple[xarray.DataArray, xarray.DataArray]:
        """returns a batch in xarray.DataArray form"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        X_batch, y_batch = self.get_dataset_batch(idx)

        # ('C', 'B', 'X', 'Y') &  ('B', 'X', 'Y')
        return X_batch.to_dataarray(dim="channels"), y_batch.to_dataarray(dim="channels")  

    def get_dataset_batch(self, idx) -> Tuple[xarray.Dataset, xarray.Dataset]:
        """returns a batch in xarray.DataArray form"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        ds = self.generator[idx]
        if self.filter:
            ds = self.apply_filter(ds)
        X_batch = ds[self.x_vars]
        y_batch = ds[self.y_vars] # with a single y_vars it becomes a dataarray automatically

        if self.transform:
            X_batch = self.transform(X_batch)
        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        return X_batch, y_batch


class MSGDatasetPoint(MSGDatasetBatched):
    def __init__(self, x_features = 'all', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_features = x_features

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns a batch in torch tensors form"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        X_batch, x_attributes, y = self.get_xarray_batch(idx)
        X_batch = X_batch.fillna(-99)  # for training we do not want any nans. 
        x_attributes = x_attributes.fillna(-99) #Very negative values should cancel with ReLU
        if y.isnull().any():
            print('nan in y')

        # y = y.fillna(-99)

        X_batch = torch.tensor(X_batch.values)  # , names=('C', 'B', 'X', 'Y'))
        x_attributes = torch.tensor(x_attributes.values)  # ('C', 'B')
        y = torch.tensor(y.values)  #  ('C','B')

        X_batch = X_batch.permute(1, 0, 2, 3)  # ('B', 'C', 'X', 'Y')
        x_attributes = x_attributes.permute(1, 0) # ('B', 'C')
        y = y.permute(1,0) # ('B', 'C')

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
                .mean(dim=("lat", "lon"))
            )
        else:
            idx = int(self.input_dims['lat'] / 2)
            y_batch_point = y_batch.isel(lat=idx, lon=idx)

        if self.transform:
            X_batch = self.transform(X_batch)
            x_attributes = self.transform(x_attributes)

        if self.target_transform:
            y_batch_point = self.target_transform(y_batch_point)

        return X_batch, x_attributes, y_batch_point
    
    def get_point_features(self, X_batch_ds):

        #interpolate between middle 4 pixels if patchsize has even sides
        if self.input_dims['lat'] % 2 == 0: 
            idx = (int(self.input_dims['lat'] / 2 - 1), int(self.input_dims['lat'] / 2))
            features = X_batch_ds.isel(
                lat=slice(idx[0], idx[1] + 1), lon=slice(idx[0], idx[1] + 1)
            )
            lat = features.lat.mean().item()
            lon = features.lon.mean().item()
            features = features.mean(dim=("lat", "lon"), keepdims=True)
        else:
            idx = int(self.input_dims['lat'] / 2)
            features = X_batch_ds.isel(lat=idx, lon=idx)
            lat = features.lat.item()
            lon = features.lon.item()

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


class MSGDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 12,
        x_vars=None,
        y_vars=['SIS'],
        transform=None,
        target_transform=None,
        filter=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.x_vars = x_vars
        self.y_vars = y_vars
        self.transform = transform
        self.target_transform = target_transform
        self.filter=filter

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MSGDataset(
                zarr_store="/scratch/snx3000/kschuurm/DATA/train.zarr", 
                y_vars=["SIS"],
                x_vars=self.x_vars,
                transform=self.tranform,
                target_transform=self.target_transform,
                filter=self.filter,
            )
            self.val_dataset = MSGDataset(
                zarr_store="/scratch/snx3000/kschuurm/DATA/valid.zarr", 
                y_vars=["SIS"],
                x_vars=self.x_vars,
                transform=self.tranform,
                target_transform=self.target_transform
            )

        if stage == "test":
            self.test_dataset = MSGDataset(
                zarr_store="/scratch/snx3000/kschuurm/DATA/test.zarr", 
                y_vars=["SIS"],
                x_vars=self.x_vars,
                transform=self.tranform,
                target_transform=self.target_transform
            )

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=None, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=None, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=None, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return self.val_dataloader()

class MSGDataModuleBatched(MSGDataModule):
    def __init__(self, patch_size: (int, int), input_overlap: Dict[Hashable, int] = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size= patch_size
        self.batch_dims = {'time':self.batch_size}
        self.input_overlap = input_overlap

    def setup(self, stage: str):
        if stage == "fit":
            # Using xbatcher to make batches of patched data
            self.train_dataset = MSGDatasetBatched(
                zarr_store="/scratch/snx3000/kschuurm/DATA/train.zarr",
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                batch_dims=self.batch_dims,
                patch_size=self.patch_size,
                input_overlap=self.input_overlap,
                transform=self.transform,
                target_transform=self.target_transform,
                filter=self.filter,
            )
            self.val_dataset = MSGDatasetBatched(
                zarr_store="/scratch/snx3000/kschuurm/DATA/valid.zarr",
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                batch_dims=self.batch_dims,
                patch_size=self.patch_size,
                input_overlap=self.input_overlap,
                transform=self.transform,
                target_transform=self.target_transform
            )
        if stage == "test":
            self.test_dataset = MSGDatasetBatched(
                zarr_store="/scratch/snx3000/kschuurm/DATA/test.zarr",
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                batch_dims=self.batch_dims,
                patch_size=self.patch_size,
                input_overlap=self.input_overlap,
                transform=self.transform,
                target_transform=self.target_transform
            )


class MSGDataModulePoint(MSGDataModuleBatched):
    def __init__(self, x_features=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_features = x_features

    def setup(self, stage: str):
        if stage == "fit":
            # Using xbatcher to make batches of patched data
            self.train_dataset = MSGDatasetPoint(
                zarr_store="/scratch/snx3000/kschuurm/DATA/train.zarr",
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                x_features=self.x_features,
                batch_dims=self.batch_dims,
                patch_size=self.patch_size,
                input_overlap=self.input_overlap,
                transform=self.transform,
                target_transform=self.target_transform,
                filter=self.filter,
            )
            self.val_dataset = MSGDatasetPoint(
                zarr_store="/scratch/snx3000/kschuurm/DATA/valid.zarr",
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                x_features=self.x_features,
                batch_dims=self.batch_dims,
                patch_size=self.patch_size,
                input_overlap=self.input_overlap,
                transform=self.transform,
                target_transform=self.target_transform
            )
        if stage == "test":
            self.test_dataset = MSGDatasetPoint(
                zarr_store="/scratch/snx3000/kschuurm/DATA/test.zarr",
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                x_features=self.x_features,
                batch_dims=self.batch_dims,
                patch_size=self.patch_size,
                input_overlap=self.input_overlap,
                transform=self.transform,
                target_transform=self.target_transform
            )


if __name__ == "__main__":
    # with Client() as client:
    #     print(client)

    # dm = MSGDataModule(batch_size=32, patch_size=(128,128), num_workers=12)
    # dm.setup('fit')
    # dl = dm.train_dataloader()

    # for i, (X, y) in tqdm(enumerate(dl)):

    #     if i> 50:
    #         break

    # plt.imshow(y.view(-1, 128))
    # plt.colorbar()
    # plt.savefig('testbatchs.png')
    from normalization import MinMax
    
    def SZA_filter(ds):
        return ds.isel(time=ds.SZA.mean(['lat','lon']) < 1.4835298641951802) # SZA below 85*
                     
    minmax_normalizer = MinMax()
    ds = MSGDataset(
        "/scratch/snx3000/kschuurm/DATA/valid.zarr",
        y_vars=["SIS", "SRTM"],
        x_vars=["channel_1", "channel_2"],
        transform=minmax_normalizer,
        target_transform=minmax_normalizer,
    )

    ds1 = MSGDatasetBatched(
        zarr_store="/scratch/snx3000/kschuurm/DATA/valid.zarr",
        y_vars=["SIS"],
        x_vars=["channel_1", "channel_2"],
        patch_size=(15, 15),
        batch_size=1024,
        transform=minmax_normalizer,
        target_transform=minmax_normalizer,
        filter = SZA_filter
    )

    ds2 = MSGDatasetPoint(
        zarr_store="/scratch/snx3000/kschuurm/DATA/valid.zarr",
        y_vars=["SIS", "SRTM"],
        x_vars=["channel_1", "channel_10"],
        patch_size=(16, 16),
        transform=minmax_normalizer,
        target_transform=minmax_normalizer
    )

    ds3 = MSGDatasetPoint(
        zarr_store="/scratch/snx3000/kschuurm/DATA/valid.zarr",
        y_vars=["SIS", "SRTM"],
        x_vars=["channel_1", "channel_10"],
        patch_size=(15, 15),
        transform=minmax_normalizer,
        target_transform=minmax_normalizer
    )

    print('test MSGDataset')
    X, y = ds[0]
    print(X.shape, y.shape)

    print('test MSGDatasetBatched')
    X, y = ds1[1]
    print(X.shape, y.shape)

    print('test MSGDatasetPoint')
    X, x, y = ds2[2]
    print(X.shape, x.shape, y.shape)
