import itertools
from typing import Any, Tuple

from tqdm import tqdm
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import xarray
from datetime import timedelta
from xbatcher import BatchGenerator
from preprocess.etc import benchmark
import pytorch_lightning as L
import numpy as np
import pandas as pd
import pvlib, ephem


import xbatcher
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def pyephem(
    datetimes,
    latitude,
    longitude,
    altitude=0,
    pressure=101325,
    temperature=12,
    horizon="+0:00",
    rad=True,
):
    """
    Calculate the solar position using the PyEphem package.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Must be localized or UTC will be assumed.
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative
        to south.
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.
    altitude : float, default 0
        Height above sea level in meters. [m]
    pressure : int or float, optional, default 101325
        air pressure in Pascals.
    temperature : int or float, optional, default 12
        air temperature in degrees C.
    horizon : string, optional, default '+0:00'
        arc degrees:arc minutes from geometrical horizon for sunrise and
        sunset, e.g., horizon='+0:00' to use sun center crossing the
        geometrical horizon to define sunrise and sunset,
        horizon='-0:34' for when the sun's upper edge crosses the
        geometrical horizon

    See also
    --------
    spa_python, spa_c, ephemeris
    """

    # Written by Will Holmgren (@wholmgren), University of Arizona, 2014
    # try:
    #     import ephem
    # except ImportError:
    #     raise ImportError('PyEphem must be installed')

    # if localized, convert to UTC. otherwise, assume UTC.
    # try:
    #     time_utc = time.tz_convert('UTC')
    # except TypeError:
    #     time_utc = time

    # sun_coords = pd.DataFrame(index=time)

    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.elevation = altitude if not np.isnan(altitude) else 0
    sun = ephem.Sun()

    # make and fill lists of the sun's altitude and azimuth
    # this is the pressure and temperature corrected apparent alt/az.
    elevs = []
    azis = []
    for thetime in datetimes:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        elevs.append(sun.alt)
        azis.append(sun.az)

    elevs = np.array(elevs)
    azis = np.array(azis)
    zens = np.pi / 2 - elevs

    if not rad:
        elevs = np.rad2deg(elevs)
        azis = np.rad2deg(azis)
        zens = n.rad2deg(zens)

    return elevs, azis, zens


def solarzenithangle(datetime, lat, lon, alt):
    """Expects datetime in UTC"""
    elevs, azis, zens = pyephem(datetime, lat, lon, alt)
    return zens  # Zenith angle


class MSGDataset(Dataset):
    def __init__(
        self, dataset, y_var, x_vars=None, transform=None, target_transform=None
    ):
        if isinstance(dataset, str):
            self.ds = xarray.open_zarr(dataset)
        elif isinstance(dataset, xarray.Dataset):
            self.ds = dataset
        else:
            raise ValueError(f"{dataset} is not a zarr store or a xarray.Dataset.")

        self.ds = self.ds.drop_vars(("lat_bnds", "lon_bnds"))
        self.attributes = self.ds.attrs

        self.transform = transform
        self.target_transform = target_transform

        vars = set(self.ds.keys())
        assert y_var in vars, "y_var not available in the dataset"
        self.y_var = y_var
        ignore_vars = set(["lon_bnds", "lat_bnds", "record_status", "crs"])
        other_vars = (
            vars
            - ignore_vars
            - set(
                [
                    y_var,
                ]
            )
        )

        if x_vars is not None:
            assert set(x_vars).issubset(
                other_vars
            ), f"{set(x_vars)^other_vars} not in the dataset"
            self.x_vars = x_vars
        else:
            self.x_vars = other_vars

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        if self.ds is not None:
            xarr = self.ds[self.x_vars].isel(time=idx)
            if self.transform:
                xarr = self.transform(xarr)
            xarr = xarr.to_dataarray()  # select time slice, reshape to t lon lat

            yarr = self.ds[self.y_var].isel(time=idx)
            x = torch.tensor(data=xarr.values).permute(1, 0, 2, 3)  # (B, C, X, Y)
            y = torch.tensor(data=yarr.values)  # (B, X, Y)
            return x, y
        else:
            raise ValueError("There is no zarr dataset to get images from.")


class MSGDatasetBatched(MSGDataset):
    def __init__(
        self,
        zarr_store,
        y_var=None,
        x_vars=None,
        patch_size=(64, 64),
        batch_size=10,
        x_features_bool=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            zarr_store,
            y_var,
            x_vars,
            transform=transform,
            target_transform=target_transform,
        )

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.generator = BatchGenerator(
            self.ds,
            input_dims={"lat": self.patch_size[1], "lon": self.patch_size[0]},
            batch_dims={"time": self.batch_size},
        )

        self.x_features_bool = x_features_bool

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
        )  # , names=('B','C','T', 'X', 'Y'))
        y_batch = torch.tensor(
            y_batch_ds.to_dataarray(dim="channels").values
        )  # , names=('B', 'T', 'X', 'Y'))

        X_batch = X_batch.permute(1, 0, 2, 3)  # Squeeze out time dimension
        y_batch = y_batch  # .squeeze(-2).unsqueeze(1) (B, T, X, Y) -> (B, C, X, Y) where dim(C)=1

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

        return X_batch.to_dataarray(dim="channels"), y_batch.to_dataarray(
            dim="channels"
        )  # ('B', 'C', 'X', 'Y') &  ('B', 'X', 'Y')

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

        a = self.generator[idx]
        X_batch = a[self.x_vars]
        y_batch = a[[self.y_var]]

        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        return X_batch, y_batch


class MSGDatasetPoint(MSGDatasetBatched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
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
        X_batch = X_batch.fillna(
            -99999
        )  # for training we do not want any nans. Very negative values should cancel with ReLU

        X_batch = torch.tensor(X_batch.values)  # , names=('C', 'B', 'X', 'Y'))
        x_attributes = torch.tensor(x_attributes.values)  # ('B', 'C')
        y = torch.tensor(y.values)  # , names=('B')

        X_batch = X_batch.permute(1, 0, 2, 3)  # ('B', 'C', 'X', 'Y')
        x_attributes = x_attributes.permute(1, 0)
        y = y.permute(1, 0)

        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        return X_batch, x_attributes, y

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
        if self.patch_size[0] % 2 == 0:
            idx = (int(self.patch_size[0] / 2 - 1), int(self.patch_size[0] / 2))
            y_batch_point = (
                y_batch.isel(
                    lat=slice(idx[0], idx[1] + 1), lon=slice(idx[0], idx[1] + 1)
                )
                .mean(dim=("lat", "lon"), keepdims=True)
                .to_dataarray(dim="channels")
            )
        else:
            idx = int(self.patch_size[0] / 2)
            y_batch_point = y_batch.isel(lat=idx, lon=idx).to_dataarray(dim="channels")

        x_attributes = self.get_point_features(X_batch).to_dataarray(dim="channels")
        X_batch = X_batch.to_dataarray(dim="channels")

        return X_batch, x_attributes, y_batch_point

    def get_point_features(self, X_batch_ds):
        if self.patch_size[0] % 2 == 0:
            idx = (int(self.patch_size[0] / 2 - 1), int(self.patch_size[0] / 2))
            features = X_batch_ds.isel(
                lat=slice(idx[0], idx[1] + 1), lon=slice(idx[0], idx[1] + 1)
            )
            lat = features.lat.mean().item()
            lon = features.lon.mean().item()
            features = features.mean(dim=("lat", "lon"))
        else:
            idx = int(self.patch_size[0] / 2)
            features = X_batch_ds.isel(lat=idx, lon=idx)
            lat = features.lat.item()
            lon = features.lon.item()

        if "SRTM" in set(X_batch_ds.keys()):
            srtm = features.SRTM.item()
        else:
            srtm = self.ds.SRTM.sel(lat=lat, lon=lon, method="nearest")

        datetimes = pd.to_datetime(features.time)
        month = features.time.dt.month.values
        day = features.time.dt.day.values
        sza = solarzenithangle(
            datetimes, lat, lon, srtm if not np.isnan(srtm) else 0
        )  # Assume altitude is zero at nans for sza calc

        srtm = (
            srtm if not np.isnan(srtm) else -99999
        )  # Very small number for ML algorithm to cancel out
        lat = np.repeat(lat, self.batch_size)
        lon = np.repeat(lon, self.batch_size)
        srtm = np.repeat(srtm, self.batch_size)

        # arr = np.hstack([month.reshape(-1,1), day.reshape(-1,1), sza.reshape(-1,1), b])

        features = xarray.Dataset(
            data_vars={
                "month": (("time"), month),
                "day": (("time"), day),
                "lat": (("time"), lat),
                "lon": (("time"), lon),
                "srtm": (("time"), srtm),
                "sza": (("time"), sza),
            },
        )

        return features  # (B, F) , F columns: [month, day, sza, lat, lon, srtm]


class MSGDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        patch_size: (int, int) = None,
        num_workers: int = 12,
        x_vars=None,
        transform=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.x_vars = x_vars
        self.tranform = transform

    def setup(self, stage: str):
        if stage == "fit":
            if self.patch_size is not None:
                # Using xbatcher to make batches of patched data
                self.train_dataset = MSGDatasetBatched(
                    "/scratch/snx3000/kschuurm/DATA/train.zarr",
                    y_var="SIS",
                    x_vars=self.x_vars
                    patch_size=self.patch_size,
                    batch_size=self.batch_size,
                    transform=self.tranform,
                )
                self.val_dataset = MSGDatasetBatched(
                    "/scratch/snx3000/kschuurm/DATA/valid.zarr",
                    y_var="SIS",
                    x_vars=self.x_vars
                    patch_size=self.patch_size,
                    batch_size=self.batch_size,
                )
            else:
                self.train_dataset = MSGDataset(
                    "/scratch/snx3000/kschuurm/DATA/train.zarr", 
                    y_var="SIS",
                    x_vars=self.x_vars
                )
                self.val_dataset = MSGDataset(
                    "/scratch/snx3000/kschuurm/DATA/validation.zarr", 
                    y_var="SIS",
                    x_vars=self.x_vars
                )

        if stage == "test":
            if self.patch_size is not None:
                self.test_dataset = MSGDatasetBatched(
                    "/scratch/snx3000/kschuurm/DATA/validation.zarr",
                    y_var="SIS",
                    x_vars=self.x_vars
                    patch_size=self.patch_size,
                    batch_size=self.batch_size,
                )
            else:
                self.test_dataset = MSGDataset(
                    "/scratch/snx3000/kschuurm/DATA/test.zarr", 
                    y_var="SIS",
                    x_vars=self.x_vars
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


class MSGDataModulePoint(MSGDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert (
            self.patch_size is not None
        ), "To create a patch to point datamodule a patchsize is required"

    def setup(self, stage: str):
        if stage == "fit":
            # Using xbatcher to make batches of patched data
            self.train_dataset = MSGDatasetPoint(
                "/scratch/snx3000/kschuurm/DATA/train.zarr",
                y_var="SIS",
                patch_size=self.patch_size,
                batch_size=self.batch_size,
            )
            self.val_dataset = MSGDatasetPoint(
                "/scratch/snx3000/kschuurm/DATA/valid.zarr",
                y_var="SIS",
                patch_size=self.patch_size,
                batch_size=self.batch_size,
            )
        if stage == "test":
            self.test_dataset = MSGDatasetBatched(
                "/scratch/snx3000/kschuurm/DATA/validation.zarr",
                y_var="SIS",
                patch_size=self.patch_size,
                batch_size=self.batch_size,
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
    
    ds = MSGDatasetPoint(
        "/scratch/snx3000/kschuurm/DATA/valid.zarr",
        y_var="SIS",
        x_vars=["channel_1", "channel_2"],
        x_features_bool=True,
        patch_size=(15, 15),
        transform=MinMax.forward_dataset,
    )

    X, x, y = ds[0]
    print(X.shape, x.shape, y.shape)

    MINMAX = {}
    for var in ds.ds.keys():
        MINMAX[var] = (ds.ds[var].min().values.item(), ds.ds[var].max().values.item())

    print(MINMAX)
