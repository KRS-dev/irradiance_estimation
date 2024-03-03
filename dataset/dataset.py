import os
import torch
import concurrent
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import xarray
from datetime import timedelta
from xbatcher import BatchGenerator
from preprocess.etc import benchmark
import lightning.pytorch as L
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


def create_singleImageDataset(**kwargs):
    return SingleImageDataset(**kwargs)


def valid_test_split(timeindex):
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

    traintimeindex = timeindex[train_ls]
    testtimeindex = timeindex[test_ls]

    return traintimeindex, testtimeindex


class ImageDataset(Dataset):
    def __init__(
        self,
        y_vars,
        x_vars,
        x_features,
        patch_size,
        transform,
        target_transform,
        random_sample=0.1,
        timeindices=None,
        batch_in_time=2,
    ):

        self._pool = concurrent.futures.ThreadPoolExecutor()

        self.seviri = (
            xarray.open_zarr(
                "/scratch/snx3000/acarpent/EumetsatData/SEVIRI_WGS_2016-2022_RSS.zarr"
            )
            .rename_dims({"x": "lon", "y": "lat"})
            .rename_vars(
                {
                    "VIS006": "channel_1",
                    "VIS008": "channel_2",
                    "IR_016": "channel_3",
                    "IR_039": "channel_4",
                    "WV_062": "channel_5",
                    "WV_073": "channel_6",
                    "IR_087": "channel_7",
                    "IR_097": "channel_8",
                    "IR_108": "channel_9",
                    "IR_120": "channel_10",
                    "IR_134": "channel_11",
                    "x": "lon",
                    "y": "lat",
                }
            )
            .drop_duplicates(dim="time")
        )
        self.dem = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/DEM.zarr").fillna(0)

        self.sarah = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/SARAH3.zarr")
        self.solarpos = xarray.open_zarr(
            "/scratch/snx3000/kschuurm/ZARR/SOLARPOS.zarr"
        ).drop_duplicates(
            dim="time"
        )  # SOLARPOS should be the same dim as SEVIRI
        self.seviri = xarray.merge(
            [self.seviri, self.solarpos], join="exact"
        )  # throws an error if time, lat, lon dim not the same
        self.seviri = xarray.merge(
            [self.seviri, self.dem], join="exact"
        )  # throws an error if lat, lon not the same

        if timeindices is not None:
            self.images = timeindices
        elif os.path.exists("/scratch/snx3000/kschuurm/ZARR/idxnotnan.npy"):
            self.images = np.load("/scratch/snx3000/kschuurm/ZARR/idxnotnan.npy")
        else:
            print("pre dropnan", len(self.seviri.time))
            self.seviri = self.seviri.dropna(dim="time")
            print("dropnan", len(self.seviri.time))
            print("pre dropnan", len(self.sarah.time))
            self.sarah = self.sarah.dropna(dim="time")
            print("dropnan", len(self.sarah.time))

            timenotnan = set(self.seviri.time.values).intersection(
                set(self.sarah.time.values)
            )
            timenotnan = np.sort(np.array(list(timenotnan)))
            print("dropnan both", len(timenotnan))
            np.save("/scratch/snx3000/kschuurm/ZARR/idxnotnan.npy", timenotnan)
            self.images = timenotnan

        self.batch_in_time = batch_in_time
        if self.batch_in_time:
            self.images = [self.images[i:i+self.batch_in_time] for i in range(0, len(self.images), self.batch_in_time)]

        self.lat = self.seviri.lat
        self.lon = self.seviri.lon
        self.patch_size = patch_size
        patch_x = patch_size["x"]
        patch_y = patch_size["y"]
        stride_x = patch_size["stride_x"]
        stride_y = patch_size["stride_y"]
        pad_x = int(np.floor(patch_x / 2))
        pad_y = int(np.floor(patch_y / 2))

        self.patches_per_image = int(
            np.floor((len(self.lat) - 2 * pad_y) / stride_y)
            * np.floor((len(self.lon) - 2 * pad_x) / stride_x)
        )

        self.random_sample = random_sample
        if self.random_sample:
            self.sample_size = int(
                np.floor(self.random_sample * self.patches_per_image)
            )
            self.samples = np.random.randint(
                0, self.patches_per_image, self.sample_size
            )
        else:
            self.sample_size = self.patches_per_image
            self.samples = range(0, self.sample_size)

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform

        self.image_i = None
        self.current_singleImageDataset = None
        self.next_images = None

    def __len__(self):
        return len(self.images) * self.sample_size * self.batch_in_time

    def load_new_image(self, image_i):
        if self.current_singleImageDataset is None:
            dt = self.images[image_i]
            self.current_singleImageDataset = self.load_singleImageDataset(dt).result()
        else:
            if self.next_images:
                self.current_singleImageDataset = self.next_images.pop().result()

        preload_n = 2
        if image_i + preload_n < len(self.images):
            self.load_next_images(image_i + 1, preload_n)

    def load_next_image(self, dt):
        self.next_singleImageDataset = self.load_singleImageDataset(dt)

    def load_next_images(self, i, preload_n):
        if self.next_images is None:
            self.next_images = [
                self.load_singleImageDataset(self.images[i]) for i in range(i, i + preload_n)
            ]
        else:
            self.next_images.append(self.load_singleImageDataset(self.images[i + preload_n - 1]))

    def load_singleImageDataset(self, dt):
        d = dict(
            hrseviri=self.seviri.sel(time=dt),
            sarah=self.sarah.sel(time=dt),
            y_vars=self.y_vars,
            x_vars=self.x_vars,
            x_features=self.x_features,
            patch_size=self.patch_size,
            transform=self.transform,
            target_transform=self.target_transform,
            extra_batch_dimension=1,
        )
        dataset = self._pool.submit(create_singleImageDataset, **d)
        return dataset

    def __getitem__(self, i):
        idx_image = int(np.floor(i / self.sample_size))
        idx_patch = self.samples[int(i % self.sample_size)]

        if self.image_i is None:
            self.image_i = 0
            self.load_new_image(idx_image)
        elif self.image_i != idx_image:
            self.image_i = idx_image
            self.load_new_image(idx_image)

        return self.current_singleImageDataset[idx_patch]


class SingleImageDataset(Dataset):
    def __init__(
        self,
        hrseviri,
        sarah,
        y_vars,
        x_vars,
        x_features,
        patch_size,
        transform,
        target_transform,
        extra_batch_dimension: int=None,
        dtype=torch.float16,
    ):
        super(SingleImageDataset, self).__init__()

        self.seviri = hrseviri
        self.sarah = sarah
        self.lat, self.lon = xarray.broadcast(
            hrseviri.lat, hrseviri.lon
        )  # size(HxW) both

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()

        self.transform = transform
        self.target_transform = target_transform
        self.extra_batch_dimension = extra_batch_dimension

        patch_x = patch_size["x"]
        patch_y = patch_size["y"]
        stride_x = patch_size["stride_x"]
        stride_y = patch_size["stride_y"]
        pad_x = int(np.floor(patch_size["x"] / 2))
        pad_y = int(np.floor(patch_size["y"] / 2))

        def unfold_tensor(X):
            # assume last two dimensions are HxW
            shape = X.shape  # * ...xHxW
            X = X.unfold(-2, patch_y, stride_y).unfold(
                -2, patch_x, stride_x
            )  # ....xH*xW*xPHxPW
            shape_unfold = (
                list(shape[:-2]) + [-1] + [patch_x, patch_y]
            )  # ....xBxPHxPW   stacking H* and W* into a batch dimension
            X = X.reshape(shape_unfold)
            permute_l = list(range(0, len(shape_unfold)))
            loc_B = permute_l.pop(-3)  # Pop dim location of the batch from the sequence
            permute_l.insert(0, loc_B)  # insert dim batch in the front
            X = X.permute(permute_l)  # Bx ... xPHxPW
            return X

        # Unfold image to patches
        self.X = torch.tensor(
            self.seviri[x_vars].to_dataarray(dim="channels").values, dtype=dtype
        )  # Cx'T'xHxW
        if self.extra_batch_dimension:
            self.X = self.X.movedim(self.extra_batch_dimension, 0)
        self.X = unfold_tensor(self.X) # BxCx'T'xPHxPW or BxTxCxPHxPH
        if self.extra_batch_dimension:
            self.X = self.X.reshape(-1, *self.X.shape[2:]) # BxT => B
          
        self.lat_patches = torch.tensor(self.lat.values.copy(), dtype=dtype)  # HxW
        self.lat_patches = unfold_tensor(self.lat_patches)  # BxPHxPW

        self.lon_patches = torch.tensor(self.lon.values.copy(), dtype=dtype)  # HxW
        self.lon_patches = unfold_tensor(self.lon_patches)  # BxPHxPW

        # Manipulate point features
        x_features_avail = set(x_features).intersection(set(self.seviri.keys()))
        self.x_features_temp = x_features.copy()
        self.x = self.seviri[x_features_avail]

        if "lat" in x_features:
            self.x = self.x.assign(
                {
                    "lat_": self.lat,
                    "lon_": self.lon,
                }
            )
            self.x_features_temp[1] = "lat_"
            self.x_features_temp[2] = "lon_"

        if "dayofyear" in x_features:
            dayofyear = self.seviri.time.dt.dayofyear.astype(int)
            dayofyear, _ = xarray.broadcast(dayofyear, self.lat)
            self.x = self.x.assign(
                {
                    "dayofyear": dayofyear,
                }
            )

        self.x = torch.tensor(
            self.x[self.x_features_temp].to_dataarray(dim="channels").values,
            dtype=dtype,
        )  # Cx...xHxW
        # reorder self.x by the original order of the x_feature list
        shape = self.x.shape
        reshape_l = list(shape)
        if self.extra_batch_dimension:
            self.x = self.x.movedim(self.extra_batch_dimension, 0)
            reshape_l.pop(self.extra_batch_dimension)
        reshape_l = reshape_l[:-2]

        permute_l = list(range(len(self.x.shape)))
        permute_l = permute_l[-2:] + permute_l[:-2]
        self.x = self.x.permute(permute_l)  # HxWxCx ....
        self.x = self.x[pad_y:-pad_y:stride_y, pad_x:-pad_x:stride_x].reshape(
            -1, *reshape_l
        )  # BxCx ..., Make sure that padding due to patchsize is incorporated, then reshape to batch dim

        assert self.x.shape[0] == self.X.shape[0], print(
            "batch size point features not equal to patches",
            self.x.shape,
            self.X.shape[0],
        )  # Batch sizes are equal

        # Manipulate output
        self.y = torch.tensor(
            self.sarah[y_vars].to_dataarray(dim="channels").values, dtype=dtype
        )  # Cx ... xHxW
        shape = self.y.shape
        reshape_l = list(shape)
        if self.extra_batch_dimension:
            self.y = self.y.movedim(self.extra_batch_dimension, 0)
            reshape_l.pop(self.extra_batch_dimension)
             # Example with shape = CxTxHxW -> TxCxHxW,  skip HxW, pop(T) results in reshape_l= [C]
        reshape_l = reshape_l[:-2]

        permute_l = list(range(len(shape)))
        permute_l = permute_l[-2:] + permute_l[:-2]
        
        self.y = self.y.permute(permute_l)  # HxWxCx...
        self.y = self.y[pad_y:-pad_y:stride_y, pad_x:-pad_x:stride_x].reshape(
            -1, *reshape_l
        )  # BxC...


        if self.transform:
            self.X = self.transform(self.X, self.x_vars)
            self.x = self.transform(self.x, self.x_features)

        if self.target_transform:
            self.y = self.target_transform(self.y, self.y_vars)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        X_element = self.X[i]
        x_element = self.x[i]
        y_element = self.y[i]

        if X_element.isnan().any():
            print("nan in X")
            print(X)
        if x_element.isnan().any():
            print("nan in x")
            print(x)
        if y_element.isnan().any():
            print("nan in y")
            print(y)

        return X_element, x_element, y_element

    def get_latlon_patch(self, i):
        lat_patch = self.lat_patches[i]
        lon_patch = self.lon_patches[i]
        return lat_patch, lon_patch

    def get_patch_xarray(self, i, transform_back=False):

        lat_patch, lon_patch = self.get_latlon_patch(i)
        X, x, y = self[i]

        X[X == -99] = torch.nan
        x[x == -99] = torch.nan

        X_xr = xarray.Dataset(
            data_vars={name: (("lat", "lon"), d) for name, d in zip(self.x_vars, X)},
            coords={
                "lat_patch": (("lat", "lon"), lat_patch),
                "lon_patch": (("lat", "lon"), lon_patch),
                "time": self.seviri.time,
            },
        )

        x_xr = xarray.Dataset(
            data_vars={name: d for name, d in zip(self.x_features, x)},
            coords={
                "time": self.seviri.time,
            },
        )

        if self.transform:
            lat = self.transform.inverse(x_xr.lat, ["lat"])
            lon = self.transform.inverse(x_xr.lon, ["lon"])
            x_xr["lat"] = lat
            x_xr["lon"] = lon

        x_xr = x_xr.set_coords(["lat", "lon"])
        y_xr = xarray.Dataset(
            data_vars={name: d for name, d in zip(self.y_vars, y)},
            coords={
                "time": self.seviri.time,
                "lat": x_xr.lat,
                "lon": x_xr.lon,
            },
        )

        if transform_back:
            if self.transform:
                X_xr = self.transform.inverse(X_xr)
                x_xr = self.transform.inverse(x_xr)
            if self.target_transform:
                y_xr = self.target_transform.inverse(y_xr)

        return X_xr, x_xr, y_xr


if __name__ == "__main__":

    np.random.seed(0)

    from types import SimpleNamespace

    config = {
        "batch_size": 512,
        "patch_size": {
            "x": 15,
            "y": 15,
            "stride_x": 1,
            "stride_y": 1,
        },
        "x_vars": [
            "channel_1",
            "channel_2",
            "channel_3",
            "channel_4",
            "channel_5",
            "channel_6",
            "channel_7",
            "channel_8",
            "channel_9",
            "channel_10",
            "channel_11",
            "DEM",
        ],
        "y_vars": ["SIS", "CAL"],
        "x_features": ["dayofyear", "lat", "lon", "SZA", "AZI"],
        "transform": None,
        "target_transform": None,
    }
    config = SimpleNamespace(**config)

    dataset = ImageDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        random_sample=0.01,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    for X, x, y in tqdm(dataloader):
        print(X.shape, x.shape, y.shape)
        break
