from glob import glob
import os
import pickle
from dataset.normalization import ZeroMinMax
import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import xarray, zarr
from datetime import timedelta
import numpy as np
import pandas as pd
import ephem
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from utils.etc import benchmark

import dask, gc
from utils.clearsky import Clearsky, SolarPosition

dask.config.set(scheduler='synchronous')

def create_singleImageDataset(**kwargs):
    return SingleImageDataset(**kwargs)

def pickle_write(obj, fn):
    with open(fn, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)
    return obj

def pickle_read(fn):
    with open(fn, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
    return obj

def get_pickled_sarah_bnds():
    if os.path.exists('/scratch/snx3000/kschuurm/ZARR/timeindices.pkl'):
        timeindices = pickle_read('/scratch/snx3000/kschuurm/ZARR/timeindices.pkl')
        max_y = pickle_read('/scratch/snx3000/kschuurm/ZARR/max_y.pkl')
        max_x = pickle_read('/scratch/snx3000/kschuurm/ZARR/max_x.pkl')
        min_y = pickle_read('/scratch/snx3000/kschuurm/ZARR/min_y.pkl')
        min_x = pickle_read('/scratch/snx3000/kschuurm/ZARR/min_x.pkl')
    else:
        sarah_bnds = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3_bnds.zarr').load()
        sarah_bnds = sarah_bnds.isel(time=sarah_bnds.pixel_count != -1)   
        timeindices = sarah_bnds.time.copy(deep=True)
        max_y = sarah_bnds.max_lat.copy(deep=True)
        max_x = sarah_bnds.max_lon.copy(deep=True)
        min_y = sarah_bnds.min_lat.copy(deep=True)
        min_x = sarah_bnds.min_lon.copy(deep=True)

        timeindices = pickle_write(timeindices, '/scratch/snx3000/kschuurm/ZARR/timeindices.pkl')
        max_y = pickle_write(max_y, '/scratch/snx3000/kschuurm/ZARR/max_y.pkl')
        max_x = pickle_write(max_x, '/scratch/snx3000/kschuurm/ZARR/max_x.pkl')
        min_y = pickle_write(min_y, '/scratch/snx3000/kschuurm/ZARR/min_y.pkl')
        min_x = pickle_write(min_x, '/scratch/snx3000/kschuurm/ZARR/min_x.pkl') 
    
    return timeindices, max_y, max_x, min_y, min_x

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
        timeindices=None,
        shuffle=True,
        batch_in_time=2,
        dtype=torch.float16,
    ):

        self._pool = ThreadPoolExecutor()

        self.seviri = (
            xarray.open_zarr(
                "/scratch/snx3000/kschuurm/ZARR/SEVIRI_FULLDISK.zarr"
            ).channel_data.to_dataset(dim='channel') 
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
            ).drop_vars(['spatial_ref'])
        )
        self.dem = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/DEM.zarr").fillna(0)

        self.sarah = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/SARAH3.zarr").channel_data.to_dataset(dim='channel') 
     
        self.seviri = xarray.merge(
            [self.seviri, self.dem], join="exact"
        )  # throws an error if lat, lon not the same

        self.sarah_nulls = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3_nulls.zip')
        timeindices_sarah = self.sarah_nulls['any'].where((self.sarah_nulls['any'] == True).compute(), drop=True).time.values

        if timeindices is not None:
            self.timeindices = timeindices
        else:
            self.timeindices = timeindices_sarah

        self.timeindices = np.array(list(set(self.timeindices).intersection(set(self.seviri.time.values))))

        if shuffle is not None:
            self.timeindices_samples = np.random.choice(range(len(self.timeindices)), size=len(self.timeindices), replace=False)

        self.lat = self.seviri.lat
        self.lon = self.seviri.lon
        self.patch_size = patch_size
        patch_x = patch_size["x"]
        patch_y = patch_size["y"]
        stride_x = patch_size["stride_x"]
        stride_y = patch_size["stride_y"]
        pad_x = int(np.floor(patch_x / 2))
        pad_y = int(np.floor(patch_y / 2))

        self.patches_per_image = ((len(self.lat)-2*pad_y )//stride_y) * ((len(self.lon) - 2*pad_x)//stride_x)

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

        self.image_i = None
        self.current_singleImageDataset = None
        self.next_images = None

    def __len__(self):
        return len(self.timeindices) * self.patches_per_image

    def load_new_image(self, image_i):
        if self.current_singleImageDataset is None:
            dt = self.timeindices[image_i]
            self.current_singleImageDataset = self.load_singleImageDataset(dt).result()
        else:
            if self.next_images:
                self.current_singleImageDataset = self.next_images.pop().result()

        preload_n = 4
        if image_i + preload_n < len(self.timeindices):
            self.load_next_images(image_i + 1, preload_n)


    def load_next_images(self, i, preload_n):
        if self.next_images is None:
            self.next_images = [
                self.load_singleImageDataset(self.timeindices[i]) for i in range(i, i + preload_n)
            ]
        else:
            self.next_images.append(self.load_singleImageDataset(self.timeindices[i + preload_n - 1]))



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
            dtype=self.dtype,
        )
        dataset = self._pool.submit(create_singleImageDataset, **d)
        return dataset

    def __getitem__(self, i):
        idx_image = int(np.floor(i // self.patches_per_image))
        idx_patch = int(i % self.patches_per_image)

        if self.image_i is None:
            self.image_i = 0
            self.load_new_image(idx_image)
        elif self.image_i != idx_image:
            self.image_i = idx_image
            self.load_new_image(idx_image)

        return self.current_singleImageDataset[idx_patch]

class SeviriDataset(Dataset):
    def __init__(
        self,
        y_vars,
        x_vars,
        x_features,
        patch_size,
        transform=None,
        target_transform=None,
        patches_per_image =2048,
        dtype=torch.float16,
        validation=False,
        rng=None
    ):


        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform
        self.patches_per_image = patches_per_image
        self.dtype=dtype
        self.validation = validation
        self.rng = rng # set random generator for all datasets in ddp

        self.seviri = (
            xarray.open_zarr(
                "/scratch/snx3000/kschuurm/ZARR/SEVIRI_FULLDISK.zarr"
            ).rename_dims({"x": "lon", "y": "lat"})
            .rename_vars(
                {
                    "x": "lon",
                    "y": "lat",
                }
            )
        )
        seviri_trans = {
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
            "IR_134": "channel_11",}
        
        nms = self.seviri.channel.values
        nms_trans = [seviri_trans[x] for x in nms]
        self.seviri['channel'] = nms_trans
        self.x_vars_available = [x for x in self.x_vars if x in nms_trans]

        if os.path.exists('/scratch/snx3000/kschuurm/ZARR/DEM.pkl'):
            with open('/scratch/snx3000/kschuurm/ZARR/DEM.pkl', 'rb') as pickle_file:
                self.dem = pickle.load(pickle_file)
        else:
            dem = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/DEM.zarr").fillna(0).load()
            with open('/scratch/snx3000/kschuurm/ZARR/DEM.pkl', 'wb') as pickle_file:
                pickle.dump(dem, pickle_file)
            with open('/scratch/snx3000/kschuurm/ZARR/DEM.pkl', 'rb') as pickle_file:
                self.dem = pickle.load(pickle_file) 

        self.sarah = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/SARAH3.zarr")

        sizes= self.seviri.sizes
        self.H = sizes['lat']
        self.W = sizes['lon']
        self.pad = int(np.floor(patch_size['x']/2))


        self.sarah_nulls = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3_nulls.zarr')
        self.sarah_nulls = self.sarah_nulls.where((self.sarah_nulls['nullssum'] > 100000).compute(), drop=True)

        if validation is True:
            self.timeindices  = self.sarah_nulls.time[self.sarah_nulls.time.dt.year == 2022].values
        else:
            self.timeindices  = self.sarah_nulls.time[self.sarah_nulls.time.dt.year < 2022].values

        self.timeindices = np.sort(np.array(list(set(self.timeindices).intersection(set(self.seviri.time.values)))))
        

        if validation is True:
            if os.path.exists('/scratch/snx3000/kschuurm/ZARR/idx_x_sampler.pkl') and os.path.exists('/scratch/snx3000/kschuurm/ZARR/idx_y_sampler.pkl'):
                self.idx_x_sampler = pickle_read('/scratch/snx3000/kschuurm/ZARR/idx_x_sampler.pkl')
                self.idx_y_sampler = pickle_read('/scratch/snx3000/kschuurm/ZARR/idx_y_sampler.pkl')
            else:
                self.idx_x_sampler = {}
                self.idx_y_sampler = {}
                for timeidx in tqdm(self.timeindices, desc='sampler setup', total=len(self.timeindices)):


                    notnulls = self.sarah_nulls.nulls.sel(time=timeidx).load()

                    coords_notnull = np.argwhere(np.array(notnulls[self.pad:-self.pad, self.pad:-self.pad]))
                    samples = coords_notnull[torch.randint(0, len(coords_notnull), (self.patches_per_image,), dtype=torch.int32, generator=self.rng)]
                    idx_x_samples = self.pad + samples[:,1]
                    idx_y_samples = self.pad + samples[:,0]

                    # min_x = int(self.min_x.sel(time=timeidx).values)
                    # min_y = int(self.min_y.sel(time=timeidx).values)
                    # max_x = int(self.max_x.sel(time=timeidx).values)
                    # max_y = int(self.max_y.sel(time=timeidx).values)
                    # idx_x_samples = torch.randint(min_x + self.pad, 
                    #                             max_x-self.pad, 
                    #                             (self.patches_per_image,), 
                    #                             dtype=torch.int32, 
                    #                             generator=self.rng)
                    # idx_y_samples = torch.randint(min_y + self.pad, 
                    #                             max_y-self.pad, 
                    #                             (self.patches_per_image,), 
                    #                             dtype=torch.int32, 
                    #                             generator=self.rng)

                    self.idx_x_sampler[timeidx] = idx_x_samples
                    self.idx_y_sampler[timeidx] = idx_y_samples
                
                pickle_write(self.idx_x_sampler, '/scratch/snx3000/kschuurm/ZARR/idx_x_sampler.pkl')
                pickle_write(self.idx_y_sampler, '/scratch/snx3000/kschuurm/ZARR/idx_y_sampler.pkl')

    def __len__(self):
        return len(self.timeindices)
    
    def __getitem__(self, i):
        timeidx= self.timeindices[i]
        subset_seviri = self.seviri.sel(time = timeidx).load()
        subset_sarah = self.sarah.sel(time = timeidx).load()

        if self.validation is True:
            idx_x_samples = self.idx_x_sampler[timeidx]
            idx_y_samples = self.idx_y_sampler[timeidx]
        else:

            notnulls = self.sarah_nulls.nulls.sel(time=timeidx).load()
            coords_notnull = np.argwhere(np.array(notnulls[self.pad:-self.pad, self.pad:-self.pad]))

            samples = coords_notnull[torch.randint(0, len(coords_notnull), (self.patches_per_image,))]

            idx_x_samples = self.pad + samples[:,1]
            idx_y_samples = self.pad + samples[:,0]
        

        idx_x_da = xarray.DataArray(idx_x_samples, dims=['z'])
        idx_y_da = xarray.DataArray(idx_y_samples, dims=['z'])

        idx_x_patch_samples = [list(range(x-self.pad, x+self.pad+1)) for x in idx_x_samples]
        idx_y_patch_samples = [list(range(y-self.pad, y+self.pad+1)) for y in idx_y_samples]
        idx_x_patch_da = xarray.DataArray(idx_x_patch_samples, dims=['sample', 'lon'])
        idx_y_patch_da = xarray.DataArray(idx_y_patch_samples, dims=['sample', 'lat'])
        idx_x_patch_da, idx_y_patch_da = xarray.broadcast(idx_x_patch_da, idx_y_patch_da) # samples x lon x lat 
        y = subset_sarah.sel(channel=self.y_vars if isinstance(self.y_vars, list) else [self.y_vars])\
                        .isel(lon=idx_x_da, lat=idx_y_da) 
        
        if 'dayofyear' in self.x_features:
            x_dayofyear = torch.tensor(y.time.dt.dayofyear.values).repeat(self.patches_per_image).view(-1,1)
            x = x_dayofyear
        if 'lat' in self.x_features:
            x_lat = torch.tensor(y.lat.values, dtype=self.dtype).view(-1,1)
            x_lon = torch.tensor(y.lon.values, dtype=self.dtype).view(-1,1)
            x = torch.cat([x_dayofyear, x_lat, x_lon], dim=1)
        
        y = torch.tensor(y.channel_data.values, dtype=self.dtype).permute(1,0)

        if 'SZA' in self.x_features:
            sun = ephem.Sun()
            szas, azis, = [], []
            for lat, lon in zip(x_lat, x_lon):
                latitude = lat.item()
                longitude = lon.item()
                altitude = 0
                thetime = pd.to_datetime(timeidx)
                obs = ephem.Observer()
                obs.date = ephem.Date(thetime)
                obs.lat = str(latitude)
                obs.lon = str(longitude)
                obs.elevation = altitude if not np.isnan(altitude) else 0

                sun.compute(obs)
                azis.append(sun.az)
                szas.append(np.pi/2 - sun.alt)

            azis = np.array(azis)
            szas = np.array(szas)

            azis = torch.tensor(azis, dtype=self.dtype).view(-1,1)
            szas = torch.tensor(szas, dtype=self.dtype).view(-1,1)

            x = torch.cat([x, szas, azis], dim=1)

        if 'DEM' in self.x_features:
            x_DEM = self.dem.DEM.isel(lon=idx_x_da, lat=idx_y_da) \
                            .values
            x_DEM = torch.tensor(x_DEM, dtype=self.dtype).view(-1,1)
            x = torch.cat([x, x_DEM], dim=1)

        X = subset_seviri.channel_data.sel(channel=self.x_vars_available) \
                            .isel(lat = idx_y_patch_da, lon=idx_x_patch_da) \
                            .values # CxBxHxW
        
        X = torch.tensor(X, dtype=self.dtype).permute(1,0,2,3) # BxCxHxW


        if 'DEM' in self.x_vars:
            D = self.dem['DEM'].isel(lat = idx_y_patch_da, lon=idx_x_patch_da).values # BxHxW
            D = torch.tensor(D, dtype=self.dtype)[:, None, :, :]
            X = torch.cat([X,D], dim=1) # BxCxHxW
        
        if self.transform:
            X = self.transform(X, self.x_vars)
            x = self.transform(x, self.x_features)

        if self.target_transform:
            y = self.target_transform(y, self.y_vars)

        if X.isnan().any():
            print("nan in X")
            print(timeidx)
            print(X)
        if x.isnan().any():
            print("nan in x")
            print(timeidx)
            print(x)
        if y.isnan().any():
            print("nan in y")
            print(timeidx)
            print(y)

        if (y == -1).any():
            print('zeros in output', sum(y == -1))


        return X.to(self.dtype), x.to(self.dtype), y.to(self.dtype)

class MemmapSeviriDataset(Dataset):
    def __init__(self, batch_size=2048):
        self.X_files = sorted(glob('/scratch/snx3000/kschuurm/irradiance_estimation/dataset/pickled/X_*.npy'))
        self.x_files = sorted(glob('/scratch/snx3000/kschuurm/irradiance_estimation/dataset/pickled/x_*.npy'))
        self.y_files = sorted(glob('/scratch/snx3000/kschuurm/irradiance_estimation/dataset/pickled/y_*.npy'))
        assert len(self.X_files) == len(self.x_files) == len(self.y_files), "number of files not equal"

        self.X_mmaps = [np.load(fn, mmap_mode='c') for fn in self.X_files]
        self.x_mmaps = [np.load(fn, mmap_mode='c') for fn in self.x_files]
        self.y_mmaps = [np.load(fn, mmap_mode='c') for fn in self.y_files]

        self.lengths = [mmap.shape[0] for mmap in self.y_mmaps]
        self.cumsum = np.cumsum(self.lengths)
        self.batch_size = batch_size

    def __len__(self):
        return sum(self.lengths)//self.batch_size

    def __getitem__(self, i):

        idx_mmap = np.searchsorted(self.cumsum, i*self.batch_size, side='right')
        idx = i*self.batch_size - self.cumsum[idx_mmap-1] if idx_mmap > 0 else i*self.batch_size


        X = torch.from_numpy(self.X_mmaps[idx_mmap][idx:idx+self.batch_size])
        x = torch.from_numpy(self.x_mmaps[idx_mmap][idx:idx+self.batch_size])
        y = torch.from_numpy(self.y_mmaps[idx_mmap][idx:idx+self.batch_size])

        return X, x, y

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
        dtype=torch.float16,
    ):
        super(SingleImageDataset, self).__init__()

        self.seviri = hrseviri.load()
        self.sarah = sarah.load()
        self.lat, self.lon = xarray.broadcast(
            hrseviri.lat, hrseviri.lon
        )  # size(HxW) both

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()

        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

        self.patch_size = patch_size
        self.pad_x = int(np.floor(patch_size["x"] / 2))
        self.pad_y = int(np.floor(patch_size["y"] / 2))

        self.X = torch.tensor(
            self.seviri[x_vars].to_dataarray(dim="channels").values, dtype=self.dtype
        )  # CxHxW or channels, lat, lon


        # "dayofyear", "lat", "lon", 'SZA', "AZI",
        # Manipulate point features

        dayofyear = self.seviri.time.dt.dayofyear.astype(int).item()
        dayofyear = dayofyear*torch.ones(1, self.X.shape[1], self.X.shape[2], dtype=self.dtype)

        

        lat = torch.tensor(self.lat.values, dtype=self.dtype).unsqueeze(0)
        lon = torch.tensor(self.lon.values, dtype=self.dtype).unsqueeze(0)
        self.x = torch.cat([dayofyear, lat, lon], dim=0)


        # Manipulate output
        self.y = torch.tensor(
            self.sarah[y_vars].to_dataarray(dim="channels").values, dtype=dtype
        )  # CxHxW

        self.sun = ephem.Sun()
        self.obs = ephem.Observer()


    def __len__(self):
        C, H, W = self.x.shape
        Y_dim = int(H - self.patch_size['y'] + 1)
        X_dim = int(W - self.patch_size['x'] + 1) 
        return Y_dim*X_dim

    def get_indices(self, i):
        Y_dim, X_dim = self.X.shape[-2:]
        Y_dim = int(Y_dim - self.patch_size['y'] + 1)
        X_dim = int(X_dim - self.patch_size['x'] + 1)
        idx_y = int(i % Y_dim)
        idx_x = int((i // Y_dim) % X_dim)

        return idx_y, idx_x

    def get_item_in_order(self, i):
        idx_y, idx_x = self.get_indices(i)
        # print(idx_y, idx_x, idx_t)
        X_element = self.X[:, 
                        idx_y:idx_y + self.patch_size['y'], 
                        idx_x:idx_x + self.patch_size['x']]
  
        pad_x = int(np.floor(self.patch_size['x']/2))
        pad_y = int(np.floor(self.patch_size['y']/2))



        x_element = self.x[:, idx_y + pad_y, idx_x + pad_x]


        if 'SZA' in self.x_features:
            
            szas, azis, = [], []
            lat, lon = x_element[1], x_element[2]
            latitude = lat.item()
            longitude = lon.item()
            altitude = 0
            thetime = pd.to_datetime(self.seviri.time.values)
        
            self.obs.date = ephem.Date(thetime)
            self.obs.lat = str(latitude)
            self.obs.lon = str(longitude)
            self.obs.elevation = altitude if not np.isnan(altitude) else 0

            self.sun.compute(self.obs)
            azis = self.sun.az
            szas = np.pi/2 - self.sun.alt
            
            azis = torch.tensor(azis, dtype=self.dtype).view(-1)
            szas = torch.tensor(szas, dtype=self.dtype).view(-1)

            x_element = torch.cat([x_element, szas, azis], dim=0)

        y_element = self.y[:, idx_y*self.patch_size['stride_y'] + pad_y, idx_x*self.patch_size['stride_x']+ pad_x]

        return X_element, x_element, y_element

    def __getitem__(self, i):
        
        X_element, x_element, y_element = self.get_item_in_order(i)

        if X_element.isnan().any():
            print("nan in X")
            print(X_element)
        if x_element.isnan().any():
            print("nan in x")
            print(x_element)
        if y_element.isnan().any():
            print("nan in y")
            print(y_element)

        if self.transform:
            X_element = self.transform(X_element.unsqueeze(0), self.x_vars).squeeze()
            x_element = self.transform(x_element.unsqueeze(0), self.x_features).squeeze()

        if self.target_transform:
            y_element = self.target_transform(y_element.unsqueeze(0), self.y_vars).squeeze()

        return X_element, x_element, y_element

    def get_latlon_patch(self, i):
        idx_y, idx_x = self.get_indices(i)
        a = slice(idx_y*self.patch_size['stride_y'], idx_y*self.patch_size['stride_y'] + self.patch_size['y'])
        b = slice(idx_x*self.patch_size['stride_x'], idx_x*self.patch_size['stride_x'] + self.patch_size['x'])
        lat_patch = self.lat[a,b]   
        lon_patch = self.lon[a,b]
        return lat_patch, lon_patch

    def get_patch_xarray(self, i, transform_back=False):

        lat_patch, lon_patch = self.get_latlon_patch(i)
        X, x, y = self[i]

        X[X == -99] = torch.nan
        x[x == -99] = torch.nan

        X_xr = xarray.Dataset(
            data_vars={name: (("lat", "lon"), d) for name, d in zip(self.x_vars, X)},
            coords={
                "lat_patch": (("lat", "lon"), lat_patch.data),
                "lon_patch": (("lat", "lon"), lon_patch.data),
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

class ForecastingDataset(Dataset):
    def __init__(
        self,
        y_vars,
        x_vars,
        x_features,
        patch_size,
        transform=None,
        target_transform=None,
        dtype=torch.float16,
        chunk_time=512,
        subset_time=None
    ):

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform
        self.chunk_time = chunk_time
        self.dtype=dtype

        input_zarr = '/scratch/snx3000/kschuurm/ZARR/SEVIRI_FULLDISK_Italy_chunk.zarr'

        self.seviri = xarray.open_zarr(
                input_zarr
            ).rename_dims({"x": "lon", "y": "lat"})\
                .rename_vars(
                {
                    "x": "lon",
                    "y": "lat",
                }
            )
        
        self.seviri_zarr = zarr.open('/scratch/snx3000/kschuurm/ZARR/SEVIRI_FULLDISK_Italy_chunk.zarr', mode='r')
        
        if subset_time is not None:
            self.start_idx = (self.seviri.time > np.datetime64(subset_time.start)).values.argmax()
            self.stop_idx = (self.seviri.time > np.datetime64(subset_time.stop)).values.argmax()
            if self.stop_idx == 0:
                self.stop_idx = len(self.seviri.time)
            print(self.start_idx, self.stop_idx)
            self.seviri = self.seviri.sel(time=subset_time)

        seviri_trans = {
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
            "IR_134": "channel_11",}
        
        nms = self.seviri.channel.values
        nms_trans = [seviri_trans[x] for x in nms]
        self.seviri['channel'] = nms_trans
        self.x_vars_available = [x for x in self.x_vars if x in nms_trans]

        self.dem = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/DEM.zarr").fillna(0).load()

        sizes= self.seviri.sizes
        self.H = sizes['lat']
        self.W = sizes['lon']
        self.pad = int(np.floor(patch_size['x']/2))

    def __len__(self):
        return int((self.H- 2*self.pad)*(self.W - 2*self.pad)*
                   (np.ceil(len(self.seviri.time)/self.chunk_time)))
    
    def get_indices(self, i):
        idx_x = int(i % (self.W - 2*self.pad))
        idx_y = int(
            (i // (self.W - 2*self.pad)) % (self.H - 2*self.pad)
            )
        idx_t = int((i // ((self.W - 2*self.pad)*(self.H - 2*self.pad))) % self.chunk_time)
        return idx_x, idx_y, idx_t
    
    def __getitem__(self, index):
        
        idx_x, idx_y, idx_t = self.get_indices(index)
        idx_x_da = xarray.DataArray([idx_x + self.pad], dims=['z'])
        idx_y_da = xarray.DataArray([idx_y + self.pad], dims=['z'])
        idx_x_patch = range(idx_x, idx_x+2*self.pad+1)
        idx_y_patch = range(idx_y, idx_y+2*self.pad+1)
        idx_x_patch_da = xarray.DataArray(idx_x_patch, dims=['lon'])
        idx_y_patch_da = xarray.DataArray(idx_y_patch, dims=['lat'])


        idx_x_patch_da, idx_y_patch_da = xarray.broadcast(idx_x_patch_da, idx_y_patch_da)



        
        X = self.seviri_zarr['channel_data'][[7,8,0,1,9,10,2,3,4,5,6], 
                                slice(self.start_idx + idx_t*self.chunk_time, 
                                    min(self.start_idx + (1+idx_t)*self.chunk_time, self.stop_idx)), 
                                slice(idx_y, idx_y + 2*self.pad + 1),
                                slice(idx_x, idx_x + 2*self.pad + 1), 
                                ]
        X = torch.tensor(X, dtype=self.dtype).permute(1,0,2,3) # BxCxHxW

        # X = torch.tensor(
        #     self.seviri.channel_data.sel(channel=self.x_vars_available) \
        #                     .isel(time=slice(idx_t*self.chunk_time, (1+idx_t)*self.chunk_time)
        #                         , lat = idx_y_patch_da, lon=idx_x_patch_da).values
        #                         , dtype=self.dtype 
        # ).permute(1,0,2,3) # CxBxHxW

        subset_x = self.seviri.isel(time=slice(idx_t*self.chunk_time, min((1+idx_t)*self.chunk_time, len(self.seviri.time))),
                                    lat = idx_y_da, lon=idx_x_da)
        N_chunk_time = len(subset_x.time)

        D = self.dem['DEM'].isel(lat = idx_y_patch_da, lon=idx_x_patch_da).values # BxHxW
        D = torch.tensor(D, dtype=self.dtype).repeat(N_chunk_time, 1, 1)[:, None, :, :]
        X = torch.cat([X,D], dim=1) # BxCxHxW

        altitude = self.dem['DEM'].isel(lat = idx_y_da, lon=idx_x_da).item()
        x_dict = {}   
        
        datetimes = pd.to_datetime(subset_x.time.values)
        x_dict['dayofyear'] = torch.tensor(subset_x.time.dt.dayofyear.values).view(-1,1)

        lat = subset_x.lat.item()
        lon = subset_x.lon.item()
        x_dict['lat'] = torch.tensor(lat, dtype=self.dtype).repeat(N_chunk_time).view(-1,1)
        x_dict['lon'] = torch.tensor(lon, dtype=self.dtype).repeat(N_chunk_time).view(-1,1)


        solarposition = SolarPosition(datetimes, lat, lon, altitude).get_solarposition()

        clearsky = Clearsky(datetimes, lat, lon, altitude, solarposition=solarposition)
        ghi_cls = torch.tensor(clearsky.get_clearsky(), dtype=self.dtype).view(-1, 1)

        x_dict['AZI'] = torch.tensor(solarposition['apparent_azimuth'], dtype=self.dtype).view(-1, 1)
        x_dict['SZA'] = torch.tensor(solarposition['apparent_zenith'], dtype=self.dtype).view(-1, 1)

        x = torch.cat([x_dict[k] for k in self.x_features], dim=1)


        if self.transform:
            X = self.transform(X, self.x_vars)
            x = self.transform(x, self.x_features)

        return X, x, ghi_cls


class ForecastingDataset2(Dataset):
    def __init__(
        self,
        y_vars,
        x_vars,
        x_features,
        patch_size,
        transform=None,
        target_transform=None,
        dtype=torch.float16,
        subset_time=None
    ):

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform
        self.dtype=dtype

        input_zarr = '/scratch/snx3000/kschuurm/ZARR/SEVIRI_FULLDISK_Italy.zarr'
        solarpos_zarr = '/scratch/snx3000/kschuurm/ZARR/SOLARPOS_Italy.zarr'

        self.seviri = xarray.open_zarr(
                input_zarr
            ).rename_dims({"x": "lon", "y": "lat"})\
                .rename_vars(
                {
                    "x": "lon",
                    "y": "lat",
                }
            )
        
        self.seviri_zarr = zarr.open(input_zarr, mode='r')

        self.solarpos = xarray.open_zarr(solarpos_zarr)
        self.solarpos_zarr = zarr.open(solarpos_zarr, mode='r')
        
        assert (self.seviri.time.values == self.solarpos.time.values).all(), "time mismatch"


        if subset_time is not None:
            self.subset_time = subset_time
            self.start_idx = (self.seviri.time >= np.datetime64(subset_time.start)).values.argmax()
            self.stop_idx = (self.seviri.time > np.datetime64(subset_time.stop)).values.argmax()
            if self.stop_idx == 0:
                self.stop_idx = len(self.seviri.time)
            print(self.start_idx, self.stop_idx)
            self.seviri = self.seviri.sel(time=subset_time)
            self.solarpos = self.solarpos.sel(time=subset_time)

        seviri_trans = {
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
            "IR_134": "channel_11",}
        
        nms = self.seviri.channel.values
        print(nms)
        nms_trans = [seviri_trans[x] for x in nms]
        self.seviri['channel'] = nms_trans
        print(self.seviri.channel.values)
        self.x_vars_available = [x for x in self.x_vars if x in nms_trans]

        self.dem = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/DEM.zarr").fillna(0).load()
        self.dem = self.dem.sel(lat=self.seviri.lat, lon=self.seviri.lon)

        sizes= self.seviri.sizes
        self.H = sizes['lat']
        self.W = sizes['lon']
        self.pad = int(np.floor(patch_size['x']/2))
        self.scan_H = self.H - 2*self.pad
        self.scan_W = self.W - 2*self.pad


    def __len__(self):
        return len(self.seviri.time)
    

    
    def __getitem__(self, i):
        
        X = self.seviri_zarr['channel_data'][[7,8,0,1,9,10,2,3,4,5,6], 
                                slice(self.start_idx + i, self.start_idx + i + 1), 
                                :,
                                :, 
                                ] # CxTxHxW
        
        
        X = torch.tensor(X, dtype=self.dtype).squeeze() # CxHxW
        X = X.unfold(1, 15, 1).unfold(2, 15, 1) # CxHxWx15x15
        X = X.permute(1,2,0,3,4).reshape(-1, 11, 15, 15) # BxCx15x15


        D = self.dem['DEM'].values # HxW
        D = torch.tensor(D, dtype=self.dtype)
        D = D.unfold(0,15,1).unfold(1,15,1) # HxWx15x15
        D = D.reshape(-1, 1, 15,15)
      
        X = torch.cat([X,D], dim=1) # BxCxHxW

        subset_x = self.seviri.isel(time=i)

        x_dict = {}   
        x_dict['dayofyear'] = torch.tensor(subset_x.time.dt.dayofyear.values).view(-1,1).repeat(X.shape[0], 1)

        llat, llon = xarray.broadcast(subset_x.lat, subset_x.lon)
        llat = torch.tensor(llat.values, dtype=self.dtype)[self.pad:-self.pad, self.pad:-self.pad].reshape(-1,1)
        llon = torch.tensor(llon.values, dtype=self.dtype)[self.pad:-self.pad, self.pad:-self.pad].reshape(-1,1)
        x_dict['lat'] = llat
        x_dict['lon'] = llon

        x_aziclssza = self.solarpos_zarr.data[:,
                                self.start_idx + i, 
                                :, :,
                                ]
        AZI = torch.tensor(x_aziclssza[0], dtype=self.dtype)[self.pad:-self.pad, self.pad:-self.pad]
        AZI = AZI.reshape(-1,1)
        SZA = torch.tensor(x_aziclssza[2], dtype=self.dtype)[self.pad:-self.pad, self.pad:-self.pad].reshape(-1,1)
        x_dict['AZI'] = AZI
        x_dict['SZA'] = SZA

        ghi_cls = torch.tensor(x_aziclssza[1], dtype=self.dtype)[self.pad:-self.pad, self.pad:-self.pad].reshape(-1,1)

        x = torch.cat([x_dict[k] for k in self.x_features], dim=1)


        if self.transform:
            X = self.transform(X, self.x_vars)
            x = self.transform(x, self.x_features)

        return X, x, ghi_cls


            

def pickle_seviri_dataset(config):
    dataset = SeviriDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        patches_per_image=config.batch_size,
    )

    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=24)

    for i in tqdm(range(0, len(dataset), 1000), desc='pickling'):
        
        subset = torch.utils.data.Subset(dataset, range(i, i+1000))
        dl = DataLoader(subset, batch_size=None, shuffle=False, num_workers=24)
        X_ls, x_ls, y_ls = [], [], []
        for j, batch in enumerate(dl):
            X_ls.append(batch[0])
            x_ls.append(batch[1])
            y_ls.append(batch[2])
        X = torch.cat(X_ls, dim=0)
        x = torch.cat(x_ls, dim=0)
        y = torch.cat(y_ls, dim=0)
        print(X.shape)
        torch.save((X,x,y), f"/scratch/snx3000/kschuurm/irradiance_estimation/dataset/pickled/seviri_{i}.pt")
        del X, x, y, dl, X_ls, x_ls, y_ls
        gc.collect()


if __name__ == "__main__":
    

    from types import SimpleNamespace

    config = {
        "batch_size": 2048,
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
        "y_vars": ["SIS"],
        "x_features": ["dayofyear", "lat", "lon", "SZA", "AZI"],
        "transform": ZeroMinMax(),
        "target_transform": ZeroMinMax(),
    }
    config = SimpleNamespace(**config)


    # timeindex = pd.DatetimeIndex(pickle_read('/scratch/snx3000/kschuurm/ZARR/timeindices.pkl'))
    # timeindex = timeindex[(timeindex.hour >10) & (timeindex.hour <17)]
    # traintimeindex = timeindex[(timeindex.year == 2016)]
    # _, validtimeindex = valid_test_split(timeindex[(timeindex.year == 2017)])
    


    # dataset = SeviriDataset(
    #     x_vars=config.x_vars,
    #     y_vars=config.y_vars,
    #     x_features=config.x_features,
    #     patch_size=config.patch_size,
    #     transform=config.transform,
    #     target_transform=config.target_transform,
    #     patches_per_image=config.batch_size,
    #     validation=True,
    # )

    dataset= ForecastingDataset(
        y_vars=config.y_vars,
        x_vars=config.x_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
    )

    dl = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0,)

    for i, batch in enumerate(tqdm(dl)):
        # print(batch)
        if i> 10000:
            print(batch[0].shape)
            break