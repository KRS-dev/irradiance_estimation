import os
import pickle
import random
import torch
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import xarray
from datetime import timedelta
import lightning.pytorch as L
import numpy as np
import pandas as pd
import ephem
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from utils.etc import benchmark

def create_singleImageDataset(**kwargs):
    return SingleImageDataset(**kwargs)

def create_singleImageDataset_generator(**kwargs):
    return SingleImageDataset_generator(**kwargs)

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
            )
        )
        self.dem = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/DEM.zarr").fillna(0)

        self.sarah = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/SARAH3.zip").channel_data.to_dataset(dim='channel') 
     
        self.seviri = xarray.merge(
            [self.seviri, self.dem], join="exact"
        )  # throws an error if lat, lon not the same

        self.sarah_nulls = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3_nulls.zip')
        timeindices_sarah = self.sarah_nulls.any.where(self.sarah_nulls.any == True, drop=True).time.values

        if timeindices is not None:
            self.timeindices = timeindices
        else:
            self.timeindices = timeindices_sarah

        self.timeindices = np.array(list(set(self.timeindices.values).intersection(set(self.seviri.time.values))))

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
        dataset = self._pool.submit(create_singleImageDataset_generator, **d)
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
        transform,
        target_transform,
        timeindices=None,
        patches_per_image =100,
        dtype=torch.float16,
        seed=None,
        rng=None
    ):

        self.seed = seed
        if self.seed is not None:
            self.rng = torch.Generator().manual_seed(self.seed)
        else:
            self.rng = rng # set random generator for all datasets in ddp

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform
        self.patches_per_image = patches_per_image
        self.dtype=dtype

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

        self.sarah = xarray.open_zarr("/scratch/snx3000/kschuurm/ZARR/SARAH3.zip")

        self.sarah_nulls = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3_nulls.zarr')

    
        sizes= self.seviri.sizes
        self.H = sizes['lat']
        self.W = sizes['lon']

        self.pad = int(np.floor(patch_size['x']/2))

        # timeindices_sarah, self.max_y, self.max_x, self.min_y, self.min_x  = get_pickled_sarah_bnds()
        timeindices_sarah = self.sarah_nulls['any'].where((self.sarah_nulls['any'] == True).compute(), drop=True).time.values

        if timeindices is not None:
            self.timeindices = timeindices
        else:
            self.timeindices = timeindices_sarah


        self.timeindices = np.array(list(set(self.timeindices.values).intersection(set(self.seviri.time.values))))
        
        self.timeindices = self.timeindices[torch.randperm(len(self.timeindices))]

        if self.seed is not None:
            with benchmark('sampler setup'):
                self.idx_x_sampler = []
                self.idx_y_sampler = []

                if os.path.exists('/scratch/snx3000/kschuurm/ZARR/idx_x_sampler.pkl') and os.path.exists('/scratch/snx3000/kschuurm/ZARR/idx_y_sampler.pkl'):
                    self.idx_x_sampler = pickle_read('/scratch/snx3000/kschuurm/ZARR/idx_x_sampler.pkl')
                    self.idx_y_sampler = pickle_read('/scratch/snx3000/kschuurm/ZARR/idx_y_sampler.pkl')
                else:
                    for timeidx in tqdm(self.timeindices):


                        notnulls = self.sarah_nulls.nulls.sel(time=timeidx).load()

                        coords_notnull = np.argwhere(np.array(notnulls[self.pad:-self.pad, self.pad:-self.pad]))
                        samples = coords_notnull[torch.randint(0, len(coords_notnull), (self.patches_per_image,), dtype=torch.int32, generator=self.rng)]
                        idx_x_samples = self.pad + samples[:,0]
                        idx_y_samples = self.pad + samples[:,1]

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
                        self.idx_x_sampler.append(idx_x_samples)
                        self.idx_y_sampler.append(idx_y_samples)
                    
                    pickle_write(self.idx_x_sampler, '/scratch/snx3000/kschuurm/ZARR/idx_x_sampler.pkl')
                    pickle_write(self.idx_y_sampler, '/scratch/snx3000/kschuurm/ZARR/idx_y_sampler.pkl')

    def __len__(self):
        return len(self.timeindices)
    
    def __getitem__(self, i):
        timeidx= self.timeindices[i]
        print(timeidx)
        subset_sarah = self.sarah.sel(time = timeidx).load()
        subset_seviri = self.seviri.sel(time = timeidx).load()

        if self.seed is not None:
            idx_x_samples = self.idx_x_sampler[i]
            idx_y_samples = self.idx_y_sampler[i]
        else:

            notnulls = self.sarah_nulls.nulls.sel(time=timeidx).load()
            print(notnulls.shape)
            

            coords_notnull = np.argwhere(np.array(notnulls[self.pad:-self.pad, self.pad:-self.pad]))
            print(coords_notnull.max(axis=0))
            print(coords_notnull.min(axis=0))

            samples = coords_notnull[torch.randint(0, len(coords_notnull), (self.patches_per_image,))]

            idx_x_samples = self.pad + samples[:,1]
            idx_y_samples = self.pad + samples[:,0]
            print(idx_x_samples)
            print(idx_y_samples)
            # min_x = int(self.min_x.sel(time=timeidx).values)
            # min_y = int(self.min_y.sel(time=timeidx).values)
            # max_x = int(self.max_x.sel(time=timeidx).values)
            # max_y = int(self.max_y.sel(time=timeidx).values)
            # min_x =0
            # min_y = 0
            # max_x = 736-1
            # max_y = 658-1
            # idx_x_samples = torch.randint(min_x + self.pad, 
            #                             max_x-self.pad, 
            #                             (self.patches_per_image,), 
            #                             dtype=torch.int32,)
            # idx_y_samples = torch.randint(min_y + self.pad, 
            #                             max_y-self.pad, 
            #                             (self.patches_per_image,), 
            #                             dtype=torch.int32,)
        
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
            print(X)
        if x.isnan().any():
            print("nan in x")
            print(x)
        if y.isnan().any():
            print("nan in y")
            print(y)

        if (y == -1).any():
            print('zeros in output', sum(y == -1))

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



if __name__ == "__main__":


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
        "y_vars": ["SIS"],
        "x_features": ["dayofyear", "lat", "lon", "SZA", "AZI"],
        "transform": None,
        "target_transform": None,
    }
    config = SimpleNamespace(**config)


    timeindex = pd.DatetimeIndex(pickle_read('/scratch/snx3000/kschuurm/ZARR/timeindices.pkl'))
    timeindex = timeindex[(timeindex.hour >10) & (timeindex.hour <17)]
    traintimeindex = timeindex[(timeindex.year == 2016)]
    _, validtimeindex = valid_test_split(timeindex[(timeindex.year == 2017)])
    

    # print(validtimeindex)

    dataset = SeviriDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        timeindices=validtimeindex,
        transform=config.transform,
        target_transform=config.target_transform,
    )


    dl = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4,)

    for i, batch in enumerate(tqdm(dl)):
        # print(batch)
        if i> 100:
            break