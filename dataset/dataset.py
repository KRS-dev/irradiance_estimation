import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import xarray, zarr
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from glob import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import ephem
import dask, gc

from dataset.normalization import ZeroMinMax
from utils.clearsky import Clearsky, SolarPosition
from utils.etc import pickle_read, pickle_write

# Essential config when working in jupyter notebooks with these datasets. Otherwise the dask scheduler will not work properly.
dask.config.set(scheduler='synchronous')


class SamplesDataset(Dataset):
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
        ).drop_duplicates('time')
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
        elif validation == 'test':
            self.timeindices  = self.sarah_nulls.time[self.sarah_nulls.time.dt.year == 2023].values
        else:
            self.timeindices  = self.sarah_nulls.time[self.sarah_nulls.time.dt.year < 2022].values

        self.timeindices = np.sort(np.array(list(set(self.timeindices).intersection(set(self.seviri.time.values)))))
        

        if validation is True or validation == 'test':
            if validation == 'test':
                nm_add = '_test'
            else:
                nm_add = ''
            fn1 = f'/scratch/snx3000/kschuurm/ZARR/idx_x_sampler{nm_add}.pkl'
            fn2 = f'/scratch/snx3000/kschuurm/ZARR/idx_y_sampler{nm_add}.pkl'

            if os.path.exists(fn1) and os.path.exists(fn2):
                self.idx_x_sampler = pickle_read(fn1)
                self.idx_y_sampler = pickle_read(fn2)
            else:
                self.idx_x_sampler = {}
                self.idx_y_sampler = {}
                for timeidx in tqdm(self.timeindices, desc='sampler setup', total=len(self.timeindices)):


                    notnulls = self.sarah_nulls.nulls.sel(time=timeidx).load()

                    coords_notnull = np.argwhere(np.array(notnulls[self.pad:-self.pad, self.pad:-self.pad]))
                    samples = coords_notnull[torch.randint(0, len(coords_notnull), (self.patches_per_image,), dtype=torch.int32, generator=self.rng)]
                    idx_x_samples = self.pad + samples[:,1]
                    idx_y_samples = self.pad + samples[:,0]

                    self.idx_x_sampler[timeidx] = idx_x_samples
                    self.idx_y_sampler[timeidx] = idx_y_samples
                
                pickle_write(self.idx_x_sampler, fn1)
                pickle_write(self.idx_y_sampler, fn2)

    def __len__(self):
        return len(self.timeindices)
    
    def __getitem__(self, i):
        timeidx= self.timeindices[i]
        subset_seviri = self.seviri.sel(time = timeidx).load()
        subset_sarah = self.sarah.sel(time = timeidx).load()

        if self.validation is True or self.validation == 'test':
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

        X = subset_seviri.channel_data.sel(channel=self.x_vars_available) \
                            .isel(lat = idx_y_patch_da, lon=idx_x_patch_da) \
                            .values # CxBxHxW
        
        X = torch.tensor(X, dtype=self.dtype).permute(1,0,2,3) # BxCxHxW

        if 'DEM' in self.x_features:
            x_DEM = self.dem.DEM.isel(lon=idx_x_da, lat=idx_y_da) \
                            .values
            x_DEM = torch.tensor(x_DEM, dtype=self.dtype).view(-1,1)
            x = torch.cat([x, x_DEM], dim=1)

        if 'DEM' in self.x_vars:
            D = self.dem['DEM'].isel(lat = idx_y_patch_da, lon=idx_x_patch_da).values # BxHxW
            D = torch.tensor(D, dtype=self.dtype)[:, None, :, :]
            X = torch.cat([X,D], dim=1) # BxCxHxW
        
        if 'SZA' in self.x_features:
            sun = ephem.Sun()
            szas, azis, = [], []
            for lat, lon in zip(x_lat, x_lon):
                latitude = lat.item()
                longitude = lon.item()
                altitude = self.dem['DEM'].sel(lat=lat, lon=lon, method='nearest').item()
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


class ImageDataset(Dataset):
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
        """Image inference dataset from ZARR datasets.
        Uses a precomputed SOLARPOS zarr to speed up the inference.
        
        
        """

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
        assert (self.seviri.lat.values == self.solarpos.lat.values).all(), "lat mismatch"
        assert (self.seviri.lon.values == self.solarpos.lon.values).all(), "lon mismatch"


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

        nms_trans = [seviri_trans[x] for x in nms]
        self.seviri['channel'] = nms_trans
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


def pickle_seviri_dataset(config, validation=False):
    dataset = SamplesDataset(
        x_vars=config.x_vars,
        y_vars=config.y_vars,
        x_features=config.x_features,
        patch_size=config.patch_size,
        transform=config.transform,
        target_transform=config.target_transform,
        patches_per_image=config.batch_size,
        validation=validation,
    )

    dl = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=24)
    for i, (X,x,y) in enumerate(tqdm(dl)):

        pickle_write((X,x,y), f"/scratch/snx3000/kschuurm/irradiance_estimation/dataset/pickled/seviri_val_{i}.pkl")
        del X, x, y
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


    dataset= ImageDataset(
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