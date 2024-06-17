
import os
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from dataset.normalization import ZeroMinMax, MinMax
import xarray
import numpy as np
from tqdm import tqdm
from utils.etc import benchmark
from utils.satellite_position import get_satellite_look_angles, coscattering_angle

import dask
dask.config.set(scheduler='synchronous')

class GroundstationDataset(Dataset):
    def __init__(self, zarr_store, y_vars, x_vars, x_features, 
                 patch_size=15, transform=None, target_transform=None, 
                 sarah_idx_only=False, subset_year=None, binned=False, bin_size=50, 
                 SZA_max=85, dtype=torch.float32):
        
        

        self.x_vars = x_vars
        self.x_features = x_features
        self.y_vars = y_vars

        self.data = xarray.open_zarr(zarr_store)

        self.data = self.data.rename_vars({
                'GHI':'SIS',
            })

        if 'lat' in self.data.variables.keys():
            self.data = self.data.rename_dims({
                'lat':'y',
                'lon':'x'
            }).rename_vars({
                'lat':'y',
                'lon':'x'
            })
            
        
        if subset_year:
            self.data = self.data.sel(time=self.data.time.dt.year == subset_year)
        
        if sarah_idx_only:
            sarah = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/SARAH3.zarr')
            sarah_time = set(self.data.time.values).intersection(set(sarah.time.values))
            sarah_time = np.sort(np.array(list(sarah_time)))
            sarah.close()
            self.data = self.data.sel(time=sarah_time)
        
        

        if SZA_max:
            self.data = self.data.isel(time=(self.data.SZA < SZA_max*np.pi/180).compute())
        
        
        
        if binned:
            SIS_max = self.data.SIS.max().values
            bins = np.arange(0, SIS_max + bin_size if SIS_max<1300 else 1300 + bin_size, bin_size)
            digitized = np.digitize(self.data.SIS, bins)
            size_bins = [np.sum(digitized == i) for i in range(1, len(bins))]
            samples_per_bin = np.quantile(size_bins, .25)
            idxs = []
            for i in range(1, len(bins)):
                sample_size =np.min([int(samples_per_bin), size_bins[i-1]])
                idxs.append(np.random.choice(np.argwhere(digitized == i).squeeze(), sample_size, replace=False))
            idxs = np.concatenate(idxs)
            self.data = self.data.isel(time=np.sort(idxs))
        
        
        
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
        
        nms = self.data.channel.values
        nms_trans = [seviri_trans[x] for x in nms]
        self.data['channel'] = nms_trans

        self.dem = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/DEM.zarr') \
                .sel(lat=self.data.y, lon=self.data.x)
        
        
        xlen = len(self.data.y)
        imiddle = int(np.floor(xlen/2))
        phalf = int(np.floor(patch_size/2))

        self.dem = self.dem.isel(y=slice(imiddle-phalf, imiddle+phalf +1),
                                x=slice(imiddle-phalf, imiddle+phalf +1))
        self.data = self.data.isel(y=slice(imiddle-phalf, imiddle+phalf +1),
                                x=slice(imiddle-phalf, imiddle+phalf +1))

        x_dict = {}
        N =len(self.data.time)

        self.lat_station = float(self.data.lat_station.values)
        self.lon_station = float(self.data.lon_station.values)

        if 'dayofyear' in x_features:
            dayofyear = self.data.time.dt.dayofyear
            x_dict['dayofyear'] = torch.tensor(dayofyear.values, dtype=dtype).view(-1, 1)

        if 'lat' in x_features:
            x_dict['lat'] = torch.tensor(self.lat_station, dtype=dtype).repeat(N, 1)
            x_dict['lon'] = torch.tensor(self.lon_station, dtype=dtype).repeat(N, 1)

        x_dict['SZA'] = torch.tensor(self.data.SZA.values, dtype=dtype).view(-1, 1) # sun zenith angle
        x_dict['AZI'] = torch.tensor(self.data.AZI.values, dtype=dtype).view(-1, 1) # azimuth angle sun

        if 'sat_AZI' in x_features:
            sat_azi, sat_sza = get_satellite_look_angles(self.lat_station, self.lon_station, degree=True, dtype=dtype)
            x_dict['sat_AZI'] = np.deg2rad(sat_azi).repeat(N,1)
            x_dict['sat_SZA'] = np.deg2rad(sat_sza).repeat(N,1)

        if 'coscatter_angle' in x_features:
            x_dict['coscatter_angle'] = coscattering_angle(sat_azi, sat_sza, 
                                                           x_dict['AZI'], x_dict['SZA'], dtype=dtype).view(-1,1)

        self.x = torch.cat([x_dict[k] for k in x_features], dim=1)

        x_vars = [v for v in self.x_vars if v in self.data.channel.values]
        self.X = torch.tensor(self.data.channel_data.sel(channel=x_vars).values, dtype=torch.float16) # CxTxHxW
        self.X = self.X.permute(1,0, 2,3) # TxCxHxW
        if 'DEM' in self.x_vars:
            X_DEM = torch.tensor(self.dem['DEM'].values, dtype=torch.float16) # HxW
            X_DEM = X_DEM[None, None, :, :].repeat(self.X.shape[0], 1,1,1)
            self.X = torch.cat([self.X, X_DEM], dim=1)

        self.y = torch.tensor(self.data[self.y_vars].to_dataarray(dim="channels").values, dtype=dtype) # CxT
        self.y = self.y.permute(1,0) # TxC

        self.timeindices = self.data.time.values

        self.transform = transform
        self.target_transform = target_transform

        if self.transform:
            self.X = self.transform(self.X, self.x_vars)
            self.x = self.transform(self.x, self.x_features)
            
        if self.target_transform:
            self.y = self.target_transform(self.y, self.y_vars)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        X = self.X[i]
        x = self.x[i]
        y = self.y[i]

        # if self.transform:
        #     X = self.transform(X.unsqueeze(0), self.x_vars).squeeze()
        #     x = self.transform(x.unsqueeze(0), self.x_features).squeeze()
            
        # if self.target_transform:
        #     y = self.target_transform(y.unsqueeze(0), self.y_vars).squeeze().view(1)
        return X.float(), x.float(), y.float()
    





if __name__ == '__main__':
    
    station_name = 'CAB'
    y_vars = ["SIS", "DNI", "SID"]
    x_vars = [
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
    ]
    x_features = ["dayofyear", "lat", "lon", "SZA", "AZI"]
    patch_size = 15
    time_window = 12 # minutes
    transform = ZeroMinMax()
    target_transform = ZeroMinMax()


