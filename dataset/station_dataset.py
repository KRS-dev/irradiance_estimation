
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

class GroundstationDataset(Dataset):
    def __init__(self, zarr_store, y_vars, x_vars, x_features, 
                 patch_size=15, transform=None, target_transform=None, 
                 subset_year=None, binned=False, bin_size=50, sarah_idx_only=False,
                 SZA_max=85):
        
        
        self.x_vars = x_vars
        self.x_features = x_features
        self.y_vars = y_vars

        self.data = xarray.open_zarr(zarr_store)

        self.data = self.data.rename_vars({
                'GHI':'SIS',
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
            self.data = self.data.where((self.data.SZA < SZA_max*np.pi/180).compute(), drop=True)
        
        if binned:
            with benchmark('binned'):
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
                .sel(lat=self.data.lat, lon=self.data.lon)

        xlen = len(self.data.lat)
        imiddle = int(np.floor(xlen/2))
        phalf = int(np.floor(patch_size/2))

        self.dem = self.dem.isel(lat=slice(imiddle-phalf, imiddle+phalf +1),
                                lon=slice(imiddle-phalf, imiddle+phalf +1))
        self.data = self.data.isel(lat=slice(imiddle-phalf, imiddle+phalf +1),
                                lon=slice(imiddle-phalf, imiddle+phalf +1))

        if 'dayofyear' in x_features:
            self.data['dayofyear'] = self.data.time.dt.dayofyear

        if 'lat' in x_features:
            self.data = self.data.rename_dims({
                'lat':'lat_',
                'lon':'lon_'
            }).rename_vars({
                'lat':'lat_',
                'lon':'lon_',
                'lat_station':'lat',
                'lon_station':'lon',
            })


        x_vars = [v for v in self.x_vars if v in self.data.channel.values]
        self.X = torch.Tensor(self.data.channel_data.sel(channel=x_vars).values) # CxTxHxW
        self.X = self.X.permute(1,0, 2,3) # TxCxHxW
        if 'DEM' in self.x_vars:
            X_DEM = torch.Tensor(self.dem['DEM'].values) # HxW
            X_DEM = X_DEM[None, None, :, :].repeat(self.X.shape[0], 1,1,1)
            self.X = torch.cat([self.X, X_DEM], dim=1)

        self.y = torch.Tensor(self.data[self.y_vars].to_dataarray(dim="channels").values) # CxT
        self.y = self.y.permute(1,0) # TxC

        self.x = torch.Tensor(self.data[self.x_features].to_dataarray(dim="channels").values) # CxT
        self.x = self.x.permute(1,0) # TxC

        self.timeindices = self.data.time.values

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        X = self.X[i]
        x = self.x[i]
        y = self.y[i]

        if self.transform:
            self.X = self.transform(X, self.x_vars)
            self.x = self.transform(x, self.x_features)
            
        if self.target_transform:
            self.y = self.target_transform(y, self.y_vars)
        return X, x, y
    





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

    dataset = GroundstationDataset(
        station_name=station_name,
        y_vars=y_vars,
        x_vars=x_vars,
        x_features=x_features,
        patch_size=patch_size,
        time_window=time_window,
        transform=transform,
        target_transform=target_transform,
    )
    

    X, x, y = dataset[0]
    print(X)
    print(x)
    print(y)

    dataloader = DataLoader(dataset, 10000, shuffle=True)

    for X, x, y in tqdm(dataloader):
        pass

