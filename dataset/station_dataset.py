
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from dataset.normalization import ZeroMinMax, MinMax
import xarray
import numpy as np
from tqdm import tqdm
from utils.etc import benchmark

class GroundstationDataset(Dataset):
    def __init__(self, station_name, y_vars, x_vars, x_features, patch_size=15, time_window=12, transform=None, target_transform=None):
        
        self.x_vars = x_vars
        self.x_features = x_features
        self.y_vars = y_vars
        self.station_seviri = xarray.open_zarr(f'/scratch/snx3000/kschuurm/ZARR/{station_name}/SEVIRI_{station_name}.zarr') \
                                    .drop_duplicates('time')
        self.station_seviri = self.station_seviri.rename(
            {'y':'lat','x':'lon'}
            ).rename_vars({
            'VIS006':'channel_1',
            'VIS008':'channel_2',
            'IR_016':'channel_3',
            'IR_039':'channel_4',
            'WV_062':'channel_5',
            'WV_073':'channel_6',
            'IR_087':'channel_7',   
            'IR_097':'channel_8',
            'IR_108':'channel_9',
            'IR_120':'channel_10',
            'IR_134':'channel_11',
        })

        self.dem = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/DEM.zarr').sel(lat=self.station_seviri.lat, lon=self.station_seviri.lon)
        # self.station_seviri = xarray.merge([self.station_seviri, self.DEM], join='exact')

        xlen = len(self.station_seviri.lon)
        imiddle = int(np.floor(xlen/2))
        phalf = int(np.floor(patch_size/2))

        self.station_seviri = self.station_seviri.isel(lat=slice(imiddle-phalf, imiddle+phalf +1),
                                                        lon=slice(imiddle-phalf, imiddle+phalf +1))
        self.dem = self.dem.isel(lat=slice(imiddle-phalf, imiddle+phalf +1),
                                lon=slice(imiddle-phalf, imiddle+phalf +1))

        self.station = xarray.open_zarr('/scratch/snx3000/kschuurm/ZARR/IEA_PVPS_europe.zarr')  \
                    .sel(station_name=station_name) \
                    .rename_vars({
                        'GHI':'SIS',
                        'DNI':'DNI',
                        # 'DIF':'SID',
                        'Kc':'KI',
                        'latitude':'lat',
                        'longitude':'lon',
                        'Azim': 'AZI',
                    }).load()

        self.station['SZA'] = (90 - self.station['Elev'] )/180*np.pi # SZA = 90 - Elev, [0, 90*] or [0, 1/2pi]
        self.station['AZI'] = self.station['AZI']/180*np.pi # AZI [0, 360*] or [0, 2pi]

        if 'dayofyear' in x_features:
            self.station['dayofyear'] = self.station.time.dt.dayofyear

        if 'lat' in x_features:
            self.station = self.station.rename_vars({
                'lat':'lat_',
                'lon':'lon_'
            })
            self.station = self.station.assign({'lat':self.station.lat_,
                                    'lon':self.station.lon_,
                                    'DEM':self.station.elevation})

        self.rolling_station = self.station.rolling(time=time_window,center=False) \
                                    .mean(skipna=True) # center=False puts the time index for the mean at the right most corner, we want the left most corner
        self.rolling_station['time'] = self.rolling_station['time'] - np.timedelta64(time_window, 'm')

        # select available time slices
        self.timeidxnotnan = np.load(f'/scratch/snx3000/kschuurm/ZARR/{station_name}/timeidxnotnan.npy')
        self.rolling_station = self.rolling_station.sel(time=self.timeidxnotnan)
        self.station_seviri = self.station_seviri.sel(time=self.timeidxnotnan)

        x_vars = [v for v in self.x_vars if v in self.station_seviri.keys()]
        self.X = torch.Tensor(self.station_seviri[x_vars].to_dataarray(dim="channels").values) # CxTxHxW
        self.X = self.X.permute(1,0, 2,3) # TxCxHxW
        if 'DEM' in self.x_vars:
            X_DEM = torch.Tensor(self.dem['DEM'].values) # HxW
            X_DEM = X_DEM[None, None, :, :].repeat(self.X.shape[0], 1,1,1)
            self.X = torch.cat([self.X, X_DEM], dim=1)

        self.y = torch.Tensor(self.rolling_station[self.y_vars].to_dataarray(dim="channels").values) # CxT
        self.y = self.y.permute(1,0) # TxC

        self.x = torch.Tensor(self.rolling_station[self.x_features].to_dataarray(dim="channels").values) # CxT
        self.x = self.x.permute(1,0) # TxC

        self.transform = transform
        self.target_transform = target_transform

        if self.transform:
            self.X = self.transform(self.X, self.x_vars)
            self.x = self.transform(self.x, self.x_features)
            
        if self.target_transform:
            self.y = self.target_transform(self.y, self.y_vars)
    
    def __len__(self):
        return len(self.timeidxnotnan)

    def __getitem__(self, i):
        X = self.X[i]
        x = self.x[i]
        y = self.y[i]
        return X, x, y

    def get_xarray(self, i):
        timeidx = self.timeidxnotnan[i]
        X_xr = self.station_seviri.sel(time=timeidx)[self.x_vars].to_dataarray(dim="channels")
        x_xr = self.station.sel(time=timeidx)[self.x_features].to_dataarray(dim='channels')
        y_xr = self.station.sel(time=timeidx)[self.y_vars].to_dataarray(dim='channels')
        return X_xr, x_xr, y_xr


class GroundstationDataset2(Dataset):
    def __init__(self, zarr_store, y_vars, x_vars, x_features, patch_size=15, transform=None, target_transform=None):
        
        self.x_vars = x_vars
        self.x_features = x_features
        self.y_vars = y_vars
        self.data = xarray.open_zarr(zarr_store)

        self.data = self.data.rename_vars({
                'GHI':'SIS',
            }).isel(time=(self.data.SZA < np.pi/2).compute())
        
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

        self.dem = xarray.open_zarr('/home/kr/Documents/Solar_Power_Forecasting/03_ground_stations/ZARR/DEM.zarr') \
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
            self.data = self.data.rename_vars({
                'lat':'lat_',
                'lon':'lon_'
            }).rename_vars({
                'lat_station':'lat',
                'lon_station':'lon',
            })
            # self.data = self.data.assign({'lat':self.data.lat_station,
            #                         'lon':self.data.lon_station,
            #                         'DEM':self.data.altitude_station})


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
        return X, x, y

    def get_xarray(self, i):
        timeidx = self.timeidxnotnan[i]
        X_xr = self.station_seviri.sel(time=timeidx)[self.x_vars].to_dataarray(dim="channels")
        x_xr = self.station.sel(time=timeidx)[self.x_features].to_dataarray(dim='channels')
        y_xr = self.station.sel(time=timeidx)[self.y_vars].to_dataarray(dim='channels')
        return X_xr, x_xr, y_xr


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

