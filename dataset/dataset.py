
import torch
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

import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, y_vars, x_vars, x_features, patch_size, transform, target_transform, random_sample=.1):

        self.hrseviri = xarray.open_zarr('/scratch/snx3000/kschuurm/DATA/HRSEVIRI_corrected.zarr')
        self.hrseviri['DEM'] = self.hrseviri.DEM.fillna(0)
        print('pre dropna', len(self.hrseviri.time))
        self.hrseviri = self.hrseviri.dropna(dim='time')
        print('dropna', len(self.hrseviri.time))
        self.sarah = xarray.open_zarr('/scratch/snx3000/kschuurm/DATA/SARAH3.zarr')
        print('pre dropna', len(self.sarah.time))
        self.sarah = self.sarah.dropna(dim='time')
        print('dropna', len(self.sarah.time))


        intersec_times = pd.DatetimeIndex(set(self.hrseviri.time.values).intersection(self.sarah.time.values)).sort_values()

        self.images = intersec_times
        self.lat = self.hrseviri.lat
        self.lon = self.hrseviri.lon
        self.patch_size = patch_size
        patch_x = patch_size['x']; patch_y = patch_size['y']
        stride_x = patch_size['stride_x']; stride_y = patch_size['stride_y']
        pad_x = int(np.floor(patch_x/2)); pad_y = int(np.floor(patch_y/2))

        self.patches_per_image = int(np.floor((len(self.lat)-2*pad_y)/stride_y) * \
            np.floor((len(self.lon)-2*pad_x)/stride_x))
        
        self.random_sample = random_sample
        if self.random_sample:
            self.sample_size = int(np.floor(self.random_sample*self.patches_per_image))
            self.samples = np.random.randint(0, self.patches_per_image, self.sample_size)
        else:
            self.sample_size=self.patches_per_image

        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        self.transform = transform
        self.target_transform = target_transform

        self.X_image = None
        self.y_image = None
        self.image_i = None
        self.current_singleImageDataset = None

    def __len__(self):
        return len(self.images)*self.sample_size

    def load_new_image(self, image_i):
        self.image_i=image_i
        if self.current_singleImageDataset is None:
            dt= self.images[image_i]
            current_X_image = self.hrseviri.sel(time=dt)  
            current_y_image = self.sarah.sel(time=dt)
            self.current_singleImageDataset = SingleImageDataset(
                hrseviri=current_X_image,
                sarah=current_y_image,
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                x_features=self.x_features,
                patch_size=self.patch_size,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            self.current_singleImageDataset = self.next_singleImageDataset
        if image_i +1 < len(self.images):
            dt_next = self.images[image_i + 1]
            self.next_X_image = self.hrseviri.sel(time=dt_next)  
            self.next_y_image = self.sarah.sel(time=dt_next)
            self.next_singleImageDataset = SingleImageDataset(
                hrseviri=self.next_X_image,
                sarah=self.next_y_image,
                y_vars=self.y_vars,
                x_vars=self.x_vars,
                x_features=self.x_features,
                patch_size=self.patch_size,
                transform=self.transform,
                target_transform=self.target_transform,
            )

    def __getitem__(self, i):
        idx_image = int(np.floor(i/self.sample_size))
        idx_patch = self.samples[int(i % self.sample_size)]

        if self.image_i is None:
            self.load_new_image(idx_image)
        elif self.image_i != idx_image:
            self.load_new_image(idx_image)
                
        return self.current_singleImageDataset[idx_patch]


class SingleImageDataset(Dataset):
    def __init__(self, hrseviri, sarah, y_vars, x_vars, x_features, patch_size, transform, target_transform):
        super(SingleImageDataset, self).__init__()
        
        self.hrseviri = hrseviri
        self.sarah = sarah
        self.lat, self.lon = xarray.broadcast(hrseviri.lat, hrseviri.lon)   # size(HxW) both     
        
        self.x_features = x_features.copy()
        self.x_vars = x_vars.copy()
        self.y_vars = y_vars.copy()
        
        self.transform = transform
        self.target_transform = target_transform

        patch_x = patch_size['x']; patch_y = patch_size['y']
        stride_x = patch_size['stride_x']; stride_y = patch_size['stride_y']
        pad_x = int(np.floor(patch_size['x']/2))
        pad_y = int(np.floor(patch_size['y']/2))
        
        def unfold_tensor(X):
            # assume last two dimensions are HxW
            shape = X.shape # * ...xHxW
            X = X.unfold(-2,patch_y,stride_y).unfold(-2,patch_x, stride_x) # ....xH*xW*xPHxPW
            shape_unfold = list(shape[:-2]) + [-1] + [patch_x, patch_y] # ....xBxPHxPW   stacking H* and W* into a batch dimension
            X = X.reshape(shape_unfold)
            permute_l = list(range(0, len(shape_unfold)))
            loc_B = permute_l.pop(-3) # Pop dim location of the batch from the sequence
            permute_l.insert(0, loc_B) # insert dim batch in the front
            X = X.permute(permute_l) # Bx ... xPHxPW
            return X

        # Unfold image to patches
        self.X_image = torch.Tensor(self.hrseviri[x_vars].to_dataarray(dim="channels").values) # Cx'T'xHxW
        self.X = unfold_tensor(self.X_image)
        
        self.lat_patches = torch.Tensor(self.lat.values.copy()) # HxW
        self.lat_patches = unfold_tensor(self.lat_patches) # BxPHxPW
        
        self.lon_patches = torch.Tensor(self.lon.values.copy()) # HxW
        self.lon_patches = unfold_tensor(self.lon_patches) # BxPHxPW

        # Manipulate point features
        x_features_avail = set(x_features).intersection(set(self.hrseviri.keys()))
        self.x_features_temp = x_features.copy()
        self.x = self.hrseviri[x_features_avail]
        
        if 'lat' in x_features:
            self.x = self.x.assign({'lat_':self.lat,
                                       'lon_':self.lon,})
            self.x_features_temp[1] = 'lat_'
            self.x_features_temp[2] = 'lon_'
            
        if 'dayofyear' in x_features:
            dayofyear = self.hrseviri.time.dt.dayofyear.astype(int)
            dayofyear, _ = xarray.broadcast(dayofyear, self.lat)
            self.x = self.x.assign({'dayofyear':dayofyear,})
        
        self.x = torch.Tensor(self.x[self.x_features_temp].to_dataarray(dim='channels').values)# Cx...xHxW
        # reorder self.x by the original order of the x_feature list
        shape = self.x.shape
        permute_l = list(range(len(self.x.shape))); permute_l = permute_l[-2:] + permute_l[:-2]
        self.x = self.x.permute(permute_l) # HxWxCx ....
        self.x = self.x[pad_y:-pad_y,pad_x:-pad_x].reshape(-1, *shape[:-2]) # BxCx ..., Make sure that padding due to patchsize is incorporated, then reshape to batch dim
        
        assert self.x.shape[0] == self.X.shape[0], print('batch size point features not equal to patches', self.x.shape, self.X.shape[0]) # Batch sizes are equal
        
        # Manipulate output 
        self.y = torch.Tensor(self.sarah[y_vars].to_dataarray(dim='channels').values) # Cx ... xHxW
        shape = self.y.shape
        permute_l = list(range(len(shape)))
        permute_l = permute_l[-2:] + permute_l[:-2]
        self.y = self.y.permute(permute_l) # HxWxCx...
        self.y = self.y[pad_x:-pad_x,pad_y:-pad_y].reshape(-1, *shape[:-2]) # BxC...
        
        
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
            print('nan in X')
            print(X)
        if x_element.isnan().any():
            print('nan in x')
            print(x)
        if y_element.isnan().any():
            print('nan in y')
            print(y)
        
        return X_element, x_element, y_element
    
    def get_latlon_patch(self, i):
        lat_patch = self.lat_patches[i]
        lon_patch = self.lon_patches[i]
        return lat_patch, lon_patch
    
    def get_patch_xarray(self, i, transform_back=False):
        
        lat_patch, lon_patch = self.get_latlon_patch(i)
        X, x, y = self[i]
        
        X[X==-99] = torch.nan
        x[x==-99] = torch.nan
        
        X_xr = xarray.Dataset(
            data_vars={
                name: (('lat','lon'),d) for name, d in zip(self.x_vars, X)
            },
            coords={
                'lat_patch': (('lat', 'lon'), lat_patch),
                'lon_patch': (('lat', 'lon'), lon_patch),
                'time': self.hrseviri.time,
            }
        )
        
        x_xr = xarray.Dataset(
            data_vars={
                name: d for name, d in zip(self.x_features, x)
            },
            coords={
                'time':self.hrseviri.time,
            }
        )
        
        if self.transform:
            lat = self.transform.inverse(x_xr.lat, ['lat'])
            lon = self.transform.inverse(x_xr.lon, ['lon'])
            x_xr['lat'] = lat
            x_xr['lon'] = lon
        
        x_xr = x_xr.set_coords(['lat','lon'])
        y_xr = xarray.Dataset(
            data_vars={
                name: d for name, d in zip(self.y_vars, y)
            },
            coords={
                'time':self.hrseviri.time,
                'lat':x_xr.lat,
                'lon':x_xr.lon,
            }
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
        "patch_size": {'x':15, 
                    'y':15,
                    'stride_x':1,
                    'stride_y':1,
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
            "DEM"
        ],
        "y_vars": ["SIS"],
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
    )

    dataloader= torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=0)

    for X, x, y in tqdm(dataloader):
        print(X.shape, x.shape, y.shape)
        break