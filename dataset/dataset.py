import itertools
from typing import Any, Tuple

from tqdm import tqdm
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, Subset
import xarray
from datetime import timedelta
# import dask
# from xrpatcher import XRDAPatcher
from xbatcher import BatchGenerator
from glob import glob
from preprocess.etc import benchmark
import pytorch_lightning as L
from dask.distributed import Client



import xbatcher
from xbatcher.loaders.torch import MapDataset as xbMapDataset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class MSGDataset(Dataset):
    def __init__(self, dataset, y_var, x_vars=None, rechunk=None):

        if isinstance(dataset, str):
            self.ds = xarray.open_zarr(dataset)
        elif isinstance(dataset, xarray.Dataset):
            self.ds = dataset
        else:
            raise ValueError(f'{dataset} is not a zarr store or a xarray.Dataset.')
            
        self.attributes = self.ds.attrs

        vars = set(self.ds.keys())
        ignore_vars = set(['lon_bnds', 'lat_bnds', 'record_status', 'crs'])
        assert y_var in vars, 'y_var not available in the dataset'
        self.y_var = y_var
        self.x_vars = vars - ignore_vars - set([y_var,])

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        if self.ds is not None:
            xarr = self.ds[self.x_vars].isel(time=idx).to_array() # select time slice, reshape to t lon lat
            yarr = self.ds[self.y_var].isel(time=idx)
            x = torch.tensor(data=xarr.data, names=('C', 'T', 'Y', 'X'))
            y = torch.tensor(data=yarr.data, names=('C', 'T', 'Y', 'X'))      
            return x, y
        else:
            raise ValueError('There is no zarr dataset to get images from.')
        
class MSGDatasetBatched(Dataset):
    def __init__(self, zarr_store, y_var =None, x_vars=None, patch_size = (64,64), transform=None, target_transform=None):

        self.ds = xarray.open_zarr(zarr_store, )
        self.generator = BatchGenerator(self.ds, input_dims={'time':1, 'lat':patch_size[1], 'lon':patch_size[0]},)
        self.transform = transform
        self.target_transform = target_transform
        self.attrs = self.ds.attrs

        vars = set(self.ds.keys())
        assert y_var in vars, 'y_var not available in the dataset'
        self.y_var = y_var
        ignore_vars = set(['lon_bnds', 'lat_bnds', 'record_status', 'crs'])
        other_vars = vars - ignore_vars - set([y_var,])

        if x_vars is not None:
            assert set(x_vars) in other_vars, f'{set(x_vars)^other_vars} not in the dataset'
            self.x_vars = x_vars
        else:
            self.x_vars = other_vars

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

        a= self.generator[idx]
        X_batch = torch.tensor(a[self.x_vars].to_array().values)#, names=('B','C','T', 'X', 'Y')) 
        y_batch = torch.tensor(a[self.y_var].values)#, names=('B', 'T', 'X', 'Y'))  
        
        X_batch = X_batch.squeeze() #Squeeze out time dimension 
        y_batch = y_batch.squeeze() #.squeeze(-2).unsqueeze(1) (B, T, X, Y) -> (B, C, X, Y) where dim(C)=1
        if self.transform:
            X_batch = self.transform(X_batch)

        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch # ('B', 'C', 'X', 'Y') &  ('B', 'X', 'Y')



class MSGDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, patch_size: (int, int)=None, num_workers:int=12):
        super().__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers

    def setup(self, stage: str ):
        if stage == 'fit':
            if self.patch_size is not None:
                # Using xbatcher to make batches of patched data
                with benchmark('set up batch generator for patches'):
                    self.msg_train = MSGDatasetBatched('/scratch/snx3000/kschuurm/DATA/train.zarr', y_var='SIS', patch_size=self.patch_size)
                    self.msg_validation = MSGDatasetBatched('/scratch/snx3000/kschuurm/DATA/valid.zarr', y_var='SIS', patch_size=self.patch_size)
            else:
                self.msg_train = MSGDataset('/scratch/snx3000/kschuurm/DATA/train.zarr', y_var='SIS')
                self.msg_validation = MSGDataset('/scratch/snx3000/kschuurm/DATA/validation.zarr', y_var='SIS')

        if stage == 'test':
            if self.patch_size is not None:
                self.msg_test = MSGDatasetBatched('/scratch/snx3000/kschuurm/DATA/validation.zarr', y_var='SIS', patch_size=self.patch_size)
            else:
                self.msg_test = MSGDataset('/scratch/snx3000/kschuurm/DATA/test.zarr', y_var='SIS')

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.msg_train, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.msg_validation, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.msg_test, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)




if __name__ == '__main__':

    # with Client() as client:
    #     print(client)


    dm = MSGDataModule(batch_size=32, patch_size=(128,128), num_workers=12)
    dm.setup('fit')
    dl = dm.train_dataloader()
    
    for i, (X, y) in tqdm(enumerate(dl)):
        
        if i> 50:
            break
    

    plt.imshow(y.view(-1, 128))
    plt.colorbar()
    plt.savefig('testsamples.png')
