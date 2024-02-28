from typing import Any, Tuple
import xarray
import numpy as np
import torch

MINMAX_old = {
    "SIS": (0.0, 1109.0),
    "CAL": (0.0, 1.0),
    "SID": (0.0, 600),
    "SRTM": (-7.6700854700854695, 3746.053675213676),
    "DEM": (-7.6700854700854695, 3746.053675213676),
    "channel_1": (-0.09116022288799286, 18.401456832885742),
    "channel_10": (13.867655754089355, 177.00460815429688),
    "channel_11": (20.31503677368164, 117.05536651611328),
    "channel_2": (-0.17268221080303192, 21.99477767944336),
    "channel_3": (-0.2678762376308441, 13.346491813659668),
    "channel_4": (-0.008304266259074211, 2.9166910648345947),
    "channel_5": (0.44394731521606445, 6.937240123748779),
    "channel_6": (1.3654084205627441, 23.604351043701172),
    "channel_7": (3.807188034057617, 108.10502624511719),
    "channel_8": (9.00473403930664, 74.256103515625),
    "channel_9": (10.230449676513672, 165.13319396972656),
    "dayofyear": (1, 365),
    "lat": (-90, 90),
    "lon": (-180, 180),
    "SZA": (0, np.pi/2),
    "AZI": (0, 2*np.pi),
}

MINMAX = {
    "SIS": (0.0, 1109.0),
    "CAL": (0.0, 1.0),
    "SID": (0.0, 600),
    "DNI": (0.0, 1100.0),
    "DEM": (-7.6700854700854695, 3746.053675213676),
    "channel_1": (0, 1),
    "channel_10": (0, 1),
    "channel_11": (0, 1),
    "channel_2": (0, 1),
    "channel_3": (0, 1),
    "channel_4": (0, 1),
    "channel_5": (0, 1),
    "channel_6": (0, 1),
    "channel_7": (0, 1),
    "channel_8": (0, 1),
    "channel_9": (0, 1),
    "dayofyear": (1, 365),
    "lat": (-90, 90),
    "lon": (-180, 180),
    "SZA": (0, np.pi/2),
    "AZI": (0, 2*np.pi),
}

class MinMax:
    def __init__(self):
        pass
    
    def __call__(self, array, vars=None):
        if isinstance(array, xarray.Dataset):
            output = self.forward_dataset(array)
        elif vars is not None:
            output = self.forward_array(array, vars)
        else:
            raise ValueError(f'{type(array)} is not a {type(xarray.Dataset)} and vars are not specified.')
        return output

    def inverse(self, array, vars=None):
        if isinstance(array, xarray.Dataset):
            output = self.inverse_dataset(array)
        elif vars is not None:
            output = self.inverse_array(array, vars)
        else:
            raise ValueError(f'{type(array)} is not a {type(xarray.Dataset)} and vars are not specified.')

        return output

    def forward_dataset(self, ds: xarray.Dataset):
        for var in ds.keys():
            minvar, maxvar = MINMAX[var]
            ds[var] = (ds[var] - minvar) / (maxvar - minvar)
        return ds

    def inverse_dataset(self, ds: xarray.Dataset):
        for var in ds.keys():
            minvar, maxvar = MINMAX[var]
            ds[var] = ds[var] * (maxvar - minvar) + minvar
        return ds

    def forward_array(self, array, vars: Tuple[str]):
        '''
        MinMax normalization
        Assumes vars are on the second dimension (Channels) in order of the vars given.
        '''
        
        
        shape = [1]*len(array.shape)
        
        if len(vars) > 1:  # Check for zero dimensional arrays
            assert array.shape[1] == len(vars), f"{len(vars)} vars are not equal to {array.shape[1]} the number of channels in dim=1."
            shape[1] = array.shape[1] # (1, C, 1, ... 1)
            
        if len(vars) == 1:
            minvars = MINMAX[vars[0]][0]
            maxvars = MINMAX[vars[0]][1]
        else:
            minvars = torch.tensor([MINMAX[x][0] for x in vars]).view(shape)
            maxvars = torch.tensor([MINMAX[x][1] for x in vars]).view(shape)

        return (array - minvars)/(maxvars - minvars)


    def inverse_array(self, array, vars: Tuple[str]):
        '''
        Inverse MinMax normalization
        Assumes vars are on the second dimension (Channels) in order of the vars given.
        '''
        
        shape = [1]*len(array.shape)
        
        
        if len(vars) > 1: # Check for zero dimensional arrays
            assert array.shape[1] == len(vars), f"{len(vars)} vars are not equal to {array.shape[1]} the number of channels in dim=1."
            shape[1] = array.shape[1] # (1, C, 1, ... 1)
        
        if len(vars) == 1:
            minvars = MINMAX[vars[0]][0]
            maxvars = MINMAX[vars[0]][1]
        else:
            minvars = torch.tensor([MINMAX[x][0] for x in vars]).view(shape)
            maxvars = torch.tensor([MINMAX[x][1] for x in vars]).view(shape)
                                                                        
        return array * (maxvars - minvars) + minvars


class ZeroMinMax:
    def __init__(self):
        pass
    
    def __call__(self, array, vars=None):
        if isinstance(array, xarray.Dataset):
            output = self.forward_dataset(array)
        elif vars is not None:
            output = self.forward_array(array, vars)
        else:
            raise ValueError(f'{type(array)} is not a {type(xarray.Dataset)} and vars are not specified.')
        return output

    def inverse(self, array, vars=None):
        if isinstance(array, xarray.Dataset):
            output = self.inverse_dataset(array)
        elif vars is not None:
            output = self.inverse_array(array, vars)
        else:
            raise ValueError(f'{type(array)} is not a {type(xarray.Dataset)} and vars are not specified.')

        return output

    def forward_dataset(self, ds: xarray.Dataset):
        for var in ds.keys():
            minvar, maxvar = MINMAX[var]
            ds[var] = 2*(ds[var] - minvar) / (maxvar - minvar) - 1
        return ds

    def inverse_dataset(self, ds: xarray.Dataset):
        for var in ds.keys():
            minvar, maxvar = MINMAX[var]
            ds[var] = (ds[var] + 1)/2 * (maxvar - minvar) + minvar
        return ds

    def forward_array(self, array, vars: Tuple[str]):
        '''
        MinMax normalization
        Assumes vars are on the second dimension (Channels) in order of the vars given.
        '''
        
        
        shape = [1]*len(array.shape)
        
        if len(vars) > 1:  # Check for zero dimensional arrays
            assert array.shape[1] == len(vars), f"{len(vars)} vars are not equal to {array.shape[1]} the number of channels in dim=1."
            shape[1] = array.shape[1] # (1, C, 1, ... 1)
            
        if len(vars) == 1:
            minvars = MINMAX[vars[0]][0]
            maxvars = MINMAX[vars[0]][1]
        else:
            minvars = torch.tensor([MINMAX[x][0] for x in vars]).view(shape)
            maxvars = torch.tensor([MINMAX[x][1] for x in vars]).view(shape)

        return 2*(array - minvars)/(maxvars - minvars) - 1


    def inverse_array(self, array, vars: Tuple[str]):
        '''
        Inverse MinMax normalization
        Assumes vars are on the second dimension (Channels) in order of the vars given.
        '''
        
        shape = [1]*len(array.shape)
        
        
        if len(vars) > 1: # Check for zero dimensional arrays
            assert array.shape[1] == len(vars), f"{len(vars)} vars are not equal to {array.shape[1]} the number of channels in dim=1."
            shape[1] = array.shape[1] # (1, C, 1, ... 1)
        
        if len(vars) == 1:
            minvars = MINMAX[vars[0]][0]
            maxvars = MINMAX[vars[0]][1]
        else:
            minvars = torch.tensor([MINMAX[x][0] for x in vars]).view(shape)
            maxvars = torch.tensor([MINMAX[x][1] for x in vars]).view(shape)
                                                                        
        return (array + 1)/2 * (maxvars - minvars) + minvars