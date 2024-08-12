from typing import Any, Tuple
import xarray
import numpy as np
import torch

MINMAX = {
    "SIS": (0.0, 1159.0),
    "CAL": (0.0, 1.0),
    "SID": (0.0, 1071),
    "KI": (0.0, 1.0),
    "DNI": (0.0, 1100.0),
    "DEM": (-7.6700854700854695, 3746.053675213676),
    "channel_1": (0, 101.6), 
    "channel_2": (0, 110.6),
    "channel_3": (0, 99.56),
    "channel_4": (204.6, 336.2),
    "channel_5": (171.6, 263.0),
    "channel_6": (194.1, 284.5),
    "channel_7": (194.2, 327.0),
    "channel_8": (201.6, 293.5),
    "channel_9": (124.06, 343.8),
    "channel_10": (191.9, 333.2),
    "channel_11": (192.9, 290.2),
    "dayofyear": (1, 365),
    "lat": (-90, 90),
    "lon": (-180, 180),
    "SZA": (0, np.pi/2),
    "AZI": (0, 2*np.pi),
    "sat_AZI": (0, 2*np.pi),
    "sat_SZA": (0, np.pi/2),
    "coscatter_angle": (0, np.pi),
    }



class ZeroMinMax:
    def __init__(self):
        pass
    
    def __str__(self):
        return 'ZeroMinMax'

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
        if isinstance(array, torch.Tensor):
            dtype=array.dtype
        else:
            dtype=torch.float32
        
        shape = [1]*len(array.shape)
        
        if len(vars) > 1:  # Check for zero dimensional arrays
            assert array.shape[1] == len(vars), f"{len(vars)} vars are not equal to {array.shape[1]} the number of channels in dim=1."
            shape[1] = array.shape[1] # (1, C, 1, ... 1)
            
        if len(vars) == 1:
            minvars = MINMAX[vars[0]][0]
            maxvars = MINMAX[vars[0]][1]
        else:
            minvars = torch.tensor([MINMAX[x][0] for x in vars], dtype=dtype).view(shape)
            maxvars = torch.tensor([MINMAX[x][1] for x in vars], dtype=dtype).view(shape)

        return 2*(array - minvars)/(maxvars - minvars) - 1


    def inverse_array(self, array, vars: Tuple[str]):
        '''
        Inverse MinMax normalization
        Assumes vars are on the second dimension (Channels) in order of the vars given.
        '''

        if isinstance(array, torch.Tensor):
            dtype=array.dtype
        else:
            dtype=torch.float32
        
        shape = [1]*len(array.shape)
        
        
        if len(vars) > 1: # Check for zero dimensional arrays
            assert array.shape[1] == len(vars), f"{len(vars)} vars are not equal to {array.shape[1]} the number of channels in dim=1."
            shape[1] = array.shape[1] # (1, C, 1, ... 1)
        
        if len(vars) == 1:
            minvars = torch.tensor(MINMAX[vars[0]][0], dtype=dtype)
            maxvars = torch.tensor(MINMAX[vars[0]][1], dtype=dtype)
        else:
            minvars = torch.tensor([MINMAX[x][0] for x in vars], dtype=dtype).view(shape)
            maxvars = torch.tensor([MINMAX[x][1] for x in vars], dtype=dtype).view(shape)
                                                                        
        return (array + 1)/2 * (maxvars - minvars) + minvars