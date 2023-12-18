import numpy as np
import math, os
from glob import glob
import xarray
import matplotlib.pyplot as plt


if __name__=='__main__':

    fns = glob('*.hgt')

    print(f'combining {len(fns)} files')
    tiles = []

    roi = [65, 34, -15, 29]
    
    combined = np.ones(shape=((roi[0]-roi[1])*1200 +1,(roi[3] - roi[2])*1200 +1))*np.nan


    for fn in fns:
        size = os.path.getsize(fn)
        dim = int(math.sqrt(size/2))
        assert dim*dim*2 == size, 'Invalid file size'

        NE_str = fn.split('.')[0]

        if 'N' in NE_str:
            mlat = int(NE_str[1:3])
        elif 'S' in NE_str:
            mlat = -1*int(NE_str[1:3])
        
        if 'E' in NE_str:
            mlon = int(NE_str[4:])
        elif 'W' in NE_str:
            mlon = -1*int(NE_str[4:])

        data = np.ndarray((dim,dim), np.dtype('>i2'), open(fn, 'rb').read())[::-1,:]
        idxlat = ((mlat - roi[1])*1200, (mlat - roi[1] +1)*1200 + 1)
        idxlon = ((mlon - roi[2])*1200, (mlon - roi[2] +1)*1200 + 1)
        combined[idxlat[0]:idxlat[1], idxlon[0]:idxlon[1]] = data
        
    combined = combined[:-1, :-1]
    combined[combined == 32768] = np.nan

    ds_combined = xarray.Dataset(coords={
            'lat':(['lat'],np.linspace(roi[1], roi[0], combined.shape[0])),
            'lon':(['lon'],np.linspace(roi[2], roi[3], combined.shape[1]),)
        },
        data_vars={
            'SRTM':(('lat', 'lon'), combined)
        })
    
    # ds_combined.where(ds_combined.SRTM < -9000)
    # ds_combined = xarray.combine_by_coords(tiles.values(), ).drop_duplicates('lat').drop_duplicates('lon')

    ds_combined.to_netcdf('SRTM.nc')

    ds_combined.SRTM.plot.imshow()
    plt.savefig('aaatest.png')


