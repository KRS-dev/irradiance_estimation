import xarray as xr
import numpy as np
import os


def check(x):
    return np.isnan(x).any()

def check_all(x):
    return np.isnan(x).all()

def check_sum(x):
    return np.isnan(x).sum()

def compute_bnds(x):
    i = 0
    j = 0
    i_max = 736
    j_max = 658
    step = 20
    angle_step = j_max//2
    if not check_all(x):
        c = check(x)
        if c:
            c_ul = check_sum(x.isel(x=slice(i, i+angle_step), y=slice(j_max-angle_step, j_max)))
            c_ur = check_sum(x.isel(x=slice(i_max-angle_step, i_max), y=slice(j_max-angle_step, j_max)))
            c_ll = check_sum(x.isel(x=slice(i, i+angle_step), y=slice(j, j+angle_step)))
            c_lr = check_sum(x.isel(x=slice(i_max-angle_step, i_max), y=slice(j, j+angle_step)))
            
            if c_ul>c_ur and c_ul>c_ll and c_ul>c_lr:
                c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
                while c:
                    j_max -= step
                    i += step
                    c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
            
            elif c_ur>c_ul and c_ur>c_ll and c_ur>c_lr:
                c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
                while c:
                    i_max -= step
                    j_max -= step
                    c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
            
            elif c_ll>c_ul and c_ll>c_ur and c_ll>c_lr:
                c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
                while c:
                    i += step
                    j += step
                    c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
            
            elif c_lr>c_ul and c_lr>c_ur and c_lr>c_ll:
                c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
                while c:
                    i_max -= step
                    j += step
                    c = check(x.isel(x=slice(i, i_max), y=slice(j, j_max)))
            else:
                return None
    else:
        return None
    
    if i_max>i+128 and j_max>j+128:
        return i, j, i_max, j_max
    else:
        return None

def clean(x):
    # remove unreliable estimations
    x = x.sel(time=x.record_status==0)
    
    # compute indeces of images with less than 50% nans
    img_x_dim = x.x.shape[0]
    img_y_dim = x.y.shape[0]
    idx = np.where(np.isnan(x.CAL).sum(axis=(1,2))/(img_x_dim*img_y_dim) < .5)[0]
    
    # initialize indices lists
    final_idx = []
    min_x_lst = []
    min_y_lst = []
    max_x_lst = []
    max_y_lst = []

    # compute image boundaries for every admitted index
    for j in idx:
        bnd = compute_bnds(x.CAL[j])
        if bnd is not None:
            if bnd[2]>bnd[0]+128 and bnd[3]>bnd[1]+128:
                final_idx.append(j)
                min_x_lst.append(bnd[0])
                min_y_lst.append(bnd[1])
                max_x_lst.append(bnd[2])
                max_y_lst.append(bnd[3])
    x = x.isel(time=final_idx)
    
    x['min_x'] = ('time', min_x_lst)
    x['min_y'] = ('time', min_y_lst)
    x['max_x'] = ('time', max_x_lst)
    x['max_y'] = ('time', max_y_lst)
    x = x[['CAL', 'min_x', 'min_y', 'max_x', 'max_y']]
    return x

def main():

    sarah_filenames = sorted(os.listdir('/scratch/snx3000/acarpent/SARAH3_2016-2022/'))
    for i in range(0, len(sarah_filenames)):
        f = sarah_filenames[i]

        try:
            sarah_dataset = clean(xr.open_dataset('/scratch/snx3000/acarpent/SARAH3_2016-2022/'+f).rename({'lon':'x', 'lat':'y'}))
            x_diff = sarah_dataset.max_x - sarah_dataset.min_x
            if len(sarah_dataset.time)>0:
                img_x_dim = sarah_dataset.x.shape[0]
                img_y_dim = sarah_dataset.y.shape[0]    
                if i == 0:
                    sarah_dataset.chunk({'y':img_y_dim, 'x':img_x_dim, 'time':1}).to_zarr('/scratch/snx3000/acarpent/SARAH3_2016-2022.zarr', mode='w')
                else:
                    sarah_dataset.chunk({'y':img_y_dim, 'x':img_x_dim, 'time':1}).to_zarr('/scratch/snx3000/acarpent/SARAH3_2016-2022.zarr', append_dim='time')
            print(i, f, x_diff)
        except:
            print(i, f, 'PROBLEM')