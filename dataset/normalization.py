import xarray


MINMAX = {
    "SIS": (0.0, 1109.0),
    "SRTM": (-7.6700854700854695, 3746.053675213676),
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
}


class MinMax:
    def forward_dataset(ds: xarray.Dataset):
        for var in ds.keys():
            minvar, maxvar = MINMAX[var]
            ds[var] = (ds[var] - minvar) / (maxvar - minvar)
        return ds

    def backward_dataset(ds: xarray.Dataset):
        for var in ds.keys():
            minvar, maxvar = MINMAX[var]
            ds[var] = ds[var] * (maxvar - minvar) + minvar
        return ds

    def forward(array, var: str):
        minvar, maxvar = MINMAX[var]
        return (array - minvar) / (maxvar - minvar)

    def backward(array, var):
        minvar, maxvar = MINMAX[var]
        return array * (maxvar - minvar) + minvar
