

def sza_filter_85(ds):
    return ds.isel(time=ds.SZA.mean(['lat','lon']) < 1.4835298641951802) # SZA below 85*

def sza_filter_95(ds):
    return ds.isel(time=ds.SZA.mean(['lat','lon']) < 1.6580627893946132) # SZA below 85*