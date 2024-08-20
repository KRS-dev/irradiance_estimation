import subprocess

import xarray

def download_files(names):
    base_url = "http://tds.webservice-energy.org/thredds/fileServer/iea-pvps/IEA_PVPS-"
    for nm in names:
        url = base_url + str(nm) + ".nc"
        print('downloading:', url)
        subprocess.run(["wget", url])


index = xarray.open_dataset('index.nc')

extent = [-8, 29, 29, 62]

index = index.where((index.longitude > extent[0]) &  
                    (index.longitude < extent[1]) &
                    (index.latitude > extent[2]) &
                    (index.latitude < extent[3]), drop=True)
names = index.station_name.values
# Example usage: download 10 files
download_files(names)
