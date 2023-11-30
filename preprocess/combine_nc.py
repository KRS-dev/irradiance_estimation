import os
from glob2 import glob
import xarray
from datetime import datetime
import argparse


def combine_netcdf_files(SAVE_PATH, dt_str, nm, verbose=False):
    # TODO: make it work aggregating weeks of nc files: list of dt_str's 20220701 - 20220701

    out_fn = os.path.join(SAVE_PATH, f"{nm}_{dt_str}_F.nc")
    HRESVIRIfns = glob(os.path.join(SAVE_PATH, f"{nm}_{dt_str}*T*.nc"))

    HRESVIRIfns = remove_doubles(HRESVIRIfns)

    if HRESVIRIfns:
        if verbose:
            print("Combining ", len(HRESVIRIfns), f" {nm}_{dt_str}.nc files.")
            print(HRESVIRIfns)

        # automatically concat the nc on the time dimension, VERY fast
        ds = xarray.open_mfdataset(
            HRESVIRIfns,
            preprocess=append_time_dim,
            parallel=True,
            concat_dim="time",
            combine="nested",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )

        ds.to_netcdf(out_fn)
        for fn in HRESVIRIfns:
            os.remove(fn)

        del ds

        return out_fn


def append_time_dim(ds):
    dt = datetime.strptime(ds.EPCT_start_sensing_time, "%Y%m%dT%H%M%SZ")
    new = ds.expand_dims(time=[dt])
    new.time.attrs.update({"axis": "T", "long_name": "time", "standard_name": "time"})
    return new


def remove_doubles(HRESVIRIfns):
    d = {a.split("Z")[0]: a for a in HRESVIRIfns}

    fns_unique = set(d.values())
    fns_double = set(HRESVIRIfns) - fns_unique

    for fn in fns_double:
        os.remove(fn)

    return fns_unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Combine netcdf4 files with h5repack .nc",
    )
    parser.add_argument("-f", "--folder", default=".")
    parser.add_argument("-o", "--outputfolder", default=None)
    parser.add_argument("-i", "--globidentifier", default="*_F.nc")
    parser.add_argument("-ex", "--exclude", default="*_FPC.nc")
    parser.add_argument("--delete", action="store_true")

    args = parser.parse_args()
    if args.outputfolder is None:
        outputfolder = args.folder

    if args.exclude is None:
        ext = "_FPC.nc"
    else:
        ext = args.exclude.strip("*")

    input_fns = glob(os.path.join(args.folder, args.globidentifier))

    compression_d_full = {
        x: x.split(args.globidentifier.strip("*"))[0] + ext for x in input_fns
    }

    compression_d = {}
    for un, comp in compression_d_full.items():
        if not os.path.exists(comp):
            compression_d[un] = comp
    print("Number of files to compress", len(compression_d.keys()))

    with benchmark(f"Compressing {len(compression_d_full.keys())} netCDF4 files:"):
        compress_nc_parallel(compression_d)

    if args.delete:
        for fn in compression_d.keys():
            os.remove(fn)
