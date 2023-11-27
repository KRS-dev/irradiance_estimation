import subprocess, os, argparse
from glob2 import glob
from etc import benchmark
from tqdm import tqdm
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    prog='Compress netcdf4 files with h5repack .nc',
    )
    parser.add_argument('-f', '--folder', default='.')
    parser.add_argument('-o', '--outputfolder', default=None)
    parser.add_argument('-i', '--globidentifier', default='*_F.nc')
    parser.add_argument('-ex', '--exclude', default='*_FPC.nc')
    parser.add_argument('--delete', action='store_true')


    args = parser.parse_args()
    if args.outputfolder is None:
        outputfolder = args.folder

    if args.exclude is None:
        ext = '_FPC.nc'
    else:
        ext = args.exclude.strip('*')

    input_fns = glob(os.path.join(args.folder, args.globidentifier))


    compression_d_full = {x : x.split(args.globidentifier.strip('*'))[0] + ext for x in input_fns}
    if outputfolder is not args.folder:
        compression_d_full = {x : os.path.join(outputfolder, os.path.basename(y)) for x, y in compression_d_full.items()}

    compression_d = {}
    for un, comp in compression_d_full.items():
        if not os.path.exists(comp):
            compression_d[un] = comp
    print('Number of files to compress', len(compression_d.keys()))

    processes = []
    for uncompressed_fn, compressed_fn in compression_d.items():
        a = subprocess.Popen(['h5repack','-f', 'SHUF', '-f', 'GZIP=1',  uncompressed_fn, compressed_fn])
        processes.append(a)
    w = tqdm([x.wait() for x in processes], desc=f'Compressing files')
    
    if args.delete:
        for fn in tqdm(compression_d.keys(), desc='Removing uncompressed files.'):
            os.remove(fn)




