
#%%
import argparse, logging, time
import subprocess
import eumdac
import os
from datetime import date, datetime, timedelta
from glob2 import glob
from epct import api
import pandas as pd
import xarray
from tqdm import tqdm
from functools import partial
from zipfile import ZipFile
from dask.distributed import as_completed, wait
import psutil

from etc import valid_date, parse_dates, benchmark
from combine_nc import combine_netcdf_files


parser = argparse.ArgumentParser(
    prog='Customize MSG .nat to aggregated .nc',
    description='Customizes the native encoded MSG level 1.5 data to a time aggregated time filtered .nc file. Aggregation normally happens per day.',
)
parser.add_argument('-f', '--folder', default='.')
parser.add_argument('-o', '--outputfolder', default='.')
parser.add_argument('-l', '--logdir', default=None)
parser.add_argument('-c', '--chain', default='HRSEVIRI', choices=['HRSEVIRI', 'HRSEVIRI_HRV', 'both'])
parser.add_argument('-roi', '--regionofinterest', default='65,35,-15,28')
parser.add_argument('-agg', '--aggregatetime', default='day', choices=['day', 'week', 'month'])
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-s', '--startdate',
    required=True,
    help='startdate in the format: "2022-07-01"',
    type=valid_date,
)
parser.add_argument('-e', '--enddate', 
    required=True, 
    help='enddate in the format: "2022-07-01"',
    type=valid_date,
)



if __name__ ==  '__main__':

    args = parser.parse_args()

    SAVE_PATH = os.path.abspath(args.outputfolder)
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    INPUT_PATH = os.path.abspath(args.folder)
    if args.logdir is None:
        LOG_DIR = os.path.abspath(os.path.join(SAVE_PATH, 'log'))
    else:
        LOG_DIR = os.path.abspath(args.logdir)

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)


    if ',' in args.regionofinterest:
        NSWE = [float(x) for x in args.regionofinterest.split(',')]
        roi = {"NSWE" : NSWE}
    else:
        roi = args.regionofinterest

    chain = eumdac.tailor_models.Chain(
        id='Western_Europe',
        product='HRSEVIRI',
        format='netcdf4',
        roi=roi,
        projection='geographic',
        # compression='internal'
    )

    chain_hrv = eumdac.tailor_models.Chain(
        id='Western_Europe_HRV',
        product='HRSEVIRI_HRV',
        format='netcdf4',
        roi=roi,
        projection='geographic',
        # compression='internal',
    )


    if args.chain == 'HRSEVIRI':
        chains = [chain]
    elif args.chain == 'HRSEVIRI_HRV':
        chains = [chain_hrv]
    elif args.chain == 'both':
        chains= [chain, chain_hrv]


    

    config = api.config()
    if not args.verbose:
        config['epct'].update({'log_level': logging.WARNING})


    aggregate_time = args.aggregatetime 

    startdate = args.startdate
    enddate = args.enddate
    if aggregate_time == 'day':
        daterange = pd.date_range(startdate, enddate, freq='D')
    elif aggregate_time == 'month':
        mind = startdate
        maxd = enddate
        minm = mind.month
        maxm = maxd.month
        miny = mind.year
        maxy = maxd.year
        if maxm == 12:
            maxy += 1
            maxm = 1
        daterange = pd.date_range(date(miny, minm, 1), date(maxy, maxm, 1), freq='M')
    elif aggregate_time == 'week':
        mind = startdate
        maxd = enddate
        #TODO

    for ch in chains:
        
        results = []
        dt_ls_running = []

        for dt in tqdm(daterange, desc='Looking for input data and checking previously customized data.'):
            if aggregate_time == 'day':
                dt_str = dt.strftime('%Y%m%d')
            elif aggregate_time == 'month':
                dt_str = dt.strftime('%Y%m')

            finished_customizations = glob(os.path.join(SAVE_PATH, f'{ch.product}_{dt_str}_*.nc'))
            if len(finished_customizations)>0:
                print(f'{dt_str} {ch.product} already customized.')
                continue

            results_nat = glob(os.path.join(INPUT_PATH, f'*-{dt_str}*.nat'))
            results_zip = glob(os.path.join(INPUT_PATH, f'*-{dt_str}*.zip'))
            results_zip_nat = []
            for fn in results_zip:
                try: 
                    with ZipFile(fn, 'r') as zf:
                        zipped_files = [name for name in zf.namelist() if name.endswith('.nat')]
                        if zipped_files:
                            results_zip_nat.append(fn)
                except Exception as e:
                    print(f'Failed to read zip file {fn}. Removing file.')
                    print(e)
                    os.remove(fn)
                    

            results_dt = results_zip_nat + results_nat


            if len(results_dt) == 0:
                print(f'{dt_str} {ch.product} No Native data available.')
                continue
            
            h = os.path.join(SAVE_PATH, ch.product + f'_{dt_str}T*.nc')
            customized = glob(h)

            if len(customized) > 0:
                results_dt_dict = {x.split('-NA-')[-1].split('.')[0] : x for x in results_dt}
                customized_dtstr = set([x.split('Z')[0].split('_')[-1].replace('T', '') for x in customized])
                if ch.product == 'HRSEVIRI_HRV':
                    offset = timedelta(minutes=12)
                    customized_dt_corrected = [datetime.strptime(x, '%Y%m%d%H%M%S') + offset for x in customized_dtstr]
                    customized_dtstr = set([x.strftime('%Y%m%d%H%M') for x in customized_dt_corrected]) # Remove seconds as they differ as well
                    results_dt_dict = {k[:-2]:v for k,v in results_dt_dict.items()} # same
                
                results_dtstr = set(results_dt_dict.keys())
                a = results_dtstr - customized_dtstr # #FIXED with 12min offset, doesnt work because sensing time of HRSEVIRI_HRV is later than HRSEVIRI or the time in the zip file
                results_to_customize = [results_dt_dict[x] for x in a]
            else:
                results_to_customize = results_dt

            if not len(results_to_customize) == 0:
                results.extend(results_to_customize)
            else:
                print(f'{dt_str} {ch.product} All individual products customized already.')
            dt_ls_running.append(dt_str) # Add day to list for the need to be aggregated and compressed.
            



        if len(results) >0:

            print(f'{ch.product} Start customisation.')
            pbar = tqdm(total=len(results), desc=f'{ch.product}')  # Init pbar
            tasks, client = api.submit_customisations(results, config=config, chain_config=ch.asdict(), target_dir=SAVE_PATH, log_dir=LOG_DIR)

            i = 0
            for t in as_completed(tasks.values()):
                pbar.update(n=1)
                i +=1
                if i% 100 == 0:
                    print('RAM memory %:', psutil.virtual_memory()[2])
                    print('CPU usage %:', psutil.cpu_percent(4))
                del t
            
            del client
      

        uncompressed_dict = {}
        for dt_str in tqdm(dt_ls_running, desc=f'{ch.product} Combining per {args.aggregatetime}.'):
            uncompressed_fn = combine_netcdf_files(SAVE_PATH, dt_str, ch.product, verbose=args.verbose)
            if args.verbose:
                print(f'{dt_str} {ch.product} Combined the .nc files.')
            compressed_fn = os.path.join(SAVE_PATH, ch.product + '_' + dt_str + '_FPC.nc')
            uncompressed_dict[uncompressed_fn] = compressed_fn

        processes = []
        for uncompressed_fn, compressed_fn in uncompressed_dict.items():
            a = subprocess.Popen(['h5repack','-f', 'SHUF', '-f', 'GZIP=1',  uncompressed_fn, compressed_fn])
            processes.append(a)
        w = tqdm([x.wait() for x in processes], desc=f'{ch.product} Compressing files per {args.aggregatetime}.')

        for uncompressed_fn in uncompressed_dict.keys():
            os.remove(uncompressed_fn)
            if args.verbose:
                print(f'{dt_str} {ch.product} Compressed the netcdf4 file.')

    print(f'Finished customisations of {INPUT_PATH} to {SAVE_PATH}')


