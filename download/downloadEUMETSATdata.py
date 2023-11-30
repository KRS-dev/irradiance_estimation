# %%
import eumdac
import os, shutil
from datetime import datetime
from glob2 import glob
from concurrent.futures import ThreadPoolExecutor

# from epct import api
import pandas as pd

# import xarray
from tqdm import tqdm
from functools import partial
from concurrent.futures import wait
import subprocess
import ephem
import argparse


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


def download_product(product, SAVE_PATH, verbose=False):
    try:
        fn = None
        fn_1 = os.path.join(SAVE_PATH, product._id + ".zip")
        if not os.path.exists(fn_1):
            with product.open() as fsrc:
                fn = os.path.join(SAVE_PATH, fsrc.name)
                if not os.path.exists(fn):
                    with open(fn, mode="wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst, length=128 * 1024)
                else:
                    print("asdfasdf", fn)

                if verbose:
                    print(f"Download of product {product} finished.")

        else:
            if verbose:
                print(f"Already downloaded: {product._id}")

        return fn_1
    except Exception as e:
        print(e)
        print("{} is not available".format(product))
        if os.path.exists(fn):
            os.remove(fn)  # remove half downloaded files


parser = argparse.ArgumentParser(
    prog="Downloads EUMETSAT products from sunrise to sunset.",
    description="Multithreads EUMETSAT product downloading through the max 16 connections.",
)
parser.add_argument("-o", "--outputfolder", default=".")
parser.add_argument("-p", "--product", default="EO:EUM:DAT:MSG:HRSEVIRI")
parser.add_argument(
    "-s",
    "--startdate",
    required=True,
    help='startdate in the format: "2022-07-01"',
    type=valid_date,
)
parser.add_argument(
    "-e",
    "--enddate",
    required=True,
    help='enddate in the format: "2022-07-01"',
    type=valid_date,
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-ch", "--checkoldfiles", action="store_true")


args = parser.parse_args()


if __name__ == "__main__":
    consumer_key = "sUBVMMtDM5Yq40GBHpNWlTwukIga"
    consumer_secret = "I6x6wAkLrt_IGQikDK13h_EoYwEa"

    credentials = (consumer_key, consumer_secret)
    token = eumdac.AccessToken(credentials)

    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(args.product)  # full spectrum
    # selected_collection = datastore.get_collection('EO:EUM:DAT:MSG:MSG15-RSS') # full spectrum rapidscan

    if args.verbose:
        print(selected_collection)

    start = args.startdate
    end = args.enddate

    ## Set up observer
    obs = ephem.Observer()
    obs.lon = str(8.5417)  # Zurich as center
    obs.lat = str(47.3769)
    obs.elev = 0
    obs.pressure = 0

    day_products = {}
    for day in pd.date_range(start, end, freq="D"):
        midday = datetime(day.year, day.month, day.day, 13)
        obs.date = midday.strftime(
            "%Y/%m/%d %H:%M:%S"
        )  # pyephem requires datetimes in string utc format
        sunrise = obs.previous_rising(ephem.Sun()).datetime()
        sunset = obs.next_setting(ephem.Sun()).datetime()

        day_products[datetime.strftime(day, "%Y%m%d")] = selected_collection.search(
            dtstart=sunrise, dtend=sunset
        )

    print(
        "Number of Images: ",
        sum([products.total_results for products in day_products.values()]),
    )

    SAVE_PATH = os.path.abspath(os.path.join(args.outputfolder, args.product))

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if args.verbose:
        print(f"Saving the downloaded products in: {SAVE_PATH}")

    downloaded_products = glob(os.path.join(SAVE_PATH, "*.zip"))
    dp_ids = {os.path.basename(x).strip(".zip"): x for x in downloaded_products}

    print("Amount of products downloaded already:", len(downloaded_products))

    for day_str, products in tqdm(day_products.items()):
        N = len(products)
        p_ids = {str(x): x for x in products}

        p_ids_left = set(p_ids.keys()) - set(dp_ids.keys())
        products = [p_ids[x] for x in p_ids_left]

        if args.checkoldfiles:
            for p_id in set.intersection(set(p_ids.keys()), set(dp_ids.keys())):
                if p_ids[p_id].size == int(os.path.getsize(dp_ids[p_id]) / 1024):
                    continue
                else:
                    print(f"Deleting old file {p_id}")
                    products.append(p_ids[p_id])
                    os.remove(dp_ids[p_id])

        print(len(products))
        part_download_product = partial(
            download_product, SAVE_PATH=SAVE_PATH, verbose=args.verbose
        )

        with ThreadPoolExecutor(max_workers=10) as executor:
            if args.verbose:
                results = list(
                    tqdm(executor.map(part_download_product, products), total=N)
                )
            else:
                results = list(executor.map(part_download_product, products))
