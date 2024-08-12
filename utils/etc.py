from time import time
from datetime import datetime
import argparse


class benchmark(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time()

    def __exit__(self, ty, val, tb):
        end = time()
        print("%s : %0.3f seconds" % (self.name, end - self.start))
        return False


def parse_dates(fns):
    dates = []
    for fn in fns:
        dt_str = fn.split("Z")[0].split("-")[-1].split(".")[0]
        dates.append(datetime.strptime(dt_str, "%Y%m%d%H%M%S"))
    return dates


def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)

def pickle_write(obj, fn):
    with open(fn, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)
    return obj

def pickle_read(fn):
    with open(fn, 'rb') as pickle_file:
        obj = pickle.load(pickle_file)
    return obj