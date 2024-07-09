

from itertools import repeat
import os
from typing import OrderedDict
import numpy as np
import pandas as pd

from collections.abc import Iterable

from pvlib import atmosphere
from pvlib.clearsky import lookup_linke_turbidity, ineichen
import ephem

from utils.etc import benchmark

class SolarPosition:
    def  __init__(self, datetime, lat, lon, altitude=0) -> None:
        

        if not isinstance(datetime, pd.DatetimeIndex):
            self.datetimes = pd.to_datetime(datetime)
        else:
            self.datetimes = datetime

        self.obs = ephem.Observer()
        self.sun = ephem.Sun()

        self.lat = lat
        self.lon = lon
        self.altitude = altitude

    def get_solarposition(self) -> OrderedDict:
    
        solarposition = OrderedDict()

        if len(self.datetimes) == 1:
    
            self.obs.date = ephem.Date(self.datetime)
            self.obs.lat = str(self.lat)
            self.obs.lon = str(self.lon)
            self.obs.elevation = self.altitude if not np.isnan(self.altitude) else 0

            self.sun.compute(self.obs)
            azis = self.sun.az
            szas = np.pi/2 - self.sun.alt

        else:
            szas = []
            azis = []
            for date, lat, lon, altitude in zip(
                self.datetimes,
                repeat(self.lat) if not isinstance(self.lat, Iterable) else self.lat, 
                repeat(self.lon) if not isinstance(self.lon, Iterable) else self.lon,
                repeat(self.altitude) if not isinstance(self.altitude, Iterable) else self.altitude,
            ):
                self.obs.date = ephem.Date(date)
                self.obs.lat = str(lat)
                self.obs.lon = str(lon)
                self.obs.elevation = altitude if not np.isnan(altitude) else 0

                self.sun.compute(self.obs)
                azis.append(self.sun.az)
                szas.append( np.pi/2 - self.sun.alt)

        solarposition['apparent_zenith'] = np.array(szas)
        solarposition['apparent_azimuth'] = np.array(azis)

        return solarposition


class Clearsky:
    def __init__(self, datetimes, latitudes, longitudes, altitudes, solarposition=None):
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.altitudes = altitudes
        self.solarposition = solarposition

        if not isinstance(datetimes, pd.DatetimeIndex):
            self.datetimes = pd.to_datetime(datetimes)
        else:
            self.datetimes = datetimes

    def get_clearsky(self):

        if self.solarposition is None:
            self.solarposition = SolarPosition(
                self.datetimes, self.latitudes, self.longitudes, self.altitudes
                ).get_solarposition()

        apparent_zenith = self.solarposition['apparent_zenith']


        if isinstance(self.latitudes, Iterable):
            print('iterable')
            linke_turbidity = pd.concat([
                lookup_linke_turbidity(pd.DatetimeIndex([datetime]), lat, lon)
                for datetime, lat, lon in zip(self.datetimes, self.latitudes, self.longitudes)
            ])
        else:
            linke_turbidity = lookup_linke_turbidity(
                self.datetimes, self.latitudes, self.longitudes
            )

        airmass_absolute = self.get_airmass(self.datetimes, self.solarposition)

        cs = ineichen(np.rad2deg(apparent_zenith), airmass_absolute, linke_turbidity, self.altitudes)

        return cs['ghi'].to_numpy()
    

    def get_airmass(self, datetimes, solarposition):
        apparent_zenith = np.rad2deg(solarposition['apparent_zenith'])

        airmass_relative = atmosphere.get_relative_airmass(apparent_zenith, 'kasten1966')

        pressure = atmosphere.alt2pres(self.altitudes)
        airmass_absolute = atmosphere.get_absolute_airmass(airmass_relative,
                                                           pressure)
        
        return airmass_absolute


if __name__ == '__main__':

    datetime = pd.date_range(start='2021-01-01', end='2021-02-01', freq='H')
    latitudes = 10
    longitudes = 0
    # latitudes = np.linspace(0, 90, len(datetime))
    # longitudes = np.linspace(-180, 180, len(datetime))
    altitudes = 1000
    print(len(datetime))

    cs = Clearsky(datetime, latitudes, longitudes, altitudes)
    with benchmark('clearsky'):
        print(cs.get_clearsky())


