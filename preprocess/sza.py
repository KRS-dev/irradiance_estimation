import pandas as pd
import xarray
import ephem
import numpy as np
from tqdm import tqdm

def pyephem(
    datetimes,
    latitude,
    longitude,
    altitude=0,
    pressure=101325,
    temperature=12,
    horizon="+0:00",
    rad=True,
):
    """
    Calculate the solar position using the PyEphem package.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        Must be localized or UTC will be assumed.
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative
        to south.
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.
    altitude : float, default 0
        Height above sea level in meters. [m]
    pressure : int or float, optional, default 101325
        air pressure in Pascals.
    temperature : int or float, optional, default 12
        air temperature in degrees C.
    horizon : string, optional, default '+0:00'
        arc degrees:arc minutes from geometrical horizon for sunrise and
        sunset, e.g., horizon='+0:00' to use sun center crossing the
        geometrical horizon to define sunrise and sunset,
        horizon='-0:34' for when the sun's upper edge crosses the
        geometrical horizon

    See also
    --------
    spa_python, spa_c, ephemeris
    """

    # Written by Will Holmgren (@wholmgren), University of Arizona, 2014
    # try:
    #     import ephem
    # except ImportError:
    #     raise ImportError('PyEphem must be installed')

    # if localized, convert to UTC. otherwise, assume UTC.
    # try:
    #     time_utc = time.tz_convert('UTC')
    # except TypeError:
    #     time_utc = time

    # sun_coords = pd.DataFrame(index=time)

    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.elevation = altitude if not np.isnan(altitude) else 0
    sun = ephem.Sun()

    # make and fill lists of the sun's altitude and azimuth
    # this is the pressure and temperature corrected apparent alt/az.
    elevs = []
    azis = []
    for thetime in datetimes:
        obs.date = ephem.Date(thetime)
        sun.compute(obs)
        elevs.append(sun.alt)
        azis.append(sun.az)

    elevs = np.array(elevs)
    azis = np.array(azis)
    zens = np.pi / 2 - elevs

    if not rad:
        elevs = np.rad2deg(elevs)
        azis = np.rad2deg(azis)
        zens = np.rad2deg(zens)

    return elevs, azis, zens


def solarzenithangle(datetime, lat, lon, alt):
    """Expects datetime in UTC"""
    elevs, azis, zens = pyephem(datetime, lat, lon, alt)
    return zens, azis  # Zenith angle


def solarzenithangle_latlon(da_temp):
    
    
    # a = [x for x in range(0, len(da_temp.time), 1)]
    # a.append(len(da_temp.time)-1)
    datetimes = pd.to_datetime(da_temp.time)
    
    lats = np.arange(da_temp.lat.min(), da_temp.lat.max()+1, 1, dtype=np.float32)
    lons = np.arange(da_temp.lon.min(), da_temp.lon.max()+1, 1, dtype=np.float32)
    
    da_sza = xarray.DataArray(coords={'time':datetimes, 'lat':lats, 'lon':lons,},
                          data=np.zeros(shape=(len(datetimes), len(lats), len(lons)),
                                       dtype=np.float16))
    da_sza.name = 'SZA'
    da_sza.attrs.update({'long_name': 'Solar Zenith Angle at sea level',
                      'standard_name': 'solar_zenith_angle',
                      'units':'rad'})
    da_azi = xarray.DataArray(coords={'time':datetimes, 'lat':lats, 'lon':lons,},
                          data=np.zeros(shape=(len(datetimes), len(lats), len(lons)), 
                                        dtype=np.float16))
    da_azi.name = 'AZI'
    da_azi.attrs.update({'long_name': 'Solar Azimuth Angle at sea level',
                      'standard_name': 'solar_azimuth_angle',
                      'units':'rad'})
    
    for i, lat in tqdm(enumerate(lats)):
        for j, lon in enumerate(lons):
            sza, azi = solarzenithangle(datetimes, lat, lon, 0)
            da_sza[:,i, j] = sza
            da_azi[:,i,j] = azi
    
    
    ds = xarray.Dataset({'SZA':da_sza, 'AZI':da_azi})
    return ds


