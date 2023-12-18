import pandas as pd
import xarray
import ephem
import numpy as np


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
    return zens  # Zenith angle





if __name__ == '__main__':
    zarr_store = '/scratch/snx3000/kschuurm/DATA/train.zarr'

    hres = xarray.open_zarr(zarr_store)

    lat = hres.lat 
    lon = hres.lon

    lat_sza = np.linspace(np.floor(lat.min()), np.ceil(lat.max()), 0.5)
    lon_sza = np.linspace(np.floor(lon.min()), np.ceil(lon.max()), 0.5)

    lonlon, latlat= np.meshgrid(lon_sza, lat_sza)

    times = hres.time.values

    datetimes = pd.to_datetime(hres.time)

    array = xarray.apply_ufunc(
        solarzenithangle,
        datetimes,
        hres.lat,
        hres.lon,
        hres.SRTM,
        input_core_dims=[['time'], [], [], []],
        output_core_dims=[['time', 'solar_position']],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={'output_sizes': {'time':len(hres.time), 'solar_position':1}},
        output_dtypes=[np.float32],
        )

    print(array)