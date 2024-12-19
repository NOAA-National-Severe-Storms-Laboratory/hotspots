import datetime as dt

import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import convolve2d

import pyart
import xradar as xd

def pyart_to_xarray_vol(radar: pyart.core.radar,
                        fields: list,
                        reindex_angle: (bool, dict)) -> xr.Dataset:
    """
    Convert a pyart radar object to a xarray dataset. Optionally you can
    reindex the azimuth angles so they align for different sweeps.

    Parameters
    ----------
    radar: pyart.core.radar
        Pyart radar object for conversion to xarray Dataset
    fields: list
        Field names to include in the xarray Dataset. For example
        fieldnames_dict = ["reflectivity", "cross_correlation_ratio"]
    reindex_angle: bool, dict
        Allows to reindex to given azimuth angles with a certain tolerance.
        For example
        reindex_angle = {"angle_res": 0.5, "start_angle": 0, "stop_angle": 360,
                 "direction": 1, "tolerance": 0.5, "method": "nearest"}

    Returns
    -------
    ds_vol: xarray.Dataset
        Radar volume data in xarray format
    """
    # TODO: can be parallelized

    vars_dict = dict()
    ds_list = list()
    radar_times = pyart.util.datetimes_from_radar(radar)

    metadata = dict(
        radar_name=radar.metadata['instrument_name'],
        radar_latitude=radar.latitude['data'][0],
        radar_longitude=radar.longitude['data'][0],
        radar_altitude=radar.altitude['data'][0],
        vcp_pattern=radar.metadata['vcp_pattern'],
        starttime=min(radar_times))

    for sweep in range(radar.nsweeps):

        for field in fields:
            # populate dictionary for xarray
            vars_dict[field] = (
                ['azimuth', 'range'], radar.get_field(sweep, field))

        sweep_slice = radar.get_slice(sweep)
        times_sweep = radar_times[sweep_slice]

        # build the xarray Dataset for each sweep
        ds_sweep = xr.Dataset(
            vars_dict,
            coords=dict(
                range=radar.range['data'],
                azimuth=radar.get_azimuth(sweep),
                elevation=("azimuth", radar.get_elevation(sweep)),
                time=("azimuth", times_sweep)
            )
        )

        # first find exact duplicates and remove
        ds_sweep = xd.util.remove_duplicate_rays(ds_sweep)

        if reindex_angle is not False:
            # second reindex according to retrieved parameters
            ds_sweep = xd.util.reindex_angle(ds_sweep, **reindex_angle)

        # FIXME: this is memory-demanding, can be added later
        # x, y, z = pyart.core.antenna_vectors_to_cartesian(
        #     ds_sweep.range, ds_sweep.azimuth, ds_sweep.elevation)
        # ds_sweep = ds_sweep.assign_coords(
        #     {"x": (["azimuth", "range"], x),
        #      "y": (["azimuth", "range"], y),
        #      "z": (["azimuth", "range"], z)}
        # )

        ds_list.append(ds_sweep)

    ds_vol = xr.concat(
        ds_list, pd.Index(radar.fixed_angle['data'], name='fixed_angle'))
    ds_vol = ds_vol.assign_attrs(metadata)

    return ds_vol.sortby(['fixed_angle'])


def smooth_radar_field(data, window_size):
    """
    Box filtering along a radar ray for a specific field. From
    pyart.correct.despeckle

    Parameters
    ----------
    data : 2D array of ints or floats
        Sweep of data for a specific field. Will be masked.
    window_size : int or None
        Number of gates included in a smoothing box filter along a ray.
        If None, no smoothing is done.

    Returns
    -------
    data : 2D array of ints or floats
        Smoothed sweep of data.
    """
    return np.ma.masked_array(convolve2d(
        data,
        np.ones((1, window_size)) / float(window_size),
        mode="same",
        boundary="symm"))


def xy_assign_context(ds, radar_lon, radar_lat):

    longridspacing = ds.LonGridSpacing
    latgridspacing = ds.LatGridSpacing

    lon_nw = ds.Longitude
    lat_nw = ds.Latitude

    ds = ds.assign_coords(Lon=(lon_nw + longridspacing * ds.Lon),
                          Lat=(lat_nw - latgridspacing * ds.Lat))

    #ds.attrs['Time'] = dt.datetime.utcfromtimestamp(ds.attrs['Time'])
    ds.attrs['Time'] = dt.datetime.strptime(ds.attrs['FilenameDateTime-value'],
                                            '%Y%m%d-%H%M%S')

    x, y = pyart.core.geographic_to_cartesian_aeqd(
        ds.Lon, ds.Lat, radar_lon, radar_lat)
    ds = ds.assign_coords(x=("Lon", x/1000), y=("Lat", y/1000))

    return ds
