# coding: utf-8
"""
Created on April 26 2024

@author: John Krause (JK)
@author: Vinzent Klaus (VK)

@license: BSD-3-Clause

@copyright Copyright 2024 John Krause

Update History:
    Version 0:
        -JK Initial version 
            Python and wrapped into function and separate driver. 
"""
import numpy as np
import pyart
import xarray as xr

from cappi.azran import find_az_index
from cappi.helpers import pyart_to_xarray_vol
from make_logger import logger


def simple_vv(prune_radar: pyart.core.Radar,
              fields: list = None,
              gate_spacing: (float, int) = None,
              num_azs: int = 360,
              az_offset: float = 0.5) -> xr.Dataset:
    """
    Pyart objects are a volume of radar data, but they often
    contain multiple sweeps at the same elevation. They are also
    ordered in a way that makes computation of vertical columns
    difficult. This is our way to create a data object that is
    both more intuitive and a little faster to access. The problem
    remains that the indexed range at any particular sweep DOES
    NOT correspond with the same indexed range at another elevation
    to do that one needs to use the indexedVV. This VV is a simple
    collection of radar data where only the azimuths are aligned.

    Parameters
    ----------
    prune_radar: pyart.core.Radar
        Radar object to perform ground range interpolation on. Must be pruned
        to include only one sweep at each elevation and ordered with the lowest
        sweep at index 0
    fields: list, optional
        List of radar fields that should be in the interpolated xarray
    gate_spacing: (float, int), optional
        Target gate spacing in m, defaults to native radar gate spacing
    num_azs: int, optional
        Number of target azimuth angles
    az_offset: float, optional
        Offset for target azimuth angles

    Returns
    -------
    vv_ds: xr.Dataset
        Dataset containing all radar fields in the simple VV

    """
    if gate_spacing is None:
        gate_spacing = (prune_radar.range['data'][1] -
                        prune_radar.range['data'][0])

    # if no specific fields are requested just do all
    if fields is None:
        fields = list(prune_radar.fields.keys())

    num_gates = prune_radar.ngates
    range_firstgate = prune_radar.range['data'][0]
    az_spacing_deg = num_azs / 360.0

    # We don't set the number of elevations or the elevations
    # themselves, because we take those from the dataset
    num_sweeps = prune_radar.nsweeps

    ##collect metadata from the pyart radar volume:
    metadata = dict(
        radar_name=prune_radar.metadata['instrument_name'],
        radar_latitude=prune_radar.latitude['data'][0],
        radar_longitude=prune_radar.longitude['data'][0],
        vcp_pattern=prune_radar.metadata['vcp_pattern'],
        missing_value=prune_radar.fields[fields[0]]["_FillValue"]
        )

    fixed_angle = prune_radar.fixed_angle['data']

    # load the target azimuth array
    azimuth = np.arange(0+az_offset, 360, az_spacing_deg)
    logger.debug("az: ", azimuth)

    # get start index
    sbri = prune_radar.sweep_start_ray_index['data']
    logger.debug(sbri)

    # Load the range variable
    grange = np.arange(
        range_firstgate, range_firstgate + num_gates*gate_spacing, gate_spacing)
    logger.debug(grange)

    field_dict = {fieldname: np.empty([num_sweeps, num_azs, num_gates]) for
                  fieldname in fields}

    for s in range(num_sweeps):
        start_index, end_index = prune_radar.get_start_end(s)

        for ia in range(num_azs):
            data_az_index = find_az_index(
                azimuth[ia], prune_radar.azimuth['data'][start_index:end_index])
            data_az_index += start_index

            for fieldname in field_dict:
                field_dict[fieldname][s, ia] = (
                    prune_radar.fields['reflectivity']['data'][data_az_index][
                    0:num_gates])

    var_dict = dict()
    for fieldname in field_dict:
        var_dict[fieldname] = (["fixed_angle", "azimuth", "grange"],
                                field_dict[fieldname])

    logger.info(f"num_sweeps: {num_sweeps}, num_azs: {num_azs}, num_gates: "
                 f"{num_gates}")

    rg, azg = np.meshgrid(grange, azimuth)
    x, y, z0 = pyart.core.transforms.antenna_to_cartesian(rg/1000, azg, 0)

    vv_ds = xr.Dataset(
        data_vars=var_dict,
        coords=dict(
            fixed_angle=fixed_angle,
            grange=grange,
            azimuth=azimuth,
            x=(("azimuth", "grange"), x/1000),
            y=(("azimuth", "grange"), y/1000)),
        attrs=metadata
        )
    return vv_ds


def get_indexed_vv(radar_vol: pyart.core.Radar,
                   fields: list = None,
                   num_gates: int = 1200,
                   gate_spacing: (int, str) = 'auto') -> xr.Dataset:
    """
    Creates a xarray Dataset that holds a virtual volume of data. We set our
    virtual volume to the coordinates we want then interpolate the data from
    the pyart volume to these coordinates.

    Parameters
    ----------
    radar_vol: pyart.core.Radar
        pyart radar object that will be converted
    fields: list, optional
        list of field names which will be converted. Defaults to all fields
        in the radar object
    num_gates: int, optional
        Number of gates in the target virtual volume
    gate_spacing
        Spacing of the gates in the target virtual volume

    Returns
    -------
    vv_ds: xarray.Dataset
        Virtual volume in xarray.Dataset

    """

    #it's best to take the range and gate spacing that is already in the data
    #num_gates = 1200

    if gate_spacing == 'auto':
        gate_spacing = radar_vol.range['data'][1] - radar_vol.range['data'][0]
        spacing_test = np.all(np.diff(radar_vol.range['data']) == gate_spacing)

        if spacing_test is False:
            raise NotImplementedError('gate spacing needs to be equal in auto '
                                      'mode')

    # if no target fields are given we just take all the fields from the
    # radar volume
    if fields is None:
        fields = radar_vol.fields.keys()

    r2fg = radar_vol.range['data'][0]
    
    num_azs = 360
    az_spacing_deg = num_azs/360.0
    az_offset = 0.5

    vv_elev = 0.0

    #We don't set the number of elevations or the elevations
    #themselves, because we take those from the dataset
    num_sweeps = radar_vol.nsweeps
    
    ##collect metadata from the pyart radar volume:
    metadata = dict(
        radar_name=radar_vol.metadata['instrument_name'],
        radar_latitude=radar_vol.latitude['data'][0],
        radar_longitude=radar_vol.longitude['data'][0],
        radar_altitude=radar_vol.altitude['data'][0],
        vcp_pattern=radar_vol.metadata['vcp_pattern'])
    
    fixed_angle = radar_vol.fixed_angle['data']
    
    #load the azimuth array:
    # because this is a virtual volume we can know the azimuths in advance
    azimuth = np.arange(0+az_offset, 360, az_spacing_deg)
    logger.debug(azimuth)

    #Load the range variable
    # if your range changes every elevation then you 
    # need to use the slower index_VV
    grange = r2fg + np.arange(0, num_gates) * gate_spacing
    logger.debug(grange)

    #for pyart the collection range never changes....
    elev_range = radar_vol.range['data']

    field_dict = {
        fieldname: np.ma.masked_array(
            np.empty((num_sweeps, num_azs, num_gates)), mask=True) for
        fieldname in fields}

    for s in range(num_sweeps):
        current_elev = radar_vol.fixed_angle['data'][s]
        start_index, end_index = radar_vol.get_start_end(s)

        logger.info(
            f"{s + 1} of {num_sweeps} current_elev: {current_elev:.2f}")

        #compute the range at the VV for the current sweep
        elev_diff = current_elev - vv_elev
        conv_range = elev_range * np.cos(np.radians(elev_diff))

        #we want to identify the index on the cappi where the current elev
        #would fall. There are probably multiple gates for higher elevations
        #that would fall into a single gate on a zero deg elev cappi. But the number
        #of gates is really quite low. You can try and average them if you want
        #but it's a waste of cycles.....If you get elevations higher than 20 Deg we
        #might want to average the data that has the same index.

        conv_index = np.rint((conv_range-r2fg)/gate_spacing).astype(int)
       
        var_index = conv_index
        #python to compare i and i+1
        #shift the conv_index one element to the right
        shift_var_index = np.insert(var_index, 0, -100, axis=0)

        #append a value onto the end of conv_index to make them match in size
        var_index = np.append(var_index, var_index[-1])

        #This will look like:
        # [.....33 34 34 35 36 36 37 38 39 39 40 41 41 42....]
        # we only want one of the duplicated values for the
        #index at the lowest level. You could average and then replace
        #all the duplicated values and then do this, but that's not
        #particularly efficient and you don't gain much for the work
        #
        #using where identify the locations we want to keep
        non_dups = np.where(((var_index != shift_var_index) & (var_index>=0)),
                            True, False)

        #chop non_dups back down (lose one value)
        non_dups = non_dups[0:len(non_dups)-1]

        #each sweep needs to be converted into the location 
        #of the virtual volume. 
    
        for ia in range(num_azs):
            data_az_index = find_az_index(
                azimuth[ia], radar_vol.azimuth['data'][start_index:end_index])
            data_az_index += start_index

            for field_name in field_dict:
                # FIXME: this will be overwritten in each loop, so the
                #  FillValue of the last field in the loop is the winner.
                #  some fields do not have a _FillValue at all. Therefore we
                #  need the if
                #  that may explain why the _FillValue did not make it into
                #  some of my commits
                if '_FillValue' in radar_vol.fields[field_name].keys():
                    metadata['missing_value'] = radar_vol.fields[field_name][
                        '_FillValue']
                variable_data = radar_vol.fields[field_name]['data'][data_az_index]

                vv_data = variable_data[np.where(non_dups==True)]
                #fixme pad vv_data to num_gates if it is short?
                field_dict[field_name][s, ia] = vv_data[0:num_gates]
    
        logger.info(f"num_sweeps: {num_sweeps}, num_azs: {num_azs}, num_gates: "
                     f"{num_gates}")

    var_dict = dict()
    for fieldname in field_dict:
        var_dict[fieldname] = (["fixed_angle", "azimuth", "grange"],
                               field_dict[fieldname])

    # get cartesian x, y coordinate info for the xarray DataArray
    x, y, z0 = pyart.core.antenna_vectors_to_cartesian(grange, azimuth, 0)

    # FIXME this calculation of z neglects different refraction. a more accurate
    #  method would be to calculate the new range of the sweeps and calculate
    #  z in the loop.
    z0_repeated = np.repeat(z0, num_sweeps).reshape(-1, num_sweeps)
    z = z0_repeated + np.sin(np.radians(fixed_angle)) * grange[:, np.newaxis]

    # get geographic coordinates as well
    #x and y from pyart.core.antenna_vectors_to_cartesian are in meters already
    lat, lon = pyart.core.cartesian_to_geographic_aeqd(
        x, y, radar_vol.longitude['data'][0], radar_vol.latitude['data'][0])

    vv_ds = xr.Dataset(
        data_vars=var_dict,
        coords=dict(
            fixed_angle=fixed_angle,
            grange=grange,
            azimuth=azimuth,
            x=(("azimuth", "grange"), x / 1000),
            y=(("azimuth", "grange"), y / 1000),
            z=(("grange", "fixed_angle"), z),
            lat=(("azimuth", "grange"), lat),
            lon=(("azimuth", "grange"), lon)),
        attrs=metadata
        )

    return vv_ds


def get_indexed_vv_experimental(
        radar_vol: pyart.core.Radar,
        num_gates: int = 1200,
        gate_spacing: (int, str) = 'auto') -> xr.Dataset:
    """
    Same as above, creates a xarray Dataset that holds a virtual volume of data.
    We set our virtual volume to the coordinates we want then interpolate the
    data from the pyart volume to these coordinates.

    Parameters
    ----------
    radar_vol: pyart.core.Radar
        pyart radar object that will be converted
    num_gates: int, optional
        Number of gates in the target virtual volume
    gate_spacing
        Spacing of the gates in the target virtual volume
    Returns
    -------
    ds_vv: xarray.Dataset
        Virtual volume in xarray.Dataset
    """

    if gate_spacing == 'auto':
        gate_spacing = radar_vol.range['data'][1] - radar_vol.range['data'][0]
        spacing_test = np.all(np.diff(radar_vol.range['data']) == gate_spacing)

        if spacing_test is False:
            raise NotImplementedError('gate spacing needs to be equal in auto '
                                      'mode')

    fields = radar_vol.fields.keys()
    reindex_dict = {"angle_res": 1.0, "start_angle": 0, "stop_angle": 360,
                    "direction": 1, "tolerance": 0.5, "method": "nearest"}

    ds_vol = pyart_to_xarray_vol(radar_vol, fields, reindex_dict)

    range_to_first_gate = ds_vol.range.values[0]

    trg_grange = xr.Coordinates(
        {"grange": range_to_first_gate + np.arange(0, num_gates) * gate_spacing})
    trg_range = trg_grange["grange"] / np.cos(np.radians(ds_vol.fixed_angle))

    # FIXME: using backfill and pad could be used for linear interpolation
    ds_vv = ds_vol.sel(range=trg_range, method='nearest')

    # let's be happy with approximate numbers?
    ds_vv["range"] = trg_range

    # get cartesian x, y coordinate info for the xarray DataArray
    x, y, z0 = pyart.core.antenna_vectors_to_cartesian(
        ds_vv.grange, ds_vv.azimuth, 0)

    # FIXME this calculation of z neglects different refraction. a more accurate
    #  method would be to calculate the new range of the sweeps and calculate
    #  z in the loop.
    z0 = xr.DataArray(z0[0, :], dims="grange")
    z0 = z0.assign_coords(grange=ds_vv.grange)
    z = z0 + np.sin(np.radians(ds_vv.fixed_angle)) * ds_vv.grange

    # get geographic coordinates as well
    #x and y from pyart.core.antenna_vectors_to_cartesian are in meters already
    lat, lon = pyart.core.cartesian_to_geographic_aeqd(
        x, y, radar_vol.longitude['data'][0], radar_vol.latitude['data'][0])

    ds_vv = ds_vv.assign_coords(
        x=(["azimuth", "grange"], x),
        y=(["azimuth", "grange"], y),
        z=z,
        lat=(["azimuth", "grange"], lat),
        lon=(["azimuth", "grange"], lon)
    )

    return ds_vv
