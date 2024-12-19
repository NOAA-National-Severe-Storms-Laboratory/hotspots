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
import math
import copy
import xarray as xr
from make_logger import logger


def find_az_index(target_az, azimuths):
    #
    # subroutine to find the nearest az index from the azimuth.
    #
    index = -1
    min_diff = 1040.0
    for i in range(len(azimuths)):
        diff = target_az - azimuths[i]
        if diff > 180:
            diff -= 360
        if diff < -180:
            diff += 360

        diff = abs(diff)

        if diff < min_diff:
            min_diff = diff
            index = i

    return index


def indexed_vv_to_cappi_agl(ds_vv: (xr.Dataset, xr.DataArray),
                            z_trg: (int, float),
                            z_cutoff: (int, float) = None,
                            idw_power: int = 2) -> (xr.Dataset, xr.DataArray):
    """
    Interpolate from indexed_vv to AzRan CAPPI for heights given in AGL. Works
    by inverse distance interpolation between the closest sweep above and below
    the data.

    Parameters
    ----------
    ds_vv: xr.Dataset
        xarray Dataset containing the indexed_vv data.
    z_trg: (int, float)
        Target height for the CAPPI.
    z_cutoff: (int, float, optional)
        Maximum height difference to source data.
        If not provided we assume that the cutoff height is the target height
        divided by 3.
    idw_power: int, optional
        Power of the inverse distance weighting. One for linear interpolation,
        two for squared inverse distance weighting

    Returns
    -------
    ds_cappi: xr.Dataset
        xarray Dataset in AzRan coordinates interpolated to a CAPPI level
    """

    ds_cappi_agl = indexed_vv_to_cappi(ds_vv, z_trg, 'agl', z_cutoff=z_cutoff,
                                       idw_power=idw_power)

    return ds_cappi_agl


def indexed_vv_to_cappi_amsl(ds_vv: (xr.Dataset, xr.DataArray),
                             z_trg: (int, float),
                             z_cutoff: (int, float) = None,
                             idw_power: int = 2) -> (xr.Dataset, xr.DataArray):

    """
    Interpolate from indexed_vv to AzRan CAPPI for heights given in AMSL. Works
    by inverse distance interpolation between the closest sweep above and below
    the data.

    Parameters
    ----------
    ds_vv: xr.Dataset
        xarray Dataset containing the indexed_vv data.
    z_trg: (int, float)
        Target height for the CAPPI.
    z_cutoff: (int, float, optional)
        Maximum height difference to source data.
        If not provided we assume that the cutoff height is the target height
        divided by 3.
    idw_power: int, optional
        Power of the inverse distance weighting. One for linear interpolation,
        two for squared inverse distance weighting

    Returns
    -------
    ds_cappi: xr.Dataset
        xarray Dataset in AzRan coordinates interpolated to a CAPPI level
    """

    ds_cappi_amsl = indexed_vv_to_cappi(ds_vv, z_trg, 'amsl', z_cutoff=z_cutoff,
                                        idw_power=idw_power)

    return ds_cappi_amsl


def indexed_vv_to_cappi(ds_vv: xr.Dataset,
                        z_trg: (int, float),
                        z_reference: str,
                        z_cutoff: (int, float) = None,
                        idw_power: int = 2) -> xr.Dataset:
    """
    Interpolate from indexed_vv to AzRan CAPPI. Works by inverse distance
    interpolation between the closest sweep above and below the data.

    Parameters
    ----------
    ds_vv: xr.Dataset
        xarray Dataset containing the indexed_vv data.
    z_trg: (int, float)
        Target height for the CAPPI.
    z_reference: str, ('amsl', 'agl')
        Specifies whether the CAPPI target height is given in amsl or agl
    z_cutoff: (int, float, optional)
        Maximum height difference to source data.
        If not provided we assume that the cutoff height is the target height
        divided by 3.
    idw_power: int, optional
        Power of the inverse distance weighting. One for linear interpolation,
        two for squared inverse distance weighting

    Returns
    -------
    ds_cappi: xr.Dataset
        xarray Dataset in AzRan coordinates interpolated to a CAPPI level
    """

    # if no z_cutoff is provided we make an assumption that could be reasonable
    if z_cutoff is None:
        z_cutoff = z_trg / 3

    num_sweeps = len(ds_vv.fixed_angle)

    # make sure that volume is ordered by fixed angle in ascending order
    ds_vv = ds_vv.sortby("fixed_angle", ascending=True)

    # get the z data from the indexed_vv
    # VERY IMPORTANT: here we either convert the radar z coordinate to height
    # AMSL or stay in the AGL framework
    if z_reference == 'amsl':
        logger.info('reminder: using AMSL for target CAPPI height')
        z_src = ds_vv.z + ds_vv.radar_altitude
    elif z_reference == 'agl':
        z_src = ds_vv.z
        logger.info('reminder: using AGL for target CAPPI height')
    else:
        raise ValueError('z_reference must be amsl or agl')

    # fill a new DataArray with the target altitude
    z_trg_arr = xr.full_like(z_src, z_trg)

    z_diff = z_src - z_trg_arr

    # get the closest index below the target height
    idx_below = xr.where(z_diff < 0, np.abs(z_diff), 99999).argmin(
        dim='fixed_angle')

    # get the closest index above the target height
    idx_above = idx_below + 1
    # we must assure that we don't try to access a sweep that doesn't exist
    idx_above[idx_above > num_sweeps-1] = num_sweeps-1

    z_diff_below = z_diff.isel(fixed_angle=idx_below)
    z_diff_above = z_diff.isel(fixed_angle=idx_above)

    # calculate the weights
    weight_below = 1 / (z_diff_below ** idw_power)
    weight_above = 1 / (z_diff_above ** idw_power)

    val_below = ds_vv.isel(fixed_angle=idx_below)
    val_above = ds_vv.isel(fixed_angle=idx_above)

    # linear interpolation between the levels
    ds_cappi = ((val_above * weight_above + val_below * weight_below) /
                (weight_above + weight_below))

    # keep only data within certain height ranges, rest is set to NaN
    height_mask = (z_diff_below > -z_cutoff) & (z_diff_above < z_cutoff)
    ds_cappi = ds_cappi.where(height_mask)

    # assign all the nice attributes
    ds_cappi.attrs = ds_vv.attrs
    ds_cappi.coords["z"] = z_trg

    return ds_cappi


def make_cappi(prune_vol, field_name, cappi_height_meters,
               max_cappi_height_dist='auto', num_gates=1200, num_az=360,
               gate_spacing_m=250, az_offset=0.5):
    """
        Parameters
        ---------
        prune_vol: pyart radar object, a volume of radar data that
                   has been pruned to include only one sweep at
                   each elevation and ordered with the lowest
                   sweep at index 0

        field_name: string, accessable name in pyart object, ex. 'reflectivity'

        cappi_height_meters: float, height AGL (of the radar) that you want
                   to compute the CAPPI at

        max_cappi_height_dist: float, unit, meters, the maximum distance the
                   data is allowed to be from the target_height above
                   to still be included in the cappi. Larger values shrink
                   the cone of silence and extend the cappi outward at a
                   loss of data accuracy in those lcoations.
                   'auto' value = 0.3*cappi_height_meters

       num_gates: int, number of gates per radial in the output CAPPI
       num_az: int, number of azimuhts per sweep in the output CAPPI

       gate_spacing_m: float, distance between gates in the output CAPPI

       Note: the output CAPPI takes the range_to_first_gate from the
       input radar volumne and utilizes it in the output volume. This is
       note requied, but reasonable

       az_offest: float, limits 0.0<1.0, the offset from zero of each azimuth
                   example: 0.2 with num_az = 360 would give CAPPI azimuths
                   of [0.2, 1.2, 2.2, 3.2 .....359.2]
                   recommend 0.5 for nexrad data


        Returns
        -------
        cappi: pyart radar object: with
               fields[] = ['cappi_<field_name>', 'cappi_height', 'work_data' ]

               'cappi_<field_name>' is the data at the height you requested from the
               input field name in the units of the requested field.
               Data were created by linear interplation to the cappi_height_meters from two
               elevations that span that height at that location

               'cappi_height' is the height that the cappi used to compute the data, rings
               of heights were no radar data exists are noraml. Missing data flags are where
               the algortihm was unable to be computed due to max_cappi_height_dist. If your
               max_cappi_height_dist is too small there will be rings in the domain
               that have missing data. Locations where the algorithm was able to compute the
               cappi will have a height=cappi_height_meters

               This output is for diagnostic purposes, you've been warned. Don't ask about it.

               'work_data' is trash.

    """

    ##find the dictionary of the field we want:
    moment_names = list(prune_vol.fields.keys())
    logger.debug(moment_names)
   
    if max_cappi_height_dist == 'auto':
        max_cappi_dist = cappi_height_meters*0.3
                     # the maximum distance in meters allowed between the target cappi height
                     # and the actual height, should vary with height because radars have
                     # wider vertical gaps aloft
    else:
        max_cappi_dist = max_cappi_height_dist
    
    target_field = prune_vol.fields[field_name]
    logger.debug(target_field)
    
    # We have a pruned volume and now we setup an empty pyart radial that we want to
    # make into a CAPPI. 
    # 
    # At this point you can use any method to make a CAPPI from the pruned volume of data
    # We implemented this method because we used it before
    
    cappi_target_height_m = cappi_height_meters 
   
    cappi_ceil = cappi_target_height_m + max_cappi_dist 
    #num_gates = 1200
    #gate_spacing_m = 250.0
    
    cappi_elev = 0.0
        
    #num_az = 360
    az_spacing_deg = num_az/360.0
    #az_offset = 0.5
        
    cappi_azimuths = np.linspace(0, (num_az-1), num_az)*az_spacing_deg + az_offset
    #print ('CAPPI azimuths: ', cappi_azimuths)
        
    range_to_first_gate = prune_vol.range['data'][0]
    #print ('Range_to_first: ', range_to_first_gate)
    
    cappi_range = range_to_first_gate + np.linspace(0,(num_gates-1), num_gates)*gate_spacing_m
    #print('CAPPI range: ', cappi_range)
        
    cappi = pyart.testing.make_empty_ppi_radar(num_gates, num_az, 1)
    
    #FIXME:
    # go ahead and add all the other relevant metadata
    # mostly because we want to be good German citizens
    cappi.time = prune_vol.time
    cappi.metadata = prune_vol.metadata
    cappi.lattidue = prune_vol.latitude
    cappi.longitude = prune_vol.longitude
    cappi.altitude = prune_vol.altitude
    cappi.altitude_agl = prune_vol.altitude_agl
    
    #print ('altitude:', cappi.altitude, ' agl: ', cappi.altitude_agl)
    
    cappi.fixed_angle["data"] = np.array([cappi_elev])
    cappi.sweep_number["data"] = np.array([0])
    cappi.sweep_start_ray_index["data"] = np.array([0])
    cappi.sweep_end_ray_index["data"] = np.array([num_az-1])
    
    cappi.azimuth['data'][:] = cappi_azimuths
    cappi.range['data'][:] = cappi_range
    cappi_data = np.empty((num_az, num_gates), dtype='float32')
    cappi_data.fill(target_field["_FillValue"])
    #cappi_data.where(0,  target_field["_FillValue"])
    cappi_name = 'cappi_' + field_name
    cappi_dict = {
        "data": cappi_data,
        "units": target_field['units'],
        "long_name": 'cappi_'+target_field["long_name"],
        "_FillValue": target_field["_FillValue"],
        "standard_name": cappi_name
    }
    cappi.add_field(cappi_name, cappi_dict)
    
    #need this for a deep copy. Learning the hard way
    work_data = np.empty((num_az, num_gates), dtype='float32')
    work_data.fill(target_field["_FillValue"])
    #let's go ahead and make the data storage forwork_data
    working_dict = {
        "data": work_data,
        "units": target_field['units'],
        "long_name": 'work_data',
        "_FillValue": target_field["_FillValue"],
        "standard_name": 'work_data'
    }
    cappi.add_field('work_data', working_dict)
   
    #need this for a deep copy. Learning the hard way
    height_data = np.empty((num_az, num_gates), dtype='float32')
    height_data.fill(target_field["_FillValue"])
    #let's go ahead and make the data storage for cappi_height
    height_dict = {
        "data": height_data,
        "units": 'meters AGL',
        "long_name": 'cappi_height',
        "_FillValue": target_field["_FillValue"],
        "standard_name": 'cappi_height'
    }
    cappi.add_field('cappi_height', height_dict)

    # Now the hard part where we iterate through the pyart object
    # which is sorted (important!) by elevation lowest elevation first and add the data
    # to a working_data and working_height array. When we encounter
    # data that spans the target height we compute the value of the cappi
    # 
    
    sweeps = prune_vol.sweep_number['data']
    cappi_r2fg = cappi_range[0]
    cappi_gate_spacing = cappi_range[1] - cappi_range[0]

    #we need the gate_z value later so
    prune_vol.init_gate_x_y_z()
    #print(prune_vol.gate_z['data'].shape)

    missing_value = target_field["_FillValue"]
    #
    #code taks some time....I find this output soothing.
    print('makeCAPPI: working...will notify at end')
    for s in range(len(sweeps)):
   
        current_elev = prune_vol.fixed_angle['data'][s]

        #code taks some time....I find this output soothing.
        print(s, 'current elev:', current_elev, 'cappi_elev', cappi_elev, 'start: ', prune_vol.get_start(s))
        #extract the current elevation range data
        #print(prune_vol.range['data'].shape)
        work_range = prune_vol.range['data']
        #print( work_range.shape)
        #convert the range for each bin in the current elevation to 
        #the range of the cappi angle (usually zero degrees)
        elev_diff = current_elev - cappi_elev
        conv_range = work_range * math.cos(math.radians(elev_diff))
        #print('conv_range:', conv_range[0:20])
        #if s == len(sweeps)-1:
        #    for i in range(len(conv_range)):
        #        print(i, work_range[i], conv_range[i])
        
        #we want to identify the index on the cappi where the current elev
        #would fall. There are probably multiple gates for higher elevations
        #that would fall into a single gate on a zero deg elev cappi. But the number
        #of gates is really quite low. You can try and average them if you want
        #but it's a waste of cycles.....If you get elevations higher than 20 Deg we
        #might want to average the data that has the same index. 
        conv_index = np.rint((conv_range-cappi_r2fg)/cappi_gate_spacing).astype(int)
        #print('conv_index:', conv_index[0:20])
        
        #compute the radar (AGL,at feedhorn) height along the radial for this elevation
        start_index, end_index = prune_vol.get_start_end(s)
        #print('start:', start_index, ' end_index: ', end_index)
        
        #height in AGL, meters
        conv_height = prune_vol.gate_z['data'][start_index] 
        #print(start_index, 'conv_height: ', conv_height[0:20])
       
        var_index = conv_index
        #shift the var_index one element to the right
        shift_var_index = np.insert(var_index,0,-100, axis=0)

        #append a value onto the end of var_index to make them match in size
        var_index = np.append(var_index,var_index[-1])

        #using where, identify the locations we want to keep
        non_dups = np.where( ((var_index != shift_var_index) & ( var_index >= 0 )), True, False)

        #chop non_dups back down:
        non_dups = non_dups[0:len(non_dups)-1]
        #print the output for the last sweep because it has the most interesting output
        #if s == len(sweeps)-1:
        #    for i in range(len(conv_range)):
        #        print(i, work_range[i], conv_range[i], conv_index[i], conv_height[i])
     
        #we have all we need start reviewing the data computing the CAPPI
        #print(prune_vol.azimuth['data'].shape)
        
        cappi_az_index = 0
        for a in cappi.azimuth['data']:
            #print('working on cappi azimuth', a)
            #find the nearest data azimuth in this sweep.
            #print('shape',prune_vol.azimuth['data'][start_index:end_index].shape)
            data_az_index = find_az_index(a, prune_vol.azimuth['data'][start_index:end_index])
            data_az_index += start_index
            #
            #convert the input radial of data at the az_index to a matching cappi radial
            #using the computed index values above
            # The input radial data
            variable_data = prune_vol.fields[field_name]['data'][data_az_index]
            #our target
            #var_data = np.empty(num_gates, dtype='float32')
            #var_data.fill(missing_value)

            #var_height = np.empty(num_gates, dtype='float32')
            #var_height.fill(missing_value)

            #Now loop through and add the data
            #index = 0
            #for d in variable_data:
            #    if index >= num_gates:
            #        break
            #    if np.isnan(d) == False:
            #        var_data[conv_index[index]] = d
                #always load height, never missing or nan (we hope)
            #    var_height[conv_index[index]] = conv_height[index]
            #    index +=1

            var_data = variable_data[np.where(non_dups==True)]
            var_height = conv_height[np.where(non_dups==True)]

            #Now we combine this data to the cappi if the height of the data looks right

            #print('working on cappi azimuth', a, ' found azimuth: ', prune_vol.azimuth['data'][data_az_index], ' data_az_index: ', data_az_index)
            #print('found az_index:', data_az_index, 'az_diff:', a-prune_vol.azimuth['data'][data_az_index])
            cappi_radial = np.array(cappi.fields[cappi_name]['data'][cappi_az_index])
            #print(cappi_radial.shape)
            #print('missing:', missing_value)
            #print('fields: ', cappi.fields)
            #print('var_radial:', var_data[0:20])
            #print('var_height:', var_height[0:20])
            #print('cappi_radial: ', cappi_radial[0:20])
            cappi_work_data = np.array(
                cappi.fields['work_data']['data'][cappi_az_index])
            #print('cappi_work_data: ', cappi_work_data[0:20])
            cappi_height = np.array(
                cappi.fields['cappi_height']['data'][cappi_az_index])
            #if a >180 and a < 181:
            #    print('working on cappi azimuth', a, ' found azimuth: ', prune_vol.azimuth['data'][data_az_index], ' data_az_index: ', data_az_index)
            #    print('found az_index:', data_az_index, 'az_diff:', a-prune_vol.azimuth['data'][data_az_index])
            #    print('var_radial:', var_data[0:20])
            #    print('var_height:', var_height[0:20])
            #    print('cappi_radial: ', cappi_radial[0:20])
            #    print('cappi_height: ', cappi_height[0:20])
            #    print('cappi_work_data: ', cappi_work_data[0:20])
            
            #if(s==0):
            #   cappi_radial.fill(missing_value)
            #   cappi_work_data.fill(missing_value)
            #   cappi_height.fill(missing_value)

            #
            #FIXME: improve this by using pandas

            for i in range(num_gates):
                #
                #an attempt to go faster
                #  but someone will come along and figure out how to do this all
                #  in parallel numpy arrays.....I hope
                if var_height[i] > cappi_ceil:
                    break
                #find the first value of index i in conv_index
                #if i > 200 and i < 220 and a > 180 and a < 181:
                #    print('data', i, cappi_work_data[i], cappi_height[i])
                #    print('var:', i, var_data[i], var_height[i])
                #    print('cappi:', i, cappi_radial[i])
                #do not run if data index is invalid of cappi radial already has a value
               
                # and (np.isnan(variable_data[data_index])==False)
                #fill if missing
                if cappi_work_data[i] == missing_value:
                    cappi_work_data[i] = var_data[i]
                        
                #fill if missing
                #if cappi_height[i] == missing_value:
                #    cappi_height[i] = var_height[i]
                
                #if cappi_radial[i] == missing_value and abs(var_height[i]-cappi_target_height_m) < max_cappi_dist:
                #    cappi_radial[i] = var_data[i]
                #    cappi_height[i] = var_height[i]
                    
                #only estimate data if we haven't made an estimate at the proper height
                if cappi_height[i] != cappi_target_height_m:     
                        
                    #test to see if the heights span the cappi height you want:
                    if cappi_height[i] < cappi_target_height_m and var_height[i] > cappi_target_height_m:
                        #and (var_height[i]-cappi_height[i])/2 < max_cappi_dist:    
                        #we have a span so linear interpolate the value between
                        if cappi_work_data[i] != missing_value and var_data[i] != missing_value:
                            cappi_radial[i] = cappi_work_data[i] + (cappi_target_height_m-cappi_height[i]) * \
                            (var_data[i]-cappi_work_data[i])/(var_height[i]-cappi_height[i])
                            #now we set the cappi_height to the value of cappi_target_height_m
                            #note that the if statement above will not execute on and =
                            cappi_height[i] = cappi_target_height_m
                            #if (a > 180 and a < 181):
                            #    print(i, cappi_work_data[i], cappi_radial[i], variable_data[i])
                            #    print(i, cappi_height[i], cappi_target_height_m, var_height[i])
                            
                    elif cappi_height[i] < cappi_target_height_m and var_height[i] < cappi_target_height_m:
                    #both the stored height and the new height are below the cappi_target_height_m
                    #take the new height
                        if (cappi_target_height_m-var_height[i]) < max_cappi_dist:
                            cappi_radial[i] = var_data[i]
                            cappi_height[i] = var_height[i]
            #
            #push the arrays back into the object
            # note that we keep the latest version of var_data and var_var_height
            # but the computed version of cappi_radial
            cappi.fields[cappi_name]['data'][cappi_az_index] = copy.deepcopy(cappi_radial)
            cappi.fields['work_data']['data'][cappi_az_index] = copy.deepcopy(var_data[0:num_gates])
            cappi.fields['cappi_height']['data'][cappi_az_index] = copy.deepcopy(cappi_height)
            cappi_az_index += 1
            
            #print('work_data:', cappi_work_data)
    
    logger.info('makeCAPPI: done.')
    return cappi 
    
    
    
    
    
