import copy

import numpy as np
import pyart
import xarray as xr

from make_logger import logger
from prepro.prepro_helpers import calc_std_texture_nORPG, \
    local_running_ave_1D, compute_SNR_nORPG, calc_std_texture_nORPG_xarray, \
    compute_SNR_nORPG_xarray
from prepro.process_phase import compute_min_system_phase, \
    compute_smoothed_phase_pzhang_xarray, compute_smoothed_phase_PZhangNSSL, \
    calc_kdp_wradlib


def preprocessor_nORPG_pyart(metadata, radar_vol):
    """
        Preprocessor_nORPG_pyart:
            This python subroutine attempts to replicate the steps needed to
            output data similar too the U.S. NEXRAD network ORPG. It should
            more closely match the data output in the Level III data if that
            data were at the original 250m gate resolution. There are significant
            differences in the treatment of the data between this subroutine and
            the ORPG. First the treatment of KDP cannot be replicated without
            both a CAPPI of Z at 3km and the Metsignal algorithm. We substitute
            the triple median method developed by PZhang, NSSL instead. Second,
            frequency correction is applied on the ORPG, but not here yet. Finally,
            data from the ORPG is "recombined" from 250m to 1km by averaging. I
            don't find this step worthy to include here.

        Errors, concerns, proposed modifications, can be sent to:
            John.Krause@noaa.gov, please help us improve this code

        Metadata:
            a dict of metadata is required for this subroutine. Pyart does not
            extract some of the things we need, specfically minmum system phase
            and other radar data may use the metadata method to input data.

            The code will run with a blank dict.
            ex.  metadata = {}

            The following inputs to the metadata will modifiy the output.
                'zdr_absolute_calibration'
                'z_absolute_calibration'
                'min_system_phase'
                'phase_window'
                'C' used in the SNR computations, receiver gain?
                'FillValue'

        Parameters
        ----------
            metadata: dict, can be blank
                contains additional information used in processing

            radar_vol: pyart Radar object
                a pyart object that contains the 6 radar fields

            outputs:
                output_vol: pyart Radar object with additional data
    """

    sweeps = radar_vol.sweep_number['data']
    z_field = copy.deepcopy(radar_vol.fields['reflectivity'])
    cc_field = copy.deepcopy(radar_vol.fields['cross_correlation_ratio'])
    zdr_field = copy.deepcopy(radar_vol.fields['differential_reflectivity'])
    phase_field = copy.deepcopy(radar_vol.fields['differential_phase'])
    vel_field = copy.deepcopy(radar_vol.fields['velocity'])
    snr_field = copy.deepcopy(radar_vol.fields['differential_reflectivity'])
    kdp_field = copy.deepcopy(radar_vol.fields['differential_phase'])
    texture_z_field = np.full(np.shape(z_field['data']), np.nan)
    texture_phi_field = np.full(np.shape(phase_field['data']), np.nan)

    num_az, num_gates = z_field['data'].shape

    # FIXME: the values below are not used anywhere?
    if 'FillValue' in metadata:
        missing_value = metadata['FillValue']
    else:
        missing_value = z_field["_FillValue"]

    output = copy.deepcopy(radar_vol)

    #metadata['radar_Frequency_in_MHz'] =
    #need help here.... where is this in the pyart radar object?
    # VK: if available, it should be in
    # radar_vol.instrument_parameters['frequency'],
    # but the Py-ART reader of NEXRAD Level II data does not include it

    metadata['gatespacing_meters'] = radar_vol.range['meters_between_gates']
    #KDP is reported in deg/km not deg/m
    gate_spacing_km = radar_vol.range['meters_between_gates']/1000.0

    metadata['distance_to_first_gate_meters'] = radar_vol.range['meters_to_center_of_first_gate']
    metadata['ngates'] = radar_vol.ngates
    metadata['rays_per_sweep'] = radar_vol.rays_per_sweep['data']

    #for SNR computations:
    if 'C' not in metadata:
        metadata['C'] = 40.0 #need help here.... where is this in the pyart radar object?
    metadata['phase_wrap_in_degrees'] = radar_vol.fields['differential_phase']['valid_max']

    ##calibration if you want it.
    if 'zdr_absolute_calibration' in metadata:
        zdr_calibration = metadata['zdr_absolute_calibration']
    else:
        zdr_calibration = 0.0

    if 'z_absolute_calibration' in metadata:
        z_calibration = metadata['z_absolute_calibration']
    else:
        z_calibration = 0.0

    #
    #I segmented the processing by sweeps for better comprehension.
    #The ORPG processes the data a radial at a time. We can too if we
    #wanted too.
    #
    for s in range(len(sweeps)):
        #which data do we use?
        start_index, end_index = radar_vol.get_start_end(s)
        num_az_s = end_index-start_index

        # The code can be slow at times, we use this to keep the user
        # aware it's working.
        print(s, 'sweep_num: ', sweeps[s], 'start: ', start_index, ' end: ', end_index)

        #for convienece we use elevations of data.
        #you could push the input data in one ray at a time,
        #but that requires knowing
        #min_system_phase, which you can compute in advance, or have in metadata

        z_data = z_field['data'][start_index:end_index]
        cc_data = cc_field['data'][start_index:end_index]
        zdr_data = zdr_field['data'][start_index:end_index]
        phase_data = phase_field['data'][start_index:end_index]
        v_data = vel_field['data'][start_index:end_index]

        elevation = radar_vol.elevation['data'][start_index:end_index] #elevation for each ray
        azimuth = radar_vol.azimuth['data'][start_index:end_index]

        #
        #returned data
        #
        prepro_z_data = np.empty([num_az_s, num_gates]) * np.nan
        prepro_cc_data = np.empty([num_az_s, num_gates]) * np.nan
        prepro_zdr_data = np.empty([num_az_s, num_gates]) * np.nan
        prepro_phase_data = np.empty([num_az_s, num_gates]) * np.nan
        prepro_kdp_data = np.empty([num_az_s, num_gates]) * np.nan

        prepro_v_data = np.empty([num_az_s, num_gates]) * np.nan
        prepro_snr_data = np.empty([num_az_s, num_gates]) * np.nan

        texture_z_data = np.empty([num_az_s, num_gates]) * np.nan
        texture_phi_data = np.empty([num_az_s, num_gates]) * np.nan

        #prepro_textureZ = np.empty([num_az_s, num_gates]) * np.nan
        #prepro_texturePhase = np.empty([num_az_s, num_gates]) * np.nan

        #### preprocessor starts here ####
        debug=False
        #correct phase for radar frequency difference to KTLX: (not implemented)
        # Dusan Zurnic, NSSL recommends that the radar frequency differenct be accounted for because
        # the corrections for horizontal attenuation are based/developed on the KOUN frequency

        #must have this if you are using radial-by-radial processing
        if 'min_system_phase' not in metadata:
            min_system_phase = compute_min_system_phase(phase_data, cc_data)

        logger.debug(f"using {min_system_phase} as minimum sys phase")

        if 'phase_window' not in metadata:
            metadata['phase_window'] = 9

        #phase_window = metadata['phase_window']

        #correct the phase to a minuimum value of zero,
        #uses the current elevation to find the min_system_phase
        #
        # The ORPG process the phase and kdp data on two different scales
        # short_gate = 9 for convective
        # long_gate = 25 for non-convective
        #
        #There is no nORPG match to phase processing on the ORPG which
        #requires a CAPPI and the MetSignal algorithm.
        #
        #The Triple median method by PZhang, NSSL is your best 2nd option.
        #
        prepro_phase_9gate = compute_smoothed_phase_PZhangNSSL(
                phase_data, cc_data, min_system_phase=min_system_phase,
                window_size=9)

        prepro_phase_25gate = compute_smoothed_phase_PZhangNSSL(
                 phase_data, cc_data, min_system_phase=min_system_phase,
                 window_size=25)

        #calculate kdp as the slope of smoothed phi. If you don't like the kdp
        #computation, change process_differential_phase not this.
        #Note the ORPG uses llse, but the implementations of this in python
        #are too slow. We use wradlib's derivate solution. No need to reinvent
        #the wheel.
        prepro_kdp_9gate = calc_kdp_wradlib(prepro_phase_9gate, 9,
                                            gate_spacing_km)

        prepro_kdp_25gate = calc_kdp_wradlib(prepro_phase_25gate, 25,
                                             gate_spacing_km)

        for a in range(0, end_index-start_index):
            metadata['elev'] = elevation[a]
            metadata['az'] = azimuth[a]

            #calculate the texture fields, used in HCA
            texture_z_data[a] = calc_std_texture_nORPG(
                z_data[a], 5, debug)
            texture_phi_data[a] = calc_std_texture_nORPG(
                phase_data[a], 9, debug)

            #calculate the smoothed (radial directioin only) dualpol
            #data, to reduce the noise
            prepro_z_data[a] = local_running_ave_1D(z_data[a], 3, debug)
            prepro_v_data[a] = local_running_ave_1D(v_data[a], 5, debug)
            prepro_zdr_data[a] = local_running_ave_1D(zdr_data[a], 5, debug)
            prepro_cc_data[a] = local_running_ave_1D(cc_data[a], 5, debug)

            #compute the SNR
            prepro_snr_data[a] = compute_SNR_nORPG(prepro_z_data[a], elevation[a], metadata)
            #correct for horizontal attenuation
            prepro_zdr_data[a] += prepro_phase_25gate[a]*0.004
            prepro_z_data[a] += prepro_phase_25gate[a]*0.04

            #apply absolute calibration
            prepro_zdr_data[a] += zdr_calibration
            prepro_z_data[a] += z_calibration

            #combine long and short kdp and phase based on Z
            """
            First Citation
    Ryzhkov, A. V., , and Zrnić D. S. , 1996: Assessment of rainfall measurement that uses specific differential phase. J. Appl. Meteor., 35 , 2080–2090.

            A Better Citation. 
    Brandes, E. A., Ryzhkov, A. V., & Zrnić, D. S. (2001). An Evaluation of Radar Rainfall Estimates from Specific Differential Phase, Journal of Atmospheric and Oceanic Technology, 18(3), 363-375.

            """
            prepro_kdp_data[a] = np.where(prepro_z_data[a] >= 40.0,
                                          prepro_kdp_9gate[a],
                                          prepro_kdp_25gate[a])
            prepro_phase_data[a] = np.where(prepro_z_data[a] >= 40.0,
                                            prepro_phase_9gate[a],
                                            prepro_phase_25gate[a])

        ###################################################################

        #store the data in the field
        z_field['data'][start_index:end_index][:] = prepro_z_data
        cc_field['data'][start_index:end_index][:] = prepro_cc_data
        zdr_field['data'][start_index:end_index][:] = prepro_zdr_data
        phase_field['data'][start_index:end_index][:] = prepro_phase_data
        vel_field['data'][start_index:end_index][:] = prepro_v_data
        snr_field['data'][start_index:end_index][:] = prepro_snr_data
        kdp_field['data'][start_index:end_index][:] = prepro_kdp_data

        texture_z_field[start_index:end_index, :] = texture_z_data
        texture_phi_field[start_index:end_index, :] = texture_phi_data

   ###################################################################

    #push the field onto the pyart object
    output.add_field('prepro_zh', z_field)
    output.add_field('prepro_cc', cc_field)
    output.add_field('prepro_zdr', zdr_field)
    output.add_field('prepro_phase', phase_field)
    output.add_field('prepro_v', vel_field)
    output.add_field('prepro_snr', snr_field)
    output.add_field('prepro_kdp', kdp_field)

    # adding the new texture fields
    output.add_field('reflectivity_texture',
                     dict(data=np.ma.masked_invalid(texture_z_field),
                          _FillValue=-9999.0))
    output.add_field('differential_phase_texture',
                     dict(data=np.ma.masked_invalid(texture_phi_field),
                          _FillValue=-9999.0))

    return output


def preprocessor_norpg_xarray(radar_vol: pyart.core.Radar,
                              metadata: dict) -> pyart.core.Radar:
    """
    xarray version of the Python nORPG processing. This should be ~3-4x
    faster than the non-xarray version.
    This python subroutine attempts to replicate the steps needed to
    output data similar too the U.S. NEXRAD network ORPG. It should
    more closely match the data output in the Level III data if that
    data were at the original 250m gate resolution. There are significant
    differences in the treatment of the data between this subroutine and
    the ORPG. First the treatment of KDP cannot be replicated without
    both a CAPPI of Z at 3km and the Metsignal algorithm. We substitute
    the triple median method developed by PZhang, NSSL instead. Second,
    frequency correction is applied on the ORPG, but not here yet. Finally,
    data from the ORPG is "recombined" from 250m to 1km by averaging. I
    don't find this step worthy to include here.

    Errors, concerns, proposed modifications, can be sent to:
        John.Krause@noaa.gov and vinzent.klaus@boku.ac.at, please help us
        improve this code

    Parameters
    ----------
    radar_vol: pyart.core.Radar
        a pyart object that must contain the radar fields 'reflectivity',
        'differential_reflectivity', 'differential_phase' and
        'cross_correlation_ratio'
    metadata: dict
        a dict of metadata is required for this subroutine. Pyart does not
        extract some of the things we need, specfically minimum system phase
        and other radar data may use the metadata method to input data.

        The code will run with a blank dict.
        ex.  metadata = {}

        The following inputs to the metadata will modify the output.
            'zdr_absolute_calibration'
            'z_absolute_calibration'
            'min_system_phase'
            'phase_window'
            'C' used in the SNR computations, receiver gain?

    Returns
    -------
    output: pyart.core.Radar
        pyart radar object with processed fields

    """
    # prepare for later
    output = copy.deepcopy(radar_vol)

    field_dict = {field_name: (["azimuth", "range"], radar_vol.fields[
        field_name]['data']) for field_name in radar_vol.fields}
    xr_vol = xr.Dataset(
        data_vars=field_dict,
        coords=dict(
            azimuth=("azimuth", radar_vol.azimuth['data']),
            range=("range", radar_vol.range['data']),
            elevation=("azimuth", radar_vol.elevation['data']))
    )

    xr_vol.attrs['sweep_start_ray_index'] = radar_vol.sweep_start_ray_index
    xr_vol.attrs['sweep_end_ray_index'] = radar_vol.sweep_end_ray_index
    xr_vol.attrs['sweep_number'] = radar_vol.sweep_number
    xr_vol.attrs['fixed_angle'] = radar_vol.fixed_angle

    xr_vol.attrs['metadata'] = metadata

    metadata['gatespacing_meters'] = radar_vol.range['meters_between_gates']
    # KDP is reported in deg/km not deg/m
    gate_spacing_km = radar_vol.range['meters_between_gates'] / 1000.0
    metadata['distance_to_first_gate_meters'] = radar_vol.range[
        'meters_to_center_of_first_gate']
    metadata['ngates'] = radar_vol.ngates
    metadata['rays_per_sweep'] = radar_vol.rays_per_sweep['data']

    # for SNR computations:
    if 'C' not in metadata:
        metadata['C'] = 40.0  # need help here.... where is this in the pyart radar object?
    metadata['phase_wrap_in_degrees'] = radar_vol.fields['differential_phase'][
        'valid_max']

    ##calibration if you want it.
    if 'zdr_absolute_calibration' in metadata:
        zdr_calibration = metadata['zdr_absolute_calibration']
    else:
        zdr_calibration = 0.0

    if 'z_absolute_calibration' in metadata:
        z_calibration = metadata['z_absolute_calibration']
    else:
        z_calibration = 0.0

    # FIXME: there are still some differences in KDP between the versions
    #  which are due to the differences in median calculation between xarray
    #  and pandas
    prepro_phase_9gate = compute_smoothed_phase_pzhang_xarray(
        xr_vol.differential_phase, xr_vol.cross_correlation_ratio,
        window_size=9)
    prepro_phase_25gate = compute_smoothed_phase_pzhang_xarray(
        xr_vol.differential_phase, xr_vol.cross_correlation_ratio,
        window_size=25)

    prepro_kdp_9gate = calc_kdp_wradlib(
        prepro_phase_9gate, 9, gate_spacing_km)
    prepro_kdp_25gate = calc_kdp_wradlib(
        prepro_phase_25gate, 25, gate_spacing_km)

    prepro_texture_z = calc_std_texture_nORPG_xarray(
        xr_vol.reflectivity, 5)
    prepro_texture_phase = calc_std_texture_nORPG_xarray(
        xr_vol.differential_phase, 9)

    data_5gate_smooth = xr_vol[["velocity", "differential_reflectivity",
                                "cross_correlation_ratio"]].rolling(
        range=5, center=True, min_periods=2).mean()
    data_3gate_smooth = xr_vol["reflectivity"].rolling(
        range=3, center=True, min_periods=1).mean().rename(
        "prepro_zh")

    data_5gate_smooth = data_5gate_smooth.rename(
        {"velocity": "prepro_v",
         "differential_reflectivity": "prepro_zdr",
         "cross_correlation_ratio": "prepro_cc"}
    )

    xr_prepro_vol = xr.merge([data_5gate_smooth, data_3gate_smooth,
                              prepro_texture_z, prepro_texture_phase])
    xr_prepro_vol.attrs = xr_vol.attrs

    # get the signal-to-noise ratio field
    snr_field = compute_SNR_nORPG_xarray(
        xr_prepro_vol.prepro_zh, xr_prepro_vol.elevation, metadata)

    # correct for horizontal attenuation
    xr_prepro_vol["prepro_zdr"] += prepro_phase_25gate * 0.004
    xr_prepro_vol["prepro_zh"] += prepro_phase_25gate * 0.04

    # apply absolute calibration
    xr_prepro_vol["prepro_zdr"] += zdr_calibration
    xr_prepro_vol["prepro_zh"] += z_calibration

    prepro_kdp_data = xr.where(xr_prepro_vol.prepro_zh >= 40.0,
                               prepro_kdp_9gate,
                               prepro_kdp_25gate)
    prepro_phase_data = xr.where(xr_prepro_vol.prepro_zh >= 40.0,
                                 prepro_phase_9gate,
                                 prepro_phase_25gate)

    xr_prepro_vol["prepro_kdp"] = prepro_kdp_data
    xr_prepro_vol["prepro_phase"] = prepro_phase_data
    xr_prepro_vol["prepro_snr"] = snr_field

    # ###################################################################

    # write the new fields to the radar object
    for var in xr_prepro_vol.data_vars:
        logger.debug(f"writing {var} to output radar object")
        new_field_dic = dict(
            data=np.ma.masked_invalid(xr_prepro_vol[var].values),
            _FillValue=-9999.0)
        output.add_field(var, new_field_dic)

    return output


def compute_nexrad_wave_form(radar_vol):
    """
    Parameters
    ---------
    radar_vol: pyart radar object

    Returns
    -------
    nexrad_wave_form: list
        each sweep identifed by sweep_type


    Identifies the type of waveform that was used to collect the data

    There are currently 3 types:
    surv, dopper, combined

    Survalence data is collected with a short nyquist velocity and usually
    has fewer pulses. The data is noiser than the dopper waveforms, but has
    fewer locations where the data is ambigous due to range folding

    Dopper data is collected with a longer nyquist velocity and usually has
    more pulses. The data is less noisy due to longer dwell time and more pulses
    but the data has areas of range folding where the data are unknown.

    Combined data is a catch all category where the data collected has been
    unified, because only one sweep at a particular elevation
    was present in the data

    Knowing all the possible permutations of nyquist velocity and VCP in the
    nexrad network is possible, but always changing. The addition of SZ phase
    coding is considered surv data for this subroutine, but if known it
    could have it's own category. Not enough information in the pyart radar
    object to identify SZ at this time.

    This determination of waveform is needed because users should not use
    a collections moments without, or simply all the moments withouth
    knowing that they are collected with nyquist velocities unsuited
    for the purpose needed.

    """

    #find the target elevations. Don't use actual because the radar is
    #sometimes ramping up and down and actual elevations won't match the
    #target elevations

    target_elev = radar_vol.fixed_angle['data']
    logger.debug(f"elev: {target_elev}")

    #we use the comparison of nyquist velocities at each target elevation
    #to deterime the waveform

    nyq_vel = radar_vol.instrument_parameters['nyquist_velocity']['data']

    #the nyquist is stored for each ray rather than by elevation. We need to
    #know when the rays start and we will use the first nyquist in the ray
    #to make the determination

    sweep_start_index = radar_vol.sweep_start_ray_index['data']
    nexrad_wave_form = ['none'] * target_elev.size

    #Test data output for verification and debug
    #for index1 in range(len(target_elev)):
    #    print('index: ', index1, ' elev: ', target_elev[index1], ' ny: ', nyq_vel[sweep_start_index[index1]] )


    #we use the index version of the for loops.
    for index1 in range(len(target_elev)):
        elev_count = 0
        matching_index = index1

        for index2 in range(len(target_elev)):
            if (target_elev[index1] == target_elev[index2]):
                elev_count += 1
                #the nexrad network has multiple cuts at the lowest
                #elvations this logic assumes that the two cuts
                #with different surveillance and doppler data
                #are adjacent
                #print(' index1: ', index1, ' index2: ', index2, ' abs: ', abs(index1-index2) )
                if (index1 != index2 and abs(index1-index2) == 1 ):
                    matching_index = index2

        #print (index1, ' matching:', matching_index)
        if elev_count == 1:
            nexrad_wave_form[index1] = 'combined'
        elif matching_index != index1:
            logger.debug(f"nyquist: {nyq_vel[sweep_start_index[index1]]} and"
                         f" {nyq_vel[sweep_start_index[matching_index]]}")
            if (nyq_vel[sweep_start_index[index1]] > nyq_vel[sweep_start_index[matching_index]]):
                nexrad_wave_form[index1] = 'doppler'
            else:
                nexrad_wave_form[index1] = 'surv'
        else:
            raise ValueError('Error in compute_nexrad_wave_form')

    return nexrad_wave_form


def prune_nexrad(prune_actions: list,
                 radar_vol: pyart.core.Radar) -> pyart.core.Radar:
    """
    Parameters
    ---------
    prune_actions: a list of actions
    exclusive: 'doppler', 'surv'
    'volume', 'drop_low_level'

    radar_vol: pyart radar object

    Returns
    -------
    pyart radar object

    removes sweeps from the input pyart object that do not match the input
    parameters. You can include more than one input parameter.

    prune_actions = ['doppler', 'volume']

    will give you a radar object that contains only doppler mode cuts and
    combined cuts where you have only one sweep per elevation in the volume

    'drop_low_level' will drop cuts below 0.5 degree
    'volume' will keep only the first cut in the volume at any elevation angle
    'doppler' will keep only doppler cuts and combined cuts
    'surv' will keep only surveillance cuts and combined cuts


    """
    #fixme: add checks for exclusive

    ###### Start the work #####
    target_elev = radar_vol.fixed_angle['data']
    logger.debug(f"orig elev: {target_elev}")

    sweep_number = radar_vol.sweep_number['data']
    logger.debug(f"orig sweep num: {sweep_number}")

    nexrad_wave_form = compute_nexrad_wave_form(radar_vol)
    logger.debug(f" nexrad_wave_form: {nexrad_wave_form}")

    # always handle the doppler and the surv first.
    for a in prune_actions:

        if a == 'doppler' or a == 'surv':
            logger.debug(f"action: {a}")

            #print('action: ', a )

            nexrad_wave = a
            index = 0
            for d in nexrad_wave_form:
                #print (index,  ' d:', d)
                if d != "combined":
                    #print('d: ', d)
                    if d != nexrad_wave:
                        sweep_number[index] = -1
                index += 1

            logger.debug(f"modified: {sweep_number}")

    for a in prune_actions:
        if a == 'volume':
            logger.debug(f"action: {a}")
            index = 0
            for e in target_elev:
                if sweep_number[index] != -1:
                    #check for this elevation later in the volume
                    #print( 'range:', range(index+1, len(target_elev)), 1)
                    for i in range(index+1, len(target_elev)):
                        #print('elev search:', i, ' target:', e, ' test: ', target_elev[i])
                        if e == target_elev[i]:
                            sweep_number[i] = -1
                index += 1
            logger.debug(f"modified 2: {sweep_number}")

        if a == 'drop_low_level':
            logger.debug(f"action: {a}")
            index = 0
            for e in target_elev:
                if e < 0.45:
                    sweep_number[index] = -1
                index += 1
            logger.debug(f"modified 2: {sweep_number}")

    #prune actions are complete, sweeps to drop are labeled -1
    final_sweeps = []
    for i in sweep_number:
        if i != -1:
            final_sweeps.append(i)

    output_radar_vol = radar_vol.extract_sweeps(final_sweeps)
    # fix inconsistency sweep number and angle
    output_radar_vol.sweep_number['data'] = np.arange(
        0, output_radar_vol.nsweeps, dtype=int)

    logger.debug(f"final_elev: {output_radar_vol.fixed_angle['data']}")

    return output_radar_vol
