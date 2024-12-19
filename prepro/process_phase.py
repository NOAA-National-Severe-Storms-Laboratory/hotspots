#!/usr/bin/env python
# coding: utf-8

"""
Created on May 13, 2024 

@author: John Krause (JK)

@license: BSD-3-Clause

@copyright Copyright 2024 John Krause

Update History:
    PreVersion: Evolved from compute_kdp_v4,py 
       by Jacob Carlin and John Krause, NSSL, circa, 2020
    Version 0:
        -JK Initial version delivered as part of a prepro suite
            Python and wrapped into function and separate driver.

"""

import numpy as np
import pandas as pd
import copy

import wradlib as wrl
import xarray as xr

from prepro.prepro_helpers import calc_std_texture_nORPG_xarray, \
    calc_std_texture_nORPG, compute_llsq


def process_differential_phase(raw_phase, cc_data, min_system_phase=None,
                               phase_window=9,
                               phase_std_thresh=10,
                               cc_thresh=0.7,
                               unwrap_phase_flag=False):
    # This is a more basic computation to compute smoothed differential phase
    # I keep it as an example that the user might want to extract and modify
   
    sm_phase = np.zeros_like(raw_phase)

    sh = raw_phase.shape
    if len(sh) == 1:
        #single ray not an elevation
        num_az = 0
        num_gates = sh[0]
    else:
        num_az = sh[0]
        num_gates = sh[1]
    #
    #Correct_system_phase removes the min_system_phase and returns phase
    #after it has been lowered to a minimum value of zero (if our min_system_phase
    # computation is correct) If you know min_system_phase, and it is accurate, not
    #always the case with the ORPG, then you should provide it to the algorithm and 
    #it will use that instead of it's own, estimate
    if ( num_az == 0 and min_system_phase==None):
        print ('raw_phase_shape: ',raw_phase.shape)
        print('exiting...single azimuth phase prcessing requires min_system_phase.')
        exit(1)

    corr_phi, min_system_phase = correct_system_phase(
        raw_phase, cc_data, min_system_phase=min_system_phase)


    for a in range(num_az):
        #after correct_system_phase, 
        #if the maximum value of phase is high compared to the wrapping point, rare, usually 360 degree or 180 degree
        #then you will need to unwrap the phase here.
        #
        #This will slow down the processing, so apply when you think it's needed
        if unwrap_phase_flag == True:
            uw_phi = unwrap_phase(corr_phi[a], cc[a])
        else:
            uw_phi = corr_phi[a]

        
        cc = np.array(cc_data[a])
        
        #We threshold by cc and std of phi, but there are many ways to threshold the phi data
        #that sort the data into good and bad regions. 
        #The ORPG uses the MetSignal Alg to do this.
        #
        window_size = phase_window
        half_width = int(np.floor(window_size/2))
        
        #compute the std of phase for the window, using pandas and dataframes, 
        #Our thanks to Jacob Carlin, NSSL, 2020
        uw_phase = copy.deepcopy(uw_phi)  # FIXME: do we need this deepcopy if we define a pd.DataFrame? also, isn't a pd.Series sufficient?
        df = pd.DataFrame(uw_phase)
        phase_std = df.rolling(
            window=window_size, center=True,
            min_periods=(half_width+1)).std().values.flatten()


        #compute the median of phase for the windwo, using pandas and dataframes, 
        #Our thanks to Jacob Carlin, NSSL, 2020
        uw_phase = copy.deepcopy(uw_phi)  # FIXME: same as above
        df = pd.DataFrame(uw_phase)
        phase_median = df.rolling(
            window=window_size, center=True,
            min_periods=(half_width+1)).median().values.flatten()
        #print(phase_median.shape)
        phase = copy.deepcopy(phase_median)
        #threshold off the bad data, keep the good data
        #
        #We use a combination of phase std and CC
        #
        phase[phase_std > phase_std_thresh] = np.nan
        #
        phase[cc < cc_thresh] = np.nan

        valid_idx = np.where(~np.isnan(phase))[0]
        valid_intervals = consecutive(valid_idx)
        
        #
        #This is where things get messy. Each tuple in the valid_intervals
        #can be trimmed to be conservative or not. How much to trim is also 
        #a bit of a guess. The edges of the good data intervals can be 
        #inaccurate causing negative values of phase, which are objectionable.
        #Another way to smooth out 
        #the phase is to require that the new interval start near where the old
        #interval ended. Which is conceptually pleasing, but hard because there
        #can be a strong positive bias at the end of the intervals. 
        final_int = []
        for interval in valid_intervals:
            if len(interval) >= phase_window:
                #possibly trim the interval here to remove
                # a few gates on the front and back to be conservative
                # usually half the min_interval_length (4) or 
                # 0.25 of min_interval_length (2)
                
                trim_num = half_width
                trim_int = interval[half_width:-half_width]
                #we'll keep this one
                final_int.append(trim_int)        
                
        #push the last value on to get interplation to the end of the radial   
        #final_int.append([num_gates-1, num_gates-1])
        
        last_int_idx = 0
        phase[0] = 0
        for f in final_int:
            #linearly interpret the phase along the radial from the last_phase_value
            #to the first phase value of the new interval.
            linear_interp_phase(phase, last_int_idx, f[0])
            last_int_idx = f[-1]
           
        if len(final_int) > 0:
            last_int = final_int[-1]
            last_valid = last_int[-1]
            last_value = phase[last_valid]
            phase[last_valid:-1] = last_value
        else:
            #no data
            phase[:] = 0.0
        
        sm_phase[a] = phase
        
    return sm_phase


# Source: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
#I'm not nearly this cleaver
def consecutive(data, stepsize=1):
    """
    Generate list of lists of consecutive indices.

    Parameters
    ----------
    data : int vector
        Input list of indices where some condition is true.
    stepsize : int, optional
        Spacing to validate against. The default (consecutive) is 1.

    Returns
    -------
    list
        List of indices grouped by consecutive indices.

    """
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def linear_interp_phase(data, first_index, last_index):
    """
    Return data after linear interpolation

    Parameters
    -------
        data array: float array
            data you want to interpolate
        first_index: int value
            index closest to zero, where interpolation begins
        second_index: int value
            index farthest from zero, where interpolation ends

    Returns
    ------
      data_interp: array modified to linear interpolate between input indexs
    """
    #send data and two indexs, returns data with linear interploation
    start_data = data[first_index]
    end_data = data[last_index]
    #This is for phase processing
    if end_data == np.nan:
        end_data = start_data
        
    #dist = last_index - first_index

    #works on the data array itself
    for i in range(first_index, last_index, 1): #slow
        data[i] = start_data + (end_data - start_data)*(i-first_index)/(last_index-first_index)

    return data


def compute_min_system_phase_xarray(raw_phase, cc, min_cc=0.7):

    phase_thresh = raw_phase.where(cc > min_cc, np.nan)

    corr_flag = phase_thresh.notnull()
    # get the count of consecutive valid values
    cons_idx = (corr_flag.cumsum(dim="range") -
                corr_flag.cumsum(dim="range").where(~corr_flag).ffill(
                    dim="range").fillna(0).astype(int))

    # get the index of the first time we have 10 valid consecutive values
    # inspiration from here: https://stackoverflow.com/questions/45964740/python-pandas-cumsum-with-reset-everytime-there-is-a-0
    first_cons_range = cons_idx.where(cons_idx == 10).idxmin(dim="range")

    # fill with dummy value because "sel" does not take NaN
    first_cons_range_filled = first_cons_range.fillna(phase_thresh.range.max())
    min_phase_ray = phase_thresh.sel(range=first_cons_range_filled)

    # get the NaNs back from before so they are not used in the calculation
    min_phase_ray = min_phase_ray.where(first_cons_range.notnull(), np.nan)

    # now we can finally remove them again because we extracted the right values
    min_phase_ray = min_phase_ray[~np.isnan(min_phase_ray)]

    if min_phase_ray.notnull().sum() < 30:
        min_phase = min_phase_ray.median()

    else:
        # We want a phase value that is in the lower part of the sorted phases
        # Not too close to zero to avoid badly computed first echos, but
        # not too close to the middle where we would have a lot of errors in
        # phase near the radar or at first echo.
        idx = int(np.floor(len(min_phase_ray) * 0.2))
        sorted_min_phase = np.sort(min_phase_ray)
        min_phase = sorted_min_phase[idx]

    return min_phase


def compute_min_system_phase(phase, cc, missing_value=-9999.0):
    #
    #If our computation of min_system_phase is too high then 
    #values of pahse near the 360 deg fold will appear close
    #to the radar. (gates with very high phase)
    #
    #If our comptuation of min_system_phase is too low then the 
    #Zdr and Z will be high after the correction for horizontal attenuation
    #
    num_az = phase.shape[0]

    #we only want to work on meteorological data.... This is our standard
    #You can have higher or lower standards, or just more complicated ones
    min_cc_threshold = 0.9 
    #data below this value are not considered in the computation
    
    min_sys_phase = []

    for a in range(num_az):

        phi_radial_raw = copy.deepcopy(phase[a,:])
        phi_thresh = np.zeros_like(phi_radial_raw) * np.nan 
        #fill missing with nan
        phi_radial_raw = np.where(
            phi_radial_raw == missing_value, np.nan, phi_radial_raw)
        #threshold based on cc
        cc_radial_raw = cc[a, :]
        
        phi_thresh = np.where(cc_radial_raw > min_cc_threshold,
                              phi_radial_raw,
                              np.nan)

        valid_idx = np.where(~np.isnan(phi_thresh))[0]
        if len(valid_idx) > 1:
            #print('a: ', a, ' cc: ', cc_radial_raw)
            #print('a: ', a, ' phi_thresh: ', phi_thresh)
            #print('a: ', a, ' valid_idx: ', valid_idx)
            valid_intervals = consecutive(valid_idx)

            #we want a first echo of at least 10 gates to make sure
            #we really have a met echo and not ground clutter
            for i in valid_intervals:
                if len(i) > 10:
                    min_sys_phase.append(phi_thresh[i[10]])
                    #print('a: ', a , ' min_sys: ', phi_thresh[i[10]], ' valid: ', i  )
                    break
       
    median_phase = np.ma.median(min_sys_phase)
    sorted_phase = np.ma.sort(min_sys_phase)
    #print('len: ', len(sorted_phase), 'sort: ', sorted_phase)

    #We want a phase value that is in the lower part of the sorted phases
    #Not too close to zero to avoid badly computed first echos, but
    #not too close to the middle where we would have a lot of errors in 
    #phase near the radar or at first echo.
    idx = int(np.floor(len(sorted_phase) * 0.2))
    #compute index into sorted phase
    #print('exit: ', median_phase, ' min_system_phase_idx: ', min_sys_phase[idx]) 

    #we also need enough estimates of phase to make a decent guess ourselves
    if len(sorted_phase) < 30:
        return median_phase
    else:
        return sorted_phase[idx]


def correct_system_phase(phase, cc, min_system_phase=None,
                         missing_value=-9999.0):
    """
       Correct the system phase to have a minimum value of zero:

       Historically, the original versions of phase processing unwrapped 
       folded phase first. It is easier to correct the system phase to 
       have a minimum value of zero and then only unwrap when neccessary.
       The WSR-88D network wraps (folds) phase at 360 and I'm not sure
       if data exists where unwrapping is still required. Older data, or
       data from other radars may still require unwrapping. Some radars wrap
       at 180 or even 90 degrees. These will require both correct_system_phase
       and unwrap_phase.

       Parameters
       -------
       phase: 1d or 2D array 
         differential phase normal range (0.0, 360)

       cc:    matching phase by row, 1D or 2D array 
         co-polar correlation coefficient normal range (0.0, 1.2)

        optional parameters:
        --------
        min_system_phase: float, degrees
            minimum system phase value, must be accurate, if you are unsure
            do not send. Will speed up the computation by including this
        missing_value: float degrees
            The value in the phase array to corresponds to missing or no data

    """
    #
    #adjust the phase to remove the min_system_phase and attempt to 
    #return phase with a bias of zero
    #

    #compute_min_system_phase, send in the system value if known and accurate
    if min_system_phase is None:
        min_system_phase = compute_min_system_phase(
            phase, cc, missing_value=missing_value)

    #print('min_system_phase: ', min_system_phase)

    #apply min_system_phase
    corr_phase = copy.deepcopy(phase) 
    #corr_phase = np.where(corr_phase == missing_value, np.nan, corr_phase)
    #....pyart hiccup.....
    #looks like missing phase is really phase with a value below zero not the missing value 
    corr_phase = np.where(phase > 0, corr_phase-min_system_phase, np.nan)
    #reapply nan
    corr_phase = np.where(((corr_phase < 0) & (corr_phase != np.nan)),
                          corr_phase + 360,
                          corr_phase)

    return corr_phase, min_system_phase


def unwrap_phase(phase, cc, fold_in_deg=360, window_size=30, min_cc=0.85,
                 min_count=15, first_bin_allowed_to_fold=100,
                 missing_value=-9999.0):
    """
       Unwrap the system phase to extend the data beyonw the fold:
       
       This python routine attempts to exend the phase data beyond the 
       collection interval when the data has wrapped around the circle.
       I have tested it for 360 degree wraps, but no other

       Parameters
       -------
       phase: 1d or 2D array 
         differential phase normal range (0.0, 360)

       cc:    matching phase by row, 1D or 2D array 
         co-polar correlation coefficient normal range (0.0, 1.2)

        optional parameters:
        fold_in_deg: float degrees
            The location on the circle where the phase returns back to zero
        window_size: int 
            The size of the area to consider if a location has wrapped or not.
        min_cc: float
            The minimum value of co-polar correlcation ration, used to idenfy
            areas of phase that are "good" and may be unwrapped.
        first_bin_allowed_to_fold: int
            The first location in phase that is allowed to be unwrapped. This
            prevents misidentifed ground clutter from breaking the process
        mising_value: float 
            The value used to identify, no-data or missing-data in the phase

    """
    #
    #This algorithm is for the rare case when there is so much phase in the beam
    #that it wraps (folds) around. You should only need this if 
    #    1) you forgot to run correct_system_phase (do that instead)
    #    2) you have a very unusal situation.
    #
    # Most radars fold at 360. I've only teseted it for 360. If you have data that 
    # folds at 180 degrees, please send it to john.krause@noaa.gov. So that I can
    # test/make it work for such data.
    
    #phi_radial_raw = copy.deepcopy(np.array(phase)) 
    phi_radial_raw = np.array(phase) 
       
    #pyart radial data needs this:
    phi_radial_raw = np.where(phi_radial_raw == missing_value,
                              np.nan, phi_radial_raw)

    #threshold based on cc
    cc_radial_raw = np.array(cc)
                                         
    phi_radial_thresh = np.where(cc_radial_raw > min_cc, phi_radial_raw, np.nan)
        
    #quick exit?
    if np.nanmax(phi_radial_thresh) < (fold_in_deg*0.8):
        uw_phi = phi_radial_raw 
        return uw_phi 
        
    #fill missing with last valid value
    last_value = phi_radial_raw[0]
    for i in range(len(phi_radial_thresh)):
        #print('i:', i, ' value: ',phi_radial_thresh[i], ' last:', last_value  )
        if np.isnan(phi_radial_thresh[i]):
            phi_radial_thresh[i] = last_value
        else:
            last_value = phi_radial_thresh[i]
            #print('i: ', i, ' final:', phi_radial_thresh[i])
            
    #This uses the dataFrame and forward fill ? FIXME
    #df = pd.DataFrame(phi_radial_thresh)
    #df_out = df.ffill(phi_radial_thresh)
    #phi_radial_thresh = fd_out.flatten()
       
    #
    phi_radial_single_fold = phi_radial_thresh + fold_in_deg
        
    #if a == ia:
         #print('phi_raw', phi_radial_raw[si:ei])
         #print('phi_thresh', phi_radial_thresh[si:ei])
         #print('cc_raw', cc_radial_raw[si:ei])
        
    #print('phi_thresh', phi_radial_thresh[100:150])
    #print('phi_fold', phi_radial_single_fold[100:150])
        
    #non-centered last 30 gates, mean value
    df = pd.DataFrame(phi_radial_thresh)
    field_mean = df.rolling(
        window=window_size, center=False, min_periods=3).mean().values
    phi_radial_mean = field_mean.flatten()

    #if a == ia:
        #print('radial_mean', phi_radial_mean[si:ei])
          
    #print('phi_radial_mean', phi_radial_mean[100:150])
    phi_diff_no_fold = np.abs(phi_radial_thresh-phi_radial_mean)
    phi_diff_single_fold = np.abs(phi_radial_single_fold-phi_radial_mean)
        
    #if a == ia:
         #print('no_fold', phi_diff_no_fold[si:ei])
         #print('no_fold', phi_diff_no_fold[si:ei])

    # FIXME: this deepcopy is unnecessary as it will be overwritten later
    # phi_final = copy.deepcopy(phi_radial_raw)
        
    fold_locations = np.where(phi_diff_single_fold < phi_diff_no_fold, 1, 0)
    fold_locations[0:first_bin_allowed_to_fold] = 0
    ind = np.where(fold_locations)[0]
        
    if len(ind) > min_count:
        #print(a, 'fold_ind: ', ind)
        start_fold_index = ind[0]
        #print('start_fold_index', start_fold_index)
        fold_locations[0:start_fold_index] = 0
        fold_locations[start_fold_index:len(fold_locations)] = 1
          
    phi_final = np.where((fold_locations > 0) &
                         (phi_radial_raw < fold_in_deg*.7),
                         phi_radial_raw + fold_in_deg, phi_radial_raw) 
        
    #Trim back to the original data
    phi_final = np.where(phi_radial_raw != np.nan, phi_final, np.nan)
        
    return phi_final


def compute_smoothed_phase_pzhang_xarray(
        raw_phase: xr.DataArray,
        cc: xr.DataArray,
        min_system_phase: int = None,
        window_size: int = 9,
        phase_std_thresh: int = 10,
        cc_thresh: float = 0.7,
        unwrap_phase_flag: bool = False) -> xr.DataArray:
    """

    Parameters
    ----------
    raw_phase: xr.DataArray
        unprocessed radar phase (2D)
    cc: xr.DataArray
        co-polar correlation coefficient (2D)
    min_system_phase: int, None
        
    window_size: int, 9
        smoothing window size along the radial
    phase_std_thresh: int, 10
        texture threshold for differential phase
    cc_thresh: float, 0.7
        co-polar correlation coefficient for differential phase
    unwrap_phase_flag: bool, False
        unwrap differential phase; CURRENTLY NOT IMPLEMENTED

    Returns
    -------
    sm_phase: xr.DataArray
        processed and smoothed differential phase
    """
    
    half_width = int(np.floor(window_size / 2))

    if min_system_phase is None:
        min_phase = compute_min_system_phase_xarray(
            raw_phase, cc, min_cc=cc_thresh)

    # TODO: phase unwrap option
    if unwrap_phase_flag:
        print("unwrapping not implemented yet")

    corr_phase = xr.where(raw_phase > 0, raw_phase-min_phase, np.nan)
    # change corr_phase when below 0
    corr_phase = corr_phase.where(corr_phase >= 0, corr_phase + 360)

    phase_texture = calc_std_texture_nORPG_xarray(corr_phase, 5)

    # FIXME: maybe remove the line below if not necessary - this is the
    #  standard deviation of the texture??
    # phase_texture = phase_texture.rolling(
    #     range=window_size, center=True, min_periods=half_width+1).std()
    phase_median = corr_phase.rolling(
        range=window_size, center=True, min_periods=half_width+1).median()

    # keep values below the texture threshold
    phase_median = phase_median.where(phase_texture <= phase_std_thresh, np.nan)
    # keep values above the cc threshold
    phase_median = phase_median.where(cc >= cc_thresh, np.nan)

    phase_double_median = phase_median.rolling(
        range=window_size, center=True, min_periods=half_width+1).median()
    phase_triple_median = phase_double_median.rolling(
        range=window_size, center=True, min_periods=half_width+1).median()

    # apply ufunc is essentially a for loop applying a function that handles
    # numpy arrays. for this we need to specify the dimensions of the
    # input variables along which the for loop is applied.
    sm_phase = xr.apply_ufunc(
        phase_trim,
        phase_triple_median,
        phase_median,
        window_size,
        input_core_dims=[["range"], ["range"], []],
        output_core_dims=[["range"]],
        vectorize=True)

    return sm_phase


def compute_smoothed_phase_PZhangNSSL(raw_phase: np.ndarray,
                                      cc_data: np.ndarray,
                                      min_system_phase=None,
                                      window_size=9, phase_std_thresh=10,
                                      cc_thresh=0.7, unwrap_phase_flag=False):

    """
    Compute the smoothed PhiDP data to be used in Kdp computations.

    There are many ways to compute Kdp, but all require identifying good Phi data
    and then smoothing that data. This one was developed by PengFei Zhang at
    the National Severe Storms Lab (NSSL), circa 2013-2017 (?) with help from
    Alexander Ryzhkov, NSSL. It is the standard method in the DRARSR group at 
    NSSL and untilizes a triple median. 

    Moved into Python (2020) by Jacob Carlin, NSSL with help from John Krause,
    NSSL.

    Parameters
    ----------
    raw_phase: 2D(az, gate) of phase values.
        Intrinsic range  (0,360) works with WSR-88D data
    cc_data: 2D(az, gate) of co-polar correlation coefficient.
        Intrinsic range (0.0, 1.0) but works with WSR-88D range of (0.0, 1.2)

    Returns
    -------
    np.array(smoothed_phi): 1D (gates)  or 2D(az, gate) of smoothed phase values
    """

    sm_phase = np.zeros_like(raw_phase)
    half_width = int(np.floor(window_size/2))

    sh = raw_phase.shape
    if len(sh) == 1:
        #single ray not an elevation
        num_az = 0
        # FIXME: num_gates variable is never used
        num_gates = sh[0]
    else:
        num_az = sh[0]
        num_gates = sh[1]
    #
    #Correct_system_phase removes the min_system_phase and returns phase
    #after it has been lowered to a minimum value of zero (if our min_system_phase
    # computation is correct) If you know min_system_phase, and it is accurate, not
    #always the case with the ORPG, then you should provide it to the algorithm and 
    #it will use that instead of it's own, estimate
    if (num_az == 0 and min_system_phase == None):
        print('raw_phase_shape: ', raw_phase.shape)
        print('exiting...single azimuth phase prcessing requires min_system_phase.')
        exit(1)

    corr_phi, min_system_phase = correct_system_phase(
        raw_phase, cc_data, min_system_phase=min_system_phase)

    for a in range(num_az):
        #after correct_system_phase, 
        #if the maximum value of phase is high compared to the wrapping
        # point, rare, usually 360 degree or 180 degree
        #if you need to unwrap phase at 180 deg, Please send data to 
        #John.Krause@noaa.gov. I do not have any data to test at 180 wrapping
        #
        #This will slow down the processing, so apply when you think it's needed
        if unwrap_phase_flag == True:
            uw_phi = unwrap_phase(corr_phi[a], cc_data[a])
        else:
            uw_phi = corr_phi[a]

        cc = np.array(cc_data[a])
        
        #We threshold by cc and std of resdiduals of phase, 
        #but there are many ways to threshold the phi data
        #that sort the data into good and bad regions. 
        #The ORPG uses the MetSignal Alg to do this.
        #
        
        #compute the std of phase for the window, using pandas and dataframes, 
        #Our thanks to Jacob Carlin, NSSL, 2020
        uw_phase = copy.deepcopy(uw_phi)  # FIXME: do we need this deepcopy if we define a pd.DataFrame?

        #In the NSSL version the phase texture, which is the std of 
        #the residuals, is used rather than the raw std of phase
        #this is at a static window size of 5
        uw_texture = calc_std_texture_nORPG(uw_phase, 5, debug=False)

        df = pd.DataFrame(uw_texture)
        phase_std = df.rolling(
            window=window_size, center=True,
            min_periods=(half_width+1)).std().values.flatten()

        #compute the median of phase for the windwow, using pandas and dataframes, 
        #Our thanks to Jacob Carlin, NSSL, 2020
        uw_phase = copy.deepcopy(uw_phi)  # FIXME: same as above
        df = pd.DataFrame(uw_phase)
        phase_median = df.rolling(
            window=window_size, center=True,
            min_periods=(half_width+1)).median()

        #
        #threshold off the bad data, keep the good data
        #We use a combination of texture phase std and CC
        #
        phase_median[phase_std > phase_std_thresh] = np.nan
        phase_median[cc < cc_thresh] = np.nan

        #
        #here is where the PZhangeNSSL method differs
        #Threshold the phase_median data

        #apply a 2nd median filter to the Phase Data
        #Yes this is median of a field where the median has already
        #been applied
        phase_double_median = phase_median.rolling(
            window=window_size, center=True,
            min_periods=(half_width+1)).median()

        # apply a third median filter to the Phase Data
        # Yes this is median of a field where the median has already
        # been applied. This method reduces negative KDP
        phase_triple_median = phase_double_median.rolling(
            window=window_size, center=True,
            min_periods=(half_width+1)).median().values.flatten()
        #
        # This is where things get messy. Each tuple in the valid_intervals
        # can be trimmed to be conservative or not. How much to trim is also
        # a bit of a guess. The edges of the good data intervals can be
        # inaccurate causing negative values of phase, which are objectionable.
        # Another way to smooth out
        # the phase is to require that the new interval start near where the old
        # interval ended. Which is conceptually pleasing, but mostly irrelevant
        # only removing negative slopes (neg KDP) between intervals where data
        # was bad anyway.
        #
        sm_phase[a] = phase_trim(phase_triple_median, phase_median,
                                 window_size)
        
    return sm_phase


def phase_trim(phi_triple_median, phi_median, window_size):

    # find the intervals where the data still exist != nan
    valid_idx = np.where(~np.isnan(phi_median))[0]
    valid_intervals = consecutive(valid_idx)

    half_width = int(np.floor(window_size / 2))

    phi_smoothed = copy.deepcopy(phi_triple_median)

    final_int = []
    for interval in valid_intervals:
        if len(interval) >= window_size:
            # possibly trim the interval here to remove
            # a few gates on the front and back to be conservative
            # usually half the min_interval_length (4) or
            # 0.25 of min_interval_length (2)
            # trim_num = half_width

            trim_int = interval[half_width:-half_width]
            # we'll keep this one
            final_int.append(trim_int)

    last_int_idx = 0
    phi_smoothed[0] = 0
    for f in final_int:
        # linearly interpolate the phase along the radial from the
        # last_phase_value to the first phase value of the new interval.
        phi_smoothed = linear_interp_phase(
            phi_smoothed, last_int_idx, f[0])
        last_int_idx = f[-1]

    if len(final_int) > 0:
        last_int = final_int[-1]
        last_valid = last_int[-1]
        last_value = phi_smoothed[last_valid]
        phi_smoothed[last_valid:-1] = last_value
    else:
        # no data
        phi_smoothed[:] = 0.0

    return phi_smoothed


def calc_kdp_wradlib(sm_phase, window_size, gate_spacing_km):
    #This is fast and it works. Nicely done wradlib! Kudos.
    kdp = wrl.util.derivate(
        sm_phase, winlen=window_size, skipna=False) * 0.5 / gate_spacing_km

    return kdp


def calc_kdp_1D(metadata, sm_phase, window_size, debug):
    # This is slow until someone figures out how to use
    # scipy.linalg import lstsq
    #read: https://stackoverflow.com/questions/55367024/fastest-way-of-solving-linear-least-squares
    if debug >= 2:
       print('    Entering calc_KDP_1D:')
    #
    kdp = np.zeros_like(sm_phase)
    #note to user, all the real decisions are made in the smoothed phase
    #computation. KDP is really just this, but most people don't realize
    #that different KDP outputs are based on different smoothed phase
    #treatments.
    hws = int(np.floor((window_size/2)))
    #the range array can be static as it always has the same distance
    gate_range = np.arange(len(sm_phase)) * metadata['gatespacing_meters']/1000.0

    #KDP output is in deg/KM not deg/m
    x = np.arange(window_size) * metadata['gatespacing_meters']/1000.0
    A = np.vstack([x, np.ones(len(x))]).T

    #if metadata['parallel']:
    #    import multiprocessing as mp
    #    pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)

    #FIXME
    #only handle full intervals, because python.....
    for i in np.arange( hws, len(sm_phase)-hws-1, 1):
        if debug >= 4:
            print('i: ', i , ' ', sm_phase[i-hws:i+hws])
        #polyfit is too slow!!!!!
        #print('i: ', i , ' ', sm_phase[i-hws:i+hws])
        #regress retuens the powers we want the slope in position '0')
        y = sm_phase[i-hws:i+hws+1]

        #everyting is too slow to be used.
        if np.max(y) > 0:
            #regress = np.polyfit(x, y,1)
            slope = compute_llsq(A,y)

            #FIXME sp_lstsq is the fastest llsq method,
            #but I can't seem to make it work......

            #res returns the the slope in position '1')
            #can't handle missing data as nan's ???
            #B = A[~np.isnan(y)]
            #y = y[~np.isnan(y)]
            #print('i: ', i, ' y: ', y, ' B: ', B)
            #res = sp_lstsq(B, y, lapack_driver='gelsy')[0]

            kdp[i] = 0.5 * slope  / metadata['gatespacing_meters']/1000.0


        #if kdp[i] > 0:
        #    print(i, ' np_lstsq: ', slope, )
        #if kdp[i] > 0:
        #    print(i, ' polyfit: ', regress[0], ' np_lstsq: ', slope, ' sp_lstsq: ', res[1] )
    #if metadata['parallel':
    #    pool.close()

    return kdp
