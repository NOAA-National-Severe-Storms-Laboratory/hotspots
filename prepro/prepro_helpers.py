# -*- coding: utf-8 -*-
"""
Created on Wed Dec 04 2020

@author: John Krause (JK)

Update History:
    Version 0:
        - JK's iinitial version of NSSL's operational C code, implemented in  
        Python and wrapped into function and separate driver. 

(c) Copyright 2020, Board of Regents of the University of Oklahoma.
All rights reserved. Not to be provided or used in any format without
the express written permission of the University of Oklahoma.
Software provided as is no warranties expressed or implied.

"""

# Import statements
import numpy as np
import xarray as xr
import pandas as pd

from make_logger import logger


def calc_atmos_nORPG(elev):
#
#Compute the atmospheric attenuation 
#
    logger.debug('        Entering calc_atmos')
    #
    #Strictly speaking the ORPG uses a static atmospheric attenuation correction
    #based on the elevation. This means that everytime there is a new elevation added
    #the code has to be altered to include it. To avoid this we use a dynamic fit to  
    #the values provided by V. Lakshmanan 
    atmos = -0.012747 + 0.001895 * elev
    atmos = np.maximum(atmos, -0.012)
    atmos = np.minimum(atmos, -0.005)
    return atmos


def compute_SNR_nORPG(ref, elev, metadata):
#
#Compute the Signal-Noise Ration the ORPG way
#
    logger.debug('        Entering compute_SNR_nORPG')

    range_km = (np.arange(metadata['ngates']) *
                metadata['gatespacing_meters']/1000.0 +
                metadata['distance_to_first_gate_meters']/1000.0)

    c = metadata['C']
    atmos = calc_atmos_nORPG(elev)

    snr = ref + -20.0*np.log10(range_km) + c + atmos*range_km
  
    return snr


def compute_SNR_nORPG_xarray(ref, elev, metadata):
    #
    # Compute the Signal-Noise Ration the ORPG way but in xarray
    #
    logger.debug('        Entering compute_SNR_nORPG')

    c = metadata['C']
    atmos = calc_atmos_nORPG(elev)
    snr = ref + -20.0 * np.log10(ref.range/1000) + c + atmos * ref.range/1000

    return snr


#apply frequencey correction for radar location
def applyFrequencyCorrection_nORPG(metadata, uw_phi, debug):
    #
    # This routine corrects for the difference in phase due to the radars
    # transmit frequency. All of the original dualpol work was done on a radar
    # (KOUN) that transmitted at 2.705 GHz. We need to adjust phase if your radar
    # transmits on a different frequency.  recommended by Dusan Zurnic, NSSL 
    #
    # Note: alters the correction for Horizontal Attenuation later

    KOUN_Frequency_MHz = 2705.
    radar_Frequency_MHz = metadata['radar_frequency_MHz']

    if debug >= 3:
       print('    Entering applyFrequencyCorrection_nORPG')
    
    return KOUN_Frequency_MHz/radar_Frequency_MHz * uw_phi


def calc_std_texture_nORPG(field, window_size, debug):
    """
    Calculate texture (standard deviation of residuals) for a specific window

    Calculate the residuals for the standard deviation calculation.
    The residuals tell us how "choppy" Z is and the standard deviation of
    them tells us how extreme that "choppyness" is. If the standard deviation 
    is high then the Z data is bouncing around alot. If low then just a little.
    The point of all this is that STD_Z of AP >> STD_Z of Birds/Insects 
    >> STD_Z of meteorological targets 

 
    Parameters
    ----------
    field : float array (1D) (straight from L2 data)
        Radar field to calculate standard deviation of.
    window_size : int
        Window length for calculating standard deviation.

    Returns
    -------
    field_texture : float array (1D)
        Standard deviation of the residuals.

    """
    if debug >= 2:
       print('    Entering calc_std_texture')
   
    half_width = int(np.floor(0.5 * window_size))

    # Step 1: average the data
    field = pd.DataFrame(field)
    field_ave = field.rolling(
        window=window_size, center=True, min_periods=half_width).mean()

    # Step 2: Calculate difference (residuals) field
    field_diff = field - field_ave
    
    # Step 3: Calculate standard deviation of the difference (residuals) field
    field_texture = field_diff.rolling(
        window=window_size, center=True,
        min_periods=half_width).std().values.flatten()
      
    return field_texture


def calc_std_texture_nORPG_xarray(field: xr.DataArray,
                                  window_size: int) -> xr.DataArray:
    """

    Calculate texture (standard deviation of residuals) for a specific window

    Calculate the residuals for the standard deviation calculation.
    The residuals tell us how "choppy" Z is and the standard deviation of
    them tells us how extreme that "choppyness" is. If the standard deviation
    is high then the Z data is bouncing around alot. If low then just a little.
    The point of all this is that STD_Z of AP >> STD_Z of Birds/Insects
    >> STD_Z of meteorological targets


    Parameters
    ----------
    field : xarray.DataArray
    window_size : int
        Window length for calculating standard deviation.

    Returns
    -------
    field_texture : float array (1D)
        Standard deviation of the residuals.

    """
    half_width = int(np.floor(0.5 * window_size))

    # Step 1: average the data
    field_ave = field.rolling(range=window_size, center=True,
                              min_periods=half_width).mean()

    # Step 2: Calculate difference (residuals) field
    field_diff = field - field_ave

    # Step 3: Calculate standard deviation of the difference (residuals) field
    field_texture = field_diff.rolling(range=window_size, center=True,
                                       min_periods=half_width).std(ddof=1)

    field_texture = field_texture.rename(field_texture.name + "_texture")

    return field_texture


def local_running_ave_1D(field, window_size, debug):
    if debug >= 2:
       print('    Entering local_running_ave:')
    
    half_width = int(np.floor(0.5 * window_size))
    # Step 1: average the data 
    df = pd.DataFrame(field)
    field_ave = df.rolling(window=window_size, center=True,
                           min_periods=half_width).mean().values
    field_ave.flatten()

    return field_ave[:,0]


def compute_llsq(A, y):
    # This is slow until someone figures out how to use
    # scipy.linalg import lstsq
    slope, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope

