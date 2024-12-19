#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyart

from make_logger import logger
from hotspots import detection
from cappi.helpers import xy_assign_context
from prepro.nexrad import preprocessor_norpg_xarray, compute_nexrad_wave_form, \
    prune_nexrad
from cappi.make_vv import get_indexed_vv
from cappi.azran import indexed_vv_to_cappi_amsl
from cappi.xy_cappi import azran_to_xy
import klaus_krause_cmap


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process Level2 data to get Zdr Hotspots")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-i", "--input_dir", type=str, help="location where the Level 2 files are located", required=True)
parser.add_argument("-o", "--output_dir", type=str, help="location where the output files are stored", required=True)
parser.add_argument("-r", "--radar_name", type=str, help="4 letter Radar name, example: KOAX used as a filename glob to select files")
parser.add_argument("-c", "--cappi_height", type=float, help="Height of the -10C CAPPI in Meters AMSL above mean sea level", required=True)
# Add optional flags
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
# Parse the command-line arguments
args = parser.parse_args()

# Define the directory containing the files
directory_path = args.input_dir
if not directory_path.endswith("/"):
    directory_path += "/"

#Define the location of the mcit output
output_path = args.output_dir
if not output_path.endswith("/"):
    output_path += "/"

if args.cappi_height < 10.0:
    print("CAPPI height is in meters not in kilometers: %f " % (args.cappi_height))
    exit(1)

#assign input data
radar_name = args.radar_name
cappi_height = args.cappi_height

# Define the pattern to match files (e.g., all .txt files)
if args.radar_name is not None:
  filepattern = radar_name + "*"
else:
  filepattern = "*"

files = glob.glob(os.path.join(directory_path, filepattern))

print("Size of files: %d %s %s\n" % (len(files), directory_path, radar_name))

#
#Setup the grid we use:
#
x_trg = np.arange(-300, 300, 1)
y_trg = np.arange(-300, 300, 1)

for filename in files:
    print(f"cappi: {cappi_height} file: {filename} \n")

    #skip the mdm files, they contain model data not radar data
    if "MDM" in filename:
       continue

    #you can hard code it here if you need too
    target_cappi_height = cappi_height
    #target_cappi_height = 5188  # m AMSL = -10°C isotherm height

    #filename = Path('exampledata/nexrad_level2/KOAX20140603_213649_V06.gz')
    radar_vol = pyart.io.read_nexrad_archive(filename)

    #nexrad_wave_form = compute_nexrad_wave_form(radar_vol)
    #logger.info(f"from file: {filename}")
    #logger.info(f"wave_forms: {nexrad_wave_form}")

    #remove unneeded data from the pyart object
    prune_actions = ['surv', 'volume']
    prune_vol = prune_nexrad(prune_actions, radar_vol)

    #The processing of the data by the python version 
    #of the NWS ORPG Preprocess is close but not the same
    #
    #This will have to do until I can wrap the c/c++ code
    #into a python library
    metadata = {}
    metadata['zdr_absolute_calibration'] = 0.0
    metadata['z_absolute_calibration'] = 0.0
    processed_vol = preprocessor_norpg_xarray(prune_vol, metadata)

    # get the circular depolarization ratio and add it to the radar object
    dr = pyart.retrieve.compute_cdr(processed_vol,
                                     rhohv_field='cross_correlation_ratio',
                                     zdr_field='differential_reflectivity')
    processed_vol.add_field('dr', dr)

    # same for the processed fields
    prepro_dr = pyart.retrieve.compute_cdr(processed_vol,
                                            rhohv_field='prepro_cc',
                                            zdr_field='prepro_zdr')
    #add circular depolarization ratio to our processed volume
    processed_vol.add_field('prepro_dr', prepro_dr)

    #indexing the volume forces the volume into a 1 degree by 250m gate format
    indexed_vv = get_indexed_vv(processed_vol)

    # make azran CAPPI interpolation
    azran_ds = indexed_vv_to_cappi_amsl(indexed_vv, target_cappi_height)

    # make xy interpolation
    xycappi = azran_to_xy(azran_ds, x_trg, y_trg, grid_spacing_meters=1000., max_dist=2)

    # same for processed radar fields
    # filter the zdr data to include only meteorological data
    # There are many ways to do this
    zdrcut_proc = detection.get_filtered_zdr(
        xycappi, refl_fieldname='prepro_zh', zdr_fieldname='prepro_zdr',
        cdr_fieldname='prepro_dr')

    #load the filtered zdr data into the xycappi
    xycappi["zdr_cut_proc"] = zdrcut_proc

    #compute zdr hotspots
    hotspot_field_proc, hotspot_features_proc = (
        detection.apply_hotspot_method(
        xycappi, x_dim="x", y_dim="y", refl_fieldname='prepro_zh',
        zdr_fieldname='zdr_cut_proc'))

    #create an xarray dataset for output 
    #Strip the Hotspot objects down to the minimum number of variables
    #for output the netcdf file.... 
    #You can of course add the other things you need/want
    HotSpot_ds = xr.merge([hotspot_field_proc, hotspot_features_proc])

    #Output the Dataset to a NetCDF file
    #
    #expect an input filename like:
    #KDDC20200525_044128_V06
    radar_file = os.path.basename(filename)
    time_str = radar_file[4:19]
    output_file = output_path + "ZdrHotSpot" + "_" + time_str + ".nc"
    print("output file: %s \n" % (output_file))

    HotSpot_ds.to_netcdf(output_file)
