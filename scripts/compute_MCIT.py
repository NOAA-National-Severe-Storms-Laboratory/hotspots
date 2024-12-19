#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse
import pyart
import xarray as xr
import numpy as np
from prepro.nexrad import prune_nexrad
#from cappi.azran import make_cappi
from cappi.make_vv import get_indexed_vv
from cappi.xy_cappi import azran_to_xy

from mcit.vil import get_vil_from_azran
from mcit.mcit_objects import mcit_objects, mcit_trim


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process Level2 data to get MCIT")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-i", "--input_dir", type=str, help="location where the Level 2 files are located")
parser.add_argument("-o", "--output_dir", type=str, help="location where the output files are stored")
parser.add_argument("-r", "--radar_name", type=str, help="4 letter Radar name, example: KOAX")
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

# Define the pattern to match files (e.g., all .txt files)
radar_name = args.radar_name 
filepattern = radar_name + "*"

files = sorted(glob.glob(os.path.join(directory_path, filepattern)))

print("Size of files: %d %s %s\n" % (len(files), directory_path, radar_name))

#
#Setup the grid we use:
#
x_trg = np.arange(-300, 300, 1)
y_trg = np.arange(-300, 300, 1)

for filename in files:
    print(f"file: {filename} \n")

    #skip the mdm files, they contain model data not radar data
    if "MDM" in filename:
       continue 

    #read in the Level 2 WSR-88D data as a pyart object
    radar_vol = pyart.io.read_nexrad_archive(filename)

    #remove extra sweeps of data. Keep only data
    #from the survelience cuts and one cut per volume 
    prune_actions = ['surv', 'volume']
    prune_vol = prune_nexrad(prune_actions, radar_vol)

    #create an indexed version of the data on 360 radials
    #to a range of 1200 (defaults) at 250m gates.
    #We also change formats here to an xarray datastore
    #working with xarrys is easier for us than pyart objects
    ds = get_indexed_vv(prune_vol, fields=['reflectivity'])

    #compute VIL from the indexed volume of reflectivity
    vil = get_vil_from_azran(ds['reflectivity'])
    # keep attributes such as radar location info
    vil.attrs.update(ds.attrs)
    #convert to VIL to xy grid
    vil_xy = azran_to_xy(vil, x_trg, y_trg, grid_spacing_meters=1000.0)

    #create MCIT objects
    mcit_obj = mcit_objects(vil_xy, 2.0, 1.25)

    #trim MCIT objects
    trimmed_mcit_obj = mcit_trim(mcit_obj, vil_xy, 5.0, 50)

    #Strip the MCIT objects down to the minimum number of variables
    MCIT_ds = xr.merge([trimmed_mcit_obj, vil_xy])

    #output this dataset to a netcdf file
    # Output the Dataset to a NetCDF file
    #expect a filename like:
    #KDDC20200525_044128_V06
    radar_file = os.path.basename(filename)
    time_str = radar_file[4:19]
    output_file = output_path + "MCITobj" + "_" + time_str + ".nc"
    print("output file: %s \n" % (output_file))
    
    MCIT_ds.to_netcdf(output_file)
