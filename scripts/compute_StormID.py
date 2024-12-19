#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse
import pyart
import xarray as xr
import numpy as np

from mcit.tracking import overlap_tracking_basic 

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process MCIT data to get StormID, provides tracking to MCIT")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-i", "--input_dir", type=str, help="location where the MCIT files are located")
parser.add_argument("-o", "--output_dir", type=str, help="location where the output files are stored")
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
filepattern = "MCITobj_*"

files = sorted(glob.glob(os.path.join(directory_path, filepattern)))

print("Size of files: %d %s \n" % (len(files), directory_path))

#
index = 0
for filename in files:

    print(f"file: {filename} \n")
    if index == 0:
        #cute trick to handle startup
        storm_id_filename = filename
    else:
        #really previous file time_str as it hasn't changed yet for this file
        storm_id_filename = output_path + "StormID" + "_" + time_str + ".nc"

    #data at time t (most_recent_time)
    mcit_data = xr.open_dataset(filename)

    #data at time t-1 
    storm_data = xr.open_dataset(storm_id_filename)

    if index == 0:
        #from the trick above.... now less cute
        #change the variable name
        storm_data = storm_data.rename( {"MCIT":"StormID"})
        max_value_id = storm_data.max() + 1

    VIL_copy = mcit_data["VIL"].copy(deep=True)

    #Assign tracking from storm_data to mcit_data
    storm_id  = overlap_tracking_basic( storm_data["VIL"], storm_data["StormID"],
                                      mcit_data["VIL"], mcit_data["MCIT"])

    #update max_value_id:

    #print("max %d: " % (storm_id.max()))

    # output new storm_data file at time t
    #output this dataset to a netcdf file
    # Output the Dataset to a NetCDF file
    #expect a filename like:
    #MCITobj_20200523_001143.nc
    radar_file = os.path.basename(filename)
    time_str = radar_file[8:23]
    output_file = output_path + "StormID" + "_" + time_str + ".nc"
    print("output file: %s \n" % (output_file))
   
    #xr.where(storm_id == np.nan, -1, storm_id)
    
    #FIXME: It works, but not sure why
    VIL_copy.encoding = {}
    
    #add the vil data into the Storm ID so that we can output them as a pair
    StormID_ds = xr.Dataset( {"StormID": storm_id, "VIL": VIL_copy} ) 

    StormID_ds.VIL.attrs["_FillValue"] = np.nan
    
    StormID_ds.to_netcdf(output_file)

    index += 1
