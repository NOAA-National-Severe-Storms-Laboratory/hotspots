#!/usr/bin/env python
# coding: utf-8
import os
import glob
import argparse

from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import klaus_krause_cmap
import pyart
import matplotlib.pyplot as plt
import imageio
from hotspots.assignment import hotspot_assignment, filter_assigned_hotspots
from stormcell.stormcell_helpers import create_stormcell_list
from matplotlib.colors import ListedColormap

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Assign hotspot data to cells and trim non-assinged hotspot results")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-c", "--input_cells_directory", type=str, help="a directory of storm cell files")
parser.add_argument("-d", "--input_hotspots_directory", type=str, help="a directory of hotspot files")
parser.add_argument("-o", "--output_directory", type=str, help="where to put the output files")
parser.add_argument("-g", "--movie",  action='store_true', help="make a movie out of the files")

# Parse the command-line arguments
args = parser.parse_args()

# Make sure output directory exists
os.makedirs(args.output_directory, exist_ok=True)

output_path = args.output_directory
if not output_path.endswith("/"):
    output_path += "/"

if args.input_cells_directory:
    cell_filenames = sorted([f for f in os.listdir(args.input_cells_directory) if f.endswith('.nc')])

if args.input_hotspots_directory:
    hs_filenames = sorted([f for f in os.listdir(args.input_hotspots_directory) if f.endswith('.nc')])

for i, filename in enumerate(hs_filenames):
    full_hs_filename = os.path.join(args.input_hotspots_directory, filename)
    # Load the NetCDF file into an xarray.Dataset
    dataset = xr.open_dataset(full_hs_filename)
    hotspot_id = dataset['hotspot_features']
    hotspot_data = dataset['hotspot_field']

    HS_object_list = create_stormcell_list(hotspot_id, hotspot_data)

    #Load the StormID file or the MCIT file for assignment
    # We like to load the cell from t-1 to match hotspots at time t
    # but you don't have too. 
    if i == 0:
        #there is no t-1 for this time, use time t
        full_cell_filename = os.path.join(args.input_cells_directory, cell_filenames[0])
    else:
        #use time t-1
        full_cell_filename = os.path.join(args.input_cells_directory, cell_filenames[i-1])

    print("Matching: %s with %s "%(full_hs_filename, full_cell_filename))
    #read in the data from the netcdf file
    dataset = xr.open_dataset(full_cell_filename)
    #did you give us mcit or StormID files?
    if "StormID" in dataset.data_vars:
        cell_id = dataset["StormID"]
    else:
        cell_id = dataset["MCIT"]

    vil_xy = dataset["VIL"]

    MCIT_object_list = create_stormcell_list(cell_id, vil_xy)
    #do assignment
    merge_dict = hotspot_assignment(MCIT_object_list, HS_object_list)

    #trim:
    assigned_hotspots = filter_assigned_hotspots(hotspot_id, merge_dict)

    #create an xarray dataset for output
    #Strip the Hotspot objects down to the minimum number of variables
    #for output the netcdf file....
    #You can of course add the other things you need/want
    HotSpot_ds = xr.merge([hotspot_data, assigned_hotspots])

    #Output the Dataset to a NetCDF file
    #
    #expect an input filename like:
    #ZdrHotSpot_20200522_235942.nc
    radar_file = os.path.basename(filename)
    time_str = radar_file[11:26]
    output_file = output_path + "FinalZdrHotSpot" + "_" + time_str + ".nc"
    print("output file: %s \n" % (output_file))

    HotSpot_ds.to_netcdf(output_file)

    if args.movie == True:
        # now let's take a look at the output
        xlim = (-100, 100)
        ylim = (-100, 100)
        obj_cmap = klaus_krause_cmap.get_obj_cmap()
        hs_cmap = klaus_krause_cmap.get_hs_cmap(3.0)
        # Create a color map that is just black
        black_cmap = ListedColormap(['black'])
        
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
        
        ax1 = fig.add_subplot(221)
        cell_id.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, vmin=0,
                           cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'Cells', 'shrink':0.9}, vmax=75.0, ax=ax1)
        assigned_hotspots.where(assigned_hotspots > 0).plot(
    x="x", y="y", xlim=xlim, ylim=ylim, cmap=black_cmap, add_colorbar=False,
    ax=ax1, vmin=0, vmax=75)
        ax1.set_title('cell values with \n timmed hotspot overlay in black', fontweight='bold')
        #
        ax2 = fig.add_subplot(222)
        hotspot_data.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap, vmin=-3.0,
                           cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'HotSpot values', 'shrink':0.9}, vmax=3.0, ax=ax2)
        ax2.set_title('HotSpot values', fontweight='bold')
        
        ax3 = fig.add_subplot(223)
        hotspot_id.plot(
            x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, vmin=0, vmax=75,
            ax=ax3, cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'HotSpot objects', 'shrink':0.9})
        ax3.set_title('HotSpot objects', fontweight='bold')
        
        
        ax4 = fig.add_subplot(224)
        assigned_hotspots.plot(
            x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, vmin=0, vmax=75,
            ax=ax4, cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'HotSpot objects', 'shrink':0.9})
        ax4.set_title('HotSpot Trimmed', fontweight='bold')
        
        # Save the plot as an image file
        image_path = os.path.join(args.output_directory, f'plot_{i:03d}.png')
        fig.suptitle(f'Frame {i+1}: {filename}')
        plt.savefig(image_path)
        plt.close()  # Close the plot to avoid overlapping
  
if args.movie == True:
    # Collect all saved images and create an animated GIF
    image_files = [os.path.join(args.output_directory, f'plot_{i:03d}.png') for i in range(len(hs_filenames))]
    gif_filename = os.path.join(args.output_directory, f'animated_gif.gif')
    
    with imageio.get_writer(gif_filename, mode='I', duration=5) as writer:  # Adjust duration as needed
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)
    
    print(f"Animated GIF saved as {gif_filename}")
