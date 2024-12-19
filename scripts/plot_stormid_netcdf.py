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

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Plot StormID data from compute_StormID.py")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-f", "--input_file", type=str, help="file to plot")
parser.add_argument("-d", "--input_directory", type=str, help="a directory of files to plot")
parser.add_argument("-o", "--output_directory", type=str, help="where to put the output files")
parser.add_argument("-g", "--movie",  action='store_true', help="make a movie out of the files")
# Parse the command-line arguments
args = parser.parse_args()

# Make sure output directory exists
os.makedirs(args.output_directory, exist_ok=True)

if args.input_directory:
    filenames = sorted([f for f in os.listdir(args.input_directory) if f.endswith('.nc')])
else:
    filenames = args.input_file

for i, filename in enumerate(filenames):
    full_filename = os.path.join(args.input_directory, filename)
    # Load the NetCDF file into an xarray.Dataset
    dataset = xr.open_dataset(full_filename)
    stormid = dataset['StormID']
    vil = dataset['VIL']
    
    # now let's take a look at the output
    xlim = (-100, 100)
    ylim = (-100, 100)
    obj_cmap = klaus_krause_cmap.get_obj_cmap()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    #
    ax2 = axs[0]
    vil.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap='pyart_NWSRef', vmin=0,
                       cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'VIL', 'shrink':0.9}, vmax=75, ax=ax2)
    ax2.set_title('VIL', fontweight='bold')
    
    ax3 = axs[1]
    stormid.plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, vmin=0, vmax=200,
        ax=ax3, cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'StormID', 'shrink':0.9})
    ax3.set_title('StormID', fontweight='bold')
    
    
    for ax in axs.flatten():
        ax.set_aspect('equal')
    
    # Save the plot as an image file
    image_path = os.path.join(args.output_directory, f'plot_{i:03d}.png')
    fig.suptitle(f'Frame {i+1}: {filename}')
    plt.savefig(image_path)
    plt.close()  # Close the plot to avoid overlapping
  
if args.movie == True:
    # Collect all saved images and create an animated GIF
    image_files = [os.path.join(args.output_directory, f'plot_{i:03d}.png') for i in range(len(filenames))]
    gif_filename = os.path.join(args.output_directory, f'animated_gif.gif')
    
    with imageio.get_writer(gif_filename, mode='I', duration=5) as writer:  # Adjust duration as needed
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)
    
    print(f"Animated GIF saved as {gif_filename}")
