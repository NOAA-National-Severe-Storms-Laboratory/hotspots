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
parser = argparse.ArgumentParser(description="Plot hotspot data from compute_zdr_hotspots.py")

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
    hotspot = dataset['hotspot_features']
    data = dataset['hotspot_field']
    
    # now let's take a look at the output
    xlim = (-200, 200)
    ylim = (-200, 200)
    obj_cmap = klaus_krause_cmap.get_obj_cmap()
    hs_cmap = klaus_krause_cmap.get_hs_cmap(3.0)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    #
    ax2 = axs[0]
    data.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap, vmin=-3.0,
                       cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'HotSpot values', 'shrink':0.9}, vmax=3.0, ax=ax2)
    ax2.set_title('HotSpot values', fontweight='bold')
    
    ax3 = axs[1]
    hotspot.plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, vmin=0, vmax=200,
        ax=ax3, cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'HotSpot objects', 'shrink':0.9})
    ax3.set_title('HotSpot objects', fontweight='bold')
    
    
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
