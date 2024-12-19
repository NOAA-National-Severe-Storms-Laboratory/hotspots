#!/usr/bin/env python
# coding: utf-8
import os
import glob
import argparse

from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import klaus_krause_cmap
# Initialize the argument parser
parser = argparse.ArgumentParser(description="Plot hotspot data from compute_zdr_hotspots.py")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-f", "--input_file", type=str, help="file to plot")
# Parse the command-line arguments
args = parser.parse_args()

filename = args.input_file


# Load the NetCDF file into an xarray.Dataset
dataset = xr.open_dataset(filename)
hotspot_field = dataset['hotspot_field']
hotspot_features = dataset['hotspot_features']

# now let's take a look at the output
xlim = (-100, 100)
ylim = (-100, 100)
hs_cmap = klaus_krause_cmap.get_hs_cmap(3)
obj_cmap = klaus_krause_cmap.get_obj_cmap()

fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

#
ax2 = axs[0]
hotspot_field.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap, vmin=-3,
                   cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'hotspot field', 'shrink':0.9}, vmax=3, ax=ax2)
ax2.set_title('Hotspot Field', fontweight='bold')

ax3 = axs[1]
hotspot_features.where(hotspot_features > 0).plot(
    x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, vmin=0, vmax=50,
    ax=ax3, cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'hotspot objects', 'shrink':0.9})
ax3.set_title('Hotspot Objects', fontweight='bold')


for ax in axs.flatten():
    ax.set_aspect('equal')

plt.show()

