#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse
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
from klaus_krause_cmap import *


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process Level2 data to get Zdr Hotspots")

# Add a positional argument (e.g., a required input file)
parser.add_argument("-f", "--filename", type=str, help="a single level 2 file")
parser.add_argument("-i", "--input_dir", type=str, help="location where the Level 2 files are located")
parser.add_argument("-o", "--output_dir", type=str, help="location where the output files are stored", required=True)
parser.add_argument("-r", "--radar_name", type=str, help="4 letter Radar name, example: KOAX, used as a file glob in the directory")
parser.add_argument("-c", "--cappi_height", type=float, help="Height of the -10C CAPPI in Meters AMSL above mean sea level", required=True)
# Add optional flags
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
# Parse the command-line arguments
args = parser.parse_args()

    

# Define the directory containing the files
if args.input_dir is not None:
    directory_path = args.input_dir
    if not directory_path.endswith("/"):
        directory_path += "/"

#Define the location of the mcit output
output_path = args.output_dir
if not output_path.endswith("/"):
    output_path += "/"

#assign input data
radar_name = args.radar_name
cappi_height = args.cappi_height

if args.filename is not None:
    files = [args.filename]
else:
    # Define the pattern to match files (e.g., all .txt files)
    filepattern = radar_name + "*"

    files = sorted(glob.glob(os.path.join(directory_path, filepattern)))

print("Size of files: %d \n" % (len(files)))

#
#Setup the grid we use:
#
x_trg = np.arange(-200, 201)
y_trg = np.arange(-200, 201)
#
#Plotting code for debug
#plots for sanity:
import klaus_krause_cmap

xlim = (-150, 150)
ylim = (-150, 150)
hs_cmap = klaus_krause_cmap.get_hs_cmap(3)
zdr_cmap = klaus_krause_cmap.get_zdr_cmap(-2, 5)
cc_cmap = klaus_krause_cmap.get_NWS_CC_ext()


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

    if args.filename is not None:
        fig = plt.figure(figsize=(12, 4))
        fig.suptitle('Raw Radar Data', fontsize=16)

        plt.rcParams.update(plt.rcParamsDefault)
        #plt.style.use('dark_background')
        
        plt.rcParams.update(
                {'font.size': 14.0,
                 'axes.titlesize': 'x-large',
                 'axes.linewidth': 2.0,
                 'axes.labelsize': 'large'}
            )
        
        
        xlim = [-100, 100]
        ylim = [-100, 100]
        axislabels=["X (km)", "Y (km)"]
    
        sweep = 0
        display = pyart.graph.RadarDisplay(radar_vol)
    
        ax1 = fig.add_subplot(131)
        ax1.set_aspect('equal')
        ax1.set_facecolor('darkgrey')
        
        
        display.plot_ppi(
                "reflectivity", sweep=sweep, ax=ax1, vmin=-32, vmax=95,
                mask_outside=True, cmap="pyart_NWSRef", title_flag=False,
                colorbar_label='Reflectivity (dBZ)', colorbar_flag=False, axislabels=axislabels )
        
        display.set_limits(xlim, ylim, ax=ax1)
        #display.plot_range_rings([50, 100, 150], ax=ax1, lw=1.0, ls='--', col='black')
        
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='Reflectivity (dBZ)')
        #
        ax3 = fig.add_subplot(132)
        ax3.set_aspect('equal')
        cc_cmap, norm = get_cc()
    
        display.plot(
                "cross_correlation_ratio", sweep=sweep, ax=ax3, vmin=0, vmax=1.2,
                cmap=cc_cmap, mask_outside=True, title_flag=False,
                colorbar_label='', colorbar_flag=False, axislabels=axislabels)
        display.set_limits(xlim, ylim, ax=ax3)
        #display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='CC')
        
        ax3.set_xlabel('')
        ax3.set_ylabel('Y (km)')
    
        ax5 = fig.add_subplot(133)
        ax5.set_aspect('equal')
        
        display.plot(
                "differential_reflectivity", sweep=sweep, ax=ax5, vmin=-2, vmax=5,
                cmap='pyart_HomeyerRainbow', mask_outside=True, title_flag=False,
                colorbar_label='', colorbar_flag=False, axislabels=axislabels)
        display.set_limits(xlim, ylim, ax=ax5)
        #display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='differential_reflectivity')
        
        ax5.set_xlabel('')
        ax5.set_ylabel('Y (km)')
    
        plt.tight_layout()
        plt.show()
 
    #The processing of the data by the python version 
    #of the NWS ORPG Preprocess is close but not the same
    #
    #This will have to do until I can wrap the c/c++ code
    #into a python library
    metadata = {}
    metadata['zdr_absolute_calibration'] = 0.0
    metadata['z_absolute_calibration'] = 0.0
    processed_vol = preprocessor_norpg_xarray(prune_vol, metadata)

    #compute the dr from the processed data, use pyart 
    prepro_dr = pyart.retrieve.compute_cdr(processed_vol,
                                           rhohv_field='prepro_cc',
                                           zdr_field='prepro_zdr')
    dr = pyart.retrieve.compute_cdr(processed_vol,
                                     rhohv_field='cross_correlation_ratio',
                                     zdr_field='differential_reflectivity')

    #add circular depolarization ratio to our processed volume
    processed_vol.add_field('prepro_dr', prepro_dr)

    if args.filename is not None:
    
        display = pyart.graph.RadarDisplay(processed_vol)
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Processed Radar Data', fontsize=16)
    
        plt.rcParams.update(plt.rcParamsDefault)
        #plt.style.use('dark_background')
        
        plt.rcParams.update(
                {'font.size': 14.0,
                 'axes.titlesize': 'x-large',
                 'axes.linewidth': 2.0,
                 'axes.labelsize': 'large'}
            )
        
        
        xlim = [-100, 100]
        ylim = [-100, 100]
        axislabels=["X (km)", "Y (km)"]
    
        sweep = 0
    
        ax1 = fig.add_subplot(221)
        ax1.set_aspect('equal')
        ax1.set_facecolor('darkgrey')
        
        
        display.plot_ppi(
                "prepro_zh", sweep=sweep, ax=ax1, vmin=-32, vmax=95,
                mask_outside=True, cmap="pyart_NWSRef", title_flag=False,
                colorbar_label='Reflectivity (dBZ)', colorbar_flag=False, axislabels=axislabels )
        
        display.set_limits(xlim, ylim, ax=ax1)
        #display.plot_range_rings([50, 100, 150], ax=ax1, lw=1.0, ls='--', col='black')
        
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='Reflectivity (dBZ)')
        #
        ax3 = fig.add_subplot(223)
        ax3.set_aspect('equal')
        cc_cmap, norm = get_cc()
    
        display.plot(
                "prepro_cc", sweep=sweep, ax=ax3, vmin=0, vmax=1.2,
                cmap=cc_cmap, mask_outside=True, title_flag=False,
                colorbar_label='', colorbar_flag=False, axislabels=axislabels)
        display.set_limits(xlim, ylim, ax=ax3)
        #display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='CC')
        
        ax3.set_xlabel('')
        ax3.set_ylabel('Y (km)')
    
        ax4 = fig.add_subplot(224)
        ax4.set_aspect('equal')
        DR_cmap = get_dr_cmap(-30, 0, precip_threshold=-10)
    
        display.plot(
                "prepro_dr", sweep=sweep, ax=ax4, vmin=-30, vmax=0,
                cmap=DR_cmap, mask_outside=True, title_flag=False,
                colorbar_label='', colorbar_flag=False, axislabels=axislabels)
        display.set_limits(xlim, ylim, ax=ax4)
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='DR')
        
        ax4.set_xlabel('')
        ax4.set_ylabel('Y (km)')
    
        ax5 = fig.add_subplot(222)
        ax5.set_aspect('equal')
        
        display.plot(
                "prepro_zdr", sweep=sweep, ax=ax5, vmin=-2, vmax=5,
                cmap=zdr_cmap, mask_outside=True, title_flag=False,
                colorbar_label='', colorbar_flag=False, axislabels=axislabels)
        display.set_limits(xlim, ylim, ax=ax5)
        #display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
        display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='differential_reflectivity')
        
        ax5.set_xlabel('')
        ax5.set_ylabel('Y (km)')
    
        plt.tight_layout()
        plt.show()

    #indexing the volume forces the volume into a 1 degree by 250m gate format
    vv_ds = get_indexed_vv(processed_vol)
    
    print("Target CAPPI height: %f " % (target_cappi_height))
    # make azran CAPPI interpolation
    azran_ds = indexed_vv_to_cappi_amsl(vv_ds, target_cappi_height)
    if args.filename is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        fig.suptitle('AzRan CAPPI Radar Data', fontsize=16)
        #base data 
        ax4 = axs[0]
        azran_ds['prepro_zdr'].plot( x="x", y="y", xlim=xlim, ylim=ylim, cmap=zdr_cmap, ax=ax4, vmin=-2, vmax=5, 
                cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'PrePro Zdr', 'shrink':0.9})
        ax4.set_title(f'CAPPI height {target_cappi_height:.0f}')
    
        ax5 = axs[1]
        azran_ds['prepro_zh'].plot( x="x", y="y", xlim=xlim, ylim=ylim, cmap='pyart_NWSRef', ax=ax5, vmin=-10, vmax=75)
        ax5.set_title(f'CAPPI height {target_cappi_height:.0f}')

        ax6 = axs[2]
        azran_ds['prepro_cc'].plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=cc_cmap, ax=ax6, vmin=0.2, vmax=1.05)
        ax6.set_title(f'CAPPI height {target_cappi_height:.0f}')
    
        plt.show()

    # make xy interpolation
    xycappi = azran_to_xy(azran_ds, x_trg, y_trg, max_dist=3)

    if args.filename is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        fig.suptitle('XY CAPPI Radar Data', fontsize=16)
        #base data 
        ax4 = axs[0]
        ax4.set_title(f'CAPPI height {target_cappi_height:.0f}')
        xycappi['prepro_zdr'].plot( x="x", y="y", xlim=xlim, ylim=ylim, cmap=zdr_cmap, ax=ax4, vmin=-2, vmax=5)
    
        ax5 = axs[1]
        xycappi['prepro_zh'].plot( x="x", y="y", xlim=xlim, ylim=ylim, cmap='pyart_NWSRef', ax=ax5, vmin=-10, vmax=75)
    
        ax6 = axs[2]
        xycappi['prepro_cc'].plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=cc_cmap, ax=ax6, vmin=0.2, vmax=1.05)
    
        plt.show()
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

    if args.filename is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        # new hotspots first
        ax4 = axs[0]
        xycappi['zdr_cut_proc'].plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=zdr_cmap, ax=ax4, vmin=-2,
        vmax=5)
    
        ax5 = axs[1]
        hotspot_field_proc.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap,
                            vmin=-3, vmax=3, ax=ax5)
    
        ax6 = axs[2]
        hotspot_features_proc.where(hotspot_features_proc > 0).plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap='tab20b', ax=ax6)
    
        plt.show()
    #Output the Dataset to a NetCDF file
    #
    #expect an input filename like:
    #KDDC20200525_044128_V06
    radar_file = os.path.basename(filename)
    time_str = radar_file[4:19]
    output_file = output_path + "ZdrHotSpot" + "_" + time_str + ".nc"
    print("output file: %s \n" % (output_file))

    HotSpot_ds.to_netcdf(output_file)

