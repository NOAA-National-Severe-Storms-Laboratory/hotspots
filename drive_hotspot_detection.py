from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyart

from hotspots import detection
from cappi.helpers import xy_assign_context
from prepro.nexrad import preprocessor_norpg_xarray, compute_nexrad_wave_form, \
    prune_nexrad
from cappi.make_vv import get_indexed_vv
from cappi.azran import indexed_vv_to_cappi_amsl
from cappi.xy_cappi import azran_to_xy
import klaus_krause_cmap
from make_logger import logger



def drive_hotspot_detection():

    filename = Path('exampledata/nexrad_level2/KOAX20140603_213649_V06.gz')
    radar_vol = pyart.io.read_nexrad_archive(filename)

    nexrad_wave_form = compute_nexrad_wave_form(radar_vol)
    logger.info(f"from file: {filename}")
    logger.info(f"wave_forms: {nexrad_wave_form}")

    prune_actions = ['surv', 'volume']

    prune_vol = prune_nexrad(prune_actions, radar_vol)

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
    processed_vol.add_field('prepro_dr', prepro_dr)

    indexed_vv = get_indexed_vv(processed_vol)
    #indexed_vv2 = get_indexed_vv_experimental(prune_vol)

    # make azran CAPPI interpolation
    target_cappi_height = 5188  # m AMSL = -10°C isotherm height
    azran_ds = indexed_vv_to_cappi_amsl(indexed_vv, target_cappi_height)

    # make xy interpolation
    x_trg = np.arange(-200, 201)
    y_trg = np.arange(-200, 201)
    xycappi = azran_to_xy(azran_ds, x_trg, y_trg, grid_spacing_meters=1000, max_dist=2)

    # get hotspot field and detections for unprocessed fields
    zdrcut = detection.get_filtered_zdr(
        xycappi, refl_fieldname='reflectivity',
        zdr_fieldname='differential_reflectivity',
        cdr_fieldname='dr')
    xycappi["zdr_cut"] = zdrcut
    hotspot_field, hotspot_features = (
        detection.apply_hotspot_method(
        xycappi, x_dim="x", y_dim="y", refl_fieldname='reflectivity',
        zdr_fieldname='zdr_cut'))

    # same for processed radar fields
    zdrcut_proc = detection.get_filtered_zdr(
        xycappi, refl_fieldname='prepro_zh', zdr_fieldname='prepro_zdr',
        cdr_fieldname='prepro_dr')
    xycappi["zdr_cut_proc"] = zdrcut_proc
    hotspot_field_proc, hotspot_features_proc = (
        detection.apply_hotspot_method(
        xycappi, x_dim="x", y_dim="y", refl_fieldname='prepro_zh',
        zdr_fieldname='zdr_cut_proc'))

    # now let's take a look at the output
    xlim = (0, 100)
    ylim = (-35, 65)
    hs_cmap = klaus_krause_cmap.get_hs_cmap(3)
    zdr_cmap = klaus_krause_cmap.get_zdr_cmap(-2, 5)
    obj_cmap = klaus_krause_cmap.get_obj_cmap()

    fig, axs = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)

    # unprocessed fields
    ax1 = axs[0, 0]
    xycappi['zdr_cut'].plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=zdr_cmap, ax=ax1, vmin=-2,
        vmax=5)
    ax1.set_title('', fontweight='bold')

    ax2 = axs[0, 1]
    hotspot_field.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap, vmin=-3,
                       vmax=3, ax=ax2)
    ax2.set_title('Python version (unprocessed)', fontweight='bold')

    ax3 = axs[0, 2]
    hotspot_features.where(hotspot_features > 0).plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap,
        ax=ax3, cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'hotspot objects'})
    ax3.set_title('', fontweight='bold')


    # new hotspots first
    ax4 = axs[1, 0]
    xycappi['zdr_cut_proc'].plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=zdr_cmap, ax=ax4, vmin=-2,
        vmax=5)
    ax4.set_title('', fontweight='bold')

    ax5 = axs[1, 1]
    hotspot_field_proc.plot(x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap,
                            vmin=-3, vmax=3, ax=ax5)
    ax5.set_title('Python version (processed)', fontweight='bold')

    ax6 = axs[1, 2]
    hotspot_features_proc.where(hotspot_features_proc > 0).plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, 
        cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'hotspot objects'}, 
        ax=ax6)
    ax6.set_title('', fontweight='bold')

    # get the data from the docker container
    filename_hotspotid = Path("exampledata/test/HotSpotID_00.00_20140603-213650.netcdf.gz")
    filename_hotspotfield = Path("exampledata/test/HotspotKlausZdrCAPPIneg10C_00.00_20140603"
                                 "-213650.netcdf.gz")
    filename_zdrcut = Path("exampledata/test/ZdrCutKlausZdrCAPPIneg10C_00.00_20140603-213650."
                           "netcdf.gz")

    hotspotid_alt = xr.open_dataset(filename_hotspotid)
    hotspotfield_alt = xr.open_dataset(filename_hotspotfield)
    zdrcut_alt = xr.open_dataset(filename_zdrcut)

    hotspotid_alt = xy_assign_context(hotspotid_alt,
                                      processed_vol.longitude['data'][0],
                                      processed_vol.latitude['data'][0])
    hotspotfield_alt = xy_assign_context(hotspotfield_alt,
                                         processed_vol.longitude['data'][0],
                                         processed_vol.latitude['data'][0])
    zdrcut_alt = xy_assign_context(zdrcut_alt,
                                   processed_vol.longitude['data'][0],
                                   processed_vol.latitude['data'][0])

    ax7 = axs[2, 0]
    zdrcut_alt['Zdr'].where(zdrcut_alt['Zdr'] > -9999).plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=zdr_cmap, ax=ax7, vmin=-2,
        vmax=5)
    ax7.set_title('', fontweight='bold')

    ax8 = axs[2, 1]
    hotspotfield_alt['Hotspot'].where(
        hotspotfield_alt['Hotspot'] > -9999).plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=hs_cmap, ax=ax8, vmin=-3,
        vmax=3)
    ax8.set_title('Docker version', fontweight='bold')

    ax9 = axs[2, 2]
    hotspotid_alt['HotspotID'].where(hotspotid_alt['HotspotID'] > 0).plot(
        x="x", y="y", xlim=xlim, ylim=ylim, cmap=obj_cmap, 
        cbar_kwargs={'extend': 'both',  'pad':0.01, 'label':'hotspot objects'}, 
        ax=ax9)
    ax9.set_title('', fontweight='bold')

    for ax in axs.flatten():
        ax.set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    drive_hotspot_detection()
