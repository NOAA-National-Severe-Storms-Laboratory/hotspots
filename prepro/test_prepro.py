#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pyart
import datetime as dt
import xarray as xr

from prepro.nexrad import preprocessor_nORPG_pyart, preprocessor_norpg_xarray, \
        prune_nexrad

#viewable plots
import matplotlib.pyplot as plt
from klaus_krause_cmap import *

#filename = Path('../data/KUDX20150620_040849_V06.gz')
filename = Path('../exampledata/nexrad_level2/KOAX20140603_213649_V06.gz')
#filename = Path('../data/KEWX20210504_020040_V06')
#filename = Path('../data/KSHV20230613_230228_V06')
#filename = Path('../data/KTLX20160526_215113_V06')
#filename = Path('../data/KTLX20200422_023145_V06')
#read in the Level 2 WSR-88D data as a pyart object
radar_vol = pyart.io.read_nexrad_archive(filename)
#radar_vol.info()

#remove extra sweeps of data. Keep only data
#from the survelience cuts and one cut per volume 
prune_actions = ['surv', 'volume']
prune_vol = prune_nexrad(prune_actions, radar_vol)
#prune_vol.info()


#Our Pyart driver:
radar_vol = prune_vol

metadata = {}
metadata['zdr_absolute_calibration'] = 0.0
metadata['z_absolute_calibration'] = 0.0

time0 = dt.datetime.now()
#output = preprocessor_nORPG_pyart(metadata, radar_vol)
output = preprocessor_norpg_xarray(radar_vol, metadata)
time1 = dt.datetime.now()

print("phase processing time", time1-time0)

#output.info()


fig = plt.figure(figsize=(12, 10))

plt.rcParams.update(plt.rcParamsDefault)
#plt.style.use('dark_background')

plt.rcParams.update(
        {'font.size': 14.0,
         'axes.titlesize': 'x-large',
         'axes.linewidth': 2.0,
         'axes.labelsize': 'large'}
    )


xlim = [-300, 300]
ylim = [-300, 300]
axislabels=["X (km)", "Y (km)"]


sweep = 0
display = pyart.graph.RadarDisplay(output)


ax1 = fig.add_subplot(321)
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
#no figure labels for paper. Information is in the
#figure caption
#
#ax1.set_title(f"Reflectivity {radarname} {cappi_time:%Y-%m-%d %H:%M}")
ax1.set_xlabel('')
ax1.set_ylabel('Y (km)')

ax2 = fig.add_subplot(322)
ax2.set_aspect('equal')
ax2.set_facecolor('darkgrey')


display.plot_ppi(
        "prepro_zh", sweep=sweep, ax=ax2, vmin=-32, vmax=95,
        mask_outside=True, cmap="pyart_NWSRef", title_flag=False,
        colorbar_label='Reflectivity (dBZ)', colorbar_flag=False,
        axislabels=axislabels)

display.set_limits(xlim, ylim, ax=ax2)
#display.plot_range_rings([50, 100, 150], ax=ax1, lw=1.0, ls='--', col='black')

display.plot_colorbar(extend='both', pad=0.01, shrink=1.0,
                      label='PrePro Reflectivity (dBZ)')
#
#no figure labels for paper. Information is in the
#figure caption
#
#ax1.set_title(f"Reflectivity {radarname} {cappi_time:%Y-%m-%d %H:%M}")
ax2.set_xlabel('')
ax2.set_ylabel('Y (km)')


ax3 = fig.add_subplot(323)
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

ax4 = fig.add_subplot(324)
ax4.set_aspect('equal')


display.plot(
        "prepro_cc", sweep=sweep, ax=ax4, vmin=0, vmax=1.2,
        cmap=cc_cmap, mask_outside=True, title_flag=False,
        colorbar_label='', colorbar_flag=False, axislabels=axislabels)
display.set_limits(xlim, ylim, ax=ax4)
#display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='prepro CC')

ax5 = fig.add_subplot(325)
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

ax6 = fig.add_subplot(326)
ax6.set_aspect('equal')


display.plot(
        "prepro_zdr", sweep=sweep, ax=ax6, vmin=-2, vmax=5, mask_outside=True,
        cmap='pyart_HomeyerRainbow', title_flag=False,
        colorbar_label='', colorbar_flag=False, axislabels=axislabels)
display.set_limits(xlim, ylim, ax=ax6)
#display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='prepro_zdr')

ax6.set_xlabel('')
ax6.set_ylabel('Y (km)')

plt.tight_layout()
plt.show()
#plt.close(fig)


fig1 = plt.figure(figsize=(12, 10))

plt.rcParams.update(plt.rcParamsDefault)
#plt.style.use('dark_background')

plt.rcParams.update(
        {'font.size': 14.0,
         'axes.titlesize': 'x-large',
         'axes.linewidth': 2.0,
         'axes.labelsize': 'large'}
    )


xlim = [-300, 300]
ylim = [-300, 300]
axislabels=["X (km)", "Y (km)"]

sweep = 0
display = pyart.graph.RadarDisplay(output)

ax1 = fig1.add_subplot(321)
ax1.set_aspect('equal')
ax1.set_facecolor('darkgrey')

display.plot_ppi(
        "reflectivity_texture", sweep=sweep, ax=ax1, vmin=0, vmax=10,
        mask_outside=True, cmap="pyart_BlueBrown11", title_flag=False,
        colorbar_flag=False, axislabels=axislabels)

display.set_limits(xlim, ylim, ax=ax1)
#display.plot_range_rings([50, 100, 150], ax=ax1, lw=1.0, ls='--', col='black')

display.plot_colorbar(extend='both', pad=0.01, shrink=1.0,
                      label='Refl. Texture')
#
#no figure labels for paper. Information is in the
#figure caption
#
#ax1.set_title(f"Reflectivity {radarname} {cappi_time:%Y-%m-%d %H:%M}")
ax1.set_xlabel('')
ax1.set_ylabel('Y (km)')

ax2 = fig1.add_subplot(322)
ax2.set_aspect('equal')
ax2.set_facecolor('darkgrey')


display.plot_ppi(
        "differential_phase_texture", sweep=sweep, ax=ax2, vmin=0, vmax=60,
        mask_outside=True, cmap="pyart_BlueBrown11", title_flag=False,
        colorbar_flag=False, axislabels=axislabels)

display.set_limits(xlim, ylim, ax=ax2)
#display.plot_range_rings([50, 100, 150], ax=ax1, lw=1.0, ls='--', col='black')

display.plot_colorbar(extend='both', pad=0.01, shrink=1.0,
                      label='Diff. Phase Texture')
#
#no figure labels for paper. Information is in the
#figure caption
#
#ax1.set_title(f"Reflectivity {radarname} {cappi_time:%Y-%m-%d %H:%M}")
ax2.set_xlabel('')
ax2.set_ylabel('Y (km)')


ax3 = fig1.add_subplot(323)
ax3.set_aspect('equal')

display.plot(
        "differential_phase", sweep=sweep, ax=ax3, vmin=0, vmax=360,
        cmap='pyart_Wild25', mask_outside=True, title_flag=False,
        colorbar_label='', colorbar_flag=False, axislabels=axislabels)
display.set_limits(xlim, ylim, ax=ax3)
#display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='Diff. Phase')

ax3.set_xlabel('')
ax3.set_ylabel('Y (km)')

ax4 = fig1.add_subplot(324)
ax4.set_aspect('equal')

display.plot(
        "prepro_phase", sweep=sweep, ax=ax4, vmin=0, vmax=360,
        cmap='pyart_Wild25', mask_outside=True, title_flag=False,
        colorbar_label='', colorbar_flag=False, axislabels=axislabels)
display.set_limits(xlim, ylim, ax=ax4)
#display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='prepro phase')

ax5 = fig1.add_subplot(325)
ax5.set_aspect('equal')


display.plot(
        "prepro_snr", sweep=sweep, ax=ax5, vmin=-5, vmax=95,
        cmap='pyart_HomeyerRainbow', mask_outside=True, title_flag=False,
        colorbar_label='', colorbar_flag=False, axislabels=axislabels)
display.set_limits(xlim, ylim, ax=ax5)
#display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='SNR')

ax5.set_xlabel('')
ax5.set_ylabel('Y (km)')


ax6 = fig1.add_subplot(326)
ax6.set_aspect('equal')


display.plot(
        "prepro_kdp", sweep=sweep, ax=ax6, vmin=-5, vmax=5,
        cmap='pyart_Wild25', mask_outside=True, title_flag=False,
        colorbar_label='', colorbar_flag=False, axislabels=axislabels)
display.set_limits(xlim, ylim, ax=ax6)
#display.plot_range_rings([50, 100, 150], ax=ax2, lw=1.0, ls='--', col='black')
display.plot_colorbar(extend='both', pad=0.01, shrink=1.0, label='KDP')

ax6.set_xlabel('')
ax6.set_ylabel('Y (km)')

plt.tight_layout()
plt.show()

# compare with the docker output
# get the data from the docker container
filename_zdrdocker = Path("../exampledata/test/Zdr_00.50_20140603-213650.netcdf.gz")
zdr_docker = xr.open_dataset(filename_zdrdocker)
r_range = zdr_docker.RangeToFirstGate + zdr_docker.Gate * zdr_docker.GateWidth

x, y, z = pyart.core.antenna_vectors_to_cartesian(
        r_range.values[:, 0], zdr_docker.Azimuth.values,
        zdr_docker.Elevation)
zdr_docker = zdr_docker.assign_coords(
        x=(["Azimuth", "Gate"], x/1000),
        y=(["Azimuth", "Gate"], y/1000))
cmap_zdr = get_zdr_cmap(-2, 5)

sweep0 = output.extract_sweeps([0])
sweep0_ds = xr.Dataset(
        data_vars=dict(
                zdr=(["Azimuth", "Gate"], sweep0.fields['prepro_zdr']['data'])
        ),
        coords=dict(
                Azimuth=sweep0.azimuth['data'],
                Gate=np.arange(0, sweep0.ngates)
        )
)
sweep0_ds = sweep0_ds.sortby(sweep0_ds.Azimuth).isel(Gate=slice(0, 1192))
zdr_docker_matched = zdr_docker.sortby(zdr_docker.Azimuth).sel(
        Azimuth=sweep0_ds.Azimuth, method='ffill')
zdr_diff = sweep0_ds['zdr'].values - zdr_docker_matched['Zdr'].values
zdr_docker_matched['zdr_diff'] = (["Azimuth", "Gate"], zdr_diff)

xlim = [0, 100]
ylim = [-35, 65]

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
display.plot("prepro_zdr", sweep=0, ax=ax1, vmin=-2, vmax=5, cmap=cmap_zdr,
             colorbar_flag=False, title_flag=False)
display.set_limits(xlim, ylim)
display.plot_colorbar(extend='both', label='Zdr')
ax1.set_title('Prepro Zdr Python')
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_aspect('equal')

ax2 = fig.add_subplot(132)
zdr_docker['Zdr'].where(zdr_docker['Zdr'] > -99).plot(
        x="x", y="y", vmin=-2, vmax=5, cmap=cmap_zdr, xlim=xlim,
        ylim=ylim)
ax2.set_title('Prepro Zdr Docker')
ax2.set_xlabel('X (km)')
ax2.set_ylabel('')
ax2.set_aspect('equal')


ax3 = fig.add_subplot(133)
zdr_docker_matched['zdr_diff'].where(zdr_docker_matched['Zdr'] > -99).plot(
        x="x", y="y", vmin=-2, vmax=2, cmap='seismic', xlim=xlim,
        ylim=ylim
)
ax3.set_title('Difference Python-Docker')
ax3.set_xlabel('X (km)')
ax3.set_aspect('equal')

plt.show()