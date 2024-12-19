#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pyart
import numpy as np

from prepro.nexrad import prune_nexrad
from cappi.make_vv import simple_vv, get_indexed_vv
from mcit.vil import get_vil_from_azran
from mcit.mcit_objects import mcit_objects
from mcit.mcit_objects import mcit_trim
from config import _EXAMPLEDATA_DIR

#viewable plots
import matplotlib.pyplot as plt
from klaus_krause_cmap import get_zdr_cmap, get_obj_cmap


#filename = Path('../data/KTLX20230227_012550_V06')
filename = Path(_EXAMPLEDATA_DIR, "nexrad_level2", "KOAX20140603_213649_V06.gz")
#filename = Path('../data/KEWX20210504_020040_V06')
#filename = Path('../data/KSHV20230613_230228_V06')

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
vil.attrs.update(ds.attrs)
#
#convert the AzRan image to XY coordinates
#
from cappi.xy_cappi import azran_to_xy

x_trg = np.arange(-300, 301, 1)
y_trg = np.arange(-300, 301, 1)

vil_xy = azran_to_xy(vil, x_trg, y_trg, grid_spacing_meters=1000.0)
mcit_raw = mcit_objects(vil_xy, 2.0, 1.5)
mcit = mcit_trim(mcit_raw, vil_xy, 5.0, 50)

#plot some results
plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams.update(
        {'font.size': 14.0,
         'axes.titlesize': 'x-large',
         'axes.linewidth': 2.0,
         'axes.labelsize': 'large'}
    )

axislabels = ["X (km)", "Y (km)"]
axisXlabels = ["X (km)", ""]
axisYlabels = ["", "Y (km)"]
axisnonelabels = ["", ""]
xlim = [-150, 150]
ylim = [-150, 150]
xdiff = xlim[1] - xlim[0]
axps = xdiff * 0.075
ayps = xdiff * 0.025
aps = [xlim[0] + axps, ylim[0] + ayps]

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(221)
ax1.set_aspect('equal')
obj_cmap = get_obj_cmap()

vil.where(vil>-10).plot(ax=ax1, x='x', y='y', vmin=0.0, vmax=75.0,
                        xlim=xlim, ylim=ylim, cmap='pyart_NWSRef',
                        add_colorbar=True,
                        cbar_kwargs={'extend':'both', 'pad':0.01, 'label':'VIL',
                                     'shrink':0.9})

ax1.set_title("")
ax2 = fig.add_subplot(222)
ax2.set_aspect('equal')
ax2.set_ylabel('')


mcit_raw.plot(ax=ax2, x='x', y='y',  vmin=-1.0, vmax=200.0, xlim=xlim, ylim=ylim,
              cmap=obj_cmap, add_colorbar=True,
              cbar_kwargs={'extend':'both', 'pad':0.01, 'label':'labels',
                           'shrink':0.9})
ax2.set_title("")


ax3 = fig.add_subplot(223)
ax3.set_aspect('equal')
ax3.set_ylabel('')


mcit.plot(ax=ax3, x='x', y='y', xlim=xlim, ylim=ylim, vmin=-1, vmax=100.0,
          cmap=obj_cmap, add_colorbar=True,
          cbar_kwargs={'extend':'both', 'pad':0.01, 'label': 'trimmed_labels',
                       'shrink':0.9})
ax3.set_title("")
plt.show()

#out_filename = Path("./test.png")
#fig.savefig(out_filename)
#plt.close(fig)
