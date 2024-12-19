import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import copy
import pyart
import wradlib as wrl

from prepro.process_phase import process_differential_phase, \
    compute_smoothed_phase_PZhangNSSL, calc_kdp_wradlib

#filename = Path("../data", "KFDR20200523_040411_V06")
filename = Path("../exampledata", "KUDX20150620_040849_V06.gz")
#filename = Path("../data", "KSHV20230613_230228_V06")

radar_vol = pyart.io.read_nexrad_archive(filename)
radar_vol0 = radar_vol.extract_sweeps(sweeps=[0])

# get JK Advanced Computing version
diff_phase = np.array(radar_vol0.fields['differential_phase']['data'])
cc = radar_vol0.fields['cross_correlation_ratio']['data']

cor_diff_phase = np.ma.empty_like(diff_phase)
cor_diff_phase = process_differential_phase(diff_phase, cc, phase_window=9)
#cor_diff_phase = compute_smoothed_phase_PZhangNSSL(diff_phase, cc, window_size=9)

radar_vol0.add_field_like('differential_phase',
                          'corrected_differential_phase',
                          cor_diff_phase, replace_existing=True)
#
#using wradlib......
#
gate_spacing_km = 0.250 
#kdp = wrl.util.derivate(cor_diff_phase, winlen=9, skipna=False) * 0.5 / gate_spacing_km
metadata = {}
metadata['gatespacing_meters'] = 250.0
metadata['phase_window'] = 9
kdp = calc_kdp_wradlib(cor_diff_phase, 9, gate_spacing_km, debug=False)

radar_vol0.add_field_like('differential_phase',
                          'kdp',
                          kdp, replace_existing=True)

display = pyart.graph.RadarDisplay(radar_vol0)

# sweep comparison
fig = plt.figure(figsize=(14, 4))
xlim = [0, 200]
ylim = [0, 200]
vmax = 400
ax1 = fig.add_subplot(131)
display.plot_ppi('differential_phase', sweep=0, vmin=0.1, vmax=vmax, mask_outside=True,
                 colorbar_flag=False, title_flag=False)
display.set_limits(xlim, ylim)
display.plot_colorbar(extend='both', shrink=0.9)
ax1.set_title('Original NEXRAD Level II')

ax2 = fig.add_subplot(132)
display.plot_ppi('corrected_differential_phase', sweep=0,
                 cmap='pyart_Wild25', vmin=0.1, vmax=vmax, mask_outside=True,
                 colorbar_flag=False, title_flag=False)
display.set_limits(xlim, ylim)
display.plot_colorbar(extend='both', shrink=0.9)
ax2.set_title("Phase - JK Advanced Computing Facility")

ax3 = fig.add_subplot(133)
display.plot_ppi('kdp', sweep=0,
                 cmap='pyart_Wild25', vmin=-0.5, vmax=5, mask_outside=True,
                 colorbar_flag=False, title_flag=False)
display.set_limits(xlim, ylim)
display.plot_colorbar(extend='both', shrink=0.9)
ax3.set_title('KDP - JK Advanced Computing Facility')

for ax in [ax1, ax2, ax3]:
    ax.set_aspect('equal')

time0 = pyart.util.datetime_from_radar(radar_vol0)
radarname = radar_vol0.metadata['instrument_name']
fig.suptitle(radarname + str(time0), fontweight='bold', fontsize=20)

plt.tight_layout()

plt.show()
#plt.savefig('phaseunwrap.png', dpi=200)
#plt.close()

# let's get KDP
# kdp = wrl.dp.kdp_from_phidp(
#     radar_vol0.fields['unwrapped_differential_phase']['data'], dr=0.250,
#     winlen=27)
# kdp_dict = {
#     'data': kdp
# }
# radar_vol0.add_field('specific_differential_phase', kdp_dict,
#                      replace_existing=True)
#
# display = pyart.graph.RadarDisplay(radar_vol0)
# display.plot_ppi('specific_differential_phase')
# plt.show()

