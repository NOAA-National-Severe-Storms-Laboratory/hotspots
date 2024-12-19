import cv2
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt  # TODO: remove in final version
from skimage.segmentation import expand_labels
import pyart

from hotspots import helpers

# TODO check code for bugs, look at differences between box and circle method


def get_hotspot_field(radar, z_neg10C):

    # mask everything < -10°C
    # remove non-meteo echoes by depolarization ratio
    # calculate detection for each elevation angle
    # find algorithm to get all values within 49 km² and 9 km² bounding boxes
    # substract from each other

    small_box_width = 1500
    large_box_width = 3500

    alt = radar.gate_altitude['data']
    alt_mask = alt < z_neg10C * 1000

    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_above('circular_depolarization_ratio', -12)
    gatefilter.exclude_outside('differential_reflectivity', -3, 6)
    gatefilter.exclude_gates(alt_mask)

    zdr_for_hotspots = np.ma.masked_where(
        gatefilter.gate_included == False, radar.fields[
            "differential_reflectivity"]["data"]
    )

    radar.add_field_like('differential_reflectivity',
                         'hotspot_differential_reflectivity',
                         zdr_for_hotspots)
    hotspot_field_method2 = np.ma.masked_all((radar.nrays, radar.ngates))

    # (2) distance method
    for sweep in radar.sweep_number['data']:
        hotspot_field_sweep = get_hotspot_distancemethod(
            radar, sweep, z_neg10C, small_radius=1800, large_radius=4100
        )
        sweep_slice = radar.get_slice(sweep)
        hotspot_field_method2[sweep_slice, :] = hotspot_field_sweep

    hotspot_field = np.ma.masked_all((radar.nrays, radar.ngates))
    # (1) box method
    for sweep in radar.sweep_number['data']:
        hotspot_field_sweep = get_hotspot_for_sweep(
            radar, sweep, z_neg10C,
            small_box_width=small_box_width,
            large_box_width=large_box_width)
        sweep_slice = radar.get_slice(sweep)
        hotspot_field[sweep_slice, :] = hotspot_field_sweep

    hotspots_dict = {'data': hotspot_field,
                     'standard_name': 'ZDR hotspot box method',
                     'coordinates': 'azimuth range'}
    radar.add_field('zdr_hotspots', hotspots_dict, replace_existing=True)
    # TODO: remove "replace_existing=True"

    hotspots_dict2 = {'data': hotspot_field_method2,
                      'standard_name': 'ZDR hotspot radius method',
                      'coordinates': 'azimuth range'}
    radar.add_field('zdr_hotspots2', hotspots_dict2, replace_existing=True)
    # TODO: remove "replace_existing=True"

    # for plotting
    display = pyart.graph.RadarDisplay(radar)
    xlim = [-30, 30]
    ylim = [50, 110]
    sweep = 7

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(221)
    display.plot_ppi('differential_reflectivity', sweep, vmin=-1, vmax=6,
                     ax=ax, mask_outside=True)
    display.set_limits(xlim, ylim, ax=ax)
    display.plot_range_rings([50, 100, 150], ax=ax, lw=0.6, ls='--', col='w')

    ax = fig.add_subplot(222)
    display.plot_ppi('reflectivity', sweep, vmin=0, vmax=75,
                     mask_outside=True, cmap='pyart_NWSRef')
    display.set_limits(xlim, ylim, ax=ax)
    display.plot_range_rings([50, 100, 150], ax=ax, lw=0.6, ls='--', col='w')

    ax = fig.add_subplot(223)
    display.plot_ppi('zdr_hotspots', sweep, vmin=-1.5, vmax=1.5, cmap='seismic')
    display.set_limits(xlim, ylim, ax=ax)
    display.plot_range_rings([50, 100, 150], ax=ax, lw=0.6, ls='--', col='w')

    ax = fig.add_subplot(224)
    display.plot_ppi('zdr_hotspots2', sweep, vmin=-1.5, vmax=1.5,
                     cmap='seismic')
    display.set_limits(xlim, ylim, ax=ax)
    display.plot_range_rings([50, 100, 150], ax=ax, lw=0.6, ls='--', col='w')

    plt.suptitle('AzRan detection', fontsize=14,
                 backgroundcolor='yellow', fontweight='bold', color='k')

    plt.tight_layout()
    plt.savefig('azran_hotspots.png', dpi=200)
    plt.show()

    return


def get_hotspot_for_sweep(radar, sweep, z_neg10C,
                          small_box_width=1500,
                          large_box_width=3500):

    zdr_sweep = radar.get_field(sweep, 'hotspot_differential_reflectivity')
    refl_sweep = radar.get_field(sweep, 'reflectivity')

    slice_sweep = radar.get_slice(sweep)
    alt_sweep = radar.gate_altitude['data'][slice_sweep]

    az_res = 360 / radar.rays_per_sweep['data'][sweep]

    mask_25dbz = np.where(refl_sweep > 25, 1, 0)
    expanded_25dbz = common.expand_labels_azran(
        mask_25dbz, 2500, max(radar.range['data']))
    zdr_sweep = np.ma.masked_where(expanded_25dbz == 0, zdr_sweep)

    rng_res = radar.range['meters_between_gates']  # valid for all sweeps
    rng_small_box = int(small_box_width / rng_res)
    rng_large_box = int(large_box_width / rng_res)
    len_rng = radar.rays_per_sweep['data'][sweep]

    hotspot_field = np.ma.masked_all(np.shape(zdr_sweep))

    for rng_i in range(rng_large_box + 1, radar.ngates-rng_large_box):
        rng_alt = alt_sweep[0, rng_i]
        if rng_alt < z_neg10C * 1000:
            continue

        rng = radar.range['data'][rng_i]

        min_rng_small = rng_i - rng_small_box
        max_rng_small = rng_i + rng_small_box

        min_rng_large = rng_i - rng_large_box
        max_rng_large = rng_i + rng_large_box

        rng_indices_small = np.arange(min_rng_small, max_rng_small+1)
        rng_indices_large = np.arange(min_rng_large, max_rng_large+1)
        rng_indices_large = rng_indices_large[
            ~np.isin(rng_indices_large, rng_indices_small)]

        az_small_box = small_box_width / (rng * np.pi/180 * az_res)
        az_large_box = large_box_width / (rng * np.pi/180 * az_res)

        az_small_residual = az_small_box - int(az_small_box)
        az_large_residual = az_large_box - int(az_large_box)

        for az_i in range(radar.rays_per_sweep['data'][sweep]):
            if np.ma.is_masked(zdr_sweep[az_i, rng_i]):
                continue

            min_az_small = (az_i - (int(az_small_box) + 1))
            max_az_small = (az_i + (int(az_small_box) + 1))

            min_az_large = (az_i - (int(az_large_box) + 1))
            max_az_large = (az_i + (int(az_large_box) + 1))

            az_ind_small = _get_indices(len_rng, min_az_small, max_az_small+1)
            az_ind_large = _get_indices(len_rng, min_az_large, max_az_large+1)

            az_ind_large = az_ind_large[
                ~np.isin(az_ind_large, az_ind_small[1:-1])]

            zdr_small = zdr_sweep[az_ind_small, min_rng_small:max_rng_small]
            zdr_large = zdr_sweep[az_ind_large.reshape(-1, 1), rng_indices_large]

            # TODO: maybe add a "if...continue" here if not enough valid points?

            center_large = int(np.shape(zdr_large)[0] / 2)
            weights_small = np.ones(np.shape(zdr_small))
            weights_large = np.ones(np.shape(zdr_large))

            weights_small[[0, -1], :] = az_small_residual
            weights_large[[0, -1], :] = az_large_residual
            weights_large[[center_large-1, center_large], :] = (
                    1-az_small_residual)

            mean_small = np.ma.average(zdr_small, weights=weights_small)
            mean_large = np.ma.average(zdr_large, weights=weights_large)

            hotspot_field[az_i, rng_i] = mean_small - mean_large

    return hotspot_field


def get_hotspot_distancemethod(radar, sweep, z_neg10C,
                               small_radius=1200,
                               large_radius=5000):

    zdr_sweep = radar.get_field(sweep, 'hotspot_differential_reflectivity')
    refl_sweep = radar.get_field(sweep, 'reflectivity')

    mask_25dbz = np.where(refl_sweep > 25, 1, 0)
    expanded_25dbz = common.expand_labels_azran(
        mask_25dbz, 2500, max(radar.range['data']))
    zdr_sweep = np.ma.masked_where(expanded_25dbz == 0, zdr_sweep)

    x, y, z = radar.get_gate_x_y_z(sweep)

    slice_sweep = radar.get_slice(sweep)
    alt_sweep = radar.gate_altitude['data'][slice_sweep]
    alt_mask = alt_sweep > z_neg10C * 1000

    zdr_mask = ~zdr_sweep.mask

    src_mask = (alt_mask & zdr_mask).ravel()

    x_src = x.ravel()[src_mask]
    y_src = y.ravel()[src_mask]
    zdr_src = zdr_sweep.ravel()[src_mask]

    src = np.c_[x_src, y_src]

    tree = spatial.KDTree(src)
    points_small_circle = tree.query_ball_point(src, small_radius, workers=-1)
    points_large_circle = tree.query_ball_point(src, large_radius, workers=-1)

    hotspots = np.zeros(np.shape(zdr_src))

    for i in range(len(src)):
        ind_small = points_small_circle[i]
        ind_large_all = points_large_circle[i]
        ind_large = np.array(ind_large_all)[~np.isin(ind_large_all, ind_small)]

        zdr_small = zdr_src[ind_small]
        zdr_large = zdr_src[ind_large]

        hotspots[i] = np.mean(zdr_small) - np.mean(zdr_large)

    hotspot_flat = np.ma.masked_all(np.shape(zdr_sweep)).ravel()
    hotspot_flat[src_mask] = hotspots

    hotspot_field = hotspot_flat.reshape(np.shape(zdr_sweep))

    return hotspot_field


def _get_indices(len_arr, start, stop):
    # fn to convert your start stop to a wrapped range

    if stop <= start:
        stop += len_arr
    return np.arange(start, stop) % len_arr


def expand_labels_azran(labels, expand_distance, max_range):

    dsize = (2000, 2000)
    center = (dsize[0]/2, dsize[1]/2)
    radius = dsize[0]/2
    dx = max_range / radius

    expand_steps = int(expand_distance/dx)

    cart_2_polar_flag = cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
    labels_cart = cv2.warpPolar(labels,
                                center=center,
                                maxRadius=radius,
                                dsize=dsize,
                                flags=cart_2_polar_flag)

    expanded_cart = expand_labels(labels_cart, expand_steps)

    expand_polar = cv2.warpPolar(expanded_cart,
                                 center=center,
                                 maxRadius=radius,
                                 dsize=(np.shape(labels)[1],
                                        np.shape(labels)[0]),
                                 flags=cv2.INTER_NEAREST)
    return expand_polar
