import numpy as np
import pyart

from prepro.nexrad import preprocessor_norpg_xarray, prune_nexrad
from cappi.make_vv import get_indexed_vv
from cappi.azran import indexed_vv_to_cappi_amsl
from cappi.xy_cappi import azran_to_xy
from mcit.vil import get_vil_from_azran
from mcit.mcit_objects import mcit_objects, mcit_trim
from hotspots import detection


def pipeline_to_cart(filename, target_cappi_height):
    """
    A wrapper to get pseudoCAPPIs in x/y coordinates from a NEXRAD volume scan.

    Parameters
    ----------
    filename
    target_cappi_height

    Returns
    -------

    """

    # We choose to work on a target grid that is 400x400 at 1km resolution
    x_trg = np.arange(-200, 201)
    y_trg = np.arange(-200, 201)

    radar_vol = pyart.io.read_nexrad_archive(filename)

    # we want to use the surveillance data and only a volumes worth
    # we don't want the doppler cut data nor the extra sweeps from
    # SAILS or Meso-Sails
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

    # add the dualpol processed fields to the pyart object
    prepro_dr = pyart.retrieve.compute_cdr(processed_vol,
                                           rhohv_field='prepro_cc',
                                           zdr_field='prepro_zdr')
    processed_vol.add_field('prepro_dr', prepro_dr)

    indexed_vv = get_indexed_vv(processed_vol)
    vil = get_vil_from_azran(indexed_vv['reflectivity'])
    indexed_vv['vil'] = vil

    # make azran CAPPI interpolation at the specified height from the indexed
    # virtual volume input at the top. this changes for each case.
    azran_ds = indexed_vv_to_cappi_amsl(indexed_vv, target_cappi_height)

    # make xy interpolation and convert to xarray
    xycappi = azran_to_xy(azran_ds, x_trg, y_trg, grid_spacing_meters=1000.0, max_dist=2)

    return xycappi


def pipeline_hotspots(xy_cappi):

    zdrcut_proc = detection.get_filtered_zdr(
        xy_cappi, refl_fieldname='prepro_zh', zdr_fieldname='prepro_zdr',
        cdr_fieldname='prepro_dr')

    xy_cappi["zdr_cut_proc"] = zdrcut_proc
    hotspot_field, hotspot_features = (
        detection.apply_hotspot_method(
        xy_cappi, x_dim="x", y_dim="y", refl_fieldname='prepro_zh',
        zdr_fieldname='zdr_cut_proc'))

    # set missing to -1
    hotspot_features = hotspot_features.where(
        hotspot_features > 0, -1)

    # add features and hotspot field to xy_cappi
    xy_cappi["hotspot_features_proc"] = hotspot_features.astype(int)
    xy_cappi["hotspot_field_proc"] = hotspot_field

    return hotspot_field, hotspot_features


def pipeline_mcit(xy_cappi):

    mcit_raw = mcit_objects(xy_cappi['vil'], 2.0, 1.25)
    mcit_final = mcit_trim(mcit_raw, xy_cappi['vil'], 5.0, 50)

    return mcit_final
