import numpy as np
import xarray as xr
from scipy import ndimage
import cv2
from skimage.segmentation import expand_labels


def get_filtered_zdr(
        xy_cappi: xr.Dataset,
        refl_threshold: (int, float) = 25,
        cdr_threshold: (int, float) = -10,
        zdr_threshold: (int, float) = 5,
        dilation_size: int = 3,
        refl_fieldname: str = 'reflectivity',
        zdr_fieldname: str = 'differential_reflectivity',
        cdr_fieldname: str = 'circular_depolarization_ratio') -> xr.DataArray:
    """
    Prepare a filtered ZDR field that can be used for calculating the hotspot
    field. This requires Z, ZDR and circular depolarization ratio (CDR).
    For the latter a proxy can be computed from ZDR and RHOHV (compare
    Ryzhkov et al. 2017: https://doi.org/10.1175/JAMC-D-16-0098.1). A
    function for the proxy calculation is included in Py-ART (see
    pyart.retrieve.compute_cdr).

    Parameters
    ----------
    xy_cappi: xr.Dataset
        xarray Dataset containing reflectivity, differential reflectivity and
        circular depolarization ratio.
    refl_threshold: int, float
        Filter all points with Z below this threshold
    zdr_threshold: int, float
        Filter all points with ZDR above this threshold
    cdr_threshold: int, float
        Filter all points with CDR above this threshold
    dilation_size: int
        Dilate the reflectivity mask by this number of grid points. Defaults
        to 3.
    refl_fieldname: str
        name of the reflectivity field in the xarray Dataset.
    zdr_fieldname: str
        name of the differential reflectivity field in the xarray Dataset.
    cdr_fieldname
        name of the circular depolarization ratio field in the xarray Dataset.

    Returns
    -------
    zdr_cut: xr.DataArray
        xarray DataArray holding the filtered ZDR data which can be used to
        compute the hotspot field.

    """
    # (1) reflectivity mask based on dilated reflectivity > 25 dBZ
    refl_mask = expand_labels(
        np.where(xy_cappi[refl_fieldname] > refl_threshold, 1, 0),
        dilation_size)

    # (2) cdr mask to avoid non-meteorological echoes
    cdr_mask = np.where(xy_cappi[cdr_fieldname] < cdr_threshold, 1, 0)

    # (3) high zdr mask
    zdr_mask = np.where(xy_cappi[zdr_fieldname] < zdr_threshold, 1, 0)

    # apply all filters together
    zdr_cut = np.where(
        ((refl_mask == 1) & (cdr_mask == 1) & (zdr_mask == 1)),
        xy_cappi[zdr_fieldname], np.nan)

    # put the filtered Zdr field back into a xarray format
    zdr_cut = xr.DataArray(
        zdr_cut,
        dims=xy_cappi.dims,
        coords=xy_cappi.coords
    )

    return zdr_cut


def apply_hotspot_method(
        xy_cappi: xr.Dataset,
        refl_fieldname: str = "reflectivity",
        zdr_fieldname: str = "zdr_cut",
        x_dim: str = "Lon",
        y_dim: str = "Lat",
        min_hotspot_value: float = 0.2,
        min_hotspot_size: int = 5,
        innerbox: int = 3,
        outerbox: int = 7) -> (xr.DataArray, xr.DataArray):
    """
    Compute the ZDR hotspot field and object identification of updraft
    features based on pre-filtered ZDR values.

    Parameters
    ----------

    xy_cappi : xr.Dataset
        xarray Dataset containing pre-filtered ZDR field and reflectivity
    refl_fieldname: str, optional
        Name of the reflectivity field in provided Xarray
    zdr_fieldname: str, optional
        Name of the ZDR field in provided Xarray
    x_dim: str
        Name of the dimension along the x axis
    y_dim: str
        Name of the dimension along the y axis
    min_hotspot_value: float
        Threshold value for hotspot detection

    Returns
    -------
    hotspot_field: xr.DataArray
        xarray DataArray containing the ZDR hotspot field
    hotspot_features: xr.DataArray
        xarray DataArray containing labeled updraft features

    """

    zdr_cut = xy_cappi[zdr_fieldname]
    dims = [y_dim, x_dim]

    # set missing value to NaN
    if 'missing_value' in xy_cappi.attrs:
        zdr_cut = zdr_cut.where(
            zdr_cut != xy_cappi.attrs['missing_value'], np.nan)

    hotspot_field = xr.apply_ufunc(
        get_hotspot_field, zdr_cut,
        kwargs={'innerbox': innerbox,
                'outerbox': outerbox},
        input_core_dims=[dims],
        output_core_dims=[dims],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[zdr_cut.dtype]).rename('hotspot_field')

    hotspot_features = xr.apply_ufunc(
        get_hotspot_features,
        hotspot_field,
        xy_cappi[refl_fieldname],
        kwargs={'min_hotspot_value': min_hotspot_value,
                'min_size': min_hotspot_size},
        input_core_dims=[dims, dims],
        output_core_dims=[dims],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[zdr_cut.dtype]).rename(
        'hotspot_features')

    return hotspot_field, hotspot_features


def get_hotspot_field(field_2d: (xr.DataArray, np.array),
                      innerbox: int = 3,
                      outerbox: int = 7) -> np.array:
    """
    Calculate the hotspot field for a given 2D array based on median
    values in the inner and outer box. This is the new way of computing
    hotspots which we encourage to use.

    Parameters
    ----------
    field_2d: xr.DataArray, np.array
        Field for which hotspot values should be calculated
    innerbox: int
        Size of the inner target box for the hotspot algorithm. Defaults to 3x3
    outerbox: int
        Size of the outer target box for the hotspot algorithm. Note that
        this excludes the grid points from the inner box. Defaults to 7x7

    Returns
    -------
    hotspot_field: np.array
        Computed hotspot field
    """

    field_nan = field_2d

    # convert to float32 for cv2.medianBlur
    field_nan = np.float32(field_nan)

    # make the outer kernel which excludes the grid points from the inner box
    outerkernel = np.ones((outerbox, outerbox))
    outerkernel[int((outerbox-innerbox)/2):int((innerbox-outerbox)/2),
                int((outerbox-innerbox)/2):int((innerbox-outerbox)/2)] = 0

    # opencv is much faster than scipy.ndimage, but at the moment it cannot
    # deal with grid boxes larger than 5x5
    if innerbox <= 5:
        small_window_median = cv2.medianBlur(field_nan, ksize=innerbox)
    else:
        innerkernel = np.ones((innerbox, innerbox))
        small_window_median = ndimage.generic_filter(
            field_nan, np.nanmedian, footprint=innerkernel)
        small_window_median = np.where(np.isnan(field_nan), np.nan,
                                       small_window_median)

    # we have to resort to scipy.ndimage for the outer box because opencv
    # cannot handle our kernel with the "hole" of the innerbox in the center
    large_window_median = ndimage.generic_filter(
        field_nan, np.nanmedian, footprint=outerkernel)

    hotspot_field = small_window_median - large_window_median

    return hotspot_field


def get_hotspot_features(hotspot_field_2d: (xr.DataArray, np.array),
                         refl_field_2d: (xr.DataArray, np.array),
                         min_hotspot_value: float = 0.20,
                         min_size: (float, int) = 5,
                         min_max_refl: (float, int) = 20) -> np.array:

    """
    Updraft object identification based on ZDR hotspot field alone. First
    identifies all closed objects with hotspot value > min_hotspot_value,
    then eliminates objects not meeting size requirements given by
    min_size, then eliminates objects with a maximum reflectivity below
    min_max_refl. Finally performs a binary closing to connect adjacent
    detections

    Parameters
    ----------
    hotspot_field_2d: xr.Dataset, np.array
        ZDR hotspot field
    refl_field_2d: xr.Dataset, np.array
    min_hotspot_value: float
    min_size: float, int
    min_max_refl: float, int

    Returns
    -------
    hotspot_labels: np.array
        Labels of all filtered hotspot objects
    """

    hotspot_field_2d = np.where(
        np.isnan(hotspot_field_2d), -9999, hotspot_field_2d)

    refl_field_2d = np.where(
        np.isnan(refl_field_2d), -9999, refl_field_2d)

    # start by finding connected areas with high hotspot value
    hotspot_labels, num_features = ndimage.label(
        hotspot_field_2d > min_hotspot_value,
        structure=ndimage.generate_binary_structure(2, 1))

    # next we apply some filtering in multiple steps

    # (1) eliminate small objects based on minimum size
    hotspot_size = ndimage.sum(hotspot_labels > 0,
                               labels=hotspot_labels,
                               index=range(1, num_features+1))

    current_hotspot_ids = np.where(hotspot_size >= min_size)[0] + 1
    # set the label field to 0 where hotspots don't meet the size criterion
    current_hotspot_labels = np.where(
        np.isin(hotspot_labels, current_hotspot_ids), hotspot_labels, 0)

    # (2) minimum reflectivity threshold
    z_max = ndimage.maximum(refl_field_2d,
                            labels=current_hotspot_labels,
                            index=current_hotspot_ids)
    current_hotspot_ids = current_hotspot_ids[z_max > min_max_refl]
    current_hotspot_labels = np.where(
        np.isin(current_hotspot_labels, current_hotspot_ids),
        current_hotspot_labels, 0)

    # (3) binary closing to connect adjacent detections
    merged_hotspots = ndimage.binary_closing(
        current_hotspot_labels > 0,
        structure=ndimage.generate_binary_structure(2, 2))

    hotspot_labels, num_features = ndimage.label(
        merged_hotspots, structure=ndimage.generate_binary_structure(2, 2))

    # TODO: (4) texture threshold? we don't want noisy ZDR in the hotspot

    return hotspot_labels


def save_hotspot_field(hotspot_field, updraft_labels):
    # TODO: save hotspot field and updraft objects to file
    return


def get_hotspot_stats(cappi_xy):
    # TODO: this should output all the hotspots stats, similar to John's
    #  scripts which return the XML files.

    cappi_xy = cappi_xy.sel(Temp=-10)

    hotspot_dict = dict()
    hotspot_labels = np.array(cappi_xy['HotspotID'])
    hotspot_ids = np.unique(hotspot_labels)[1:]

    target_vars = [("Zdr_filtered", "ZDR"),
                   ("DR", "DR"),
                   ("LogCC", "LogCC"),
                   ("Reflectivity", "Z"),
                   ("HotspotField", "HS")]

    for field_name, dict_name in target_vars:

        field_data = np.array(cappi_xy[field_name])
        hotspot_dict['Hotspotneg10C_Max'+dict_name] = ndimage.maximum(
            field_data, labels=hotspot_labels, index=hotspot_ids)
        hotspot_dict['Hotspotneg10C_Min'+dict_name] = ndimage.minimum(
            np.where(field_data == -99900, 99900, field_data),
            labels=hotspot_labels, index=hotspot_ids)
        hotspot_dict['Hotspotneg10C_Med'+dict_name] = ndimage.median(
            np.where(field_data == -99900, np.nan, field_data),
            labels=hotspot_labels, index=hotspot_ids)
        hotspot_dict['Hotspotneg10C_Std'+dict_name] = (
            ndimage.labeled_comprehension(
            np.where(field_data == -99900, np.nan, field_data),
            hotspot_labels, hotspot_ids, np.nanstd, float, 0))

    return

