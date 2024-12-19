import numpy as np
import pyart
import wradlib
import xarray as xr


def raw_3dvol_to_xy(ds_vol: xr.Dataset,
                    x_trg: np.array,
                    y_trg: np.array,
                    grid_spacing_meters: (int, float),
                    z_trg: (int, float)) -> xr.Dataset:
    """
    Interpolate raw volume data to xy CAPPI at specified altitude.

    Parameters
    ----------
    ds_vol: xr.Dataset
        xarray Dataset holding xy-referenced radar volume data
    x_trg: np.array
        target x coordinates
    y_trg: np.array
        target y coordinates
    grid_spacing_meters (int, float)
        grid_spacing in meters for target grid, asssume the same distance between
        x and y grid points (square grid)
    z_trg: (int, float)
        target height given in m AMSL

    Returns
    -------
    ds_grid: xr.Dataset
        xarray Dataset holding interpolated xy CAPPI data
    """

    # we would like to consider only source points target_height +/- cutoff
    # height. This is just a guess for a good cutoff
    cutoff_height = z_trg / 3

    # x_rad = np.repeat(ds_vol.x.data[np.newaxis, :, :],
    #                   len(ds_vol.elevation), axis=0)
    # y_rad = np.repeat(ds_vol.y.data[np.newaxis, :, :],
    #                   len(ds_vol.elevation), axis=0)
    # z_rad = np.repeat(ds_vol.z.data[:, np.newaxis, :],
    #                   len(ds_vol.azimuth), axis=1) + ds_vol.radar_altitude

    # get the source coordinates from the raw volume xarray
    x_rad = ds_vol.x.data
    y_rad = ds_vol.y.data
    z_rad = ds_vol.z.data + ds_vol.radar_altitude

    # make a target point meshgrid
    xx_trg, yy_trg, zz_trg = np.meshgrid(x_trg, y_trg, z_trg)

    # get the interpolator for each target point
    ip, valid_pts = get_ipolator_3d(xx_trg, yy_trg, zz_trg,
                                    x_rad, y_rad, z_rad,
                                    cutoff_height=cutoff_height)
    grid_shape = (len(y_trg), len(x_trg))

    # apply the function to the xarray volume data in parallel
    ds_grid = xr.apply_ufunc(
        ipol_3d, ds_vol,
        input_core_dims=[["fixed_angle", "azimuth", "range"]],
        output_core_dims=[["y", "x"]],
        dask_gufunc_kwargs=dict(output_sizes={
            "y": grid_shape[0],
            "x": grid_shape[1]}),
        output_dtypes=[float],
        dask="parallelized",
        kwargs=dict(interpolator=ip,
                    valid_mask=valid_pts,
                    grid_shape=grid_shape))

    #
    #convert x and y from index values to meters
    #
    xm_trg = x_trg*grid_spacing_meters
    ym_trg = y_trg*grid_spacing_meters

    # for adding lon, lat to xarray
    lon_trg, lat_trg = pyart.core.cartesian_to_geographic_aeqd(
        xm_trg, ym_trg, radar_lon, radar_lat)
    # for adding lon, lat to xarray
    #lon_trg, lat_trg = pyart.core.cartesian_to_geographic_aeqd(
    #    x_trg, y_trg, ds_vol.radar_longitude, ds_vol.radar_latitude, R=6370.9970)

    ds_grid = ds_grid.assign_coords(
        {'x': x_trg,
         'y': y_trg,
         'lon': ('x', lon_trg),
         'lat': ('y', lat_trg)}
    )

    # assign the radar metadata from the volume data
    ds_grid = ds_grid.assign_attrs(ds_vol.attrs)

    return ds_grid


def azran_to_xy(ds_azran: xr.Dataset,
                x_trg: np.array,
                y_trg: np.array,
                grid_spacing_meters: (int, float) = 1000, 
                max_dist: (int, float) = 500,
                radar_lon: (float, None) = None,
                radar_lat: (float, None) = None) -> xr.Dataset:
    """
    get a 2D xy CAPPI from a 2D AzRan CAPPI.

    Parameters
    ----------
    ds_azran: xr.Dataset
        xarray Dataset holding AzRan CAPPI data
    x_trg: np.array
        target x coordinates
    y_trg: np.array
        target y coordinates
    
    grid_spacing_meters (int, float, optional)
        grid_spacing in meters for target grid, asssume the same distance between
        x and y grid points (square grid)
    max_dist: (int, float, optional)
        Maximum allowed distance between target and source points. The default
        of 500 should obviously be used with coordinates in meters. Change if
        using kilometers.
    radar_lat: float, optional
        necessary if radar_latitude is not an attribute of the input
        dataset/dataarray
    radar_lon: float, optional
        necessary if radar_longitude is not an attribute of the input
        dataset/dataarray

    Returns
    -------
    ds_grid: xr.Dataset
        xarray Dataset holding 2D xy CAPPI data

    """

    if radar_lon is None and "radar_longitude" in ds_azran.attrs:
        radar_lon = ds_azran.radar_longitude
    else:
        raise ValueError("radar longitude needs to be specified")
    if radar_lat is None and "radar_latitude" in ds_azran.attrs:
        radar_lat = ds_azran.radar_latitude
    else:
        raise ValueError("radar latitude needs to be specified")

    # get the source coordinates from the raw volume xarray
    x_rad = ds_azran.x.data
    y_rad = ds_azran.y.data

    # make a target meshgrid
    xx_trg, yy_trg = np.meshgrid(x_trg, y_trg)

    # get the interpolator for each target point
    ip = get_ipolator_2d(xx_trg, yy_trg, x_rad, y_rad)

    # hint for John - take a look at these outputs:
    # print(ip.dists) --> distances from target points to closest 4 points
    # print(ip.ix) --> indices of 4 closest points in the src data to the target point

    # target grid shape, needed for cor
    grid_shape = (len(y_trg), len(x_trg))

    # apply the function to the xarray volume data in parallel
    ds_grid = xr.apply_ufunc(
        ipol_2d, ds_azran,
        input_core_dims=[["azimuth", "grange"]],
        output_core_dims=[["y", "x"]],
        dask_gufunc_kwargs=dict(output_sizes={
            "y": grid_shape[0],
            "x": grid_shape[1]}),
        output_dtypes=[float],
        dask="parallelized",
        kwargs=dict(interpolator=ip, grid_shape=grid_shape, maxdist=max_dist))
    #
    #convert x and y from index values to meters
    #
    xm_trg = x_trg*grid_spacing_meters
    ym_trg = y_trg*grid_spacing_meters

    # for adding lon, lat to xarray
    lon_trg, lat_trg = pyart.core.cartesian_to_geographic_aeqd(
        xm_trg, ym_trg, radar_lon, radar_lat)

    ds_grid = ds_grid.assign_coords(
        {'x': x_trg,
         'y': y_trg,
         'lon': ('x', lon_trg),
         'lat': ('y', lat_trg)}
    )

    # assign the radar metadata from the volume data
    ds_grid = ds_grid.assign_attrs(ds_azran.attrs)

    return ds_grid


def get_ipolator_3d(x_trg, y_trg, z_trg,
                    x_rad, y_rad, z_rad,
                    num_nearest=4, p=2, cutoff_height=500):
    
    # yields the 3D inverse distance interpolator object that relates the radar
    # source coordinates to the grid target coordinates

    # prepare grid - we get an array of x,y,z coordinates
    grid_xyz = np.vstack(
        (x_trg.ravel(),
         y_trg.ravel(),
         z_trg.ravel())).transpose()

    # valid source points are between the target_height and +/- the cutoff
    # height
    valid_pts_radar = ((z_rad > z_trg[0, 0, 0] - cutoff_height) &
                       (z_rad < z_trg[0, 0, 0] + cutoff_height))
    x_rad, y_rad, z_rad = (x_rad[valid_pts_radar], y_rad[valid_pts_radar],
                           z_rad[valid_pts_radar])

    # prepare the source points in the format the interpolator needs (tuples
    # of x,y,z)
    rad_xyz = np.vstack(
        (x_rad.ravel(),
         y_rad.ravel(),
         z_rad.ravel())).transpose()

    # get the interpolator method that matches each source point with each
    # grid point
    ip = wradlib.ipol.Idw(rad_xyz, grid_xyz, nnearest=num_nearest,
                          remove_missing=True, p=p)

    return ip, valid_pts_radar


def ipol_3d(radar_field, interpolator=None, valid_mask=None,
            grid_shape=None, maxdist=2000):

    # apply the interpolation to a 3d source array (e.g., reflectivity) in
    # radar coordinates
    vals = radar_field[valid_mask].ravel()

    # apply the interpolator method to the field values. The interpolator
    # contains the context between source and target coordinates
    ip_field = interpolator(vals, maxdist=maxdist)

    # the result needs to be reshaped to 2D
    gridded_field = ip_field.reshape(grid_shape)

    return gridded_field


def get_ipolator_2d(x_trg, y_trg, x_rad, y_rad, num_nearest=4, p=2,
                    remove_missing=True):

    # yields the 2D inverse distance interpolator object that relates the radar
    # source coordinates to the grid target coordinates

    # get the target coordinates as tuples of x,y
    grid_xy = np.vstack(
        (x_trg.ravel(),
         y_trg.ravel())).transpose()

    # get the source coordinates as tuples of x,y
    rad_xy = np.vstack(
        (x_rad.ravel(),
         y_rad.ravel())).transpose()

    # inverse distance weighting; nnearest is the number of closest points
    # from the source coordinates that go into the interpolation. P is the
    # power of the inverse distance weighting
    ip = wradlib.ipol.Idw(rad_xy, grid_xy, nnearest=num_nearest,
                          remove_missing=remove_missing, p=p)

    return ip


def ipol_2d(data, interpolator=None,
            grid_shape=None, maxdist=400):

    # interpolate a 2d field from source to target coordinates. The
    # interpolator function is required for mapping the points

    # all field values from 2d to 1d because that's the input format needed
    # by the interpolator
    vals = data.ravel()

    # interpolate values from source to target coordinates with a maximum
    # distance given by maxdist
    ip_field = interpolator(vals, maxdist=maxdist)

    # reshape back to 2d field
    gridded_field = ip_field.reshape(grid_shape)

    return gridded_field
