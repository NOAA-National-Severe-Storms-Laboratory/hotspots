import numpy as np
import xarray as xr


def get_vil_from_azran(zh: xr.DataArray) -> xr.DataArray:
    """
    Compute Vertically Integrated Liquid (VIL) from reflectivity.

    Parameters
    ----------
    zh: xr.DataArray
        Reflectivity in dBZ, given in Az,Range coordinates

    Returns
    -------
    vil: xr.DataArray
        Vertically integrated liquid
    """
    # make sure that volume is ordered by fixed angle in ascending order
    zh = zh.sortby("fixed_angle", ascending=True)

    zh_capped = xr.where(zh > 56, 56, zh)
    zh_capped = xr.where(zh_capped < 0, 0, zh_capped)
    
    num_a = zh.sizes['azimuth']
    num_g = zh.sizes['grange']
    num_s = zh.sizes['fixed_angle']

    vil = np.zeros((num_a,num_g), dtype=float)

    for i in range(num_s-1):
        refl0 = zh_capped.isel(fixed_angle=i)
        refl1 = zh_capped.isel(fixed_angle=i+1)
        refl_dbz_avg = (refl0 + refl1) / 2

        z0 = np.sin(zh.fixed_angle[i]*np.pi/180) * zh.grange
        z1 = np.sin(zh.fixed_angle[i+1]*np.pi/180) * zh.grange
        dh = z1 - z0

        vil += vil_formula(refl_dbz_avg, dh)

    # special treatment for lowest and highest elevation angle
    refl_dbz_low = zh_capped.isel(fixed_angle=0)
    dh0 = np.sin(zh.fixed_angle[0] * np.pi/180) * zh.grange
    vil += vil_formula(refl_dbz_low, dh0)

    # add the top layer
    refl_dbz_high = zh_capped.isel(fixed_angle=-1)
    ztop =  np.sin(zh.fixed_angle[-1] * np.pi/180) * zh.grange
    ztop1 = np.sin(zh.fixed_angle[-2] * np.pi/180) * zh.grange
    dhtop = (ztop - ztop1) / 2
    vil += vil_formula(refl_dbz_high, dhtop)

    vil = xr.where(vil < 0.1, np.nan, vil)

    # name the xr.DataArray so it can be easily merged into xr.Datasets
    vil.name = "VIL"

    # drop the fixed_angle coordinate, due to vertical integration not needed
    vil = vil.drop_vars("fixed_angle")

    return vil
 

def vil_formula(refl_dbz, dh):
    ref_lin_avg = 10 ** (0.1 * refl_dbz)
    ref_lin_avg = xr.where(np.isnan(ref_lin_avg), 0, ref_lin_avg)

    return (3.44*10**-6) * ref_lin_avg**(4/7) * dh
