# -*- coding: utf-8 -*-
"""
Created on August 21 2024

@author: John Krause (JK)

@license: BSD-3-Clause

@copyright Copyright 2024 John Krause

Update History:
    Errors, concerns, proposed modifications, can be sent to:
        John.Krause@noaa.gov, please help us improve this code

    Version 0:
        -JK Initial version, needs to be tested against the c++ version
            
"""

import numpy as np
import xarray as xr
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def mcit_objects(vil_xy: xr.DataArray,
                 valley_depth: float = 2.0,
                 min_vil_value: float = 1.5) -> xr.DataArray:
    """
    Processes the VIL data into mcit style objects using watershed and
    recombing the data based on a valley_depth. 

    Parameters:
    vil_xy (xr.DataArray)
        Vertically Integrated Liquid in x,y coordinates.
    valley_depth (float)
        The minimum mdepth of the valley. Default is 2.0
    min_value (float):
        The minimum value of VIL. (in VIL units) Default is 1.5

    Returns:
    trimmed_mcit_objects (xr.DataArray):
        Final MCIT objects after recombination
    
    Reference:

        MCIT (multi-cell identification and tracking):
        Jiaxi Hu, Daniel Rosenfeld, Dusan Zrnic, Earle Williams, Pengfei Zhang,
            Jeffrey C. Snyder, Alexander Ryzhkov, Eyal Hashimshoni, 
            Renyi Zhang, Richard Weitz,
        Tracking and characterization of convective cells through their 
            maturation into stratiform storm elements using polarimetric 
            radar and lightning detection,
        Atmospheric Research,
        Volume 226,
        2019,
        Pages 192-207,
        ISSN 0169-8095,
        https://doi.org/10.1016/j.atmosres.2019.04.015.

        Notes: 
           We don't remove small cells (<5 gates) before 
           applying the recombination of watershed regions.

           We don't have any distance or location limits on the 
           centers of watershed regions that are recombined. 
           
           Therefore, any two adjacent regions can be combined if
           the saddle point (valley_depth) criteria is meet

           We added a min_VIL_value threshold to cut the data into
           smaller objects
    """

    if not isinstance(vil_xy, xr.DataArray):
        raise ValueError("VIL must be provided as xr.DataArray")

    #Compute the linear version of VIL

    #This transform changes the data in an important way
    #lowering the peaks and stretching out the tails. The VIL data
    #becomes less spiky. We think this allows the watershed to do
    #a better job
    linear_vil = 10.0 * np.log10(vil_xy.data)

    #min value must be in linear units as well
    min_value = 10.0 * np.log10(min_vil_value)

    #We smooth the vil to make the objects more contiguous using a 3x3 box
    # average filter
    kernel = np.ones((3, 3)) / 9
    smlin_vil = cv2.filter2D(linear_vil, ddepth=-1, kernel=kernel)
    #remove data that is below threshold
    #smlin_vil = np.where(smlin_vil<=min_value, np.nan, smlin_vil)
   
    #We follow the techinque described in:
    #https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    #https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    #  We use a similar technique to the above without all the prework. 
    #  We can't know starting locations so we use all values not equal to the missing value as or unknown region 
    
    #set the background data (white in image above) to 0 and the foreground data (non-white) to 1
    valid = np.where(smlin_vil>=min_value, 1, 0)
    
    #the c++ code uses the vincent-sollie method and c library which is slightly different from 
    #the openCV interpretation 
    #Open CV requires "starting points" to begin the watershed. We generate those by finding the local
    #maximums in the data.This will oversegment the image. We will join segments in a later step
    #a footprint of 5x5 is a 25km-sq local max distance
    #FIXME does this match the description above? -JK
    coords = peak_local_max(smlin_vil, footprint=np.ones((3, 3)),
                            threshold_abs=1.0, min_distance=7)
    mask = np.zeros(smlin_vil.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels_arr = watershed(-smlin_vil, markers, mask=valid)

    #note that this happens after the watershed is complete
    #note the min_value must be in linVIL which is converted 
    #at the top of the subroutine
    #note: 'Not an object' locations are labeled -1 
    labels_arr = np.where(smlin_vil >= min_value, labels_arr, -1).astype(int)

    #Using the trimmed labels we want to identify watersheds that are adjacent 
    #to one another. We do this by walking the image pixel by pixel. When we 
    #find adjacent watersheds we determine if those watersheds should be 
    #combined into a single watershed based on the "valley depth". This 
    #parameter compares the highest value of the data in each watershed to the
    #"valley" location at the pixel. If the difference between the max data 
    # and the  "valley" depth isn't low enough then the watersheds are combined.

    vildata = vil_xy.values

    #exit after no move ids are combined
    # dummy variable for the check below
    old_number_of_ids = 9999
    removed_ids = [] #once an id is removed it stays removed

    # iterate as long as we find new connections
    while len(np.unique(labels_arr)[1:]) < old_number_of_ids:
        ids = np.unique(labels_arr)[1:]
        old_number_of_ids = len(ids)
        max_vals = ndi.maximum(vildata, labels=labels_arr, index=ids)

        #Lets create a dict of ids/max_vil values for ease
        #of use and understanding
        idx = dict(zip(ids, max_vals))
        # Sort by values in descending order
        idx_by_max = dict(sorted(idx.items(), key=lambda item: item[1], reverse=True))

        #idx_sort = np.argsort(max_vals)
        #ids_by_maximum = ids[idx_sort[::-1]]
        #once an id is removed it stays removed
        #removed_ids = []

        for target_id in idx_by_max.keys():
            if target_id in removed_ids:
                continue

            neighboring_labels, border_mask = get_neighboring_labels(
                labels_arr, target_id)

            if neighboring_labels.size == 0:
                continue

            for neighbor_id in neighboring_labels:
                #we removed the -1 values in get_neighboring_labels
                #if neighbor_id == -1:
                #    continue

                # Find the border between label and neighbor
                border = (border_mask & (labels_arr == neighbor_id))

                # our reference the weakest peak that is being checked
                #peak_val = np.min(
                #    [max_vals[ids == neighbor_id][0],
                #     max_vals[ids == target_id][0]])
                peak_val = np.min(
                    [idx_by_max[neighbor_id],
                    idx_by_max[target_id]]
                                 )

                # Calculate the maximum value along the border in the data
                max_val_along_watershed = np.max(vildata[border])

                # if the maximum value is close enough to the smaller peak VIL,
                # then we combine the objects
                if (peak_val - max_val_along_watershed) < valley_depth:
                    #print("Combine: %d and %d"%(target_id, neighbor_id))
                    labels_arr[labels_arr == neighbor_id] = target_id
                    removed_ids.append(neighbor_id)

    mcit_objects = xr.DataArray(
        labels_arr,
        dims=vil_xy.dims,
        coords=vil_xy.coords,
        name='Raw MCIT')

    return mcit_objects


def mcit_trim(mcit_objects: xr.DataArray,
              vil: xr.DataArray,
              min_strength: (int, float) = 5.0,
              min_size: int = 25) -> xr.DataArray:

    """
    Process the MCIT_objects data into MCIT_objects by combing or removing
    small objects and/or weak objects.

    Parameters
    ----------
    mcit_objects: xr.DataArray
        labeled MCIT objects
    vil: xr.DataArray
        VIL field on the same grid as mcit_objects
    min_strength: (int, float)
        Peak vil required for an object not to be removed or combined with its
        neighbor. Set to -1 to skip this process.
    min_size: int
        Objects smaller than min_size will be removed or combined with their
        neighbor. Set to -1 to skip this process.
    """

    if not isinstance(mcit_objects, xr.DataArray):
        raise ValueError("mcit objects must be provided as xr.DataArray")
    if not isinstance(vil, xr.DataArray):
        raise ValueError("VIL must be provided as DataArray")

    labels_arr = mcit_objects.values

    ids = np.unique(labels_arr)[1:]
    max_vils = ndi.maximum(vil, labels=labels_arr, index=ids)
    obj_sizes = ndi.sum(labels_arr > 0, labels=labels_arr, index=ids)

    # dict with maximum VIL of all labelled cells
    id_maxvil = {test_id: max_vils[i] for i, test_id in enumerate(ids)}
    # dict with size of all labelled cells
    id_size = {test_id: int(obj_sizes[i]) for i, test_id in enumerate(ids)}

    ids_to_remove = []
    # now remove weak MCIT cells unless they have stronger neighbors
    for target_id in id_maxvil.keys():
        if id_maxvil[target_id] < min_strength:
            # mark the id for removal later
            ids_to_remove.append(target_id)

            neighboring_ids, bordermask = get_neighboring_labels(
                labels_arr, target_id)

            # all neighbor labels except for -1, which is background
            neighboring_ids = neighboring_ids[~(neighboring_ids == -1)]

            if len(neighboring_ids) == 0:
                labels_arr = np.where(labels_arr == target_id, -1, labels_arr)
            else:
                neighbor_dict = {i: id_maxvil[i] for i in neighboring_ids}
                sorted_neighbors = sorted(
                    neighbor_dict.items(), key=lambda x: x[1])
                assigned_id = sorted_neighbors[-1][0]
                labels_arr = np.where(labels_arr == target_id, assigned_id,
                                      labels_arr)

    # now delete the removed labels from the dictionary
    for target_id in ids_to_remove:
        del id_maxvil[target_id]
        del id_size[target_id]

    ids_to_remove = []
    for target_id in id_size.keys():
        if id_size[target_id] < min_size:
            # mark the id for removal later
            ids_to_remove.append(target_id)

            neighboring_ids, bordermask = get_neighboring_labels(
                labels_arr, target_id)
            # all neighbor labels except for -1, which is background
            neighboring_ids = neighboring_ids[~(neighboring_ids == -1)]

            if len(neighboring_ids) == 0:
                labels_arr = np.where(labels_arr == target_id, -1, labels_arr)
            else:
                neighbor_dict = {i: id_maxvil[i] for i in neighboring_ids}
                sorted_neighbors = sorted(
                    neighbor_dict.items(), key=lambda x: x[1])
                assigned_id = sorted_neighbors[-1][0]
                labels_arr = np.where(labels_arr == target_id, assigned_id,
                                      labels_arr)

    # now delete the removed labels from the dictionary
    for target_id in ids_to_remove:
        del id_maxvil[target_id]
        del id_size[target_id]

    trimmed_mcit = xr.DataArray(
        labels_arr,
        dims=mcit_objects.dims,
        coords=mcit_objects.coords,
        name='MCIT')

    return trimmed_mcit


def get_neighboring_labels(labels_arr: np.ndarray,
                           target_id: int):
    structure = ndi.generate_binary_structure(2, 2)

    # Create a mask for the current region
    region_mask = labels_arr == target_id

    # Dilate the region to find neighbors
    border_mask = ndi.binary_dilation(
        region_mask, structure=structure) & (labels_arr != target_id)

    # Find neighboring labels
    neighboring_labels = np.unique(labels_arr[border_mask])

    #remove the -1 values
    neighboring_labels = neighboring_labels[~(neighboring_labels == -1)]

    return neighboring_labels, border_mask
