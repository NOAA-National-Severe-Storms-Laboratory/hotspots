#!/usr/bin/env python
# coding: utf-8

"""
Created on October 16, 2024

@authors: John Krause (JK), Vinzent Klaus (VK)

@license: BSD-3-Clause

@copyright Copyright 2024 John Krause

Update History:
    Version 0:
        -JK Initial version, copy of c++ code base
"""
import numpy as np
import xarray as xr
from scipy import ndimage

from stormcell.StormcellXY import LocXY, Stormcell


def create_stormcell_list(labels: xr.DataArray,
                          data: xr.DataArray) -> list:
    """
    Get a list of stormcell objects from the label and data arrays.

    Parameters
    ----------
    labels: xr.DataArray
        Integer labels of the hotspot objects. All locations with the same
        integer value are part of the same object
    data: xr.DataArray
        Data from which stormcell center locations and maximum value are
        determined

    Returns
    -------
    object_list: list
        List of stormcell objects.
    """
    # usually -1 or 0 does not indicate a label, therefore we start with the
    # second entry
    ids = np.unique(labels)[1:]

    #from the shift in the object lables
    #Nan values in the lables interfere with the codebase
    ids = ids[~np.isnan(ids)]

    # fill NaN values that interfere with the maximum calculation
    data = data.fillna(-9999)

    # we define the center by the maximum value
    obj_center = ndimage.maximum_position(data, labels=labels, index=ids)
    # get the maximum value
    obj_max = ndimage.maximum(data, labels=labels, index=ids)
    # we also get the hotspot size here
    obj_size = ndimage.sum(labels > 0, labels=labels, index=ids)

    object_list = []
    # loop through each object
    for id, center, max_val, size in zip(ids, obj_center, obj_max, obj_size):
        y_ind, x_ind = np.where(labels == id)
        y_center_ind, x_center_ind = (center[0], center[1])

        y, x  = (data.y.values[y_ind], data.x.values[x_ind])
        y_center, x_center = (data.y.values[y_center_ind],
                              data.x.values[x_center_ind])

        loc = [LocXY(xi, yi) for xi, yi in zip(x, y)]
        object_list.append(
            Stormcell(id, loc, LocXY(x_center, y_center), max_val, size))

    return object_list


def sort_object_list(object_list, key):
    if key == 'size':
        return sorted(object_list, key=lambda x: x.size)
    elif key == 'max_value':
        return sorted(object_list, key=lambda x: x.max_value)
    else:
        raise ValueError('invalid key')


def print_stormcell_list(celllist):
    for o in celllist:
        print(
            f"{o.id} size: {o.size} max: {o.max_value} "
            f"loc: {o.center_location.x},{o.center_location.y}")


def find_stormcell_index(id, celllist):
    for i in range(len(celllist)):
        #print("index: %d cellist: %d id: %d" % (i, celllist[i].id, id))
        if celllist[i].id == id:
            return i

def rank_stormcell_list(celllist):
    ranked_by_size = sort_object_list(celllist, "size")
    #reverse it so the largest is first
    ranked_by_size.reverse()

    ranked_by_strength = sort_object_list(celllist, "max_value")
    #reverse it so the strongest is first
    ranked_by_strength.reverse()

    #combine the ranks as a formulae with rank = stength_rank + 0.4*size_rank
    index = 0
    final_ranks = {}
    for obj in ranked_by_strength:
        rank = index + 0.4*find_stormcell_index(obj.id, ranked_by_size)
        #make list
        final_ranks[obj.id]=rank
        index +=1
  
    final_ranks = dict(sorted(final_ranks.items(), key=lambda item: item[1]))
   
    return final_ranks
