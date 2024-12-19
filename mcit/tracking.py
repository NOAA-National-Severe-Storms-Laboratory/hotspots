# -*- coding: utf-8 -*-
"""
Created on October 31 2024

@author: John Krause (JK)

@license: BSD-3-Clause

@copyright Copyright 2024 John Krause

Update History:
    Errors, concerns, proposed modifications, can be sent to:
        John.Krause@noaa.gov, please help us improve this code

    Version 0:
        -JK Initial version, needs to be tested against the c++ version
            
"""

#from pathlib import Path
import xarray as xr
from stormcell.stormcell_helpers import *


def compute_shiftscore(std1: float, mean1: float, thresh_data1: xr.DataArray, std2: float, mean2: float, shifted_data: xr.DataArray) -> float:

    """
    Parameters
    ----------
    std1: float 
        standard deviation of the thresh_data1 field
    mean1: float
        mean of the thresh_data1 field
    thresh_data1: xr.DataArray
        data usually from time t-1

    std2: float 
        standard deviation of the thresh_data2 field
    mean2: float
        mean of the thresh_data2 field
    thresh_data2: xr.DataArray
        data usually from time t

    Returns
    ------
    score: float 
       a measure of how aligned the data is between t-1 and 1

    """
    # Tested same by JK on 11/11/2024
    #
    #Sorry Vincent, I only know the C method here....
    # This is incrediably slow.....
    #   Maybe replace with a set fo where methods?
    #
    #There are many scoring methods....But they involve intercomparing
    #
    score = 0.0
    slow = False
    if slow == True:
        for x in range(thresh_data1.shape[0]):
            for y in range(thresh_data1.shape[1]):
                val1 = thresh_data1[x,y]
                val2 = shifted_data[x,y]
                if val1 > mean1 and val2 > mean2:
                    score = score + 1.0
                    if val1 > (mean1 + std1) and val2 > (mean2 + std2):
                        score = score + 1.0 #adds another +1
                        if val1 > (mean1 + 2*std1) and val2 > (mean2 + 2*std2):
                            score = score + 1.0 #adds a 3rd +1
    else:
       td_mean = xr.where(thresh_data1 > mean1, thresh_data1, np.nan) 
       sd_mean = xr.where(shifted_data > mean2, shifted_data, np.nan) 
       comb_mean = xr.where(~np.isnan(td_mean), sd_mean, np.nan)

       score = score + comb_mean.count()

       td_mean = xr.where(thresh_data1 > mean1+std1, thresh_data1, np.nan) 
       sd_mean = xr.where(shifted_data > mean2+std2, shifted_data, np.nan) 
       comb_mean = xr.where(~np.isnan(td_mean), sd_mean, np.nan)
            
       score = score + comb_mean.count()
    
       td_mean = xr.where(thresh_data1 > mean1+2*std1, thresh_data1, np.nan) 
       sd_mean = xr.where(shifted_data > mean2+2*std2, shifted_data, np.nan) 
       comb_mean = xr.where(~np.isnan(td_mean), sd_mean, np.nan)
            
       score = score + comb_mean.count()

    return score 

def computeStormCellOverlap(obj2, obj1):
    count=0
    for loc2 in obj2.locs:
        for loc1 in obj1.locs:
            if loc2 == loc1:
                count +=1
    if count == 0:
        return 0.0
    else:
        return count/len(obj2.locs) * 100.0 #overlap as a percentage

def overlap_tracking_basic( data1_xy: xr.DataArray,
                            obj1_xy: xr.DataArray,
                            data2_xy: xr.DataArray,
                            obj2_xy: xr.DataArray) -> (xr.DataArray):
    """
    Match objects from time t-1 (obj1) and their underlying data (data1) to 
    objects at time t (obj2) and their underlying data(data2). Return a new
    set of obects that carries forward the obj ids that are matched from obj1. 

    parameters:
    data1_xy (xr.DataArray)
       The xy data that corresponds to the obj1_xy object ids. (timestep is t-1)
    obj1_xy (xr.DataArray)
       The xy data that defines the objects such that data points with the 
       same value are part of the same object. values <= 0 are considered 
       missing or no object or no data (timestep is t-1)
    data2_xy (xr.DataArray)
       The xy data that corresponds to the obj1_xy object ids. (timestep is t)
    obj2_xy (xr.DataArray)
       The xy data that defines the objects such that data points with the 
       same value are part of the same object. values <= 0 are considered 
       missing or no object or no data (timestep is t)

    Note that timestep t is more recent than timestep t-1

    """

    """
        Compute the optimal shift vector:
           There are many ways to compute the movement of storm objects between t and t-1
           

    """
    #
    #find the max id value in the old data
    #
    max_value_id = obj1_xy.max()+1
    if max_value_id < 1:
            max_value_id = 1

    #
    #Threshold the data arrays based on valid objects. We only want to compute
    #the cost function based on valid object locations
    thresh_data1 = xr.where(obj1_xy > 0, data1_xy, np.nan)
    thresh_data2 = xr.where(obj2_xy > 0, data2_xy, np.nan)

    std1 = thresh_data1.std()
    mean1 = thresh_data1.mean()

    std2 = thresh_data2.std()
    mean2 = thresh_data2.mean()

    #
    #Compute the value of each shift
    #  assume the maximum value is the best
    #  value assigned is:
    #       +1 for any location where shifted data and data1 are both > mean value 
    #       +2 for any location where shifted data > mean1 + std1 and data2 > mean2 + std2
    #       +3 for any location where shifted data > mean1 + 2*std1 and data2 > mean2 + 2*std2
    #
    print("mean1: %f std1: %f mean2: %f std2: %f \n" % ( mean1, std1, mean2, std2))
    max_score = 0.0
    if  np.isnan(std1) or np.isnan(mean1) or np.isnan(std2) or np.isnan(mean2):
        x_shift = 0
        y_shift = 0
    else:
        #sanity please
        #
        #FIXME maybe the shift is by reference and we need a copy? (no...)
        #
        for x in range(-8,8):
            for y in range(-8,8):
                #copy_data2 = thresh_data2.copy()
                shifted_data = thresh_data2.shift( {'x': x, 'y': y} )
                score = compute_shiftscore(std1, mean1, thresh_data1, std2, mean2, shifted_data)
                if score > max_score:
                    x_shift = x
                    y_shift = y
                    max_score = score

                #print("%d %d %f \n" %(x,y,score))
    
    print("Final shift: %d %d %f\n" %(x_shift, y_shift, max_score))
    #
    #apply the shift that we found
    #
    shifted_data2 = thresh_data2.shift( {'x':x_shift, 'y':y_shift} )
    shifted_obj2 = obj2_xy.shift( {'x':x_shift, 'y':y_shift} )
    xr.where( np.isnan(shifted_obj2), -1, shifted_obj2)
    #
    #Create a list of the objects, from the data and object dataarrays
    #
    obj1_list = create_stormcell_list(obj1_xy, data1_xy)
    obj2_list = create_stormcell_list(obj2_xy, data2_xy)
    #
    #Create and rank objects
    #
    obj1_ranks = rank_stormcell_list(obj1_list)
    #print("Obj1: ")
    #print_stormcell_list(obj1_list)

    obj2_ranks = rank_stormcell_list(obj2_list)
    #print("\nObj2: ")
    #print_stormcell_list(obj2_list)
    #
    #Now we do the assignment part of the tracking code
    # Starting with the highest ranked obj2 we try and find an acceptable obj1.
    # if we do we assign the id from obj1 to obj2.
    #   (note that an obj1 id is only used once)
    #
    # We look for objects that have a 75% overlap between obj2 and obj1
    #    Then we lower this threshold to 50%
    #       Then we lower this threshold to 25%
    #         Then we give up and assign the obj2 a new ID
    #
    #Note the obj locations we want are those from the shifted data
    #
    #important!
    #make the obj2_list is from the shifted data so that the compare
    #happens at time t-1
    obj2_list = create_stormcell_list(shifted_obj2, shifted_data2)
    
    overlap_thresholds = [75.0, 50.0, 25.0]

    #setup found_dic[obj2_id] = obj1_id
    found_dic = {}

    for overlap_thresh in overlap_thresholds:
        for id2 in obj2_ranks:
            if not id2 in found_dic:
                for id1 in obj1_ranks:
                    if not id1 in found_dic.values():
                        obj2_o = obj2_list[find_stormcell_index(id2, obj2_list)]
                        obj1_o = obj1_list[find_stormcell_index(id1, obj1_list)]
                        overlap = computeStormCellOverlap(obj2_o, obj1_o)
                        print("Thresh: %f overlap between %d and %d is %f"%(overlap_thresh, obj2_o.id, obj1_o.id, overlap))
                        if overlap >= overlap_thresh:
                            #add some sanity checks
                            #size_check = len(obj2_o.locs)/len(obj1_o.locs)
                            #str_check = obj2_o.max_value/obj1_o.max_value
                            #if size_check >= 0.5 and size_check <= 1.5: 
                            #    if str_check >= 0.75 and str_check <= 1.25:
                            found_dic[int(id2)] = int(id1)
                            print("obj2: %d obj1: %d overlap: %f " % (id2, id1, overlap))
                            continue

    #identify the new storm IDs with a new ID
    #
    rotate_id_value = 1000
            
    for id2 in obj2_ranks:
        if not id2 in found_dic:
            print("new id: %s is %d "%(id2, max_value_id))
            found_dic[int(id2)] = int(max_value_id)
            max_value_id += 1
            if max_value_id >= rotate_id_value:
                max_value_id = 1

    print("Found: ")
    print(found_dic)
    #
    #Assign new object id values to the obj2 data
    #
    #Do a where replacement on the original obj2 data and return it as the new
    #tracked obj data
    #
    for key, value in found_dic.items():
        #print("replaceing: %d with %d "%(key, value))
        #
        #FIXME: does it work?
        #
        obj2_xy = xr.where(obj2_xy == int(key), int(value), obj2_xy)
        #print("inside obj1 max: %d obj2 max: %d" % (obj1_xy.max(), obj2_xy.max())) 
    #
    return obj2_xy
