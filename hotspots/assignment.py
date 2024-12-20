
import numpy as np
import xarray as xr

from stormcell.StormcellXY import Stormcell
from stormcell.stormcell_helpers import sort_object_list, rank_stormcell_list, find_stormcell_index

def hotspot_assignment(stormid_obj: list[Stormcell],
                       hotspot_obj: list[Stormcell]) -> dict:
    """
    Assign hotspot objects to MCIT cell objects.

    Parameters
    ----------
    stormid_obj: list
        List of MCIT objects
    hotspot_obj: list
        List of hotspot objects

    Returns
    -------
    merge_dict: dict
        Dictionary with MCIT cells and hotspot objects assigned to each MCIT
        cell

    """
    StormID_ranks = rank_stormcell_list(stormid_obj)

    HS_ranks = rank_stormcell_list(hotspot_obj) 

    # make a list of unassigned hotspots
    unassigned_hotspots = hotspot_obj.copy()

    # make a dictionary: StormID as key, value: assigned hotspot object
    StormIDHS_dict = dict()

    for storm_id in StormID_ranks.keys():
        stormcell =  stormid_obj[find_stormcell_index(storm_id, stormid_obj)]
        storm_center = stormcell.center_location
    
        #print("storm id %d size: %d" % (stormcell.id, stormcell.size))
        StormIDHS_dict[storm_id] = dict(hotspots=[])
    
        hs_matched=[]
    
        #for the first detection
        primary_distance_threshold = max(
                np.sqrt(stormcell.size / np.pi), 10)
    
        #for additional detections, FIXME moved from 12 to 15 --jmk
        secondary_distance_threshold = 15
    
        primary_hotspot = None
    
        HS_ranks = rank_stormcell_list(unassigned_hotspots)
        #First pass objective find the primary hotspot within the distance thresh
        for hotspot_id in HS_ranks.keys():
            hotspot_cell = unassigned_hotspots[find_stormcell_index(hotspot_id, unassigned_hotspots)]
            hotspot_center = hotspot_cell.center_location
    
            #without overlap we don't consider this hotspot
            if stormcell.check_overlap(hotspot_cell) is False:
                continue
    
            # if no hotspot has been added yet, we check if it is close enough
            # to the MCIT cell center and then decide if we add it as primary
            # hotspot
            if primary_hotspot is None:
                distance = hotspot_center.get_distance(storm_center)
                #print("stormid: %d distance to hs %d:%f" %(stormcell.id, hotspot_cell.id, distance))
                if distance > primary_distance_threshold:
                    continue
    
                primary_hotspot = hotspot_cell.id
                # now let's finally do the assignment if the hotspot object has
                # passed all checks up to now
                #print(f"assigning to MCIT ID {stormcell.id} (size {stormcell.size}): "
                #  f"Hotspot ID {hotspot_cell.id} (size {hotspot_cell.size}) as first hotspot")
                #StormIDHS_dict[stormcell.id]['hotspots'][hotspot_cell.id] = (hotspot_cell)
                hs_matched.append(hotspot_cell.id)
    
                # remove from the list of available hotspots
                unassigned_hotspots.remove(hotspot_cell)
                del HS_ranks[hotspot_cell.id]
                break
    
        #rerank so that we only look at unassigned hotspots
        HS_ranks = rank_stormcell_list(unassigned_hotspots)
        #Second pass objective find the primary hotspot within the distance thresh or add additional hotspots
        #to the primary one. FIXME why not a third pass with additional instead
        #
        min_dist = 99999
        min_id = 99999
        for hotspot_id in HS_ranks.keys():
            hotspot_cell = unassigned_hotspots[find_stormcell_index(hotspot_id, unassigned_hotspots)]
            hotspot_center = hotspot_cell.center_location
    
            #we don't check for overlap on the second pass only distance.....This might change
            #without overlap we don't consider this hotspot
            #if stormcell.check_overlap(hotspot_cell) is False:
            #    continue
    
            # if no hotspot has been added yet, we check if it is close enough
            # to the MCIT cell center and then decide if we add it as primary
            # hotspot
            if primary_hotspot is None:
                distance = hotspot_center.get_distance(storm_center)
                if distance < min_dist:
                    min_dist = distance
                    min_id = hotspot_id
                #print("stormid: %d distance to hs %d:%f" %(stormcell.id, hotspot_cell.id, distance))
                if distance > secondary_distance_threshold:
                    continue
    
                primary_hotspot = hotspot_cell.id
                # now let's finally do the assignment if the hotspot object has
                # passed all checks up to now
                #print(f"assigning to MCIT ID {stormcell.id} (size {stormcell.size}): "
                #  f"Hotspot ID {hotspot_cell.id} (size {hotspot_cell.size}) as first hotspot")
                #StormIDHS_dict[stormcell.id]['hotspots'][hotspot_cell.id] = (hotspot_cell)
                hs_matched.append(hotspot_cell.id)
    
                # remove from the list of available hotspots
                unassigned_hotspots.remove(hotspot_cell)
                #del HS_ranks[hotspot_cell.id]
    
            else:
                #we have sometime so go back to requireing overlap
                if stormcell.check_overlap(hotspot_cell) is False:
                    continue

                #print("Primary:", primary_hotspot)
                #storm has a primary hotspot. Use that for distance computations
                primary_hotspot_cell = hotspot_obj[find_stormcell_index(primary_hotspot, hotspot_obj)]
                primary_hotspot_center = primary_hotspot_cell.center_location

                distance = hotspot_center.get_distance(primary_hotspot_center)
                if distance > secondary_distance_threshold:
                    continue

                #print(f"assigning to MCIT ID {stormcell.id} (size {stormcell.size}): "
                #  f"Hotspot ID {hotspot_cell.id} (size {hotspot_cell.size}) as additional")

                hs_matched.append(hotspot_cell.id)
                # remove from the list of available hotspots
                unassigned_hotspots.remove(hotspot_cell)
                #del HS_ranks[hotspot_cell.id]
    
    
        if primary_hotspot is None:
            print("StormCell: ", stormcell.id)
            print(" NONE min_dist: ", min_dist, " hs_id: ", min_id)
        else:
            print("StormCell: ", stormcell.id)
            for h in hs_matched:
                print(" hs:", h)
    
        StormIDHS_dict[stormcell.id]['hotspots'] = hs_matched


    return StormIDHS_dict

def hotspot_assignment_old(mcit_obj: list[Stormcell],
                       hotspot_obj: list[Stormcell]) -> dict:
    """
    Assign hotspot objects to MCIT cell objects.

    Parameters
    ----------
    mcit_obj: list
        List of MCIT objects
    hotspot_obj: list
        List of hotspot objects

    Returns
    -------
    merge_dict: dict
        Dictionary with MCIT cells and hotspot objects assigned to each MCIT
        cell

    """

    hotspots_sorted = sort_object_list(hotspot_obj, 'size')
    #reverse it so the largest is first
    hotspots_sorted.reverse()

    mcit_sorted = sort_object_list(mcit_obj, 'size')
    #reverse it so the largest is first
    mcit_sorted.reverse()

    # make a list of unassigned hotspots
    unassigned_hotspots = hotspots_sorted.copy()

    # make a dictionary: MCIT ID as key, values: MCIT object + assigned hotspot objects
    # can be merged to common object later?
    merge_dict = dict()


    for mcit_cell in mcit_sorted:
        # when MCIT tracking is finished check here if the cell is new or has a
        # previous location
        # if mcit_cell.type == 'New':
        # ... get the current location
        # elif mcit_cell.type == 'Old':
        # ... get the old location

        mcit_center = mcit_cell.center_location
        mcit_size = mcit_cell.size
        print("mcit id %d size: %d" % (mcit_cell.id, mcit_cell.size))

        primary_distance_threshold = max(
            np.sqrt(mcit_size / np.pi), 10)
        secondary_distance_threshold = 12

        # assign MCIT cell to dictionary
        merge_dict[mcit_cell.id] = dict(mcit=mcit_cell, hotspots=dict())

        # save the primary hotspot of each MCIT cell to query center distance
        # needs to be assigned separately to each MCIT cell!
        primary_hotspot = None

        for hotspot_cell in unassigned_hotspots:
            hotspot_center = hotspot_cell.center_location

            # without overlap we don't consider this hotspot
            if mcit_cell.check_overlap(hotspot_cell) is False:
                continue

            # if no hotspot has been added yet, we check if it is close enough
            # to the MCIT cell center and then decide if we add it as primary
            # hotspot
            if primary_hotspot is None:
                distance = hotspot_center.get_distance(mcit_center)
                if distance > primary_distance_threshold:
                    continue

                # we found our primary hotspot (will be assigned to dict later)
                primary_hotspot = hotspot_cell

            # we have already found the primary hotspot before, now apply the
            # assignment logic for adding other updrafts
            else:
                # compare with location of first assigned hotspot
                distance = hotspot_center.get_distance(
                    primary_hotspot.center_location)

                if distance > secondary_distance_threshold:
                    continue

            # now let's finally do the assignment if the hotspot object has
            # passed all checks up to now
            print(f"assigning to MCIT ID {mcit_cell.id} (size {mcit_cell.size}): "
                  f"Hotspot ID {hotspot_cell.id} (size {hotspot_cell.size})")
            merge_dict[mcit_cell.id]['hotspots'][hotspot_cell.id] = (
                hotspot_cell)

            # remove from the list of available hotspots
            unassigned_hotspots.remove(hotspot_cell)

    return merge_dict


def filter_assigned_hotspots(hotspot_labels: xr.DataArray,
                             merged_obj: dict) -> xr.DataArray:
    """
    Filter the hotspot labels, leaving only hotspots that have been assigned to
    MCIT cells

    Parameters
    ----------
    hotspot_labels: xr.DataArray
        Hotspot label field
    merged_obj: dict
        Dictionary of MCIT cells with assigned hotspot objects

    Returns
    -------
    hotspot_labels: xr.DataArray
        Filtered hotspot label field
    """

    valid_hotspots = []
    for cell_id in merged_obj:
        for hotspot_id in merged_obj[cell_id]['hotspots']:
            valid_hotspots.append(hotspot_id)

    return hotspot_labels.where(np.isin(hotspot_labels, valid_hotspots), -1)
