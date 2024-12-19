#!/usr/bin/env python
# coding: utf-8

"""
Created on October 14, 2024

@author: John Krause (JK) and Vinzent Klaus (VK)
@license: BSD-3-Clause
@copyright Copyright 2024 John Krause

Update History:
    Version 0:
        -JK Initial version, copy of c++ code base
    Version 0.1:
        - VK additional methods to compare two objects
"""

import numpy as np


class LocXY:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Override the __eq__ method to define custom equality logic
    #use as "=="
    def __eq__(self, other):
        if isinstance(other, LocXY):
            return self.x == other.x and self.y == other.y
        else:
            raise TypeError('must provide locations as LocXY object')

    def get_distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Stormcell:

    def __init__(self, storm_id, locations=None, center=None, max_value=None,
                 size=None):
        self.id = int(storm_id)
        self.max_value = max_value
        self.center_location = center
        self.size = size
        self.prev_location = None # this is reserved for later development of MCIT tracking
        self.type = None # reserved for later, can be "New" or "Old"
        
        #self.locs = []
        #adjusted to check for LocXY

        if locations is None:
            self.locs = []
        else:
            if isinstance(locations[0], LocXY):
                self.locs = locations 
            else:
                print("The Stormcell class requires locations to use the LocXY "
                      "object\n")
                exit()


    def add_locations(self, locations):
        #append adds a list to a list as a list
        #extend adds values to the end of the list
        for l in locations:
            if not isinstance(l, LocXY):
                print("The Stormcell class requires locations to use the LocXY "
                      "object\n")
                exit()

        self.locs.extend(locations)

    def set_max_value(self, max_value):
        self.max_value = max_value

    def set_size(self, size):
        self.size = size

    def set_center_loc(self, location):
        if isinstance(location, LocXY):
            self.center_location = location
        elif isinstance(location, tuple) and len(location) == 2:
            self.center_location = LocXY(location[0], location[1])
        else:
            print("The Stormcell class requires locations to use the LocXY "
                  "object or 2d tuples\n")
            exit() 

    def check_overlap(self, other):
        for loc in self.locs:
            for loc_other in other.locs:
                if loc == loc_other:
                    return True
        return False
