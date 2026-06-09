from copy import deepcopy

import numpy as np


class Chiplet:
    # definiation of chiplet object, including the physical infomation, and 
    # the microbumps, may extend to contain the macro&IO pin info. in the future

    def __init__(self, chiplet_name):
        self.chiplet_name = chiplet_name
        self.width = None
        self.height = None
        self.power = 0
        self.rotation = 0
        # Rotation angle theta recorded by [0,1,2,3], multiplied by pi/2 when used 
        self.x = 0 # The location of the center of the chiplet
        self.y = 0

        self.ubumps = []
        self.ubump_offset_x = [] # 1D array, ubump offset x to the center of chiplet
        self.ubump_offset_y = [] # 1D array, ubump offset y to the center of chiplet

    def set_chiplet_size(self, width, height):
        self.width = width
        self.height = height

    def set_chiplet_loc(self, x, y):
        self.x = x
        self.y = y

    def set_chiplet_power(self, power):
        self.power = power
        
    def append_ubumps(self, ubump_cords):
        return