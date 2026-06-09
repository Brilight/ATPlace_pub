import math
from copy import deepcopy

import numpy as np

import Chiplet


class System_25D:
    # definiation of 2.5D system object, including all chiplets, nets, and
    # the interposer, may extend to contain the RDL info. in the future
    # All the length/size included are by the unit of um
    
    def __init__(self, num_chiplets, num_terminals):
        self.dtype = np.float32
        self.num_chiplets = num_chiplets
        self.num_terminals = num_terminals
        self.num_nodes = num_chiplets+num_terminals
        self.node_names = []
        self.node_name2id_map = {}
        self.chiplets = []
        self.node_x = []
        self.node_y = []
        self.node_size_x = []
        self.node_size_y = []
        self.node_orient = []
        self.node_flip = []
        # Rotation angle theta recorded by [0,1,2,3], multiplied by pi/2 when used 
        
        self.pin_offset_x = [] 
        # 1D array, pin offset x to its node, node can be either chiplet or terminal
        self.pin_offset_y = [] # 1D array, pin offset y to its node

        self.net_id = [] # net name
        self.net_weights = [] # weights for each net
        self.net2pin_map = [] # array of 1D array, each row stores pin idx of a net
        self.node2pin_map = [] # array of 1D array, contains pin idx of each node
        self.pin2node_map = [] # 1D array, contain parent node idx of each pin
        self.pin2net_map = [] # 1D array, contain parent net idx of each pin
        self.netid_2pin = []
        self.netid_multipin = []
    
    def update_pos(self, node_x, node_y, node_orient):
        self.node_x = node_x
        self.node_y = node_y
        self.node_orient = node_orient

    def append_chiplet(self, chiplet_name, chiplet_new):
        self.node_names.append(chiplet_name)
        self.node_name2id_map[chiplet_name] = len(self.node_names)-1
        self.chiplets.append(chiplet_new)
        self.node_x.append(0)
        self.node_y.append(0)
        self.node_size_x.append(chiplet_new.width)
        self.node_size_y.append(chiplet_new.height)
        self.node_orient.append(0)
        self.node2pin_map.append([])
        
    def append_terminal(self, terminal_name, terminal_loc):
        self.node_names.append(terminal_name)
        self.node_name2id_map[terminal_name] = len(self.node_names)-1
        self.node_x.append(terminal_loc[0])
        self.node_y.append(terminal_loc[1])
        self.node_size_x.append(0)
        self.node_size_y.append(0)
        self.node_orient.append(0)
        self.node2pin_map.append([])

    def set_connection_matrix(self, connection):
        self.connection_matrix = connection

    def set_granularity(self, granularity):
        # granularity: the minimum resolution of interposer in each direction, unit: um
        self.granularity = granularity

    def set_interposer_size(self, fence, interposer):
            self.xlow, self.xhigh = fence[0], fence[1]
            self.ylow, self.yhigh = fence[2], fence[3]
            self.intp_width, self.intp_height = interposer.width, interposer.height
    
    def initialize(self):
        # initialize class variables as numpy arrays
        self.node_x = np.array(self.node_x, dtype=self.dtype)
        self.node_y = np.array(self.node_y, dtype=self.dtype)
        self.node_size_x = np.array(self.node_size_x, dtype=self.dtype)
        self.node_size_y = np.array(self.node_size_y, dtype=self.dtype)
        self.node_area = self.node_size_x*self.node_size_y
        self.node_orient = np.array(self.node_orient, dtype=self.dtype)
        self.pin_offset_x = np.array(self.pin_offset_x, dtype=self.dtype)
        self.pin_offset_y = np.array(self.pin_offset_y, dtype=self.dtype)
        for pin in range(len(self.pin2node_map)):
            node = self.pin2node_map[pin]
            self.pin_offset_x[pin] *= self.node_size_x[node]/100
            self.pin_offset_y[pin] *= self.node_size_y[node]/100
        self.net_weights = np.array(self.net_weights)
        self.pin2node_map = np.array(self.pin2node_map, dtype=int)
        self.pin2net_map = np.array(self.pin2net_map, dtype=int)
        # convert node2pin_map to array of array 
        for i in range(len(self.node2pin_map)):
            self.node2pin_map[i] = np.array(self.node2pin_map[i]).astype(int)
        #self.node2pin_map = np.array(self.node2pin_map, dtype=object)

        # convert net2pin_map to array of array 
        self.pinid_2pin = []
        for i in range(len(self.net2pin_map)):
            self.net2pin_map[i] = np.array(self.net2pin_map[i]).astype(int)
            if len(self.pin2node_map[self.net2pin_map[i]])==2:
                if len(self.net2pin_map[i])==2:
                    self.netid_2pin.append(i)
                    self.pinid_2pin.append(self.net2pin_map[i])
                else:
                    self.netid_multipin.append(i)
                    self.pinid_multipin.append(self.net2pin_map[i])
        #self.net2pin_map = np.array(self.net2pin_map, dtype=object)
        
        self.netid_2pin = np.array(self.netid_2pin).astype(int)
        self.pinid_2pin = np.array(self.pinid_2pin).astype(int)        
        self.netid_multipin = np.array(self.netid_multipin).astype(int)

        self.bin_center_x = np.array(self.bin_center_x, dtype=self.dtype)
        self.bin_center_y = np.array(self.bin_center_y, dtype=self.dtype)
        
    def bin_centers(self, low, high, num_bins):
        """
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return number of bins
        """
        bin_size = (high-low)/num_bins
        centers = np.zeros(num_bins+2, dtype=self.dtype)
        for id_x in range(num_bins+2):
            bin_low = low+(id_x-1)*bin_size
            bin_high = min(bin_low+bin_size, high+bin_size)
            centers[id_x] = (bin_low+bin_high)/2
        return centers

    def set_bins(self, params):
        # compute number of bins and bin size
        # derive bin dimensions by keeping the aspect ratio
        aspect_ratio = (self.xlow-self.xhigh)/(self.ylow-self.yhigh)
        try:
            if 0.95 < params.num_bins_x/params.num_bins_y <1.05:
                self.num_bins_x = params.num_bins_x
                self.num_bins_y = params.num_bins_y
        except:
            self.num_bins_x = int(math.pow(2, max(np.ceil(math.log2(self.num_chiplets / aspect_ratio)/2), 0)))
            self.num_bins_y = int(math.pow(2, max(np.ceil(math.log2(self.num_chiplets * aspect_ratio)/2), 0)))
            print("num_bins_x and num_bins_y in the params cannot keep the same aspect ratio as interposer.")
            print(f"They are changed to be {self.num_bins_x}, {self.num_bins_y}")
            
        self.bin_size_x = (self.xhigh-self.xlow)/self.num_bins_x
        self.bin_size_y = (self.yhigh-self.ylow)/self.num_bins_y

        # bin center array
        self.bin_center_x = self.bin_centers(self.xlow, self.xhigh, self.num_bins_x)
        self.bin_center_y = self.bin_centers(self.ylow, self.yhigh, self.num_bins_y)
        self.num_bins_x += 2
        self.num_bins_y += 2
        print(f"Number of bins_x and bins_y are {self.num_bins_x}, {self.num_bins_y}.")
        
    def flatten_nested_map(self, net2pin_map): 
        """
        @brief flatten an array of array to two arrays like CSV format 
        @param net2pin_map array of array 
        @return a pair of (elements, cumulative column indices of the beginning element of each row)
        """
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin2net_map), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        flat_net2pin_start_map = np.zeros(len(net2pin_map)+1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count+len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count 
            count += len(net2pin_map[i])
        assert flat_net2pin_map[-1] != 0
        flat_net2pin_start_map[len(net2pin_map)] = len(pin2net_map)

        return flat_net2pin_map, flat_net2pin_start_map
    
    def rotate(self, idx, theta):
        
        r = np.int32((self.node_orient[idx]-theta)/np.pi*2)%2 #np.abs(np.cos(self.node_orient[idx])))
        self.node_orient[idx] = theta
        tmp_size = self.node_size_x[idx]+self.node_size_y[idx]
        self.node_size_x[idx] = np.where(r==1, self.node_size_y[idx], self.node_size_x[idx])
        self.node_size_y[idx] = tmp_size - self.node_size_x[idx]
        if hasattr(self, 'distance'):
            self.node_size_x_dis = self.node_size_x+self.distance
            self.node_size_y_dis = self.node_size_y+self.distance
        return self.node_size_x, self.node_size_y
        
    def net_hpwl(self, x, y, r, net_id):
        """
        @brief compute HPWL of a net
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of a net
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        if len(np.unique(nodes))<2:
            return 0
        angle = self.node_orient[nodes]
        cos, sin = np.int32(np.cos(angle)), np.int32(np.sin(angle))
        pinx = x[nodes]+self.pin_offset_x[pins]*cos-self.pin_offset_y[pins]*sin
        hpwl_x = np.amax(pinx) - np.amin(pinx)
        piny = y[nodes]+self.pin_offset_x[pins]*sin+self.pin_offset_y[pins]*cos
        hpwl_y = np.amax(piny) - np.amin(piny)
        #print(self.net_weights[net_id], nodes, x[nodes], y[nodes], pinx, hpwl_x, piny, hpwl_y)
        return (hpwl_x+hpwl_y)*self.net_weights[net_id]
    
    def hpwl(self):
        """
        @brief compute total HPWL
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of all nets
        """
        wl = 0
        #print(self.node_x, self.node_y, self.node_orient)
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(self.node_x, self.node_y, self.node_orient, net_id)
        return wl
    
    def Maxwl(self):
        """
        @brief compute max WL
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl = max(wl, self.net_hpwl(self.node_x, self.node_y, self.node_orient, net_id))
        return wl