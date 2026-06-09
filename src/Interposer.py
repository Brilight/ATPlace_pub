from copy import deepcopy

import numpy as np

import Chiplet


class Passive_Interposer:
    # definiation of the interposer object, including all terminals and nets,
    # going to contain the C4Bump and TSV info. in the future

    def __init__(self):
        
        self.num_terminals = 0
        self.terminal_name = []
        self.terminal_idx = {}
        self.terminal_x = []
        self.terminal_y = []
        
    def set_interposer_type(self, intp_type):
        self.intp_type = intp_type

    def set_interposer_size(self, intp_size):
        if isinstance(intp_size, (tuple, list)):
            self.width, self.height = intp_size
        elif isinstance(intp_size, (int, float, np.number)):
            self.width = self.height = intp_size
        else:
            raise ValueError("intp_size should be a number (square) or tuple/list (rectangular)")
    
    def append_terminal(self, terminal_name, terminal_loc):
        self.terminal_name.append(terminal_name)
        self.terminal_idx[terminal_name] = len(self.terminal_name)-1
        self.terminal_x.append(terminal_loc[0])
        self.terminal_y.append(terminal_loc[1])
        