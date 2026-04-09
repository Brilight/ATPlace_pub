import os
import sys
import time
import copy
import inspect #inspect.signature(func).parameters
import logging

import numpy as np

from utils.blocks_parser import parse_blocks
from utils.nets_parser import parse_nets
from utils.pl_parser import parse_pls
from utils.uscs_parser import parse_uscs
from Chiplet import Chiplet
from System import System_25D
from Interposer import Passive_Interposer
from Params import Params


def read_data(param_file, spec_param_file=None, param_print=False):

    params = Params()
    if param_print:
        if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
            params.printHelp()
        elif len(sys.argv) != 2:
            print("[E] One input parameters in json format in required")
            params.printHelp()
    params.load(param_file)
    if spec_param_file is not None:
        params.load(spec_param_file)
    np.random.seed(params.random_seed)

    try:
        prefix = params.prefix
        args = [prefix+".blocks", prefix+".nets", prefix+".pl"]
        powerfile = prefix+'.power'
    except:
        params.prefix = params.block_file.split("/")[-1]
        powerfile = params.power_file
        args = [params.block_file, params.net_file, params.pl_file]        
    # read database 
    configs = parse_uscs(None, args)
    modules, headers = parse_blocks(configs)
    num_chiplets = headers['Headers']['NumHardRectilinearBlocks']
    num_terminals = headers['Headers']['NumTerminals']
    locations = parse_pls(configs)
    nets, headers = parse_nets(configs)
    num_nets = headers['Headers']['NumNets']
    num_pins = headers['Headers']['NumPins']
    system = System_25D(num_chiplets, num_terminals)
    interposer = Passive_Interposer()
    for module_name in modules['Modules'].keys():
        if 'rectangles' in modules['Modules'][module_name].keys():
            chiplet_new = Chiplet(module_name)
            chiplet_new.set_chiplet_size(*modules['Modules'][module_name]['rectangles'][0][-2:])
            chiplet_new.set_chiplet_loc(*modules['Modules'][module_name]['rectangles'][0][:2])
            system.append_chiplet(module_name, chiplet_new)
        elif 'terminal' in modules['Modules'][module_name].keys():
            interposer.append_terminal(module_name, locations['Modules'][module_name]['center'])
            system.append_terminal(module_name, locations['Modules'][module_name]['center'])

    num_nodes = system.num_nodes
    t1 = time.time()
    net_idx = 0
    pin_id = 0
    for net in nets['Nets']:
        system.net_id.append(net_idx)
        system.net_weights.append(1.0)
        system.net2pin_map.append([])
        net_idx += 1
        for pin in net:
            node_id = system.node_name2id_map[pin[0]]
            if pin[2] is not None:
                pin_offset_x = float(pin[1])
                pin_offset_y = float(pin[2])
            else:
                continue
                pin_offset_x = 0.0
                pin_offset_y = 0.0
            pin_exists = False
            _pin_id = -1
            for _pid in system.node2pin_map[node_id]:
                if system.pin_offset_x[_pid] == pin_offset_x and system.pin_offset_y[_pid] == pin_offset_y:
                    pin_exists = True
                    _pin_id = _pid
            if pin_exists:
                system.net2pin_map[net_idx-1].append(_pin_id)
                system.pin2net_map[_pin_id].append(net_idx-1)
            else:
                system.net2pin_map[net_idx-1].append(pin_id)
                system.pin2net_map.append([net_idx-1])
                system.node2pin_map[node_id].append(pin_id)
                system.pin2node_map.append(node_id)
                system.pin_offset_x.append(pin_offset_x)
                system.pin_offset_y.append(pin_offset_y)
                pin_id += 1
    
    system.num_nets = len(system.net_id)
    system.num_pins = len(system.pin2net_map) - system.num_nodes + num_chiplets
    
    """
    if prefix is not None:
        with open(prefix+'.txt', 'r') as descript:
            for line in descript:
                if len(line)<=2:
                    continue
                _, intp_width, intp_height = line.split()
    """
    intp_width, intp_height = params.interposer_size
    intp_size = [float(intp_width), float(intp_height)]
    logging.info(f"Interposer size: {intp_size[0]:.2f}, {intp_size[1]:.2f} um\n")
    interposer.set_interposer_size(intp_size)
    fence = [params.fence_width, interposer.width-params.fence_width,
             params.fence_height, interposer.height-params.fence_height]
    system.set_interposer_size(fence, interposer)
    system.set_bins(params)
    system.initialize()
    system.set_granularity(params.reso_interposer)
    system.area_cplt = (np.array(system.node_size_x)*np.array(system.node_size_y)).sum()
    system.grid_size = min(system.bin_size_x, system.bin_size_y)
    system.num_grid_x, system.num_grid_y = params.num_grid_x, params.num_grid_y

    system.powermap = np.zeros(num_chiplets)
    
    with open(powerfile, 'r') as descript:
        for line in descript:
            name, power = line.split()
            system.powermap[system.node_names.index(name)] = power
    print("Total power: ", system.powermap)
    print(f"Parse and init time {time.time()-t1:.3f} s")
    return params, system, interposer
    