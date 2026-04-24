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


def read_data(param_file=None, spec_param_file=None, param_print=False, case_dir=None):

    params = Params()
    if param_print:
        if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
            params.printHelp()
        elif len(sys.argv) != 2:
            print("[E] One input parameters in json format in required")
            params.printHelp()
    if param_file is not None:
        params.load(param_file)
    if spec_param_file is not None:
        params.load(spec_param_file)
    np.random.seed(params.random_seed)

    if case_dir is not None:
        case_dir = os.path.abspath(case_dir)
        case_name = os.path.basename(os.path.normpath(case_dir))
        params.prefix = os.path.join(case_dir, case_name)

    try:
        prefix = params.prefix
        args = [prefix+".blocks", prefix+".nets", prefix+".pl"]
        powerfile = prefix+'.power'
    except:
        if hasattr(params, "block_file") and getattr(params, "block_file"):
            params.prefix = params.block_file.split("/")[-1]
            powerfile = params.power_file
            args = [params.block_file, params.net_file, params.pl_file]
        else:
            raise ValueError(
                "Cannot resolve input files. Provide case_dir (recommended) or set "
                "params.prefix or params.block_file/net_file/pl_file/power_file."
            )
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
    fence_w = getattr(params, "fence_width", 0) or 0
    fence_h = getattr(params, "fence_height", 0) or 0

    intp_size_cfg = getattr(params, "interposer_size", None)
    if (
        intp_size_cfg is None
        or (isinstance(intp_size_cfg, (list, tuple)) and len(intp_size_cfg) < 2)
        or (isinstance(intp_size_cfg, str) and not intp_size_cfg.strip())
    ):
        chiplet_areas = []
        max_w = 0.0
        max_h = 0.0
        for module_name in modules["Modules"].keys():
            if "rectangles" in modules["Modules"][module_name].keys():
                _, _, w, h = modules["Modules"][module_name]["rectangles"][0]
                w = float(w)
                h = float(h)
                chiplet_areas.append(w * h)
                max_w = max(max_w, w)
                max_h = max(max_h, h)
        area_sum = float(np.sum(chiplet_areas)) if chiplet_areas else 0.0
        side = max((area_sum**0.5) * 1.5, max(max_w, max_h) * 2.0)
        intp_width = side + 2.0 * float(fence_w)
        intp_height = side + 2.0 * float(fence_h)
        params.interposer_size = [float(intp_width), float(intp_height)]
    else:
        intp_width, intp_height = intp_size_cfg

    intp_size = [float(intp_width), float(intp_height)]
    logging.info(f"Interposer size: {intp_size[0]:.2f}, {intp_size[1]:.2f} um\n")
    interposer.set_interposer_size(intp_size)
    fence = [float(fence_w), interposer.width - float(fence_w),
             float(fence_h), interposer.height - float(fence_h)]
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
    
