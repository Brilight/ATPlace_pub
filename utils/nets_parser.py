# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import re
import typing
from typing import Union, List
from argparse import ArgumentParser

from utils.uscs_parser import word_split, blank_line
from utils.uscs_parser import Net, Nets, Headers


def parse_header(lines: list[str], i: int, headers: Headers):
    defined_headers = {
        'NumNets': (int, 0),
        'NumPins': (int, 0) }

    if len(headers.keys()) == 0:
        for header in defined_headers.keys():
            headers[header] = defined_headers[header][1]

    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if len(words) < 2 or words[1] != ':' or words[0] == 'NetDegree':
        return False
    if len(words) != 3:
        raise Exception("Error parsing line " + str(i+1) + ": Unknown Header:" + lines[i])
    if words[0] in defined_headers:
        headers[words[0]] = defined_headers[words[0]][0](words[2])
    else:
        headers[words[0]] = words[2]
    return True

def parse_net_line(lines: list[str], i: int, net: Net):
    if blank_line(lines[i]):
        return False
    words = word_split(lines[i])
    pos = re.search(r"(\w+)\s+(\w+)\s*(:?\s*\%?)([+-]?(\d+(\.\d*)?|\.\d+)?([eE][+-]?\d+)?)"+\
                    "\s*(\%([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?))?\s*",lines[i])
    if len(words) < 1:
        raise Exception("Line number " + str(i+1) + " is empty!")
    try:
        net.append([words[0],pos.group(4),pos.group(9)])
    except:
        net.append(words[0])
    return True

def parse_net(lines: list[str], i: int, nets: Nets) -> tuple[bool, int]:
    if blank_line(lines[i]):
        return True, i+1
    words = word_split(lines[i])
    if len(words) != 3 or words[0] != 'NetDegree' or words[1] != ":":
        raise Exception("Unknown format on line " + str(i+1) + ": " + lines[i])
    net_size = int(words[2])
    net: Net = []
    i += 1
    j = 0
    while j < net_size:
        if parse_net_line(lines, i, net):
            j += 1
        i += 1
    nets.append(net)
    return True, i

def parse_nets(options):
    nets: Nets = []
    headers: Headers = {}
    file_path = options['filename_nets']
    f = open(file_path, "r")
    raw_text = f.read()
    lines = re.split('\n', raw_text)
    i = 0
    while i < len(lines) and parse_header(lines, i, headers):
        i += 1
    while i < len(lines):
        cont, i = parse_net(lines, i, nets)
        if not cont:
            break
    return {'Nets': nets}, {'Headers': headers}
