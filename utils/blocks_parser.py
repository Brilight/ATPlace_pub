# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import typing
import re
from argparse import ArgumentParser
from typing import Union, List

from utils.uscs_parser import word_split, blank_line
from utils.uscs_parser import Modules, Headers


def parse_terminals(lines: list[str], i: int, modules: Modules):
    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if len(words) == 2 and words[1] == "terminal":
        modules[words[0]] = {
            'terminal': True
        }
        return True
    return False


def parse_rectangles(lines: list[str], i: int, modules: Modules) -> bool:
    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    # This separation of "soft" + "rectangular" is dumb, but it's the only way to get
    # PyCharm's typo detector to shut up
    if len(words) >= 5 and words[1] == "soft" + "rectangular":
        low, high = float(words[3]), float(words[4])
        if low > high:
            low, high = high, low
        modules[words[0]] = {
            'area': float(words[2]),
            'aspect_ratio': [low, high]
        }
        return True
    elif len(words) == 11 and words[1] == "hard" + "rectilinear":
        num_vertices = int(words[2])
        joint_string = ""
        for i in range(3, len(words)):
            joint_string += " " + words[i]
        vertices = []
        index = 0
        for i in range(0, num_vertices):
            while index < len(joint_string) and joint_string[index] != '(':
                index += 1
            index += 1
            point_string = ""
            while index < len(joint_string) and joint_string[index] != ')':
                point_string += joint_string[index]
                index += 1
            if index >= len(joint_string) or joint_string[index] != ')':
                raise Exception("Runaway argument at line " + str(i), lines)
            point_split = point_string.split(",")
            if len(point_split) != 2:
                raise Exception("Point on line " + str(i) + " has the wrong number of dimensions")
            x, y = float(point_split[0]), float(point_split[1])
            vertices.append((x, y))
        if len(vertices) != 4:
            raise Exception("Shape not tolerated on line " + str(i) + ": Only quads allowed")
        # TODO: Check whether the input shape is actually an *orthogonal* quad
        min_x = min(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
        max_x = max(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
        min_y = min(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
        max_y = max(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        dims = (max_x - min_x, max_y - min_y)
        modules[words[0]] = {
            'rectangles': [[center[0], center[1], dims[0], dims[1]]],
            'fixed': True
        }
        return True
    else:
        return False


def parse_header(lines: list[str], i: int, headers: Headers):
    defined_headers = {
        'NumSoftRectangularBlocks': (int, 0),
        'NumHardRectilinearBlocks': (int, 0),
        'NumTerminals': (int, 0)
    }

    if len(headers.keys()) == 0:
        for header in defined_headers.keys():
            headers[header] = defined_headers[header][1]

    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if len(words) < 2 or words[1] != ':':
        return False
    if len(words) != 3:
        raise Exception("Error parsing line " + str(i+1) + ": Unknown Header:" + lines[i])
    if words[0] in defined_headers:
        headers[words[0]] = defined_headers[words[0]][0](words[2])
    else:
        headers[words[0]] = words[2]
    return True


def parse_blocks(options):
    modules: Modules = {}
    headers: Headers = {}
    file_path = options['filename_blocks']
    f = open(file_path, "r")
    raw_text = f.read()
    lines = re.split('\n', raw_text)
    i = 0
    while i < len(lines) and parse_header(lines, i, headers):
        i += 1
    while i < len(lines) and parse_rectangles(lines, i, modules):
        i += 1
    while i < len(lines) and parse_terminals(lines, i, modules):
        i += 1
    return {'Modules': modules}, {'Headers': headers}
