# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import re
import typing
from typing import Union, List
from argparse import ArgumentParser

from utils.uscs_parser import word_split, blank_line
from utils.uscs_parser import Modules


def parse_pl(lines: list[str], i: int, modules: Modules) -> bool:
    if blank_line(lines[i]):
        return True
    words = word_split(lines[i])
    if len(words) != 3:
        raise Exception("Don't know how to parse line (" + str(i + 1) + "): " + lines[i])
    modules[words[0]] = {
        'fixed': True,
        'terminal': True,
        'center': [float(words[1]), float(words[2])]
    }
    return True

def parse_pls(options):
    file_path = options['filename_pl']
    modules: Modules = {}
    f = open(file_path, "r")
    raw_text = f.read()
    lines = re.split('\n', raw_text)
    i = 0
    while i < len(lines) and parse_pl(lines, i, modules):
        i += 1
    return {'Modules': modules}
