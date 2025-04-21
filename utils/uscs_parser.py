# (c) Víctor Franco Sanchez 2022
# For the FRAME Project.
# Licensed under the MIT License (see https://github.com/jordicf/FRAME/blob/master/LICENSE.txt).

import re
import typing
from typing import Union, List
from argparse import ArgumentParser

Modules = dict[str, dict[str, typing.Any]]
Headers = dict[str, typing.Any]
Net = list[typing.Any]
Nets = list[Net]


def blank_line(line: str):
    if len(line) == 0:
        return True
    if line[0] == '#':
        return True
    if line.startswith('UCLA') or line.startswith('UCSC'):
        return True
    return False


def word_split(line: str) -> list[str]:
    return re.split(r'[\s\n\t]+', line)


def parse_uscs(prog: Union[str, None], args: Union[List[str], None]):
    """
    Parse the command-line arguments for the tool
    :param prog: tool name
    :param args: command-line arguments
    :return: a dictionary with the arguments
    """
    parser = ArgumentParser(prog=prog, description="USCS dice format parser",
                            usage='%(prog)s [options]')
    parser.add_argument("filename_blocks", type=str,
                        help="Input file 1 (.blocks)")
    parser.add_argument("filename_nets", type=str,
                        help="Input file 2 (.nets)")
    parser.add_argument("filename_pl", type=str,
                        help="Input file 3 (.pl)")
    parser.add_argument("--output", dest="output", default=None, type=str,
                        help="(optional) Output file")
    options = vars(parser.parse_args(args))
    return options
