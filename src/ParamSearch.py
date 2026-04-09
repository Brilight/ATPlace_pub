import re
import os
import sys
import time
import argparse
import logging

import numpy as np
import torch
import optuna
from tqdm import tqdm, trange

from Parser import read_data
from utils.Visualize import plot_fp, plot_temp_fp, plot_double_y, plot_temp, plot_temp_pos
from utils.Utils import read_effective_data, identify_pareto

from Thermal.ThermalSolver import Thermal_solver
from ATPlace2_5D import ATPlace_core, hyper_param_search
from SA.SACore import anneal, thermal_sim


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler("./hpolog.txt"), 
                                  logging.StreamHandler()])
    default_params = '../Test_qipan/params.json'
    Res_atp = []
    SpecParams = [f'../Test_qipan/Case{i}/configs.json' for i in [1]]
    study = hyper_param_search('../Test_qipan/params.json', 100, SpecParams)
    for trial in study.trials:
        print(trial.params, trial.value)