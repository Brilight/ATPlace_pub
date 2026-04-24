import re
import os
import sys
import time
import copy
import argparse
import logging

import torch
import numpy as np
from tqdm import tqdm, trange
import optuna

from Parser import read_data
from utils.Utils import read_effective_data, identify_pareto

from System import System_25D
from Chiplet import Chiplet

from ATPLACE.NonlinearPlace import NonlinearPlace
from ATPLACE.PlaceObj import PlaceObj
from ATPLACE.Legalization import Legalization
from ATPLACE.PlaceFlow import placeflow
from SA.SAInit import Initialization
from SA.SACore import thermal_sim
from Thermal.ThermalSolver import HotSpot_solver, ATSim_solver
from Thermal.TModelTrain import Datagen, train_model


def hyper_param_search(default_params, n_trials, SpecParams):

    log_file = os.path.join(os.path.dirname(default_params), "hpo_res/log.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_file), 
                                  logging.StreamHandler()])
    def objective(trial):
        
        param_dict = {
            "@trial ": trial.number,
            "lambda_iter": trial.suggest_int("lambda_iter", 1, 5),
            "sub_iter": trial.suggest_int("sub_iter", 1, 5),
            "LR": [trial.suggest_float("lr_pos", 5e-2, 1e0, log=False),
                   trial.suggest_float("lr_ang", 1e-2, 1e-1, log=False)],
            "eta": trial.suggest_float("eta", 0.1, 1, log=False),
            "target_density": trial.suggest_float("target_density", 0.4, 1.0, log=False),
            "overflow_init": trial.suggest_float("overflow_init", 0.2, 0.5, log=False),
            "wl_weight" : trial.suggest_float("wl_weight", 1e-2, 1, log=False),
            "density_weight_init": trial.suggest_float("density_weight_init", 1e-1, 100, log=True),
        }
        for key, value in param_dict.items():
            if type(value) is int:
                print(f"{key} {value}\n\t", end='')
            if type(value) is float:
                print(f"{key}: {value:.3f}", end=', ')
            if type(value) is list:
                print(f"{key}: [{value[0]:.3f}, {value[1]:.3f}]", end=', ')
        print()
        
        wl_counter = 0
        wlres = []
        for spec_params in SpecParams:
            params, system, interposer = read_data(default_params, spec_params)
            system.temp_path = params.thermal_dir
            system.params = params
            params.temp_aware_opt = False

            fp_parms = params.floorplan_stages[0]
            fp_parms["iteration"] = 500
            fp_parms["Llambda_density_weight_iteration"] = param_dict["lambda_iter"]
            fp_parms["Lsub_iteration"] = param_dict["sub_iter"]
            fp_parms["learning_rate"] = [
                param_dict['LR'][0]*system.grid_size, param_dict['LR'][1]*np.pi
            ]

            params.eta = param_dict['eta']
            params.target_density = param_dict["target_density"]
            params.overflow_init = param_dict["overflow_init"]
            params.wl_weight = param_dict["wl_weight"]/system.num_nets
            params.density_weight_init = param_dict["density_weight_init"]
            #params.plot_freq = -1
            
            try:
                hpwl, _, _ = ATPlace_core(params, system)
                wl_counter += hpwl/system.num_nets/system.grid_size
                wlres.append(hpwl)
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                import traceback
                traceback.print_exc()
                raise optuna.TrialPruned()
        return wl_counter
    
    study = optuna.create_study(directions=["minimize"])
    study.optimize(
        #lambda trial: objective(params, system, trial), n_trials=100, catch=(ValueError,)
        objective, n_trials=n_trials, catch=(ValueError,)
    )
    return study

def best_params_search(system):

    def objective(trial):
        fp_parms = params.floorplan_stages[0]
        fp_parms["iteration"] = 100
        fp_parms["learning_rate"] = [
            trial.suggest_float("lr_pos", 0.01, 10, log=True)*system.grid_size, 
            trial.suggest_float("lr_ang", 1e-3, 1e-1, log=True)*np.pi
        ]
        fp_parms["Llambda_density_weight_iteration"] = 2 #trial.suggest_int("lambda_iter", 1, 5)
        fp_parms["Lsub_iteration"] = 4 # trial.suggest_int("sub_iter", 1, 5)

        params.gamma = trial.suggest_int("gamma", 2, 8)
        params.eta = 0.1 #trial.suggest_float("eta", 0.1, 5, log=True)
        params.target_density = 0.5 #trial.suggest_float("target_density", 0.6, 1.0, log=False)
        params.random_center_init_flag = False
        params.num_bins_x = 2**8 #trial.suggest_int("num_bins", 5, 8)

        params.overflow_init = trial.suggest_float("overflow_init", 0.2, 0.5, log=False)
        params.wl_weight = trial.suggest_float("wl_weight", 0, 5, log=False)/system.num_nets
        params.temp_weight_init = trial.suggest_float("temp_weight_init", 1e1, 1e4, log=True)
        params.density_weight_init = 1 #trial.suggest_float("density_weight_init", 0.5, 10, log=False)
        params.temp_threshold = trial.suggest_float("temp_threshold", model.amb, 100, log=False)
        output_dict = {
            "LR": f'[{fp_parms["learning_rate"][0]:.2e}, {fp_parms["learning_rate"][1]:.2e}]',
            "Gamma": params["gamma"], "\n": None,
            "W_temp": f'{params["temp_weight_init"]:.1e}',
            "W_dens": f'{params["density_weight_init"]:.1e}',
            "Overlap_ratio": f'{params["overflow_init"]:.2f}',
            "WL_weight": f'{params["wl_weight"]:.2e}',
            "Temp_threshold": f'{params["temp_threshold"]:.1f}',
            "@trial ": trial.number
        }
        for key, value in output_dict.items():
            print(f"{key}: {value}", end=',')
            if not value:
                print()

        return placeflow(params, system, tempmodel, temp_solver)

    params = system.params
    Res_log_atp = []
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=100, catch=(ValueError,))
    #get_ipython().run_line_magic('time', 'study.optimize(objective, n_trials=100, catch=(ValueError,))')
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")

    Res_atp = []
    for trial in study.trials:
        try:
            Res_atp.append([trial.values[0]/1e6, trial.values[1]])
            print(f"[{trial.values[0]/1e6:.1f}, {trial.values[1]:.1f}], ",end='')
        except:
            print(end='')

    np.save(os.path.join(system.respath,'ATP.npy'), np.array(Res_atp))


def ATPlace_core(params, system):

    np.random.seed(params.random_seed)
    torch.set_num_threads(params.random_seed)

    system.grid_size = min(
        (system.xhigh-system.xlow)/params.num_bins_x, 
        (system.yhigh-system.ylow)/params.num_bins_y
    )
    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    system.num_grid_x, system.num_grid_y = params.num_grid_x, params.num_grid_y
    #planer = NonlinearPlace(params, system)
    #model = PlaceObj([0, 0], params, planer.datacollection, params.floorplan_stages[0])
    
    if params.temp_aware_opt:
        assert params.thermal_solver.lower() in ('hotspot', 'atsim'), (
            f"Thermal solver can be ATSim or HotSpot only, rather than {params.thermal_solver}"
        )
        if params.thermal_solver.lower()=='hotspot':
            temp_solver = HotSpot_solver(system)
            with open(system.temp_path+"hotspot.config", 'r') as file:
                config_contents = file.read()
            config_contents = re.sub(
                r'-grid_rows			\d+', 
                f'-grid_rows			{system.num_grid_x}', 
                config_contents
            )
            config_contents = re.sub(
                r'-grid_cols			\d+', 
                f'-grid_cols			{system.num_grid_y}', 
                config_contents
            )
            with open(system.temp_path+"hotspot.config", 'w') as file:
                file.write(config_contents)
        else:
            temp_solver = ATSim_solver(system)             
            thermal_configs = os.path.join(system.temp_path, "config.json")
            temp_solver.read_configs(thermal_configs)

        temp_solver.amb = params.ambient
        num_iters = 1000
        lr = 1e-1
        data_loader = Datagen(params.num_thermal_dataset, system, temp_solver)
        criterion = lambda out,label,mask: ((out-label).pow(2)).mean()
        tempmodel = train_model(
            system.intp_width, system.intp_height, system.num_chiplets, 
            system.num_grid_x, system.num_grid_y, lr, data_loader, criterion, num_iters
        )
        for param in tempmodel.parameters():
            param.requires_grad_(False)

        train_model(
            system.intp_width, system.intp_height, system.num_chiplets, 
            system.num_grid_x, system.num_grid_y, lr, data_loader, criterion, num_iters
        )

    else:
        tempmodel = temp_solver = None
        
    t1 = time.perf_counter()
    hpwl, best_metric, best_pos = placeflow(
        params, system, tempmodel, temp_solver
    )    
    print(f"Final wirelength: {hpwl/1e6:.3f}m within {time.perf_counter()-t1:.3f}s")
    return hpwl, best_metric, best_pos

    max_intp_size = int(system.area_cplt**0.5)/2
    init_pos = Initialization(system, max_intp_size, params.overflow_init, 10)
    while init_pos is None:
        max_intp_size *= 1.1
        init_pos = Initialization(system, max_intp_size, params.overflow_init, 10)
    system.node_x[:num_chiplets] = init_pos[0][0]
    system.node_y[:num_chiplets] = init_pos[0][1]
    system.node_orient[:num_chiplets] = init_pos[1]+np.random.rand()*0.2+0.2

    placer = NonlinearPlace(params, system)
    model = PlaceObj([0, 0], params, placer.datacollection, params.floorplan_stages[0])
    placer.temp_weight_init = params.temp_weight_init 
    placer.density_weight_init = params.density_weight_init
    placer.temp_aware_opt = True
    placer.gp_noise_ratio = params.gp_noise_ratio #0.05
    placer.tempmodel = tempmodel
    placer.powermap = torch.Tensor(system.powermap).reshape(1,-1,1)
    placer.amb = model.amb = params.ambient

    best_metric, best_pos, Lgamma_metrics = placer(params, system)
    _, theta, size_x, size_y = model.legalize_theta(best_pos[0][1].data.clone())
    size_x, size_y = size_x.cpu().numpy(), size_y.cpu().numpy()
    x_init = best_pos[0][0][:num_chiplets].detach().cpu().numpy()
    y_init = best_pos[0][0][num_nodes:num_nodes+num_chiplets].detach().cpu().numpy()
    final_pos = Legalization(system, params.wl_weight, [[x_init, y_init], ], 
                             [size_x+params.dis_bet_chips, size_y+params.dis_bet_chips])
    if final_pos is None:
        final_pos = Legalization(system, 0, [[x_init, y_init],], 
                                 [size_x+params.dis_bet_chips, size_y+params.dis_bet_chips])
    best_pos[0][0][:num_chiplets] = torch.from_numpy(final_pos[0])
    best_pos[0][0][num_nodes:num_nodes+num_chiplets] = torch.from_numpy(final_pos[1])
    best_pos[0][1] = theta
    
def compare_res(system):

    Res_date = np.load(os.path.join(system.respath,'SA.npy'))#np.array(Res_log)[:,:2]
    prt_sa = identify_pareto(Res_date)
    ATPlace = np.load(os.path.join(system.respath,'ATP.npy'))
    prt_at = identify_pareto(ATPlace)
    array_bb = np.load("/".join(system.prefix.split('/')[:-1])+'/aspdac.npy')
    system.node_x, system.node_y, node_orient = array_bb
    system.rotate(np.arange(num_chiplets), node_orient)
    length_bb = system.hpwl()
    temp_bb = thermal_sim(system, -1)
    print(length_bb, temp_bb)

    thermal_sim(system, -1)
    temp_single = np.loadtxt(system.temp_path+"step-1.grid.steady")[:,1]
    temp_asp1 = np.transpose(temp_single.reshape(int(temp_single.size**0.5),-1),(1,0))[:,::-1]-273.15
    plot_temp_pos(system, temp_asp1)

    thermal_sim(system, -1)
    temp_single = np.loadtxt(system.temp_path+"step-1.grid.steady")[:,1]
    temp_asp2 = np.transpose(temp_single.reshape(int(temp_single.size**0.5),-1),(1,0))[:,::-1]-273.15
    plot_temp_pos(system, temp_asp2)

    fig, ax = plt.subplots(figsize=(9, 8))
    plt.tick_params(labelsize=20)
    plt.scatter(Res_date[:,1], Res_date[:,0], marker='+', edgecolor='blue', c='lightskyblue', s=100,  alpha=1)
    plt.scatter(ATPlace[:,0], ATPlace[:,1], edgecolor='green', marker='+', c='gold', s=100, alpha=1)
    plt.scatter(prt_sa[:,1], prt_sa[:,0], marker='o', c='dodgerblue', s=100, alpha=.9, label='DATE')
    plt.scatter(prt_at[:,0], prt_at[:,1], c='orange', marker='o', s=100, label='ATPlace')
    plt.scatter((length_bb)/1e6, (temp_bb), marker='o', s=100, c='red', label='ASPDAC')
    plt.xlabel("TotWL/m", fontsize=20); plt.ylabel("MaxTemp/C", fontsize=20)
    #plt.xticks(np.arange(2,7)*10); plt.xlim([15,70])
    #plt.yticks([140,150,160,170,]); plt.ylim([136,175])
    plt.legend(fontsize=20,frameon=False)
    plt.grid(True, linestyle='--', color='black', linewidth=0.5)
    plt.show()
