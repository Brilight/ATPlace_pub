import os
import time
import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any
from torch.utils.data import DataLoader

from Parser import read_data
from utils.Visualize import plot_temp_and_warpage, visualize_placement_results
from MPA.PhysicsSolver import ATSim_solver, Warpage_solver
from MPA.ModelTrain import MultiOutputDataset, train_tmodel, train_wmodel
from ATPLACE.PlaceFlow import placeflow_core


@dataclass
class PlacementConfig:
    num_grid_x: int = 64
    num_grid_y: int = 64
    granularity: int = 500
    num_dataset_train: int = 10
    num_iters: Tuple[int, int] = (2000, 6000)
    lrs: Tuple[float, float] = (1e-2, 1e-1)
    criterion: Optional[Any] = None
    temp_threshold: float = 100.0
    warpage_threshold: float = 50.0
    penalty_coeff: float = 1e6
    sa_baseline: Optional[list] = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unknown config key: {k}")


def evaluate_placement(
    system, physics_solver, best_fp_pos, best_fp_size, file_dir=None
    ):
    """
    Evaluate placement using physical solver.
    Returns: (hpwl, temp_max, warpage_max, temp_field, warpage_field)
    """
    try:
        if file_dir:
            temp_file_name = os.path.basename(os.path.normpath(file_dir)) + '_best'
        else:
            temp_file_name = "ATPlace"
        physics_solver.set_pos(system.powermap, best_fp_pos, best_fp_size)
        physics_solver.run(temp_file_name, 0)

        res = physics_solver.getres(temp_file_name)
        temp_whole = res['temperature'] - 273.15
        warpage_whole = res.get('warpage', None)

        temp_max = temp_whole.max()
        warpage = warpage_whole.max()-warpage_whole.min() if warpage_whole is not None else 0.0

        return temp_max, warpage, temp_whole, warpage_whole

    except Exception as e:
        raise RuntimeError(f"Physical simulation failed: {e}")

class PlacementOptimizer:
    def __init__(
        self,
        file_dir: str,
        default_params_path: str,
        therm_dir: str,
        thermal_configs_path: str,
        config: Optional[PlacementConfig] = None, 
        optimize_pure_wl=False,
    ):
        """
        Initialize optimizer for one case.
        """
        self.file_dir = file_dir
        self.case_name = os.path.basename(os.path.normpath(file_dir))
        self.optimize_pure_wl = optimize_pure_wl

        self.log_dir = os.path.join(file_dir, "../logs_all")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        existing_logs = [f for f in os.listdir(self.log_dir) if f.endswith(".log")]
        prefix = "WL-opt" if self.optimize_pure_wl else "TW-opt"
        log_file = os.path.join(self.log_dir, f"{prefix}-{len(existing_logs)+1}.log")
        print(f"Log saved to {log_file}")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        spec_params = os.path.join(file_dir, "configs.json")
        params, system, interposer = read_data(default_params_path, spec_params)
        system.thermal_dir = therm_dir
        physics_solver = Warpage_solver(system)
        physics_solver.read_configs(thermal_configs_path)
        physics_solver.amb = params.ambient

        self.params = params
        self.system = system
        self.physics_solver = physics_solver

        self.config = config or PlacementConfig()
        for k, v in self.config.__dict__.items():
            setattr(self, k, v)
        
        if (params.temp_aware_opt or params.warp_aware_opt) and (not optimize_pure_wl):
            tmodel, wmodel = self._load_and_train_models()
            self.compact_model = {"Thermal": tmodel, "Mechanical": wmodel}
        else:
            self.compact_model = {}

    def _load_and_train_models(self):
        """Load dataset and train thermal/mechanical compact models."""
        dataset_path = os.path.join(self.file_dir, "dataset_train.pt")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        dataset = MultiOutputDataset(
            num_chiplets=self.system.num_chiplets,
            W=self.system.intp_width,
            H=self.system.intp_height,
            num_grid_x=self.num_grid_x,
            num_grid_y=self.num_grid_y,
            data_path=dataset_path
        )
        data_loader_train = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        tmodel = train_tmodel(
            W=self.system.intp_width, H=self.system.intp_height,
            num_chiplets=self.system.num_chiplets,
            num_grid_x=self.num_grid_x, num_grid_y=self.num_grid_y,
            criterion=self.criterion, lr=self.lrs[0], num_iter=self.num_iters[0],
            data_loader=data_loader_train
        )
        wmodel = train_wmodel(
            W=self.system.intp_width, H=self.system.intp_height,
            num_chiplets=self.system.num_chiplets,
            num_grid_x=self.num_grid_x, num_grid_y=self.num_grid_y,
            criterion=self.criterion, lr=self.lrs[1], num_iter=self.num_iters[1],
            data_loader=data_loader_train
        )

        for param in tmodel.parameters(): param.requires_grad_(False)
        for param in wmodel.parameters(): param.requires_grad_(False)
        return tmodel, wmodel

    def run_single_trial(
        self, placerdict: Dict[str, Any], plot: bool = False, trial_id: Optional[int] = None
        ) -> Dict[str, Any]:
        """
        Run one placement trial with given parameters.
        Returns dict with metrics and optionally saves plots.
        """
        # Update params
        for k, v in placerdict.items():
            setattr(self.params, k, v)
        for stage in self.params.floorplan_stages:
            if 'optimizer' in placerdict:
                stage['optimizer'] = placerdict['optimizer']
            if 'lr_pos' in placerdict:
                stage['learning_rate'][0] = placerdict['lr_pos']
            if 'lr_ang' in placerdict:
                stage['learning_rate'][1] = placerdict['lr_ang']
            if 'iteration' in placerdict:
                stage['iteration'] = placerdict['iteration']
            if 'Llambda_density_weight_iteration' in placerdict:
                stage['Llambda_density_weight_iteration'] = placerdict['Llambda_density_weight_iteration']
            if 'Lsub_iteration' in placerdict:
                stage['Lsub_iteration'] = placerdict['Lsub_iteration']
                
        result_dict = placeflow_core(self.params, self.system, self.compact_model)

        best_pos = result_dict["best_fp_pos"]
        size_x, size_y = result_dict["best_fp_size"]
        best_metric = result_dict["best_metric"]
        Lgamma_metrics = result_dict["Lgamma_metrics"]
        hpwl = result_dict['hpwl']

        num_chiplets = self.system.num_chiplets
        num_nodes = self.system.num_nodes

        x_final = best_pos[0][0][:num_chiplets].detach().cpu().numpy()
        y_final = best_pos[0][0][num_nodes:num_nodes + num_chiplets].detach().cpu().numpy()
        theta_final = best_pos[0][1].detach().cpu().numpy()
        wl_norm = hpwl / self.sa_baseline[0] / 1e6 
        result = {}
        
        if (not self.optimize_pure_wl) or plot:
            temp_max, warpage, temp_field, warpage_field = evaluate_placement(
                self.system, self.physics_solver,
                (x_final, y_final), (size_x, size_y), self.file_dir
            )
            penalty = 0.0
            if temp_max > self.temp_threshold:
                penalty += self.penalty_coeff * (temp_max/self.temp_threshold)
            if warpage > self.warpage_threshold:
                penalty += self.penalty_coeff * (warpage/self.warpage_threshold)
            objective_value = wl_norm + penalty
            result.update({
                'penalty': penalty, 
                'temp_max': temp_max, 
                'temp_field': temp_field, 
                'warpage': warpage, 
                'warpage_field': warpage_field
            })
        else:
            objective_value = wl_norm   

        result.update({
            'hpwl': hpwl, 'wl_norm': wl_norm, 
            'objective': objective_value,
            'best_pos': (x_final, y_final, theta_final),
            'size_x': size_x, 'size_y': size_y,
            'Lgamma_metrics': Lgamma_metrics,
        })
        
        if plot:
            self.system.node_x[:num_chiplets] = x_final
            self.system.node_y[:num_chiplets] = y_final
            self.system.node_orient[:num_chiplets] = theta_final
            visualize_placement_results(
                self.system, result, self.log_dir, trial_id,
                plot_Lgamma=True, plot_thermal=True
            )

        return result
    
    def objective(self, trial: optuna.Trial) -> float:

        placerdict = {
            "iteration": 500,
            "Llambda_density_weight_iteration": 2,
            "Lsub_iteration": 5,
            "lr_pos": trial.suggest_float("lr_pos", 1e0, 1e3, log=True),
        }

        if self.optimize_pure_wl:
            placerdict.update({
                "eta": trial.suggest_float("eta", 0.1, 2.0),
                "lr_ang": trial.suggest_float("lr_ang", 1e-3, 1e-1, log=True),
                "overflow_init": trial.suggest_float("overflow_init", 0.2, 0.5),
                "target_density": trial.suggest_float("target_density", 0.2, 0.8),
                "density_weight_init": trial.suggest_float("density_weight_init", 1e0, 1e2, log=True),
                "wl_weight": trial.suggest_float("wl_weight", 1e-5, 1e-1, log=True),
            })
        else:
            if self.params.temp_aware_opt:
                placerdict.update({
                    "gamma": 2, #trial.suggest_int("gamma", 1, 6),
                    "temp_weight_init": trial.suggest_float("temp_weight_init", 1e0, 1e2, log=True),
                    "temp_threshold": trial.suggest_float("temp_threshold", 0, 80),
                })
            if self.params.warp_aware_opt:
                placerdict.update({
                    "warp_weight_init": trial.suggest_float("warp_weight_init", 1e0, 1e2, log=True),
                    "warp_threshold": trial.suggest_float("warp_threshold", 0, 70),
                })

        try:
            result = self.run_single_trial(placerdict, plot=False)
        except Exception as e:
            self.logger.error(f"Trial failed: {str(e)}")
            result = {"objective": float("inf"), "error": str(e)}

        objective_value = result["objective"]
        trial.set_user_attr("hpwl", result.get("hpwl", 0))
        trial.set_user_attr("temp_max", result.get("temp_max", 0))
        trial.set_user_attr("warpage", result.get("warpage", 0))
        trial.set_user_attr("penalty", result.get("penalty", 0))
        self.logger.info(self.params)
        self.logger.info(
            f"Trial {trial.number} @ {self.case_name}: Obj={objective_value:.4f}, "
            f"hpwl={result.get('hpwl', 0):.3e} um, "
            f"Temp={result.get('temp_max', 0):.2f} °C, "
            f"Warpage={result.get('warpage', 0):.2f} um. \n"
            f"\t{placerdict}"
        )
        return objective_value

    def optimize(self, n_trials: int = 100, seed_trial: Optional[Dict] = None) -> optuna.Study:
        """
        Run Optuna optimization.
        """
        study = optuna.create_study(direction="minimize", study_name=self.case_name)
        if seed_trial:
            study.enqueue_trial(seed_trial)
        study.optimize(self.objective, n_trials=n_trials, catch=(Exception,))
        return study
