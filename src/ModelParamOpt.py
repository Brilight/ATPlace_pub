import optuna
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
import torch
from tqdm import trange

from Parser import read_data
from MPA.PhysicsSolver import ATSim_solver, Warpage_solver
from MPA.ModelTrain import MultiOutputDataset, train_tmodel, train_wmodel, calc_error
from MPA.CompactModel import TModel, WModel


def train_case_model(model, data_loader, criterion, optimizer, scheduler, num_iter=10000):
    model.train()
    for iters in range(num_iter):
        optimizer.zero_grad()
        batch = next(iter(data_loader))
        input_data, labels = batch
        temp_label, warpage_label = labels
        x, y, w, h, power_tensor, mask = input_data

        out_warpage = model((x, y, w, h, temp_label))
        loss = criterion(out_warpage, warpage_label, mask)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if iters%1000==1:
            err = calc_error(out_warpage, warpage_label)
            print(f"\tMax RAE: {err[0]:.1f}%, Mean RAE: {err[1]:.1f}%, "
                  f"Max AE: {err[2]:.1f}, Mean AE: {err[3]:.1f}, Corr: {err[4]:.4f}")

    model.eval()
    with torch.no_grad():
        out_warpage = model((x, y, w, h, temp_label))
        err = calc_error(out_warpage, warpage_label)
        print(f"[WModel] Max RAE: {err[0]:.1f}%, Mean RAE: {err[1]:.1f}%, "
              f"Max AE: {err[2]:.1f}, Mean AE: {err[3]:.1f}, Corr: {err[4]:.4f}")

    return model


def evaluate_case_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        input_data, labels = batch
        temp_label, warpage_label = labels
        x, y, w, h, power_tensor, mask = input_data
        out_warpage = model((x, y, w, h, temp_label))
        pred_np = out_warpage.cpu().numpy().flatten()
        label_np = warpage_label.cpu().numpy().flatten()
        relative_error = (pred_np[label_np!=0] / label_np[label_np!=0]) - 1.0
        relative_error_sq = np.abs(relative_error).mean()
    return relative_error_sq


def objective(trial):
    total_relative_error_sq = 0.0

    lr = trial.suggest_float("lr", 1e-4, 1e0, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["Adam", "AdamW", "RMSprop"]
    )

    scheduler_name = trial.suggest_categorical(
        "scheduler", ["StepLR", "CosineAnnealingWarmRestarts", "None"]
    )
    criterion_name = trial.suggest_categorical(
        "criterion", ["MSE", "MAE", "Max", "Huber",]
    )

    if criterion_name == "MSE":
        criterion = lambda out, label, mask: ((out - label).pow(2)).mean()
    elif criterion_name == "MAE":
        criterion = lambda out, label, mask: (out - label).abs().mean()
    elif criterion_name == "Max":
        criterion = lambda out, label, mask: (out - label).pow(2).max()
    elif criterion_name == "Huber":
        delta = trial.suggest_float("huber_delta", 0.1, 2.0)
        criterion = lambda out, label, mask: F.huber_loss(out, label, delta=delta)

    print(f"Case {trial.number} with lr={lr:.2e}, {optimizer_name}, {criterion_name}")
    
    for i, (data_loader, system) in enumerate(zip(all_datasets, all_systems)):

        model = WModel(
            system.intp_width, system.intp_height,
            num_chiplets=system.num_chiplets,
            num_grid_x=system.num_grid_x, num_grid_y=system.num_grid_y
        )
        
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "AdamW":
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "SGD":
            momentum = trial.suggest_float("momentum", 0.1, 0.999)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)

        # 选择 scheduler
        if scheduler_name == "StepLR":
            step_size = 100
            gamma = trial.suggest_float("gamma", 0.5, 0.99)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif scheduler_name == "CosineAnnealingLR":
            T_max = trial.suggest_int("T_max", 100, 1000)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

        elif scheduler_name == "CosineAnnealingWarmRestarts":
            T_0 = trial.suggest_int("T_0", 50, 500)
            T_mult = trial.suggest_int("T_mult", 1, 2)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult)

        else:
            scheduler = None

        model = train_case_model(model, data_loader, criterion, optimizer, scheduler, 
                                 num_iter=5000)

        case_error = evaluate_case_model(model, data_loader)
        total_relative_error_sq += case_error

        print(f"{i}th Error: {case_error:.6f} | {total_relative_error_sq:.6f}")

        if np.isnan(case_error) or np.isinf(case_error):
            print(f"Case {i+1} FAILED: Invalid error = {case_error}. Returning inf.")
            return float('inf')
            
    return total_relative_error_sq  
    


if __name__=="__main__":
    default_params = '../Test_qipan/params.json'
    therm_dir = "/home/qpwang/Chiplet_place/ATPlace2.5D/ATSim/"
    thermal_configs = os.path.join(therm_dir, "config.json")

    all_datasets = []
    all_systems = []
    for index in range(1, 11):
        spec_params = f'../Test_qipan/Case{index}/configs.json'
        params, system, interposer = read_data(default_params, spec_params)
        system.thermal_dir = therm_dir

        dataset_path = f'../Test_qipan/Case{index}/dataset_train.pt'
        dataset = MultiOutputDataset(
            num_chiplets=system.num_chiplets,
            W=system.intp_width, H=system.intp_height,
            num_grid_x=system.num_grid_x, num_grid_y=system.num_grid_y,
            data_path=dataset_path
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        all_datasets.append(data_loader)
        all_systems.append(system)

    print("Starting Optuna hyperparameter optimization across all 10 Cases...")
    study = optuna.create_study(direction="minimize", study_name="Global_WModel_Opt")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Objective Value (Total Relative Error Squared): {study.best_value:.6f}")
