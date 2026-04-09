import os
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.set_printoptions(suppress=True,precision=5)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from copy import copy
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Any

from utils.Visualize import plot_double_y, plot_temp_pos, plot_fp
from ATPLACE.RandomPlace import Random_init_gurobi as Random_init
from ATPLACE.Legalization import Legalization
from SA.SAInit import Initialization as InitSA
from MPA.CompactModel import TModel, WModel


class MultiOutputDataset(Dataset):
    def __init__(self, num_chiplets, W, H, num_grid_x, num_grid_y, data_path=None):
        self.data = []
        self.num_chiplets = num_chiplets
        self.W, self.H = W, H
        self.num_grid_x, self.num_grid_y = num_grid_x, num_grid_y
        
        x_coords = (torch.arange(num_grid_x) + 0.5) / num_grid_x * self.W
        y_coords = (torch.arange(num_grid_y) + 0.5) / num_grid_y * self.H
        X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        self.X = X[None, ...].repeat(num_chiplets, 1, 1)
        self.Y = Y[None, ...].repeat(num_chiplets, 1, 1)

        if data_path is not None:
            self.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def add_data(self, geo_data, power, temp_label, warpage_label=None):
        x, y, width, height = geo_data
        x = torch.Tensor(x).float().reshape(-1, 1, 1)
        y = torch.Tensor(y).float().reshape(-1, 1, 1)
        w = torch.Tensor(width).float().reshape(-1, 1, 1)
        h = torch.Tensor(height).float().reshape(-1, 1, 1)

        mask = (((self.X - x).abs() <= w / 2) & ((self.Y - y).abs() <= h / 2)).int()

        if warpage_label is not None:
            warpage_tensor = torch.from_numpy(warpage_label).float().unsqueeze(0)
        else:
            warpage_tensor = None

        temp_tensor = torch.from_numpy(temp_label).float().unsqueeze(0)
        power_tensor = torch.Tensor(power).float().reshape(-1)

        self.data.append((
            (x, y, w, h, power_tensor, mask),
            (temp_tensor, warpage_tensor)
        ))

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.data, file_path)
        print(f"Dataset saved to {file_path}")

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        self.data = torch.load(file_path, map_location='cpu')
        print(f"Dataset loaded from {file_path} with {len(self.data)} samples")


def Multi_Physics_Data_Gen(
    system,
    physics_solver,
    num_dataset_train: int,
    num_dataset_test: int = 0,
    save_dir: Optional[str] = None,
    plot_flag: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate combined train/test dataset.
    - Total samples = num_dataset_train + num_dataset_test
    - Only train dataset is saved to disk (if save_dir provided)
    - If num_dataset_test <= 0, test_loader == train_loader
    """

    num_total = num_dataset_train + max(0, num_dataset_test)
    if num_total <= 0:
        raise ValueError("Total dataset size must be > 0")

    Temp_data_log = []
    idx = 0
    num_chiplets = system.num_chiplets
    amb = physics_solver.amb
    t1 = time.time()
    dis_bet_chips = (system.xhigh - system.xlow) / 100 / num_chiplets
    fense_size = (system.xhigh - system.xlow) / 10

    while idx < num_total:

        temp_file_name = f'CTM{idx}'
        x_init = np.random.normal(
            loc=(system.xhigh + system.xlow) / 2,
            scale=(system.xhigh - system.xlow) / 2,
            size=num_chiplets
        )
        y_init = np.random.normal(
            loc=(system.yhigh + system.ylow) / 2,
            scale=(system.yhigh - system.ylow) / 2,
            size=num_chiplets
        )
        init_angle = np.random.uniform(0, 2 * np.pi, size=num_chiplets)
        size_x, size_y = system.rotate(np.arange(num_chiplets), init_angle)
        size_x, size_y = np.array(size_x), np.array(size_y)

        pos = Random_init(system, [x_init, y_init], [size_x, size_y], fencesize=0)
        while pos is None:
            pos = Random_init(
                system, [x_init, y_init], [size_x, size_y],
                fencesize=fense_size * np.random.rand()
            )

        if plot_flag:
            plot_fp(system, [np.concatenate((pos[0], pos[1])), np.zeros(num_chiplets)])

        t2 = time.time()
        try:
            powermap_tmp = np.array(system.powermap)
            physics_solver.set_pos(
                powermap_tmp, [np.array(pos[0]), np.array(pos[1])], [size_x, size_y]
                )
            physics_solver.run(temp_file_name, default=int(plot_flag))
            print(f"Takes {time.time() - t2:.2f} s for simulation.")
            res = physics_solver.getres(temp_file_name)
            Temp_whole = res['temperature'] - 273.15 - amb
            Warpage_whole = res.get('warpage', None)
        except Exception as e:
            print(f"Simulation failed for CTM{idx}: {e}")
            continue

        flp_x = torch.from_numpy(pos[0]).type(torch.float32)
        flp_y = torch.from_numpy(pos[1]).type(torch.float32)
        flp_sx = torch.from_numpy(size_x).type(torch.float32)
        flp_sy = torch.from_numpy(size_y).type(torch.float32)

        Temp_data_log.append((
            (flp_x, flp_y, flp_sx, flp_sy),
            powermap_tmp, Temp_whole, Warpage_whole
        ))

        if plot_flag:
            system.node_x, system.node_y = pos[0], pos[1]
            plot_temp_pos(system, Temp_whole + amb)
            if Warpage_whole is not None:
                plot_temp_pos(system, Warpage_whole)

        idx += 1
        print(f"Takes {time.time() - t1:.2f} sec for generating {idx}/{num_total} cases.")

    # Initialize full dataset
    dataset = MultiOutputDataset(
        num_chiplets, system.intp_width, system.intp_height,
        system.num_grid_x, system.num_grid_y
    )
    for item in Temp_data_log:
        dataset.add_data(*item)

    # Split dataset
    if num_dataset_test <= 0:
        # No test set → return same loader for both
        train_loader = DataLoader(dataset, batch_size=num_dataset_train, shuffle=True)
        test_loader = train_loader
    else:
        # Create Subset for train and test
        train_indices = list(range(num_dataset_train))
        test_indices = list(range(num_dataset_train, num_total))

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=num_dataset_train, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=num_dataset_test, shuffle=False)

    # Save ONLY train dataset (or full if no test) to disk
    if save_dir:
        if not isinstance(save_dir, str) or not save_dir.strip():
            raise ValueError("save_dir must be a non-empty string")
        os.makedirs(save_dir, exist_ok=True)
        dataset_to_save = dataset if num_dataset_test <= 0 else train_subset.dataset
        dataset_path = os.path.join(save_dir, "dataset_train.pt")
        dataset_to_save.save(dataset_path)

    return train_loader, test_loader


def calc_error(pred, target):
    pred_flat = pred.flatten().detach().cpu().numpy()
    targ_flat = target.flatten().cpu().numpy()
    corr = np.corrcoef(pred_flat, targ_flat)[0, 1]
    rae = np.abs(pred_flat / targ_flat - 1) * 100
    max_rae = np.nanmax(rae)
    mean_rae = np.nanmean(rae)
    ae = np.abs(pred_flat - targ_flat)
    max_ae = np.max(ae)
    mean_ae = np.mean(ae)
    return [max_rae, mean_rae, max_ae, mean_ae, corr]


def train_tmodel(
    W, H, num_chiplets, num_grid_x, num_grid_y, lr, data_loader, criterion, num_iter, 
    data_loader_test=None
):
    model = TModel(W, H, num_chiplets, num_grid_x, num_grid_y)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    train_log = np.array([]).reshape(5, -1)
    pbar = trange(num_iter, desc="Training TModel")

    for iters in pbar:
        model.train()
        optimizer.zero_grad()
        batch = next(iter(data_loader))
        input_data, labels = batch
        temp_label, warpage_label = labels
        (x, y, w, h, power_tensor, mask) = input_data
        
        out_temp = model((x, y, w, h, power_tensor))
        loss = criterion(out_temp, temp_label, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        err = calc_error(out_temp, temp_label)
        train_log = np.column_stack((train_log, np.array(err)))
        pbar.set_description(f"TModel Loss: {loss.item():.4f}")
        if iters%1000==1:
            logging.info(f"[TModel] Max RAE: {err[0]:.1f}%, Mean RAE: {err[1]:.1f}%, "
              f"Max AE: {err[2]:.1f}, Mean AE: {err[3]:.1f}, Corr: {err[4]:.3f}")

    plot_keys = ['MaxRAE%', 'MeanRAE%', 'MaxAE', 'MeanAE', 'Corr']
    plot_double_y(train_log[0], train_log[3], ['MaxRAE%', 'MeanAE'], log=[1, 0])

    model.eval() # Final evaluation
    with torch.no_grad():
        if data_loader_test is not None:
            input_data, labels = next(iter(data_loader_test))
            temp_label, warpage_label = labels
            (x, y, w, h, power_tensor, mask) = input_data
        out_temp = model((x, y, w, h, power_tensor))
        err = calc_error(out_temp, temp_label)
        print(f"[TModel] Max RAE: {err[0]:.1f}%, Mean RAE: {err[1]:.1f}%, "
              f"Max AE: {err[2]:.1f}, Mean AE: {err[3]:.1f}, Corr: {err[4]:.3f}")

    return model


def train_wmodel(
    W, H, num_chiplets, num_grid_x, num_grid_y, lr, data_loader, criterion, num_iter, 
    data_loader_test=None
):
    model = WModel(W, H, num_chiplets, num_grid_x, num_grid_y)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    train_log = np.array([]).reshape(5, -1)
    pbar = trange(num_iter, desc="Training WModel")

    for iters in pbar:
        model.train()
        optimizer.zero_grad()
        
        batch = next(iter(data_loader))
        input_data, labels = batch
        temp_label, warpage_label = labels
        (x, y, w, h, power_tensor, mask) = input_data

        sum_temp = (temp_label * mask).sum(dim=[2, 3])
        count = mask.sum(dim=[2, 3])
        avg_temp = (sum_temp / count).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        out_warpage = model((x, y, w, h, temp_label))
        loss = criterion(out_warpage, warpage_label, mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        err = calc_error(out_warpage, warpage_label)
        train_log = np.column_stack((train_log, np.array(err)))
        pbar.set_description(f"WModel Loss: {loss.item():.4f}")
        if iters%1000==1:
            logging.info(f"[WModel] Max RAE: {err[0]:.1f}%, Mean RAE: {err[1]:.1f}%, "
              f"Max AE: {err[2]:.1f}, Mean AE: {err[3]:.1f}, Corr: {err[4]:.3f}")

    plot_keys = ['MaxRAE%', 'MeanRAE%', 'MaxAE', 'MeanAE', 'Corr']
    #plot_double_y(train_log[0], train_log[1:], plot_keys, log=[1]*4 + [0])
    plot_double_y(train_log[0], train_log[3], ['MaxRAE%', 'MeanAE'], log=[1, 1])

    model.eval()
    with torch.no_grad():
        if data_loader_test is not None:
            input_data, labels = next(iter(data_loader_test))
            temp_label, warpage_label = labels
            (x, y, w, h, power_tensor, mask) = input_data
        out_warpage = model((x, y, w, h, temp_label))
        err = calc_error(out_warpage, warpage_label)
        print(f"[WModel] Max RAE: {err[0]:.1f}%, Mean RAE: {err[1]:.1f}%, "
              f"Max AE: {err[2]:.1f}, Mean AE: {err[3]:.1f}, Corr: {err[4]:.3f}")

    return model


def get_models(
    system, physics_solver, criterion, 
    num_dataset_train=10, num_dataset_test=0, lr=1e-1, num_iters=1000, warpage_opt=False
):
    # Generate data loaders
    data_loader_train = Multi_Physics_Data_Gen(
        num_dataset_train, system, physics_solver, plot_flag=True
    )
    data_loader_test = Multi_Physics_Data_Gen(
        num_dataset_test, system, physics_solver, plot_flag=False
    ) if num_dataset_test > 0 else data_loader_train
    
    tmodel = train_tmodel(
        W=system.intp_width,
        H=system.intp_height,
        num_chiplets=system.num_chiplets,
        num_grid_x=system.num_grid_x,
        num_grid_y=system.num_grid_y,
        lr=lr,
        data_loader=data_loader_train,
        criterion=criterion,
        num_iter=num_iters,
        data_loader_test=data_loader_test
    )
    for param in tmodel.parameters():
        param.requires_grad_(False)
        
    if warpage_opt:
        wmodel = train_wmodel(
            W=system.intp_width,
            H=system.intp_height,
            num_chiplets=system.num_chiplets,
            num_grid_x=system.num_grid_x,
            num_grid_y=system.num_grid_y,
            lr=lr,
            data_loader=data_loader_train,
            criterion=criterion,
            num_iter=num_iters,
            data_loader_test=data_loader_test
        )

        for param in wmodel.parameters():
            param.requires_grad_(False)

        return tmodel, wmodel

    return tmodel
    