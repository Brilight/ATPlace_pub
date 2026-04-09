import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.set_printoptions(suppress=True,precision=5)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from copy import copy
from tqdm import tqdm, trange

from utils.Visualize import plot_double_y, plot_temp_pos
from ATPLACE.RandomPlace import Random_init
from ATPLACE.Legalization import Legalization
from SA.SAInit import Initialization as InitSA
from Thermal.TModel import TModel, WModel


class TempDataset(Dataset):
    def __init__(self, num_chiplets, W, H, num_grid_x, num_grid_y):
        self.data = []
        self.num_chiplets = num_chiplets
        self.W, self.H = W, H
        self.num_grid_x, self.num_grid_y = num_grid_x, num_grid_y
        X, Y = torch.meshgrid((torch.arange(self.num_grid_x)+0.5)/self.num_grid_x*self.W, 
                              (torch.arange(self.num_grid_y)+0.5)/self.num_grid_y*self.H, indexing='ij')
        self.X = X[..., None].repeat(1, 1, num_chiplets)
        self.Y = Y[..., None].repeat(1, 1, num_chiplets)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def add_data(self, power, new_data, label):
        # Add new data to the dataset
        for data in new_data:
            assert data.shape[0]==self.num_chiplets, "Must provide info of all chiplets!"
        x, y, width, height = new_data
        weights = (torch.where(((self.X-x).abs()<=width/2),1,0)*torch.where(((self.Y-y).abs()<=height/2),1,0))
        mask = (weights).sum(-1)         #*power/width/height*self.W*self.H
        power = torch.from_numpy(power).type(torch.float32).reshape(-1,1)
        self.data.append(((x, y, width, height,  power, mask), label)) 

        
def Datagen(num_dataset, system, temp_solver, plot_flag=True):
    '''
    num of train set does not affect the final max error
    '''
    Temp_data_log = []
    idx = 0
    num_chiplets = system.num_chiplets
    amb = temp_solver.amb
    t1 = time.time()
    dis_bet_chips = (system.xhigh-system.xlow)/100/num_chiplets
    idx = 0
    fencesize = 0
    while idx < num_dataset:
        temp_file_name = f'CTM{idx}'
        x_init = np.random.normal(loc=(system.xhigh+system.xlow)/2, 
                                  scale=(system.xhigh-system.xlow)/2, size=num_chiplets)
        y_init= np.random.normal(loc=(system.yhigh+system.ylow)/2, 
                                 scale=(system.yhigh-system.ylow)/2, size=num_chiplets)
        init_angle = np.random.normal(loc=np.pi, scale=np.pi, size=num_chiplets)
        size_x, size_y = system.rotate(np.arange(num_chiplets), init_angle)
        #model.legalize_theta(torch.from_numpy(init_angle))
        size_x, size_y = np.array(size_x), np.array(size_y)
        pos = Random_init(system, [x_init, y_init], [size_x, size_y], fencesize, 10)
        if pos is None:
            fencesize = 100*(1+np.random.rand()*idx)
            continue
        fencesize = 0
        powermap_tmp = np.array(system.powermap)        
        temp_solver.set_pos(powermap_tmp, [np.array(pos[0]), np.array(pos[1])], [size_x, size_y])
        t2 = time.time()
        temp_solver.run(temp_file_name)
        print(f"Takes {time.time()-t2:.2f} s for thermal simulation.")
        Temp_whole = temp_solver.getres(temp_file_name)-273.15-amb      
        flp_x = torch.from_numpy(pos[0]).type(torch.float32)
        flp_y = torch.from_numpy(pos[1]).type(torch.float32)
        flp_sx = torch.from_numpy(size_x).type(torch.float32)
        flp_sy = torch.from_numpy(size_y).type(torch.float32)
        Temp_data_log.append((
            powermap_tmp, (flp_x, flp_y,flp_sx, flp_sy), 
            torch.from_numpy(Temp_whole).type(torch.float32)
        ))
        if plot_flag:
            system.node_x, system.node_y = pos[0], pos[1]
            plot_temp_pos(system, Temp_whole+amb)
        idx += 1
        print(f"Takes {time.time()-t1:.2f} sec for generate {idx} cases.")

    np.random.shuffle(Temp_data_log)
    dataset_test = TempDataset(
        num_chiplets, system.intp_width, system.intp_height, 
        system.num_grid_x, system.num_grid_y
    )
    for i in range(num_dataset):
        dataset_test.add_data(*Temp_data_log[i])
    data_loader = DataLoader(dataset_test, batch_size=num_dataset, shuffle=True)
    return data_loader


def train_model(W, H, num_chiplets, num_grid_x, num_grid_y, lr, data_loader, criterion, num_iter, data_loader_test=None):
    #input data: 
    #   the x,y,width,height of all chiplets
    #labels: 
    #   temperature map simulated by HotSpot - ambient temperature

    Tmodel = TModel(W, H, num_chiplets, num_grid_x, num_grid_y)    
    optimizer = optim.Adam(Tmodel.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
    
    train_log = np.array([]).reshape(6,-1)
    for iters in trange(num_iter):
        optimizer.zero_grad()
        input_data, labels = next(iter(data_loader))
        weights = input_data[-1]
        outputs = Tmodel(input_data)
        Loss = criterion(outputs, labels, weights)
        Loss.backward()
        optimizer.step(); scheduler.step()
        train_err = [Loss.item(), 
                     (outputs/labels-1).abs().max().item()*100, (outputs/labels-1).abs().mean().item()*100, 
                     (outputs-labels).abs().max().item(), (outputs-labels).abs().mean().item(),
                   np.corrcoef(np.stack((outputs.detach().numpy().flatten(), labels.flatten())))[0,1]]
        train_log = np.column_stack((train_log, np.array(train_err).reshape(-1,1)))

    plot_double_y(train_log[0],[train_log[1], train_log[2]],['Loss', 'Error/%'],log=[1,1])
    print(f"Train error (Loss {train_err[0]:.4f}, Max RAE {train_err[1]:.4f}, Mean RAE {train_err[2]:.4f}, "+\
          f"Max AE {train_err[3]:.4f}, Mean AE {train_err[4]:.4f}, Correlation {train_err[5]:.4f})")
    
    if data_loader_test is None:
        data_loader_test = data_loader
    input_data, labels = next(iter(data_loader_test))
    with torch.no_grad():
        outputs = Tmodel(input_data)
    test_err = [(outputs/labels-1).abs().max().item()*100, (outputs/labels-1).abs().mean().item()*100, 
                (outputs-labels).abs().max().item(), (outputs-labels).abs().mean().item(),
                np.corrcoef(np.stack((outputs.detach().numpy().flatten(), labels.flatten())))[0,1]]
    print(f"Test error (Max RAE {test_err[0]:.4f}, Mean RAE {test_err[1]:.4f}, "+\
          f"Max AE {test_err[2]:.4f}, Mean AE {test_err[3]:.4f}, Correlation {test_err[4]:.4f})")
    
    return Tmodel
    
def get_model(system, model, num_dataset, lr = 1e-1):
    #num of train set almost does not affect the final max error
    data_loader = Datagen(num_dataset, system, model, therm_dir_path, temp_solver)
    num_iters = 800
    criterion = lambda out,label,mask: ((out-label).pow(2)).mean()
    tempmodel = train_model(system.intp_width, system.intp_height, system.num_chiplets, system.num_grid_x, 
                            system.num_grid_y, lr, data_loader, criterion, num_iters)
    for param in tempmodel.parameters():
        param.requires_grad_(False)
    return tempmodel