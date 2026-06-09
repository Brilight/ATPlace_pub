#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import torch 
#import torch.nn.functional as F

import math
import os

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '35'
plt.rcParams['font.family'] = 'DejaVu Serif'
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import mpl_toolkits.mplot3d as Axes3D
linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot'] 


def plot_temp_pos(system, node_x, node_y, node_size_x, node_size_y, Temp):
    fig, ax = plt.subplots(figsize=(12, 10))
    for spine in ax.spines.values():
        spine.set_visible(False)
    x_scale, y_scale = system.intp_width, system.intp_height
    X, Y = np.meshgrid(np.arange(0, Temp.shape[0])*x_scale/(Temp.shape[0]-1), 
                       np.arange(0, Temp.shape[1])*y_scale/(Temp.shape[1]-1), indexing='ij')
    cmap = plt.get_cmap(cm.jet)
    levels = np.linspace(Temp.min(), Temp.max(), 50)
    plt.tick_params(labelsize=20)
    for i in range(system.num_chiplets):
        name = system.node_names[i]#+f'\n{powermap[i]:.1f}W'
        length, width = node_size_x[i], node_size_y[i]
        x, y = node_x[i], node_y[i]
        rect = plt.Rectangle((x-length/2, y-width/2), length, width, angle = 0, 
                       edgecolor='k', linewidth=2, facecolor='black', alpha=.6)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', alpha=0.9, fontsize=25)
    print(Temp.max())
    cset = ax.contourf(X, Y, Temp, levels, cmap=cmap, alpha=0.7) 
    position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
    cbar.set_ticks((np.round(np.linspace(levels[0]+0.05, levels[-1]-0.05, 5),1)).tolist())
    plt.show()
    #length, width = system.node_size_x[cp_idx], system.node_size_y[cp_idx]
    #    x, y = pos[0][cp_idx].item(), pos[0][cp_idx+system.num_nodes].item()
    #    angle = pos[1][cp_idx].item()
    
def plot_double_y(y1,y2,name,log=None):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(y1, color='b')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(name[0], color='b')
    if log[0]:
        ax1.set_yscale("log")
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    if isinstance(y2,list):
        for i in range(len(y2)):
            ax2.plot(y2[i], color='k', linestyle=linestyle_str[i])
    else:
        ax2.plot(y2, color='k', linestyle=linestyle_str[0])
    ax2.set_ylabel(name[1], color='k')
    if log[1]:
        ax2.set_yscale("log")
    ax2.tick_params(axis='y', labelcolor='k')

    plt.show()

def plot_temp_fp(flp, power, Temps, x_scale, y_scale, color_error=0):

    fig, ax = plt.subplots(figsize=(12, 10))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tick_params(labelsize=20)
    X, Y = np.meshgrid(np.arange(0, Temps.shape[0])*x_scale/(Temps.shape[0]-1), 
                       np.arange(0, Temps.shape[1])*y_scale/(Temps.shape[1]-1), indexing='ij')
    if color_error:
        cmap = plt.get_cmap("coolwarm")
        val_range = max(abs(Temps.min()), abs(Temps.max()))+0.03
        levels = np.linspace(-val_range, val_range, 50)
    else:
        cmap = plt.get_cmap(cm.jet)
        levels = np.linspace(Temps.min()-0.1, Temps.max()+0.1, 50)
    cset = ax.contourf(X, Y, Temps, levels, cmap=cmap) 

    for i in range(len(flp)):
        name = flp[i][0]
        if name.startswith("Edge"):
            continue
        if name.startswith("WS"):
            name = ''
        if name.startswith("Ubump"):
            name = f"u"#int(name.split('_')[-1])%4}"
        if name.startswith("Chiplet"):
            name = f"$C_{name[-1]}$\n{power[int(name[-1])]:.1f}W"
        length, width = flp[i][1], flp[i][2]
        x, y = flp[i][3], flp[i][4]
        rect = plt.Rectangle((x, y), length, width, linestyle="-", linewidth=1, edgecolor='k', #facecolor='none',
                             facecolor=cmap(.005/flp[i][-1]), alpha=.3)
        ax.add_patch(rect)
        ax.text(x + 0.5*length, y+0.5*width, name, ha='center', va='center', alpha=0.8, fontsize=24)

    Nx, Ny = Temps.shape
    plt.xlim([0, x_scale]); plt.ylim([0, y_scale])
    plt.xticks([]); plt.yticks([])
    position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
    cbar.set_ticks((np.round(np.linspace(levels[0]+0.05, levels[-1]-0.05, 5),1)).tolist())
    plt.show()

def plot_dens(system, pos, density, terminal_plot=False):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    x_scale, y_scale = system.node_x.max(), system.node_y.max()
    legal_angles = np.arange(5)*np.pi/2
    levels = np.linspace(density.min().item()*1.01, density.max().item()*1.01, 50)
    plt.tick_params(labelsize=20)
    for cp_idx in range(system.num_chiplets):
        name = system.node_names[cp_idx]
        length, width = system.node_size_x[cp_idx], system.node_size_y[cp_idx]
        x, y = pos[0][cp_idx].detach().numpy().item(), pos[0][cp_idx+system.num_nodes].detach().numpy().item()
        angle = pos[1][cp_idx].detach().numpy().item()
        ax.text(x, y, name, ha='center', va='center', alpha=0.7, fontsize=25)
        angle = legal_angles[np.argmin(np.abs(angle-legal_angles))].item()
        ax.add_patch(plt.Rectangle((x-length/2, y-width/2), length, width, 
                       angle = angle/np.pi*180, rotation_point='center', facecolor='grey', alpha=.5))
    cset = ax.contourf(system.bin_center_x, system.bin_center_y,
                       density.detach().numpy(), cmap="coolwarm", alpha=.6)
    position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
    cbar.set_ticks((np.round(np.linspace(levels[0]+0.05, levels[-1]-0.05, 5),1)).tolist())
    plt.show()
    
def plot_fp(system, pos, terminal_plot=False, net_plot=False):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    x_scale, y_scale = system.intp_width, system.intp_height
    legal_angles = np.arange(5)*np.pi/2
    plt.tick_params(labelsize=20)
    cmap = plt.get_cmap(cm.jet)
    for i in range(system.num_chiplets):
        name = system.node_names[i]
        length, width = system.node_size_x[i], system.node_size_y[i]
        x, y = pos[0][i].item(), pos[0][i+system.num_nodes].item()
        angle = pos[1][i].item()
        ax.scatter(x+length/2*np.cos(angle), y+length/2*np.sin(angle), color=cmap(.12*i), marker='x', s=100)
        plt.plot([x, x+length/2*np.cos(angle)],[y, y+length/2*np.sin(angle)],linewidth=2.5,color=cmap(.12*i),alpha=.9)
        angle = legal_angles[np.argmin(np.abs(angle-legal_angles))].item()
        rect = plt.Rectangle((x-length/2, y-width/2), length, width, angle = angle/np.pi*180, 
                      rotation_point='center', edgecolor='k', linewidth=1, facecolor='grey', alpha=.5)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', alpha=0.7, fontsize=25)
        ax.scatter(x, y, color=cmap(.12*i), marker='o', s=100)

    if terminal_plot:
        for i in range(system.num_chiplets, system.num_nodes):
            x, y = pos[0][i].item(), pos[0][i+system.num_nodes].item()
            plt.scatter(x, y, marker='x', s=50, color='r', alpha=.9)

    if net_plot:
        connection = np.zeros((system.num_nodes, system.num_nodes))
        for net_id in system.net_id:
            pin_id = system.net2pin_map[net_id]
            node_id = system.pin2node_map[pin_id]
            for idx in range(len(node_id)):
                for suc in range(len(node_id)):
                    connection[node_id[idx], node_id[suc]] += 2/len(node_id)    
        for idx in range(system.num_chiplets):
            for suc in range(idx, system.num_nodes):
                if connection[idx,suc]!=0:
                    linewidth = np.round((connection[idx,suc]/connection.max())**0.7*15,1)
                    plt.plot([pos[0][idx].item(), pos[0][suc].item()], 
                             [pos[0][idx+system.num_nodes].item(), pos[0][suc+system.num_nodes].item()], 
                             linewidth=linewidth, linestyle='--', alpha=.5, color='purple')


    plt.axvline(x=system.xlow, color='k', linestyle='-.',)
    plt.axvline(x=system.xhigh, color='k', linestyle='-.',)
    plt.axhline(y=system.ylow, color='k', linestyle='-.',)
    plt.axhline(y=system.yhigh, color='k', linestyle='-.',)

    plt.axvline(x=0, color='k', linestyle='-',)
    plt.axvline(x=system.intp_width, color='k', linestyle='-',)
    plt.axhline(y=0, color='k', linestyle='-',)
    plt.axhline(y=system.intp_height, color='k', linestyle='-',)

    #plt.xlim([-20, x_scale+20]); plt.ylim([-20, y_scale+20])
    plt.xticks([0, x_scale/3//100*100, x_scale*2/3//100*100, x_scale//100*100])
    plt.yticks([y_scale/3//100*100, y_scale*2/3//100*100, y_scale//100*100])
    plt.show()

def plot_temp(Temps, flp, power, color_error=0):
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = '35'
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(12, 10))
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tick_params(labelsize=20)
    X, Y = np.meshgrid(np.arange(0, Temps.shape[0])/(Temps.shape[0]-1), 
                       np.arange(0, Temps.shape[1])/(Temps.shape[1]-1), indexing='ij')
    if color_error:
        cmap = plt.get_cmap("coolwarm")
        val_range = max(abs(Temps.min()), abs(Temps.max()))+0.03
        levels = np.linspace(-val_range, val_range, 50)
    else:
        cmap = plt.get_cmap(cm.jet)
        levels = np.linspace(Temps.min()-0.1, Temps.max()+0.1, 50)
    cset = ax.contourf(X, Y, Temps, levels, cmap=cmap) 

    fx, fy, fwidth, fheight = flp
    for i in range(fx.shape[0]):
        x, y = fx[i].item(), fy[i].item()
        width, height = fwidth[i].item(), fheight[i].item()
        rect = plt.Rectangle((x-0.5*width, y-0.5*height), width, height, 
                             linestyle="-", linewidth=1, edgecolor='k', facecolor='g', alpha=.4)
        ax.add_patch(rect)
        ax.text(x, y, f'{i}_{power[i]:.1f}', ha='center', va='center', alpha=0.8, fontsize=24)

    Nx, Ny = Temps.shape
    #plt.xlim([0, x_scale]); plt.ylim([0, y_scale])
    plt.xticks([0,]); plt.yticks([0,])
    position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
    cbar.set_ticks((np.round(np.linspace(levels[0]+0.05, levels[-1]-0.05, 5),1)).tolist())
    plt.show()

def error_plot(error, length, width):

    plt.rcParams['font.size'] = '30'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=35)
    Nx, Ny = error.shape
    X, Y = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), indexing='ij')
    bound = np.max(abs(error))
    levels = np.linspace(-bound, bound, 50)
    cset = ax.contourf(X, Y, error, levels, cmap="coolwarm")#"Spectral_r")     
    plt.xlim([0, Nx]); plt.ylim([0, Ny])
    plt.xticks([0,Nx//3,Nx//3*2,Nx], [0, np.round(length/3, 4),np.round(length/3*2, 4), np.round(length, 4)])
    plt.yticks([Ny//3,Ny//3*2,Ny], [np.round(width/3, 4),np.round(width/3*2, 4), np.round(width, 4)])

    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    bound *= 0.99
    cbar.set_ticks(np.round([-bound, -bound/2, 0, bound/2, bound], 2))
    cbar.ax.set_title('$\Delta$T(℃)', fontsize=30, pad=25)
    plt.show()

def temp_plot(Temps, length, width, reso=1, rdbu=False, name=None):

    plt.rcParams['font.size'] = '30'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=35)
    Nx, Ny = Temps.shape
    X, Y = np.meshgrid(np.arange(0, Nx, reso), np.arange(0, Ny, reso), indexing='ij')
    levels = np.linspace(np.floor(Temps.min()*10)/10, np.ceil(Temps.max()*10)/10+0.1, 50)
    if rdbu:
        cset = ax.contourf(X, Y, Temps, levels, cmap="RdBu")    
    else:
        cset = ax.contourf(X, Y, Temps, levels, cmap=plt.cm.jet)    
    plt.xlim([0, Nx]); plt.ylim([0, Ny])
    plt.xticks([0,Nx//3,Nx//3*2,Nx], [0, np.round(length/3, 4),np.round(length/3*2, 4), np.round(length, 4)])
    plt.yticks([Ny//3,Ny//3*2,Ny], [np.round(width/3, 4),np.round(width/3*2, 4), np.round(width, 4)])
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    cbar.set_ticks([np.round(val, 1) for val in np.linspace(levels[0], levels[-1], 6).tolist()])
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=30)
    cbar.ax.set_title('T(℃)', fontsize=30, pad=25)
    plt.show()

def power_plot(flp_df):
    from matplotlib.ticker import FuncFormatter
    plt.rcParams['font.size'] = '30'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    length = np.max(flp_df['X'] + flp_df['Length (m)'])
    width = np.max(flp_df['Y'] + flp_df['Width (m)'])
    flp_df["Powerdens"] = (flp_df["Power_dyn"]+flp_df["Power_leak"])/flp_df["Length (m)"]/flp_df["Width (m)"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=24)
    cmap = plt.cm.get_cmap("turbo")

    for i in range(flp_df.shape[0]):
        name = flp_df.iloc[i]['UnitName']
        x, y = flp_df.iloc[i]["X"], flp_df.iloc[i]["Y"]
        rect = plt.Rectangle((x, y), flp_df.iloc[i]["Length (m)"], flp_df.iloc[i]["Width (m)"], 
                linewidth=0, facecolor=cmap(flp_df.iloc[i]["Powerdens"]/flp_df["Powerdens"].max()))
        ax.add_patch(rect)

    ax.set_xlim(0, length); ax.set_xticks([0,length/3,length/3*2,length])
    ax.set_ylim(0, width); ax.set_yticks([width/3,width/3*2,width])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([]) # 此行是为了使ScalarMappable正常工作

    position = fig.add_axes([0.99, 0.15, 0.02, 0.65])
    cbar = plt.colorbar(sm, ax=ax, pad=0, fraction=0, cax=position)

#    cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_format_func))
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbarvals = np.linspace(0, np.ceil(flp_df["Powerdens"].max()), 6)
    cbar.set_ticklabels(['{:.1e}'.format(val) for val in cbarvals])
    #cbar.set_ticklabels([sci_format_func(val, None) for val in cbarvals])
    #cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=30)
    cbar.ax.set_title('W/m$^2$', fontsize=30, pad=15 )
    plt.show()

def Temp_with_FP(Temps, reso, Trange, flp_df, x_scale, y_scale, barname='Temp(°C)'):
    
    fig, ax = plt.subplots(figsize=(10, 9))
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tick_params(labelsize=20)
    X, Y = np.meshgrid(np.arange(0, Temps.shape[0], reso),
                       np.arange(0, Temps.shape[1], reso), indexing='ij')
    levels = np.linspace(Trange[0], Trange[1], 50)
    cmap = cm.jet if Trange[0]>1 else "OrRd"
    cset = ax.contourf(X, Y, Temps, levels, cmap=cmap)   
    for i in range(flp_df.shape[0]):
        name = flp_df.iloc[i]['UnitName']
        x = flp_df.iloc[i]["X"]*x_scale
        y = flp_df.iloc[i]["Y"]*y_scale
        length = flp_df.iloc[i]["Length (m)"]*x_scale
        width = flp_df.iloc[i]["Width (m)"]*y_scale
        rect = plt.Rectangle((x, y), length, width, linestyle="--", linewidth=1, edgecolor='grey', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + 0.5*length, y+0.5*width, name, ha='center', va='center', alpha=0.8, fontsize=26)

    Nx, Ny = Temps.shape
    plt.xlim([-1, Nx+1]); plt.ylim([-1, Ny+1])
    plt.xticks([]); plt.yticks([])
    position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
    cbar.set_label(barname,fontsize=24,x=0.5)
    cbar.set_ticks((np.linspace(levels[0], levels[-1], 5)*100//10/10).tolist())
    plt.show()

def Plot3d(data, z_reso=5):
    if data.max()<=data.min() or np.nan in data:
        return "The maximum of data must be larger than the minimum"
    # Create 3D plot
    scale = min(3,100/max(data.shape))
    data = F.interpolate(torch.Tensor(data).reshape(1,1,*data.shape), size=(int(scale*data.shape[0]),
                int(scale*data.shape[1]),data.shape[-1]), mode='trilinear', align_corners=False,).numpy().squeeze()
    Lx, Ly, Lz = data.shape
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    zcord = np.linspace(0,Lz-1,z_reso,dtype="int")
    X,Y,Z = np.meshgrid(np.arange(Lx), np.arange(Ly), zcord, indexing='ij')
    im = ax.scatter(X,Y,Z, c=data[...,zcord], cmap=cm.coolwarm, alpha=0.2)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1., 1.0, 1.0, 0.0))
    ax.set_xlim([0, Lx])
    ax.xaxis.set_ticks([Lx//4, Lx//4*3])
    ax.xaxis.set_ticklabels(['Left', 'Right'])
    ax.set_xlabel('X', labelpad=10)
    
    ax.set_ylim([0, Ly])
    ax.yaxis.set_ticks([Ly//4, Ly//4*3])
    ax.yaxis.set_ticklabels(['Front', 'Back'])
    ax.set_ylabel('Y', labelpad=10)
    
    ax.set_zlim([0, Lz])
    ax.zaxis.set_ticklabels([])
    ax.set_zlabel('Z')
    
    ax.grid(True)               # remove grid lines
    ax.view_init(elev=30, azim=-120)  # adjust view angle to show 3D structure
    
    # Set axis labels as unseen
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    cbar = plt.colorbar(im, shrink=0.8, aspect=16)
    cbar.set_label('Temp', fontsize=22)
    cbar.set_ticks(np.linspace(data.min(), data.max(), 5)*1000//10/100, fontsize=24)
    #cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'] )
    
    plt.show()
    
def Temp2d_plot(Temps, reso=1, rdbu=False, name=None):

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.tick_params(labelsize=20)
    X, Y = np.meshgrid(np.arange(0, Temps.shape[0], reso),
                       np.arange(0, Temps.shape[1], reso), indexing='ij')
    levels = np.linspace(Temps.min(), Temps.max(), 50)
    if rdbu:
        cset = ax.contourf(X, Y, Temps, levels, cmap="RdBu")    
    else:
        cset = ax.contourf(X, Y, Temps, levels, cmap=cm.jet)    
    ax.set_title('{} profile, Reso={:.0f}'.format(name, reso))
    Nx, Ny = Temps.shape
    plt.xlim([0, Nx]); plt.ylim([0, Ny])
    plt.xticks([0,Nx//2,Nx]); plt.yticks([Ny//2,Ny])
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5).tolist())
    plt.show()
    
    