# for initial placement, I don't think it's very necessary to define a new class. 
# use a seperate module is clean enough

import random, math, os, time
from copy import deepcopy

import numpy as np
import pulp
import gurobipy as gp
from gurobipy import GRB

import SA.BlockOccupy as block_occupation


def Initialization(system, dis_bet_chips, intp_size, timelimit=10):
    t1 = time.time()
    gpmodel = gp.Model("initialization")
    gpmodel.setParam("OutputFlag", 1)  # Fobidden middle state output
    gpmodel.setParam('TimeLimit', timelimit)

    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    size_x = system.node_size_x+dis_bet_chips
    size_y = system.node_size_y+dis_bet_chips
    intp_width, intp_height = intp_size
    granularity = system.granularity
    
    x, y, r = {}, {}, {}
    for i in range(num_chiplets):
        x[i] = gpmodel.addVar(lb=0, ub=GRB.INFINITY, 
                              vtype=GRB.CONTINUOUS, name=f"x_{i}")
        y[i] = gpmodel.addVar(lb=0, ub=GRB.INFINITY, 
                              vtype=GRB.CONTINUOUS, name=f"y_{i}")
        r[i] = gpmodel.addVar(vtype=GRB.BINARY, name=f"r_{i}")
    gpmodel.update()

    gpmodel.setObjective(1, sense=GRB.MINIMIZE)

    width, height = {}, {}
    for i in range(num_chiplets):
        
        width[i] = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                                   vtype=GRB.CONTINUOUS, name=f"width_{i}")
        gpmodel.addConstr(width[i] == (size_y[i]*r[i]+size_x[i]*(1-r[i])))
        gpmodel.addConstr(x[i]-width[i]/2 >= granularity)
        gpmodel.addConstr(x[i]+width[i]/2 <= intp_width-granularity)

        height[i] = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                                   vtype=GRB.CONTINUOUS, name=f"height_{i}")
        gpmodel.addConstr(height[i] == (size_x[i]*r[i]+size_y[i]*(1-r[i])))
        gpmodel.addConstr(y[i]-height[i]/2 >= granularity)
        gpmodel.addConstr(y[i]+height[i]/2 <= intp_height-granularity)

    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            delta1 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta1_{i}_{j}")
            delta2 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta2_{i}_{j}")
            delta3 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta3_{i}_{j}")
            delta4 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta4_{i}_{j}")
            gpmodel.addConstr(delta1 + delta2 + delta3 + delta4 <=3)
            
            gpmodel.addConstr(x[i] + (width[i]/2+width[j]/2) <= x[j] + intp_width*delta1)
            gpmodel.addConstr(x[j] + (width[i]/2+width[j]/2) <= x[i] + intp_width*delta2)
            gpmodel.addConstr(y[i] + (height[i]/2+height[j]/2) <= y[j] + intp_height*delta3)
            gpmodel.addConstr(y[j] + (height[i]/2+height[j]/2) <= y[i] + intp_height*delta4)
    
    gpmodel.optimize()

    if gpmodel.status == GRB.OPTIMAL:
        new_pos = np.zeros((3,num_chiplets))
        for i in range(num_chiplets):
            new_pos[0,i] = x[i].x
            new_pos[1,i] = y[i].x
            new_pos[2,i] = r[i].x
    else:
        print("\t!Optimal Floorplan not found!")
        new_pos = None

    gpmodel.dispose()
    return new_pos

def Initialization_gurobi(system, dis_bet_chips, intp_size, timelimit=10):
    t1 = time.time()
    gpmodel = gp.Model("initialization")
    gpmodel.setParam("OutputFlag", 1)  # Fobidden middle state output
    gpmodel.setParam('TimeLimit', timelimit)

    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    size_x = system.node_size_x+dis_bet_chips
    size_y = system.node_size_y+dis_bet_chips
    intp_width, intp_height = intp_size
    granularity = system.granularity
    
    x, y, r = {}, {}, {}
    for i in range(num_chiplets):
        x[i] = gpmodel.addVar(lb=0, ub=GRB.INFINITY, 
                              vtype=GRB.CONTINUOUS, name=f"x_{i}")
        y[i] = gpmodel.addVar(lb=0, ub=GRB.INFINITY, 
                              vtype=GRB.CONTINUOUS, name=f"y_{i}")
        r[i] = gpmodel.addVar(vtype=GRB.BINARY, name=f"r_{i}")
    gpmodel.update()

    gpmodel.setObjective(1, sense=GRB.MINIMIZE)

    width, height = {}, {}
    for i in range(num_chiplets):
        
        width[i] = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                                   vtype=GRB.CONTINUOUS, name=f"width_{i}")
        gpmodel.addConstr(width[i] == (size_y[i]*r[i]+size_x[i]*(1-r[i])))
        gpmodel.addConstr(x[i]-width[i]/2 >= granularity)
        gpmodel.addConstr(x[i]+width[i]/2 <= intp_width-granularity)

        height[i] = gpmodel.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, 
                                   vtype=GRB.CONTINUOUS, name=f"height_{i}")
        gpmodel.addConstr(height[i] == (size_x[i]*r[i]+size_y[i]*(1-r[i])))
        gpmodel.addConstr(y[i]-height[i]/2 >= granularity)
        gpmodel.addConstr(y[i]+height[i]/2 <= intp_height-granularity)

    for i in range(num_chiplets):
        for j in range(i+1, num_chiplets):
            delta1 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta1_{i}_{j}")
            delta2 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta2_{i}_{j}")
            delta3 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta3_{i}_{j}")
            delta4 = gpmodel.addVar(vtype=GRB.BINARY, name=f"delta4_{i}_{j}")
            gpmodel.addConstr(delta1 + delta2 + delta3 + delta4 <=3)
            
            gpmodel.addConstr(x[i] + (width[i]/2+width[j]/2) <= x[j] + intp_width*delta1)
            gpmodel.addConstr(x[j] + (width[i]/2+width[j]/2) <= x[i] + intp_width*delta2)
            gpmodel.addConstr(y[i] + (height[i]/2+height[j]/2) <= y[j] + intp_height*delta3)
            gpmodel.addConstr(y[j] + (height[i]/2+height[j]/2) <= y[i] + intp_height*delta4)
    
    gpmodel.optimize()

    if gpmodel.status == GRB.OPTIMAL:
        new_pos = np.zeros((3,num_chiplets))
        for i in range(num_chiplets):
            new_pos[0,i] = x[i].x
            new_pos[1,i] = y[i].x
            new_pos[2,i] = r[i].x
    else:
        print("\t!Optimal Floorplan not found!")
        new_pos = None

    gpmodel.dispose()
    return new_pos

'''
I use occupation grid (the matrix) to present and check if the unit grid is available or
if it is already occupied. The grid (x, y) represent a unit square centered at (x,y) 
(area from [x-0.5, x+0.5] * [y-0.5, y+0.5])
'''

# width and height here include microbump overhead. We did addition before calling this module
def slide_x_direction(grid, granularity, xx, yy, width, height):
    while block_occupation.check_left_occupation(grid, granularity, xx-granularity, yy, width, height):
        xx -= granularity
    return xx

def slide_y_direction(grid, granularity, xx, yy, width, height):
    while block_occupation.check_down_occupation(grid, granularity, xx, yy-granularity, width, height):
        yy -= granularity
    return yy

def init_place_tight(intp_size, granularity, chiplet_count, width, height):
    intp_width, intp_height = intp_size
    x, y, rotation = [0] * chiplet_count, [0] * chiplet_count, [0] * chiplet_count
    grid = block_occupation.initialize_grid(int(intp_width/granularity), int(intp_height/granularity))

    for i in range(chiplet_count):
        xx = int((intp_width - 0.5 - width[i] / 2)/granularity)*granularity
        yy = int((intp_height -0.5 - height[i] / 2)/granularity)*granularity #where does the 0.5 term come from?
        if block_occupation.check_block_occupation(grid, granularity, xx, yy, width[i], height[i]) == False:
            print ('can\'t find tightly arraged initial placement')
            break
        xx = slide_x_direction(grid, granularity, xx, yy, width[i], height[i])
        yy = slide_y_direction(grid, granularity, xx, yy, width[i], height[i])
        # print ('slide to ', xx, yy)
        grid = block_occupation.set_block_occupation(grid, granularity, xx, yy, width[i], height[i], i)
        x[i], y[i] = xx, yy
    return x, y, rotation

if __name__ == "__main__":
    width = [3, 4, 	 2, 2, 	 1,   4, 3, 4]
    height =[2, 1.5, 3, 1.5, 1,   1, 2, 2]
    connection_matrix = [[0,128,128,0,0,0,0,128],
                        [128,0,128,0,0,0,128,0],
                        [128,128,0,128,128,128,128,128],
                        [0,0,128,0,0,0,0,0],
                        [0,0,128,0,0,0,0,0],
                        [0,0,128,0,0,0,0,0],
                        [0,128,128,0,0,0,0,128],
                        [128,0,128,0,0,0,128,0]]
    x, y, width, height = init_place_bstree(40, 1, 8, width, height, connection_matrix, 'outputs/bstree/')
