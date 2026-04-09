import sys
import numpy as np

def initialize_grid(intp_width, intp_height, granularity):
    grid = np.zeros((int(intp_width/granularity)+1, 
                     int(intp_height/granularity)+1))
    grid[0,:] = 1
    grid[-1,:] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid

def check_block_occupation(grid, granularity, xx, yy, width, height):
    tot = np.sum(grid[int(xx/granularity)-int(width/2/granularity+0.49):\
                      int(xx/granularity)+int(width/2/granularity+0.49)+1, 
                      int(yy/granularity)-int(height/2/granularity+0.49):\
                      int(yy/granularity)+int(height/2/granularity+0.49)+1])
    return tot


def check_left_occupation(grid, granularity, xx, yy, width, height):
    i = int(xx/granularity) - int(width/2/granularity+0.49)
    if i<=0:
        return False
    if (sum(grid[i][int(yy/granularity)-int(height/2/granularity+0.49):int(yy/granularity)+int(height/2/granularity+0.49)+1])):
        return False
    else:
        return True

def check_right_occupation(grid, granularity, xx, yy, width, height):
    i = int(xx/granularity) + int(width/2/granularity+0.49)
    intp_size = (grid.shape[0] - 1) * granularity
    if i >= intp_size:
        return False
    if (sum(grid[i][int(yy/granularity)-int(height/2/granularity+0.49):int(yy/granularity)+int(height/2/granularity+0.49)+1])):
        return False
    else:
        return True

def check_down_occupation(grid, granularity, xx, yy, width, height):
    j = int(yy/granularity) - int(height/2/granularity+0.49)
    if j<=0:
        return False
    for i in range(int(xx/granularity)-int(width/2/granularity+0.49), int(xx/granularity)+int(width/2/granularity+0.49)+1):
        if grid[i][j]:
            return False
    return True

def check_up_occupation(grid, granularity, xx, yy, width, height):
    j = int(yy/granularity) + int(height/2/granularity+0.49)
    intp_size = (grid.shape[1] - 1) * granularity
    if j >= intp_size:
        return False
    for i in range(int(xx/granularity)-int(width/2/granularity+0.49), int(xx/granularity)+int(width/2/granularity+0.49)+1):
        if grid[i][j]:
            return False
    return True

def set_block_occupation(grid, granularity, xx, yy, width, height, chiplet_index):
    grid[int(xx/granularity)-int(width/2/granularity+0.49):\
         int(xx/granularity)+int(width/2/granularity+0.49)+1, 
         int(yy/granularity)-int(height/2/granularity+0.49):\
         int(yy/granularity)+int(height/2/granularity+0.49)+1] = chiplet_index + 2
    return grid

def clear_block_occupation(grid, granularity, xx, yy, width, height, chiplet_index):
    grid[int(xx/granularity)-int(width/2/granularity+0.49):\
         int(xx/granularity)+int(width/2/granularity+0.49)+1, 
         int(yy/granularity)-int(height/2/granularity+0.49):\
         int(yy/granularity)+int(height/2/granularity+0.49)+1] = 0
    return grid

def replace_block_occupation(grid, granularity, xx_new, yy_new, width, height, chiplet_index):
    for i in range(int(xx_new/granularity)-int(width/2/granularity+0.49), 
                   int(xx_new/granularity)+int(width/2/granularity+0.49)+1):
        for j in range(int(yy_new/granularity)-int(height/2/granularity+0.49), 
                       int(yy_new/granularity)+int(height/2/granularity+0.49)+1):
            if (grid[i][j] != chiplet_index + 2) and (grid[i][j] != 0):
                return False
    return True