import os
import time

from SA.SAInit import Initialization
from SA.SACore import anneal

def SA(system):
    
    # set annealing parameters
    T = 1
    T_min = 0.01
    alpha = 0.9 #system.decay
    jumping_ratio = 0.9 # fixed to 10% chance to jump
    system.decay = 0.95
    system.weight = 'adpTWv2'
    system.respath = os.path.join("/".join(system.prefix.split('/')[:-1]),'res_SA')
    os.system('mkdir '+system.respath)

    t1 = time.perf_counter()
    init_pos = Initialization(
        system, system.params.dis_bet_chips, [system.intp_width, system.intp_height], 10
    )
    system.rotate(np.arange(system.num_chiplets), init_pos[2].astype(int))
    system.node_x = init_pos[0]+system.intp_width/2-(
        (init_pos[0]+system.node_size_x/2).max()+(init_pos[0]-system.node_size_x/2).min()
    )/2
    system.node_y = init_pos[1]+system.intp_height/2-(
        (init_pos[1]+system.node_size_y/2).max()+(init_pos[1]-system.node_size_y/2).min()
    )/2
    system_best, step_best, _, _, Loc_log, Res_log_sa = anneal(
        system, T, T_min, alpha, jumping_ratio
    )
    print("SA time:", time.perf_counter()-t1)
    np.save(os.path.join(system.respath,'SA.npy'), np.array(Res_log_sa))
    return Loc_log, Res_log_sa
