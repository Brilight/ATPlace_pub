import re, os
import sys
import math, random

sys.path.append(os.path.abspath('../'))
os.environ["OMP_NUM_THREADS"] = "32"

import time
from copy import deepcopy
import logging

import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import SA.BlockOccupy as block_occupation
from ATPLACE.Legalization import Legalization
from MPA.PhysicsSolver import ATSim_solver


def plot_fp(system, val=None):
    tp = time.time()
    fig, ax = plt.subplots(figsize=(10, 8))
    x_scale, y_scale = system.intp_width, system.intp_height
    gralt = system.granularity
    plt.tick_params(labelsize=20)

    if val is not None:
        from matplotlib import cm
        cmap = plt.get_cmap(cm.jet)
        levels = np.linspace(val.min(), val.max(), 50)
        X, Y = np.meshgrid((np.arange(val.shape[0])+0.5)/(val.shape[0])*x_scale, 
                          (np.arange(val.shape[1])+0.5)/(val.shape[1])*y_scale, indexing='ij')
        cset = ax.contourf(X, Y, val, levels, cmap=cmap, alpha=0.7)
        position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
        cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
        cbar.set_ticks((np.round(np.linspace(levels[0]+0.05, levels[-1]-0.05, 5),1)).tolist())

    for i in range(system.num_chiplets):
        size_x, size_y = system.node_size_x[i], system.node_size_y[i]
        x, y = system.node_x[i], system.node_y[i]
        rect = plt.Rectangle((x-size_x/2, y-size_y/2), size_x, size_y, 
                             edgecolor='k', linewidth=0, facecolor='black', alpha=.4)
        ax.add_patch(rect)
        name = system.node_names[i].replace("_","")
        ax.text(x, y, name, ha='center', va='center', alpha=0.8, fontsize=10)
        
        pins = np.where(system.pin2node_map==i)[0]
        angle = system.node_orient[i]
        pinx = x+system.pin_offset_x[pins]*np.cos(angle)-system.pin_offset_y[pins]*np.sin(angle)
        piny = y+system.pin_offset_y[pins]*np.cos(angle)+system.pin_offset_x[pins]*np.sin(angle)
        plt.scatter(pinx, piny, s=1, marker='.', color='b' if i>10 else 'r')

    ax.add_patch(plt.Rectangle((gralt/2, 0), x_scale-gralt/2, gralt/2, facecolor='grey', alpha=.5))
    ax.add_patch(plt.Rectangle((gralt/2, y_scale-gralt/2), x_scale-gralt/2, gralt/2, facecolor='grey', alpha=.5))
    ax.add_patch(plt.Rectangle((0, 0), gralt/2, y_scale, facecolor='grey', alpha=.5))
    ax.add_patch(plt.Rectangle((x_scale-gralt/2, 0), gralt/2, y_scale, facecolor='grey', alpha=.5))
    ax.set_xticks([0, x_scale/3//100*100, x_scale*2/3//100*100, x_scale//100*100])
    ax.set_yticks([y_scale/3//100*100, y_scale*2/3//100*100, y_scale//100*100])
    plt.show()
    logging.info(f"Plot_fp time {time.time()-tp}")
    
def thermal_sim(system, solver, judge_weight=True):
    if judge_weight and system.weight == 'WL-driven':
        return {'temp': 0.0, 'warp': 0.0}
    solver.set_pos(
        system.powermap, (system.node_x, system.node_y), (system.node_size_x, system.node_size_y)
    )
    solver.run(solver.temp_file_name, 0)
    res = solver.getres(solver.temp_file_name)
    temp_whole = res['temperature'] - 273.15
    temp_max = float(temp_whole.max())
    warpage = float(res['warpage'].max()) - float(res['warpage'].min())
    return {'temp': temp_max, 'warp': warpage}
    
def boundary_check(system, x, y, w, h):
    if (x - w / 2) < 0:
        return False
    if (x + w / 2) > system.intp_width:
        return False
    if (y - h / 2) < 0:
        return False
    if (y + h / 2) > system.intp_height:
        return False
    return True

def close_neighbor(system, grid):
    ''' slightly moving chiplets, do not consider rotation'''
    chiplet_count = system.num_chiplets
    chiplet_order = np.random.permutation(range(chiplet_count))
    granularity = system.granularity
    for p in chiplet_order:
        direction_order = np.random.permutation(['up', 'down', 'left', 'right'])
        xx, width = system.node_x[p], system.node_size_x[p]
        yy, height = system.node_y[p], system.node_size_y[p]
        for d in direction_order:
            # re-connect the direction with the appropriate function in order to easily visulize
            #using print-grid(). The dirctions are referring to the grid printed on screen, 
            #the directions are referring to conventional x-y coordinates, origin in the left-bottom corner.
            if d == 'down' and (yy - granularity - height/2 >= 0):
                if block_occupation.check_down_occupation(grid, granularity, xx, yy - granularity, width, height):
                    return p, xx, yy - granularity, 0
            elif (d == 'up') and (yy + granularity + height/2 <= system.intp_height):
                if block_occupation.check_up_occupation(grid, granularity, xx, yy + granularity, width, height):
                    return p, xx, yy + granularity, 0
            elif d == 'left' and (xx - granularity - width/2 >= 0):
                if block_occupation.check_left_occupation(grid, granularity, xx - granularity, yy, width, height):
                    return p, xx - granularity, yy, 0
            elif (d == 'right') and (xx + granularity + width/2 <= system.intp_width):
                if block_occupation.check_right_occupation(grid, granularity, xx + granularity, yy, width, height):
                    return p, xx + granularity, yy, 0
    logging.info('No chiplet can be moved.')
    exit()
    
def jumping_neighbor(system, grid):
    '''define a neighbor placement as move one chiplet to anywhere can be located. 
    rotate if needed. We do not consider swapping, since can't gaurantee the placement
    is still legal (no overlap) after swapping'''

    chiplet_count = system.num_chiplets
    granularity = system.granularity
    count = 0
    while True:
        pick_chiplet = random.randint(0, chiplet_count - 1)
        x_new = random.randint(1, system.intp_width / granularity - 1) * granularity
        y_new = random.randint(1, system.intp_height / granularity - 1) * granularity
        rotation = random.randint(0,3)
        if int(rotation-system.node_orient[pick_chiplet]/np.pi*2)%2 == 0:
            chiplet_width, chiplet_height = system.node_size_x[pick_chiplet], system.node_size_y[pick_chiplet]
        else:
            chiplet_height, chiplet_width = system.node_size_x[pick_chiplet], system.node_size_y[pick_chiplet]
        if boundary_check(system, x_new, y_new, chiplet_width, chiplet_height) and \
            block_occupation.replace_block_occupation(grid, granularity, x_new, y_new, 
                                                      chiplet_width, chiplet_height, pick_chiplet):
            print ('found a random placement at', count, 'th trial')
            break
        count += 1
        if count > 10000:
            # it's not easy to find a legal placement using random method. 
            #try move each chiplet (in random order) slightly until find a legal solution
            print ('cannot find a legal random placement, go with close_neighbor')
            return close_neighbor(system, grid)
    return pick_chiplet, x_new, y_new, rotation

def accept_probability(old_temp, new_temp, old_warp, new_warp, old_length, new_length, T, weight, objective_mode):
    use_temp = objective_mode in ('WL_temp', 'WL_temp_warp')
    use_warp = objective_mode == 'WL_temp_warp'

    if weight == 'equal':
        a = 0.5 if use_temp else 0.0
        b = 0.5 if use_warp else 0.0
        c = 1.0 - a - b
    elif weight == 'adpT':
        a = max(0.0, min(0.1 + (max(old_temp, new_temp) - 45) * 0.01, 1.0)) if use_temp else 0.0
        b = 0.0
        c = 1.0 - a
    elif weight == 'adpTW':
        a = min(0.1 + (max(old_temp, new_temp) - 45) * 0.01, 0.5) if use_temp else 0.0
        b = (1.0 - a) if use_warp else 0.0
        c = 1.0 - a - b
    elif weight == 'adpTWv2':
        if use_temp and (old_temp >= 85 or new_temp >= 85):
            a = min(0.1 + (max(old_temp, new_temp) - 60) * 0.01, 0.5)
        else:
            a = 0.0
        b = (1.0 - a) if use_warp else 0.0
        c = 1.0 - a - b
    elif weight == 'WL-driven':
        a = b = 0.0
        c = 1.0
    else:
        a = b = 0.0
        c = 1.0

    def norm(val, vmin, vmax):
        return (val - vmin) / (vmax - vmin) if vmax != vmin else val - vmin

    if temp_max != temp_min and length_max != length_min and (not use_warp or warp_max != warp_min):
        old_cost = a * norm(old_temp, temp_min, temp_max) + \
                   b * norm(old_warp, warp_min, warp_max) + \
                   c * norm(old_length, length_min, length_max)
        new_cost = a * norm(new_temp, temp_min, temp_max) + \
                   b * norm(new_warp, warp_min, warp_max) + \
                   c * norm(new_length, length_min, length_max)
    else:
        old_cost = a * (old_temp - temp_min) + \
                   b * (old_warp - warp_min) + \
                   c * (old_length - length_min)
        new_cost = a * (new_temp - temp_min) + \
                   b * (new_warp - warp_min) + \
                   c * (new_length - length_min)

    delta = -(new_cost - old_cost)
    ap = 1.0 if delta > 0 else math.exp(delta / T)

    logging.info(f"temp: old={old_temp:.1f}, new={new_temp:.1f}, min={temp_min:.1f}, max={temp_max:.1f}")
    logging.info(f"warp: old={old_warp:.4f}, new={new_warp:.4f}, min={warp_min:.4f}, max={warp_max:.4f}")
    logging.info(f"length: old={old_length:.4f}, new={new_length:.4f}, min={length_min:.4f}, max={length_max:.4f}")
    logging.info(f"T={T:.4f}, delta={delta:.4f}, ap={ap:.4f}")
    return ap

def update_minmax(temp, length, warp):
    global temp_max, temp_min, length_max, length_min, warp_max, warp_min
    if temp > temp_max: temp_max = temp
    if temp < temp_min: temp_min = temp
    if length > length_max: length_max = length
    if length < length_min: length_min = length
    if warp > warp_max: warp_max = warp
    if warp < warp_min: warp_min = warp


def register_log(system_best, step_best, temp_best, warp_best, length_best, T, step):
    with open(os.path.join(system_best.respath, 'log.txt'), 'a') as LOG:
        LOG.write(f'T = {T}\t step = {step}\n')
        LOG.write(f'{step_best}\n{temp_best}\n{warp_best}\n{length_best}\n')
        LOG.write(f'{system_best.node_x}\n{system_best.node_y}\n')


def register_step(system, step, temp, warp, length, T, Loc_log):
    Loc_log.append([
        step,
        system.node_x.copy(),
        system.node_y.copy(),
        system.node_size_x.copy(),
        system.node_size_y.copy(),
        system.node_orient.copy(),
        temp,
        warp,
        length
    ])
    with open(os.path.join(system.respath, 'step.txt'), 'a') as LOG:
        LOG.write(f'T = {T}\t step = {step}\n')
        LOG.write(f'{temp}\n{warp}\n{length}\n')
        LOG.write(f'{system.node_x}\n{system.node_y}\n')
        LOG.write(f'{system.node_size_x}\n{system.node_size_y}\n')


def anneal(system, T, T_min, alpha, jumping_ratio, thermal_solver, objective_mode='WL_temp_warp'):
    assert objective_mode in ('WL_only', 'WL_temp', 'WL_temp_warp')

    global temp_max, temp_min, length_max, length_min, warp_max, warp_min
    temp_max = warp_max = length_max = 0.0
    temp_min = warp_min = 200.0
    length_min = 200.0

    t1 = time.time()
    Res_log = []
    Loc_log = []

    system_new = deepcopy(system)
    system_best = deepcopy(system)
    step = 1
    step_best = 1

    thermo_res = thermal_sim(system, thermal_solver)
    temp_current = thermo_res['temp']
    warp_current = thermo_res['warp']
    length_current = system.hpwl() / 1e6
    temp_best, warp_best, length_best = temp_current, warp_current, length_current
    update_minmax(temp_best, warp_best, length_best)
    logging.info(f'step_{step} temp={temp_current:.2f} warp={warp_current:.4f} length={length_current:.4f}')

    x_best, y_best = deepcopy(system.node_x), deepcopy(system.node_y)
    intp_width, intp_height = system.intp_width, system.intp_height
    granularity = system.granularity
    grid = block_occupation.initialize_grid(intp_width, intp_height, granularity)
    for i in range(system.num_chiplets):
        grid = block_occupation.set_block_occupation(
            grid, granularity, system.node_x[i], system.node_y[i],
            system.node_size_x[i], system.node_size_y[i], i
        )

    register_log(system_best, step_best, temp_current, warp_current, length_current, T, step)
    register_step(system, step, temp_current, warp_current, length_current, T, Loc_log)

    while T > T_min:
        i = 1
        while i <= 30:
            jump_or_close = random.random()*0.9
            if 1 - jumping_ratio > jump_or_close:
                chiplet_moving, x_new, y_new, rotation = jumping_neighbor(system, grid)
            else:
                chiplet_moving, x_new, y_new, rotation = close_neighbor(system, grid)
            system_new = deepcopy(system)
            system_new.node_x[chiplet_moving], system_new.node_y[chiplet_moving] = x_new, y_new
            system_new.rotate(chiplet_moving, rotation*np.pi/2)  
            try:
                thermo_new = thermal_sim(system_new, thermal_solver)
            except:
                logging.info("!!!Come into simulation problems, will jump to the next step!")
                continue
            temp_new = thermo_new['temp']
            warp_new = thermo_new['warp']
            length_new = system_new.hpwl() / 1e6
            maxwl_new = system_new.Maxwl() / 1e3

            step += 1
            logging.info(
                f'moving chiplet {chiplet_moving} from ({system.node_x[chiplet_moving]:.2f}, '
                f'{system.node_y[chiplet_moving]:.2f}) to ({x_new:.2f}, {y_new:.2f}), '
                f'orient {system.node_orient[chiplet_moving]:.2f} -> rotation {rotation}'
            )
            logging.info(
                f'step_{step} T={T:.4f} i={i} Temp={temp_new:.2f} Warp={warp_new:.4f} '
                f'Length={length_new:.4f}'
            )
            Res_log.append([temp_new, length_new, warp_new, maxwl_new])
            update_minmax(temp_new, length_new, warp_new)
            register_step(system_new, step, temp_new, warp_new, length_new, T, Loc_log)

            ap = accept_probability(
                temp_current, temp_new,
                warp_current, warp_new,
                length_current, length_new,
                T, system.weight, objective_mode
            )
            r = random.random()

            if ap > r:
                grid = block_occupation.clear_block_occupation(
                    grid, granularity,
                    system.node_x[chiplet_moving], system.node_y[chiplet_moving],
                    system.node_size_x[chiplet_moving], system.node_size_y[chiplet_moving],
                    chiplet_moving
                )
                grid = block_occupation.set_block_occupation(
                    grid, granularity, x_new, y_new,
                    system_new.node_size_x[chiplet_moving],
                    system_new.node_size_y[chiplet_moving],
                    chiplet_moving
                )
                system = deepcopy(system_new)
                temp_current, warp_current, length_current = temp_new, warp_new, length_new

                if (objective_mode == 'WL_only' and length_new < length_best) or \
                   (objective_mode == 'WL_temp' and (temp_new < temp_best or length_new < length_best)) or \
                   (objective_mode == 'WL_temp_warp' and (temp_new < temp_best or warp_new < warp_best or length_new < length_best)):
                    temp_best, warp_best, length_best = temp_new, warp_new, length_new
                    system_best = deepcopy(system_new)
                    step_best = step

                logging.info(f'AP={ap:.4f} > {r:.4f} Accept!')
            else:
                logging.info(f'AP={ap:.4f} < {r:.4f} Reject!')

            i += 1

        register_log(system_best, step_best, temp_best, warp_best, length_best, T, step)
        T *= alpha

    final_pos = Legalization(
        system_best, 'grb', 0,
        [[system_best.node_x, system_best.node_y]],
        [system_best.node_size_x, system_best.node_size_y],
        TimeLimit=100
    )
    system_best.node_x, system_best.node_y = final_pos[0], final_pos[1]
    result = (
        f"SA time: {time.time() - t1:.2f} s\n"
        f"node_x {system_best.node_x}\n"
        f"node_y {system_best.node_y}\n"
        f"node_size_x {system_best.node_size_x}\n"
        f"node_size_y {system_best.node_size_y}\n"
        f"node_orient {system_best.node_orient}"
    )
    logging.info(result)
    print(result)
    return system_best, step_best, Loc_log, Res_log


def rebuild_grid(system):
    """Rebuild grid and verify that current placement is legal."""
    g = block_occupation.initialize_grid(
        system.intp_width, system.intp_height, system.granularity
    )
    gr = system.granularity
    for i in range(system.num_chiplets):
        x, y = float(system.node_x[i]), float(system.node_y[i])
        w, h = float(system.node_size_x[i]), float(system.node_size_y[i])
        if not boundary_check(system, x, y, w, h):
            raise ValueError(f"Out-of-bound chiplet {i}")
        if not block_occupation.replace_block_occupation(g, gr, x, y, w, h, i):
            raise ValueError(f"Overlap when rebuilding grid at chiplet {i}")
    return g

def try_place(system, grid, idx, x, y, rot):

    def size_after_rotation(idx, rot):
        if int(rot - system.node_orient[idx] / np.pi * 2) % 2 == 0:
            w, h = system.node_size_x[idx], system.node_size_y[idx]
        else:
            h, w = system.node_size_x[idx], system.node_size_y[idx]
        return float(w), float(h)

    gr = system.granularity
    w, h = size_after_rotation(idx, rot)
    if not boundary_check(system, x, y, w, h):
        return False
    ok = block_occupation.replace_block_occupation(grid, gr, x, y, w, h, idx)
    if not ok:
        return False
    system.node_x[idx] = x
    system.node_y[idx] = y
    system.rotate(idx, rot * np.pi / 2)
    return True

def find_legal_random_position(system, grid, idx, max_trials=10000):
    gr = system.granularity
    for _ in range(max_trials):
        x = random.randint(1, int(system.intp_width / gr - 1)) * gr
        y = random.randint(1, int(system.intp_height / gr - 1)) * gr
        rot = random.randint(0, 3)
        if try_place(system, grid, idx, x, y, rot):
            return True
    chiplet, x_new, y_new, rotation = close_neighbor(system, grid)
    return try_place(system, grid, chiplet, x_new, y_new, rotation)

def ensure_legal(system):
    """Rebuild occupancy from scratch; fail fast on any boundary/conflict."""
    g = block_occupation.initialize_grid(
        system.intp_width, system.intp_height, system.granularity
    )
    gr = system.granularity
    for i in range(system.num_chiplets):
        x, y = float(system.node_x[i]), float(system.node_y[i])
        w, h = float(system.node_size_x[i]), float(system.node_size_y[i])
        if not boundary_check(system, x, y, w, h):
            return False
        # replace_block_occupation returns a bool; also mutates g on success
        if not block_occupation.replace_block_occupation(g, gr, x, y, w, h, i):
            return False
    return True

def fitness(system, solver, mode='WL_temp_warp'):
    l = system.hpwl() / 1e6
    if mode == 'WL_only':
        f = l
        t, w = 0, 0
    else:
        thermo = thermal_sim(system, solver)
        t, w = thermo['temp'], thermo['warp']
        if mode == 'WL_temp':
            f = 0.5 * l + 0.5 * t / 100
        else:
            f = 0.5 * l + 0.3 * t / 100 + 0.2 * w / 10
    return f, t, w, l

def crossover(p1, p2):
    child = deepcopy(p1)
    g = rebuild_grid(child)
    n = max(1, child.num_chiplets // 3)
    idxs = random.sample(range(child.num_chiplets), n)
    for i in idxs:
        x, y = p2.node_x[i], p2.node_y[i]
        placed = False
        for rot in (0, 1, 2, 3):
            if try_place(child, g, i, x, y, rot):
                placed = True
                break
        if not placed:
            find_legal_random_position(child, g, i)
    return child

def mutate(system, mutation_rate=0.2):
    s = deepcopy(system)
    g = rebuild_grid(s)
    for i in range(s.num_chiplets):
        if random.random() < mutation_rate:
            find_legal_random_position(s, g, i)
    return s

def select_parents(pop, scores, n):
    sc = np.array(scores)
    p = 1 / (sc + 1e-6)
    p /= p.sum()
    idx = np.random.choice(len(pop), size=n, p=p)
    return [deepcopy(pop[i]) for i in idx]

def init_population(system, n):
    pop = []
    for _ in range(n):
        s = deepcopy(system)
        g = block_occupation.initialize_grid(
            s.intp_width, s.intp_height, s.granularity
        )
        for i in np.random.permutation(range(s.num_chiplets)):
            find_legal_random_position(s, g, i)
        pop.append(s)
    return pop


def genetic_optimize(
    system, solver, pop_size=20, generations=80, 
    mutation_rate=0.3, elite_ratio=0.2, mode='WL_temp_warp'
):
    t0 = time.time()
    pop = init_population(system, pop_size)
    best, best_score = None, float('inf')
    best_temp = best_warp = best_length = None

    for gen in trange(generations, desc='GA Optimization'):
        scores, temps, warps, lengths = [], [], [], []
        # Evaluate all individuals in current population
        for s in pop:
            f, t, w, l = fitness(s, solver, mode)
            scores.append(f)
            temps.append(t)
            warps.append(w)
            lengths.append(l)
            # Register every individual (as a "step")
            if mode != 'WL_only':
                register_step(s, gen, t, w, l, T=0.0, Loc_log=[])  # T unused in GA, set to 0.0

            if f < best_score:
                best, best_score = deepcopy(s), f
                best_temp, best_warp, best_length = t, w, l

        # Select elites and breed next generation
        num_elite = max(1, int(elite_ratio * pop_size))
        elite_idx = np.argsort(scores)[:num_elite]
        new_pop = [deepcopy(pop[i]) for i in elite_idx]
        parents = select_parents(pop, scores, pop_size - num_elite)

        while len(new_pop) < pop_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            if not ensure_legal(child):
                child = init_population(system, 1)[0]
            new_pop.append(child)
        pop = new_pop

        if mode != 'WL_only':
            # Log global best at this generation
            register_log(best, gen, best_temp, best_warp, best_length, T=0.0, step=gen)

        logging.info(f'Gen {gen}: best_fitness={best_score:.4f}, '
                     f'temp={best_temp:.2f}, warp={best_warp:.6f}, length={best_length:.4f}')

    # Final legalization
    final_pos = Legalization(best, 'grb', 0,
                             [[best.node_x, best.node_y]],
                             [best.node_size_x, best.node_size_y],
                             TimeLimit=100)
    best.node_x, best.node_y = final_pos[0], final_pos[1]
    print(f'GA done in {time.time()-t0:.2f}s ')
    return best
