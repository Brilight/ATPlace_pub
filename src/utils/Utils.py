import numpy as np
import torch

def monitor(pos, system, with_grad=0):
    torch.set_printoptions(sci_mode=False, precision=0)
    print("\npos data:", pos[0].data[:system.num_chiplets+1]) 
    print("pos data:", pos[0].data[system.num_nodes:system.num_nodes+num_chiplets+1]) 
    print("theta data:", pos[1].data)
    #print("theta data:", (torch.round(pos[1].data, decimals=3)).tolist())

    if with_grad:
        torch.set_printoptions(sci_mode=True, precision=0)
        print("\npos grad_x:", pos[0].grad[:system.num_chiplets+1]) 
        print("pos grad_y:", pos[0].grad[system.num_nodes:system.num_nodes+num_chiplets+1]) 
        print("theta grad:", pos[1].grad)
        torch.set_printoptions(profile="default")
        
def read_effective_data(file_path):
    effective_data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = line.split('\t')
            unit_name, rest = values[0], list(filter(lambda x: x != "", values[1:]))
            effective_data.append([unit_name, *list(map(float,rest))])
    return effective_data

def identify_pareto(scores):
    population_size = scores.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(scores[j, :2] <= scores[i, :2]) and any(scores[j, :2] < scores[i, :2]):
                pareto_front[i] = 0
                break
    return scores[pareto_front]