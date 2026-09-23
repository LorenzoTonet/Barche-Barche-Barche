import numpy as np
import random

from .environment import SailingEnv
from .map_elements import Checkpoint

def create_random_environment_old(config: dict, verbose:bool = True) -> SailingEnv:
    """
    Create a random SailingEnv environment with random checkpoints and wind field.
    """
    # Randomly generate checkpoints
    checkpoints = []

    if config["train"]["env"]["variable_n_checkpoints"]:
        n_checkpoints = np.random.randint(1, config["train"]["env"]["n_checkpoints"] + 1)  # Random number of checkpoints between 1 and n_checkpoints
    else:
        n_checkpoints = config["train"]["env"]["n_checkpoints"]

    zone_width = config["map_width"] / (n_checkpoints + 1)
    
    if verbose: print("=====NEW ENVIRONMENT=====")
    for i in range(n_checkpoints):
        # devo dividere la mappa in n_checkpoints+1 zone per evitare che i checkpoint siano troppo vicini. Dopodiché prendo un punto random in quella zona
        # devo ricordarmi che n+1 perché lo start sarà nella zona 0, il primo checkpoint nella zona 1, ecc. L'altezza è quella di tutta la mappa, diciamo 
        # che la stiamo dividendo in fasce
        
        current_zone_start = (i + 1) * zone_width
        current_zone_end = (i + 2) * zone_width

        x = int(np.random.randint(current_zone_start, current_zone_end))
        y = int(np.random.randint(0, config["map_height"]))

        radius = 5.0
        checkpoints.append(Checkpoint(position=np.array([int(x), int(y)]), radius=radius, number = i+1))
        
        if verbose:print(f"Checkpoint {i+1}: position=({x:.2f}, {y:.2f}), radius={radius}, number={i+1}")
    if verbose:print("=========================")
    # Create the environment
    env = SailingEnv(config=config, checkpoints=checkpoints, render_mode=config["mode"])

    # change randomly the initial position of the boat 
    env.initial_boat_position = np.array([
        int(np.random.randint(0, zone_width)),
        int(np.random.randint(0, config["map_height"]))
    ])

    return env

def create_random_environment(config: dict, verbose:bool = True) -> SailingEnv:
    """
    Create a random SailingEnv environment with random checkpoints and wind field.
    """
    # Randomly generate checkpoints
    checkpoints = []

    if config["train"]["env"]["variable_n_checkpoints"]:
        n_checkpoints = np.random.randint(1, config["train"]["env"]["n_checkpoints"] + 1)  # Random number of checkpoints between 1 and n_checkpoints
    else:
        n_checkpoints = config["train"]["env"]["n_checkpoints"]
    
    if verbose: print("=====NEW ENVIRONMENT=====")

    # generate n+1 random positions for checkpoints (n checkpoints + 1 for the starting position)
    ckp_positions = []
    while len(ckp_positions) < n_checkpoints+1:
        # random position in the map
        x = np.random.randint(0, config["map_width"])
        y = np.random.randint(0, config["map_height"])

        # compute the distance from all existing checkpoints
        distances = [np.linalg.norm(np.array([x, y]) - np.array(pos)) for pos in ckp_positions]
        if all(dist > 20 for dist in distances):  # minimum distance of 20
            ckp_positions.append((x, y))

    # choose a random starting position for the boat from the generated positions
    start_pos = ckp_positions.pop(np.random.randint(0, len(ckp_positions)))

    # choose the next checkpoints from the remaining positions. 
    # config['train']['env']['ckp_probability'] times the next checkpoint will be the closest one, 
    # otherwise it will be a random one from the remaining ones
    next_checkpoints = []
    while len(next_checkpoints) < n_checkpoints:
        if np.random.rand() < config['train']['env']['ckp_probability']:
            # choose the closest checkpoint
            distances = [np.linalg.norm(np.array(start_pos) - np.array(pos)) for pos in ckp_positions]
            closest_idx = np.argmin(distances)
            next_checkpoints.append(ckp_positions.pop(closest_idx))
        else:
            # choose a random checkpoint
            random_idx = np.random.randint(0, len(ckp_positions))
            next_checkpoints.append(ckp_positions.pop(random_idx))

    radius = 5.0
    for i, (x, y) in enumerate(next_checkpoints):
        checkpoints.append(Checkpoint(position=np.array([int(x), int(y)]), radius=radius, number = i+1))
        if verbose:print(f"Checkpoint {i+1}: position=({x:.2f}, {y:.2f}), radius={radius}, number={i+1}")
    if verbose:print("=========================")

    # change randomly the initial wind vector of the environment
    config["initial_wind"] = np.array([np.random.uniform(config["train"]["env"]["wind_range"][0], config["train"]["env"]["wind_range"][1]), np.random.uniform(config["train"]["env"]["wind_range"][0], config["train"]["env"]["wind_range"][1])])
    if random.random() < 0.5:
        config["initial_wind"][0] *= -1
    if random.random() < 0.5:
        config["initial_wind"][1] *= -1

    # Create the environment
    env = SailingEnv(config=config, checkpoints=checkpoints, render_mode=config["mode"])

    # change randomly the initial position of the boat 
    env.initial_boat_position = np.array(start_pos)

    return env