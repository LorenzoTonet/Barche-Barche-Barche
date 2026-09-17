import argparse
import yaml
import numpy as np

import pygame

from source_code.environment import SailingEnv, create_random_environment
from source_code.vector_field import VecField
from source_code.map_elements import Checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Exiting.")
        raise SystemExit(1)

    env = create_random_environment(cfg)

    observation, info = env.reset()
    env.render()

    if cfg["mode"] == "human":

        action = 0.0
    
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    
            keys = pygame.key.get_pressed()
            boat_rotation = 0.0
            if keys[pygame.K_LEFT]:
                boat_rotation = 1.0
            if keys[pygame.K_RIGHT]:
                boat_rotation = -1.0
    
            action = boat_rotation
    
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            print(obs["next_checkpoint_relative"][2])
            if terminated or truncated:
                obs, info = env.reset()
    
        env.close()


    elif cfg["mode"] == "agent":
        pass


    elif cfg["mode"] == "random":
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = env.action_space.sample()
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    
            if terminated or truncated:
                obs, info = env.reset()
    
        env.close()

