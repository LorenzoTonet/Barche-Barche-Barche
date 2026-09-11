import argparse
import yaml
import numpy as np

import pygame

from source_code.environment import SailingEnv
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

    vector_field = VecField(cfg["map_width"], cfg["map_height"], None)
    goal = Checkpoint(np.array([80.0, 80.0]), radius=5.0, number=3)
    cp1 = Checkpoint(np.array([20.0, 20.0]), radius=5.0, number=1)
    cp2 = Checkpoint(np.array([50.0, 50.0]), radius=5.0, number=2)
    checkpoints = [cp1, cp2, goal]

    env = SailingEnv(cfg, vector_field, checkpoints=checkpoints, render_mode="human")
    observation, info = env.reset()
    env.render()

    if cfg["mode"] == "human":

        action = {
            "boat_rotation": np.array([0.0], dtype=np.float32),
        }
    
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
    
            action["boat_rotation"][0] = boat_rotation
    
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

