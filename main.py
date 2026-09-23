import argparse
import yaml
import numpy as np

import pygame

from source_code.environment import SailingEnv, create_random_environment
from source_code.vector_field import VecField
from source_code.map_elements import Checkpoint
from source_code.PPO import PPOAgent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
    parser.add_argument('--agent', type=str, default='./checkpoints/ppo_sailing.pt', help="Path to agent model to load")
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
        step = 0
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
            step += 1
            env.render()
            print(f"Step: {step}  | Reward: {reward:.3f}  |")
            if terminated or truncated:
                obs, info = env.reset()
    
        env.close()

    elif cfg["mode"] == "agent":
        agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

        agent.load(args.agent)

        if cfg['evaluate']['mode'] == "single_run":
            step = 0
            state, _ = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = agent.get_action(state, deterministic=True)
                state, reward, done, truncated, info = env.step(action)
                step += 1
                print(f"Step: {step}  | Reward: {reward:.3f}  |")
                env.render()

            env.close()

    elif cfg["mode"] == "random":
        step = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = float(env.action_space.sample()[0])
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            print(f"Step: {step}  | Reward: {reward:.3f}  |")
            env.render()
    
            if terminated or truncated:
                obs, info = env.reset()
    
        env.close()

    else:
        print("Invalid play mode. Choose between:")
        print("[random', 'human', 'agent']")

