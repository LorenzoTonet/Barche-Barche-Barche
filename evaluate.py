"""
Lorenzo Tonet SM3800123
Emanuele Toso SM3800114

evaluate.py

This file contains simply the logic to evaluate a pre-trained PPO agent on a randomized sailing environment. Like other files, 
all the parameters/hyperparameters are loaded from a YAML config file. The evaluation can be done in two modes:
- single_run: the agent is evaluated on a single run and the cumulative reward is plotted at the end of the run
- save_multi_run: the agent is evaluated on multiple runs and the results are rendered on a single video and saved to disk. 
  The video is saved in the path specified in the config file under the key "evaluate" -> "video_path".

This file has been used to visualize the performance of the agent on a single run and to generate a video of multiple runs for the final report.
"""


import yaml
import numpy as np
import imageio
import matplotlib.pyplot as plt
import argparse

from source_code.Environment.environment import SailingEnv, FlattenSailingObs
from source_code.Environment.environment_generators import create_random_environment_old, create_random_environment
from source_code.Environment.map_elements import Checkpoint
from source_code.Agent.PPO import PPOAgent
from source_code.Other.render_multi_boats import render_multi_boats


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
parser.add_argument('--agent', type=str, default='./experiments/Orfeo/ppo_sailing.pt', help="Path to agent model to load")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

#cp1 = Checkpoint(np.array([20.0, 20.0]), radius=5.0, number=1)
#cp2 = Checkpoint(np.array([30.0, 40.0]), radius=5.0, number=2)
#cp3 = Checkpoint(np.array([40.0, 10.0]), radius=5.0, number=3)
#checkpoints = [cp1, cp2, cp3]

env = create_random_environment(cfg)
env = FlattenSailingObs(env)

agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

agent.load("experiments/Orfeo/ppo_sailing.pt")
step = 0

# in single run mode we simulate one run and render it
if cfg['evaluate']['mode'] == "single_run":
    cum_reward = []
    state, _ = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = agent.get_action(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)
        step += 1
        cum_reward.append(reward+cum_reward[-1] if len(cum_reward) > 0 else reward)
        print(f"Step: {step}  | Reward: {reward:.3f}  | Cumulative: {cum_reward[-1]:.3f}")
        env.render()
    plt.plot(cum_reward)
    plt.title("Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid()
    plt.show()

# in save multi run mode we simulate multiple runs on the same env, render them on a single video and save the video to disk
if cfg['evaluate']['mode'] == "save_multi_run":

    render_multi_boats(cfg, env, agent)

env.close()