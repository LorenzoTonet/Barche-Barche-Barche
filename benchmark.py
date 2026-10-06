"""
Lorenzo Tonet SM3800123
Emanuele Toso SM3800114

benchmark.py

This file contains the logic to benchmark a pre-trained PPO agent on a random environment.
Further details on the benchmark functions can be found in source_code/Agent/benchmark_utils.py.

To run the benchmark, use the following command:
    python benchmark.py --agent checkpoints/ppo_sailing.pt

"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse

from source_code.Environment.environment import SailingEnv, FlattenSailingObs
from source_code.Environment.environment_generators import create_random_environment
from source_code.Environment.map_elements import Checkpoint
from source_code.Agent.PPO import PPOAgent
from source_code.Agent.benchmark_utils import benchmark_agent

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
parser.add_argument('--agent', type=str, default='./checkpoints/ppo_sailing.pt', help="Path to agent model to load")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

agent.load(args.agent)

benchmark_agent(agent=agent, cfg=cfg, save_dir="./", filename="benchmark_det.png", checkpoint_counts=(1,2,3,4), n_tests=200, max_steps=1500, seed = 7, deterministic = True)

benchmark_agent(agent=agent, cfg=cfg, save_dir="./", filename="benchmark_notdet.png", checkpoint_counts=(1,2,3,4), n_tests=200, max_steps=1500, seed = 7, deterministic = False)