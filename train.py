"""
Lorenzo Tonet SM3800123
Emanuele Toso SM3800114

train.py

This file contains the logic to train a PPO agent on the sailing environment.
First we load the configuration from a YAML file, then we initialize the PPO agent with the specified 
parameters. The training process is executed from the `train_ppo_agent` function, written in `source_code/Agent/training.py`. 
After training, the model is saved in the experiments folder, and the training run details are saved for future reference.
We also save the benchmarking results of the training run in the same folder as the model checkpoint.
"""

import argparse
import os
import yaml
import torch
import time
import os

from source_code.Agent.PPO import PPOAgent
from source_code.Agent.training import train_ppo_agent, save_training_run
from source_code.Agent.benchmark_utils import benchmark_agent


if __name__ == "__main__":
    experiments_dir = "experiments"
    torch.set_num_threads(1)  # con batch=1 il threading intra-op è solo overhead, peggio ancora su cluster
    torch.distributions.Distribution.set_default_validate_args(False)

    # parse command line arguments (YAML config file path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Exiting.")
        raise SystemExit(1)

    # initialize the PPO agent with the parameters from the config file
    print("Initializing Proximal Policy Optimization (PPO-Clip) Agent...")
    ppo_agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

    # if needed to fine tune a pre-trained model
    #ppo_agent.load("checkpoints/really_good_model_copy.pt")

    # train the PPO agent and save the model and training run details

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(experiments_dir, f"Training_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    
    returns_ppo = train_ppo_agent(cfg, ppo_agent)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    ppo_agent.save("checkpoints/ppo_sailing.pt")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(experiments_dir, f"Training_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    save_training_run(returns= returns_ppo, model_path="checkpoints/ppo_sailing.pt", cfg=cfg, save_dir=run_dir)
    benchmark_agent(agent=ppo_agent, cfg=cfg, save_dir=run_dir, filename="benchmark.png", checkpoint_counts=(1,2,3,4), n_tests=200, max_steps=1500, seed = 7)


