import yaml
import numpy as np
import imageio

from source_code.Environment.environment import SailingEnv, FlattenSailingObs
from source_code.Environment.environment_generators import create_random_environment_old, reate_random_environment
from source_code.Environment.map_elements import Checkpoint
from source_code.Agent.PPO import PPOAgent
from source_code.Other.render_multi_boats import render_multi_boats
import imageio

with open("./config.yaml") as f:
    cfg = yaml.safe_load(f)

#cp1 = Checkpoint(np.array([20.0, 20.0]), radius=5.0, number=1)
#cp2 = Checkpoint(np.array([30.0, 40.0]), radius=5.0, number=2)
#cp3 = Checkpoint(np.array([40.0, 10.0]), radius=5.0, number=3)
#checkpoints = [cp1, cp2, cp3]

env = create_random_environment_old(cfg)
env = FlattenSailingObs(env)

agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

agent.load("experiments/good model/really_good_model.pt")

# in single run mode we simulate one run and render it
if cfg['evaluate']['mode'] == "single_run":
    state, _ = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = agent.get_action(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)
        print(f"Reward = {reward}")
        env.render()

# in save multi run mode we simulate multiple runs on the same env, render them on a single video and save the video to disk
if cfg['evaluate']['mode'] == "save_multi_run":

    render_multi_boats(cfg, env, agent)

env.close()