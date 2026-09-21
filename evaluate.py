import yaml
import numpy as np
from source_code.environment import SailingEnv, FlattenSailingObs
from source_code.map_elements import Checkpoint
from source_code.PPO import PPOAgent

with open("./config.yaml") as f:
    cfg = yaml.safe_load(f)

cp1 = Checkpoint(np.array([20.0, 20.0]), radius=5.0, number=1)
checkpoints = [cp1]

env = SailingEnv(cfg, checkpoints=checkpoints, render_mode="human")
env = FlattenSailingObs(env)

agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

agent.load("checkpoints/ppo_sailing.pt")

state, _ = env.reset()
done = truncated = False
while not (done or truncated):
    action, _ = agent.get_action(state, deterministic=True)
    state, reward, done, truncated, info = env.step(action)
    print(f"Reward = {reward}")
    env.render()

env.close()