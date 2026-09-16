import yaml
import numpy as np
from source_code.environment import SailingEnv, FlattenSailingObs
from source_code.map_elements import Checkpoint
from source_code.PPO_beta import PPOAgent

with open("./config.yaml") as f:
    cfg = yaml.safe_load(f)

cp1 = Checkpoint(np.array([20.0, 20.0]), radius=5.0, number=1)
cp2 = Checkpoint(np.array([40.0, 40.0]), radius=5.0, number=2)
checkpoints = [cp1, cp2]

env = SailingEnv(cfg, checkpoints=checkpoints, render_mode="human")
env = FlattenSailingObs(env)

agent = PPOAgent(state_dim=13, action_dim=1)
agent.load("checkpoints/ppo_sailing.pt")

state, _ = env.reset()
done = truncated = False
while not (done or truncated):
    action, _ = agent.get_action(state, deterministic=True)
    state, reward, done, truncated, info = env.step(action)
    print(f"Reward = {reward}")
    env.render()

env.close()