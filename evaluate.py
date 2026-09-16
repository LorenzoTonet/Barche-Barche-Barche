import yaml
import numpy as np
from source_code.environment import SailingEnv, FlattenSailingObs
from source_code.map_elements import Checkpoint
from source_code.PPO_beta import PPOAgent

with open("./config.yaml") as f:
    cfg = yaml.safe_load(f)

cp1 = Checkpoint(np.array([15.0, 15.0]), radius=5.0, number=1)
checkpoints = [cp1]

env = SailingEnv(cfg, checkpoints=checkpoints, render_mode="human")
env = FlattenSailingObs(env)

agent = PPOAgent(state_dim=13, action_dim=1)
agent.load("checkpoints/ppo_sailing.pt")

state, _ = env.reset()
done = truncated = False
while not (done or truncated):
    action, _ = agent.get_action(state, deterministic=True)
    print(action)
    state, reward, done, truncated, info = env.step(action)
    env.render()

env.close()