import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from source_code.environment import SailingEnv, FlattenSailingObs
from source_code.vector_field import VecField
from source_code.map_elements import Checkpoint
from source_code.PPO_beta import train_ppo_agent, PPOAgent


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

    cp1 = Checkpoint(np.array([20.0, 20.0]), radius=5.0, number=1)
    goal = Checkpoint(np.array([40.0, 20.0]), radius=5.0, number=2)
    checkpoints = [cp1, goal]

    env = SailingEnv(cfg, checkpoints=checkpoints, render_mode="human")
    env = FlattenSailingObs(env)

    observation, info = env.reset()

    print("Initializing Proximal Policy Optimization (PPO-Clip) Agent...")
    ppo_agent = PPOAgent(
        state_dim=13,
        action_dim=1,
        lr=2e-3,
        clip_ratio=0.2,
        epochs=4
    )
    
    # Train for fewer episodes because PPO converges much faster than basic Actor-Critic
    n_episodes = 100
    returns_ppo = train_ppo_agent(ppo_agent, env, n_episodes=n_episodes, update_timestep=2000)

    ppo_agent.save("checkpoints/ppo_sailing.pt")

    env.close()
    plt.figure(figsize=(8, 5))
        
    window = 10
    smoothed_ppo = np.convolve(returns_ppo, np.ones(window)/window, mode='valid')
    
    plt.plot(smoothed_ppo, label='PPO-Clip (Neural)', color='teal', linewidth=2)
    
    plt.title('Proximal Policy Optimization on CartPole-v1')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards (Moving Average)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

