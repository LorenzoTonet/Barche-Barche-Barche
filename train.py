import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from source_code.environment import SailingEnv, FlattenSailingObs, create_random_environment
from source_code.vector_field import VecField
from source_code.map_elements import Checkpoint
from source_code.PPO_beta import train_ppo_agent, PPOAgent


def train_ppo_agent(config, agent):
    """
    Train a Proximal Policy Optimization (PPO-Clip) agent in the SailingEnv environment.
    """
    
    optimizer = optim.Adam(agent.network.parameters(), lr=config['train']['lr'])

    # TMP
    loss = 0

    # PARTE 1: Collect trajectories
    
    returns = np.zeros(config['train']['n_episodes'])
    timestep_counter = 0

    for i in range(config['train']['n_episodes']):
        print("Starting episode {}/{}".format(i + 1, config['train']['n_episodes']))

        # create random environment for each episode
        env = create_random_environment(config)
        state = env.reset()
        done = False
        episode_return = 0

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, log_prob, value)
            state = next_state
            episode_return += reward
            timestep_counter += 1

        returns[i] = episode_return



    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return agent


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


    print("Initializing Proximal Policy Optimization (PPO-Clip) Agent...")
    ppo_agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )

    # Train for fewer episodes because PPO converges much faster than basic Actor-Critic
    returns_ppo = train_ppo_agent(cfg, ppo_agent)

    plt.figure(figsize=(8, 5))
        
    window = 10
    smoothed_ppo = np.convolve(returns_ppo, np.ones(window)/window, mode='valid')
    
    plt.plot(smoothed_ppo, label='PPO-Clip (Neural)', color='teal', linewidth=2)
    
    plt.title('Proximal Policy Optimization on Boat Sailing Environment')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards (Moving Average)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

