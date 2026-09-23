import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.distributions import Beta
import torch.nn as nn

from source_code.environment import FlattenSailingObs, create_random_environment
from source_code.PPO import PPOAgent

import os
import time
import shutil

def policy_update(config, agent, optimizer, scheduler):
    """ Copiato da Panizzon
    """
    if len(agent.buffer) == 0:
        return

    # 1. Unpack the buffer
    states = torch.tensor(np.array([t[0] for t in agent.buffer]), dtype=torch.float32)
    actions = torch.tensor(np.array([t[1] for t in agent.buffer]), dtype=torch.float32)
    rewards = [t[2] for t in agent.buffer]
    next_states = np.array([t[3] for t in agent.buffer])
    dones = [t[4] for t in agent.buffer]
    truncateds = [t[5] for t in agent.buffer] 
    old_log_probs = torch.tensor(np.array([t[6] for t in agent.buffer]), dtype=torch.float32)

    gamma = config['train']['discount_factor']
    lamb = config['train']['gae_lam']
    # RETURNS 
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    with torch.no_grad():
        _, _, next_state_values = agent.network(next_states_tensor)
    next_state_values = next_state_values.squeeze()

    returns = []
    discounted_sum = 0 if (dones[-1] or truncateds[-1]) else next_state_values[-1].item()
    for i in reversed(range(len(rewards))):
        if dones[i]:
            discounted_sum = 0
        elif truncateds[i]:
            discounted_sum = next_state_values[i].item()
        discounted_sum = rewards[i] + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    returns = torch.tensor(returns, dtype=torch.float32)

    # PART 2: Compute Advantages
    with torch.no_grad():
        _, _, state_values = agent.network(states)
    state_values = state_values.squeeze()
    if config['train']['advantages_mode'] == 'Naive':
        advantages = agent.compute_advantages(returns, state_values, old_log_probs, old_log_probs)
    elif config['train']['advantages_mode'] == 'GAE':
        advantages= agent.compute_advantages_gae(
        rewards, state_values.numpy(), next_state_values.numpy(),
        dones, truncateds, gamma, lamb)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
    else:
        raise Exception("advantage mode not valid. Choose from Naive or GAE. Go to config.yaml")

    # PART 3: Update Policy
    batch_size = config['train']['buffer_size']//config['train']['n_batches']

    for _ in range(agent.epochs):
        # shuffle for every epoch
        indices = torch.randperm(config['train']['buffer_size'])

        for start_idx in range(0, config['train']['buffer_size'], batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_returns = returns[batch_indices]
            b_old_log_probs = old_log_probs[batch_indices]
            b_advantages = advantages[batch_indices]

            # Recalculate probabilities and values under the CURRENT, continually updating network
            alpha, beta, state_values = agent.network(b_states)
            state_values = state_values.squeeze()
            
            dist = Beta(alpha, beta)
            
            actions_raw = ((b_actions + 1) / 2).clamp(1e-6, 1 - 1e-6).unsqueeze(-1)  # (N,) -> (N,1)
            curr_log_probs = dist.log_prob(actions_raw).squeeze(-1)                
            
            # 4. Calculate PPO Ratio: r(theta) = pi_new / pi_old = exp(log_new - log_old)
            ratios = torch.exp(curr_log_probs - b_old_log_probs)
            
            # 5. Calculate Clipped Surrogate Objective
            surr1 = ratios * b_advantages
            surr2 = torch.clamp(ratios, 1.0 - agent.clip_ratio, 1.0 + agent.clip_ratio) * b_advantages
            
            # Actor Loss: maximize surrogate (minimize negative surrogate)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss: MSE between V(s) and returns
            critic_loss = nn.MSELoss()(state_values, b_returns)
            
            # Entropy Bonus (optional, encourages exploration)
            entropy = dist.entropy().mean()
            
            # Total Loss formulation
            loss = actor_loss + config['train']['critic_loss_parameter'] * critic_loss - config['train']['entropy_loss_parameter'] * entropy
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
                
    # Clear the buffer after the batch update is complete
    agent.clear_buffer()


def train_ppo_agent(config, agent):
    """
    Train a Proximal Policy Optimization (PPO-Clip) agent in the SailingEnv environment.
    
    PARTE 1: Collect trajectories
    for a number of episodes, collect trajectories by interacting with the environment using the current policy. 
    store the states in a buffer that, every buffer_size steps, will be used to update the policy.
    every env_reset_every episodes, re-generate the environment to introduce variability in the training process.
    
    
    PARTE 2: Compute Advantages
    written in compute_advantages function, called inside policy_update.

    PARTE 3: Update Policy
    written in policy_update function, called every buffer_size steps.
    """
    
    optimizer = optim.Adam(agent.network.parameters(), lr=config['train']['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['lr_step_size']*config['PPO']['epochs']*config['train']['n_batches'], gamma=config['train']['lr_gamma'])


    returns = np.zeros(config['train']['n_episodes'])
    timestep_counter = 0

    # PARTE 1: Collect Trajectories
    for i in range(config['train']['n_episodes']):
        print("Starting episode {}/{}".format(i + 1, config['train']['n_episodes']))
        if i % config["train"]["env_reset_every"] == 0:
            env_cfg = config.deepcopy()
            if i < 100:
                env_cfg["train"]["env"]["n_checkpoints"] = 1 if config["train"]["env"]["n_checkpoints"] >= 1 else config["train"]["env"]["n_checkpoints"]
            elif i < 200:
                env_cfg["train"]["env"]["n_checkpoints"] = 2 if config["train"]["env"]["n_checkpoints"] >= 2 else config["train"]["env"]["n_checkpoints"]
            elif i < 300:
                env_cfg["train"]["env"]["n_checkpoints"] = 3 if config["train"]["env"]["n_checkpoints"] >= 3 else config["train"]["env"]["n_checkpoints"]
            elif i < 400:
                env_cfg["train"]["env"]["n_checkpoints"] = 4 if config["train"]["env"]["n_checkpoints"] >= 4 else config["train"]["env"]["n_checkpoints"]
            env = create_random_environment(env_cfg)
            env = FlattenSailingObs(env) 

        state, _ = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            is_terminal = terminated and not truncated

            agent.store_transition((state, action, reward, next_state, is_terminal, truncated, log_prob))
            state = next_state
            returns[i] += reward
            timestep_counter += 1

            if timestep_counter % cfg["train"]["buffer_size"] == 0:
                policy_update(config, agent, optimizer, scheduler)

    return returns

def save_training_run(returns, model_path, cfg, base_dir="experiments"):
    """
    TODO documentation
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"Training_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
 
    model_dst = os.path.join(run_dir, os.path.basename(model_path))
    shutil.copy(model_path, model_dst)
 
    window = 10
    smoothed_returns = np.convolve(returns, np.ones(window) / window, mode='valid')
 
    plt.figure(figsize=(8, 5))
    plt.plot(smoothed_returns, label='PPO-Clip (Neural)', color='teal', linewidth=2)
    plt.title(f'Proximal Policy Optimization on Boat Sailing Environment ({cfg["train"]["advantages_mode"]})')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards (Moving Average)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "returns.png"))
    plt.close()
 
    config_path = os.path.join(run_dir, "config.txt")
    with open(config_path, "w") as f:
        for section, params in cfg.items():
            f.write(f"[{section}]\n")
            if isinstance(params, dict):
                for k, v in params.items():
                    f.write(f"{k} = {v}\n")
            else:
                f.write(f"{params}\n")
            f.write("\n")
 
    return run_dir

if __name__ == "__main__":
    torch.set_num_threads(1)  # con batch=1 il threading intra-op è solo overhead, peggio ancora su cluster
    torch.distributions.Distribution.set_default_validate_args(False)
    
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
    ppo_agent.save("checkpoints/ppo_sailing.pt")

    save_training_run(returns= returns_ppo, model_path="checkpoints/ppo_sailing.pt", cfg=cfg)

