import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Beta
import gymnasium as gym
import numpy as np

class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dimension = 128, action_dim = 1):
        super(ActorCriticNetwork, self).__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU()
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, 1)
        )

        self.actor_alpha = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, action_dim),
            torch.nn.Softplus()
        )
        self.actor_beta = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, action_dim),
            torch.nn.Softplus()
        )

    def forward(self, x):
        action_features = self.actor(x)
        action_alpha = self.actor_alpha(action_features) + 1.0 + 1e-3
        action_beta = self.actor_beta(action_features) + 1.0 + 1e-3
        state_value = self.critic(x)
        return action_alpha, action_beta, state_value


class SharedActorCriticNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dimension = 128, action_dim = 1):
        super(SharedActorCriticNetwork, self).__init__()
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dimension, hidden_dimension),
            torch.nn.ReLU()
        )
        self.actor_alpha = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, action_dim),
            torch.nn.Softplus()
        )
        self.actor_beta = torch.nn.Sequential(
            torch.nn.Linear(hidden_dimension, action_dim),
            torch.nn.Softplus()
        )

        self.critic = torch.nn.Linear(hidden_dimension, 1)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        
        state_value = self.critic(shared_output)

        action_alpha = self.actor_alpha(shared_output) + 1.0 + 1e-3
        action_beta = self.actor_beta(shared_output) + 1.0 + 1e-3


        return action_alpha, action_beta, state_value

class PPOAgent():
    def __init__(self,
                    state_dim,
                    action_dim,
                    hidden_dim: int = 128, 
                    discount_factor: float = 0.99, 
                    lr: float = 0.01, 
                    clip_ratio: float = 0.2, 
                    epochs: int = 5, 
                    shared_net: bool = False,
                    critic_loss_paramter: float = 0.5,
                    entropy_loss_parameter: float = 0.01):
        
        self.state_dimension = state_dim
        self.action_dimension = action_dim
        self.hidden_dimention = hidden_dim
        self.discount_factor = discount_factor
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.critic_loss_parameter = critic_loss_paramter
        self.entropy_loss_parameter = entropy_loss_parameter

        if not shared_net:
            self.network = ActorCriticNetwork(input_dim=self.state_dimension, hidden_dimension= self.hidden_dimention, action_dim= self.action_dimension)
        else:
            self.network = SharedActorCriticNetwork(input_dim=self.state_dimension, hidden_dimension= self.hidden_dimention, action_dim= self.action_dimension)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffer to store trajectories for the batch update
        self.buffer = []

    def get_action(self, state, deterministic: bool = False):
        s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
                if deterministic:
                    alpha, beta, _ = self.network(s_tensor)
                    action = (alpha - 1) / (alpha + beta - 2)
                    action = 2 * action - 1  # Scale to [-1, 1]
                    return action.item(), None  # Return the mean action for deterministic behavior
                else:
                    alpha, beta, _ = self.network(s_tensor)
                    dist = Beta(alpha, beta)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    action = 2 * action - 1  # Scale to [-1, 1]
                    action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
                    return action.item(), log_prob.item()

    def store_transition(self, transition):
        self.buffer.append(transition)

    def clear_buffer(self):
        self.buffer = []

    def update(self):
        """ Copiato da Panizzon
        """
        if len(self.buffer) == 0:
            return

        # 1. Unpack the buffer
        states = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.float32)
        rewards = [t[2] for t in self.buffer]
        next_states = np.array([t[3] for t in self.buffer])
        dones = [t[4] for t in self.buffer]
        truncateds = [t[5] for t in self.buffer] 
        old_log_probs = torch.tensor(np.array([t[6] for t in self.buffer]), dtype=torch.float32)

        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        with torch.no_grad():
            _, _, next_state_values = self.network(next_states_tensor)
        next_state_values = next_state_values.squeeze()

        returns = []
        discounted_sum = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_sum = 0
            elif truncateds[i]:
                discounted_sum = next_state_values[i].item()
            discounted_sum = rewards[i] + self.discount_factor * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(self.epochs):
            # Recalculate probabilities and values under the CURRENT, continually updating network
            alpha, beta, state_values = self.network(states)
            state_values = state_values.squeeze()
            
            dist = Beta(alpha, beta)
            
            actions_raw = (actions + 1) / 2
            actions_raw = actions_raw.clamp(1e-6, 1 - 1e-6)
            curr_log_probs = dist.log_prob(actions_raw)
            
            # Calculate Advantage
            # Advantage must be detached so gradients don't flow backward through the target calculation
            advantages = returns - state_values.detach()
            
            # 4. Calculate PPO Ratio: r(theta) = pi_new / pi_old = exp(log_new - log_old)
            ratios = torch.exp(curr_log_probs - old_log_probs)
            
            # 5. Calculate Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Actor Loss: maximize surrogate (minimize negative surrogate)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss: MSE between V(s) and returns
            critic_loss = nn.MSELoss()(state_values, returns)
            
            # Entropy Bonus (optional, encourages exploration)
            entropy = dist.entropy().mean()
            
            # Total Loss formulation
            loss = actor_loss + self.critic_loss_parameter * critic_loss - self.entropy_loss_parameter * entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                    
        # Clear the buffer after the batch update is complete
        self.clear_buffer()

    def save(self, path: str):
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dimension,
            "action_dim": self.action_dimension,
            "hidden_dim": self.hidden_dimention,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



def train_ppo_agent(agent, env: gym.Env, n_episodes: int = 1500, update_timestep: int = 2000) -> list:
    r"""
    Executes the PPO training loop.
    Collects a fixed number of timesteps across episodes before triggering the epoch update.
    """
    returns = np.zeros(n_episodes)
    timestep_counter = 0
    
    for i in range(n_episodes):
        print("Starting episode {}/{}".format(i + 1, n_episodes))
        state, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action, log_prob = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            is_terminal = done and not truncated
            
            # Store data in agent's rollout buffer
            agent.store_transition((state, action, reward, next_state, is_terminal, truncated, log_prob))
            
            state = next_state
            returns[i] += reward
            timestep_counter += 1
            
            # Trigger PPO update if we have collected enough timesteps
            if timestep_counter % update_timestep == 0:
                agent.update()
                
        if (i + 1) % 500 == 0:
            print(f"[PPO-Clip] Episode {i+1:4d} | Last Return: {returns[i]:.0f}")

    
    return returns
        
            
