import torch
from torch.distributions import Beta

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
                    state_dim: int = 17,
                    action_dim: int = 1,
                    hidden_dim: int = 128,
                    clip_ratio: float = 0.2, 
                    epochs: int = 5, 
                    shared_net: bool = False):
        
        self.state_dimension = state_dim
        self.action_dimension = action_dim
        self.hidden_dimention = hidden_dim
        self.clip_ratio = clip_ratio
        self.epochs = epochs

        if not shared_net:
            self.network = ActorCriticNetwork(input_dim=self.state_dimension, hidden_dimension= self.hidden_dimention, action_dim= self.action_dimension)
        else:
            self.network = SharedActorCriticNetwork(input_dim=self.state_dimension, hidden_dimension= self.hidden_dimention, action_dim= self.action_dimension)

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

    def save(self, path: str):
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "state_dim": self.state_dimension,
            "action_dim": self.action_dimension,
            "hidden_dim": self.hidden_dimention,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.network.load_state_dict(checkpoint["network_state_dict"])

    def compute_advantages(self, returns, state_values, old_log_probs, curr_log_probs):
        """
        TODO: implement a more sophisticated advantage estimation method, such as GAE (Generalized Advantage Estimation) or other methods
        """
        # Advantage must be detached so gradients don't flow backward through the target calculation
        advantages = returns - state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
    
    def compute_advantages_gae(self, rewards, values, next_values,
                           terminated, truncated, gamma, lamb):
        
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(terminated[t])                  
            continues   = 1.0 - float(terminated[t] or truncated[t])
            delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
            gae = delta + gamma * lamb * continues * gae
            adv[t] = gae

        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv_norm

