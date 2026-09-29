import torch
from torch.distributions import Beta
import numpy as np

'''

In this file there is the logic about the Machine Learning models, including the definition of the neural backbone divided in two categories:
- Actor Critic Network: An architechture that uses 2 different network for the critic and the actor
- Shared AC Network: An architechture where both the A and C utilize the same net to do all the work - better efficency but noisier

PPOAgent class is the actual actor that will behave in the environment during both training and inference fase. It contains the logic for both 
the policy (and the "decision making") and for the update done during the trining phase.

- get_action() : use the neural backbone to predict the parameters for a Beta distribution. Behavior is different during train and inference.
- store_transition() and clear_buffer() : function to manage the internal memory of the agent (used during the training)
- save() and load() : utils for model persistency
- compute_advantange() : compute A(s,a) with the naive method
- compute_advantange_gae() : compute A(s,a) with the Generalized Advantage Estimation Technique

'''
class ActorCriticNetwork(torch.nn.Module):
    '''
    In this architechture there are 2 separate nets for the actor and for the critic
    '''
    def __init__(self, input_dim, hidden_dimension = 128, action_dim = 1):
        '''
        Inputs:
        - input_dim: dimension of the observation
        - hidden_dimension: dimension of the 2 hidden layers
        - action_dim: dimension of the action space
        
        '''
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
        '''
        Inputs:
        - x : observation vector
        
        Output:
        - action_alpha : alpha parameter for the beta distribution (>1)
        - action_alpha : aeta parameter for the beta distribution (>1)
        - state_value : state value for observation x 
        '''
        action_features = self.actor(x)
        action_alpha = self.actor_alpha(action_features) + 1.0 + 1e-3
        action_beta = self.actor_beta(action_features) + 1.0 + 1e-3
        state_value = self.critic(x)
        return action_alpha, action_beta, state_value


class SharedActorCriticNetwork(torch.nn.Module):
    '''
    In this architechture there are is only 1 network for both the actor and for the critic.
    '''
    def __init__(self, input_dim, hidden_dimension = 128, action_dim = 1):
        '''
        Input:
        - input_dim: dimension of the observation
        - hidden_dimension: dimension of the 2 hidden layers
        - action_dim: dimension of the action space
        
        '''
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
        '''
        Inputs:
        - x : observation vector
        
        Output:
        - action_alpha : alpha parameter for the beta distribution (>1)
        - action_alpha : aeta parameter for the beta distribution (>1)
        - state_value : state value for observation x 
        '''
        shared_output = self.shared_layers(x)
        
        state_value = self.critic(shared_output)

        action_alpha = self.actor_alpha(shared_output) + 1.0 + 1e-3
        action_beta = self.actor_beta(shared_output) + 1.0 + 1e-3


        return action_alpha, action_beta, state_value


class PPOAgent():
    '''
    PPOAgent class is the actual actor that will behave in the environment during both training and inference fase. It contains the logic for both 
    the policy (and the "decision making") and for the update done during the trining phase.
    '''
    def __init__(self,
                    state_dim: int = 17,
                    action_dim: int = 1,
                    hidden_dim: int = 128,
                    clip_ratio: float = 0.2, 
                    epochs: int = 5, 
                    shared_net: bool = False):
        '''
        Inputs:
        - state_dim: dimension of the observation
        - hidden_dim: dimension of the 2 hidden layers
        - action_dim: dimension of the action space
        
        '''
        
        self.state_dimension = state_dim
        self.action_dimension = action_dim
        self.hidden_dimention = hidden_dim
        self.clip_ratio = clip_ratio
        self.epochs = epochs

        if not shared_net:
            self.network = ActorCriticNetwork(input_dim = self.state_dimension, hidden_dimension = self.hidden_dimention, action_dim = self.action_dimension)
        else:
            self.network = SharedActorCriticNetwork(input_dim = self.state_dimension, hidden_dimension = self.hidden_dimention, action_dim= self.action_dimension)

        # Buffer to store trajectories for the batch update
        self.buffer = []

    def get_action(self, state, deterministic: bool = False):
        '''
        Inputs:
        - state: the observation vector
        - deterministic: True if the actor should act deterministically
        
        Outputs:
        - action: a number between -1 and 1
        - log_prob: the probability of that action (None if deterministic)
        
        The function works differently depending on 
        '''
        s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
                if deterministic:
                    alpha, beta, _ = self.network(s_tensor)
                    action = (alpha - 1) / (alpha + beta - 2)
                    action = 2 * action - 1  # Scale to [-1, 1]
                    #print(f"Alpha = {alpha.item()} | Beta = {beta.item()} | action = {action.item()}")
                    return action.item(), None  # Return the mode action for deterministic behavior
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

    def compute_advantages(self, returns, state_values):
        """
       
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

