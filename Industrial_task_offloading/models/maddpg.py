import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class SelfAttention(nn.Module):
    """
    Self-attention mechanism applied to the input sequence of joint states and actions.
    Implements Q = HW_Q, K = HW_K, V = HW_V and the Softmax scoring.
    """
    def __init__(self, feature_dim: int, attention_dim: int):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        
        # Learnable weight matrices W_Q, W_K, W_V
        self.W_Q = nn.Linear(feature_dim, attention_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H shape: (batch_size, seq_len, feature_dim)
        
        Q = self.W_Q(H)
        K = self.W_K(H)
        V = self.W_V(H)
        
        # Compute similarity scores: (Q * K^T) / sqrt(d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
        
        # Normalize into attention weights using Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Calculate attention output
        attention_output = torch.bmm(attention_weights, V)
        
        return attention_output

class CriticNetwork(nn.Module):
    """
    The Critic evaluates the joint state and action space using Self-Attention.
    """
    def __init__(self, state_dim: int, action_dim: int, num_agents : int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        feature_dim = state_dim + action_dim
        
        # Self-attention requires a sequence. We will treat the concatenated
        # joint state and action as a sequence of length 1 for simplicity, 
        # or you can split features if treating agents as sequence elements.
        self.attention = SelfAttention(feature_dim=feature_dim, attention_dim=hidden_dim)
        
        self.fc1 = nn.Linear(num_agents*feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Outputs a single Q-value

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)
        
        # Apply self-attention (unsqueeze to add sequence dimension)
        # x = x.unsqueeze(1)
        x = self.attention(x)
        # x = x.squeeze(1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value

class ActorNetwork(nn.Module):
    """
    The Actor decides the offloading location based on the agent's local state.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer maps to action probabilities (discrete offloading locations)
        self.fc3 = nn.Linear(hidden_dim, action_dim) 

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Softmax to get probabilities for each offloading destination
        action_probs = F.softmax(self.fc3(x), dim=-1) 
        return action_probs

class EpsilonATNMADDPGAgent:
    """
    Wrapper for an individual agent managing its Actor, Critic, and epsilon-greedy logic.
    """
    def __init__(self, state_dim: int, action_dim: int, num_agents: int,
                 lr: float = 0.0001, epsilon_init: float = 1.0, epsilon_min: float = 0.01, decay: float = 0.995):
        self.action_dim = action_dim
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(state_dim, action_dim, num_agents)
        self.target_critic = CriticNetwork(state_dim, action_dim, num_agents)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers (learning rate of 0.0001 as proven optimal in the paper)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.decay = decay

    def select_action(self, state: torch.Tensor) -> int:
        """
        Implements the epsilon-greedy action selection strategy.
        """
        # Random exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation based on policy
        with torch.no_grad():
            action_probs = self.actor(state)
            return torch.argmax(action_probs).item()

    def update_epsilon(self):
        """Decays epsilon over time to shift from exploration to exploitation."""
        self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)

    def soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float = 0.01):
        """Soft updates target network parameters."""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)