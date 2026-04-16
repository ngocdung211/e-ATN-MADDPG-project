"""MADDPG-based agents with optional self-attention and epsilon-greedy."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SelfAttention(nn.Module):
    """Self-attention mechanism for joint state-action sequences."""

    def __init__(self, feature_dim: int, attention_dim: int):
        """Initialize the self-attention module.

        Args:
            feature_dim: Input feature dimension.
            attention_dim: Attention projection dimension.
        """
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        
        # Learnable weight matrices W_Q, W_K, W_V
        self.W_Q = nn.Linear(feature_dim, attention_dim, bias=False)
        self.W_K = nn.Linear(feature_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Tensor shaped (batch_size, seq_len, feature_dim).

        Returns:
            Attention-weighted features.
        """
        
        Q = self.W_Q(features)
        K = self.W_K(features)
        V = self.W_V(features)
        
        # Compute similarity scores: (Q * K^T) / sqrt(d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
        
        # Normalize into attention weights using Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Calculate attention output
        attention_output = torch.bmm(attention_weights, V)
        
        return attention_output

class CriticNetwork(nn.Module):
    """Critic that evaluates joint state-action values."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        use_attention: bool = True,
    ):
        """Initialize the critic network.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            num_agents: Number of agents in the system.
            hidden_dim: Hidden layer width.
            use_attention: Whether to apply self-attention.
        """
        super(CriticNetwork, self).__init__()
        feature_dim = state_dim + action_dim
        self.use_attention = use_attention
        self.attention = SelfAttention(feature_dim=feature_dim, attention_dim=hidden_dim) if use_attention else None
        
        self.fc1 = nn.Linear(num_agents*feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Outputs a single Q-value

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            states: Joint states tensor.
            actions: Joint actions tensor.

        Returns:
            Q-value estimates.
        """
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)

        if self.use_attention:
            x = self.attention(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value

class ActorNetwork(nn.Module):
    """Actor that outputs offloading action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize the actor network.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            hidden_dim: Hidden layer width.
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer maps to action probabilities (discrete offloading locations)
        self.fc3 = nn.Linear(hidden_dim, action_dim) 

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: Per-agent state tensor.

        Returns:
            Action probability distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Softmax to get probabilities for each offloading destination
        action_probs = F.softmax(self.fc3(x), dim=-1) 
        return action_probs

class EpsilonATNMADDPGAgent:
    """Agent wrapper for Actor, Critic, and epsilon-greedy logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 0.0001,
        epsilon_init: float = 1.0,
        epsilon_min: float = 0.01,
        decay: float = 0.998,
        use_attention: bool = True,
        use_epsilon_greedy: bool = True,
    ):
        """Initialize the agent.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            num_agents: Number of agents in the system.
            lr: Learning rate.
            epsilon_init: Initial epsilon for exploration.
            epsilon_min: Minimum epsilon value.
            decay: Epsilon decay factor.
            use_attention: Whether to use attention in the critic.
            use_epsilon_greedy: Enable epsilon-greedy exploration.
        """
        self.action_dim = action_dim
        self.use_epsilon_greedy = use_epsilon_greedy
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(state_dim, action_dim, num_agents, use_attention=use_attention)
        self.target_critic = CriticNetwork(state_dim, action_dim, num_agents, use_attention=use_attention)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers (learning rate of 0.0001 as proven optimal in the paper)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.decay = decay

    def select_action(self, state: torch.Tensor) -> int:
        """Select an action using epsilon-greedy strategy.

        Args:
            state: Per-agent state tensor.

        Returns:
            Selected action index.
        """
        # Random exploration
        if self.use_epsilon_greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation based on policy
        with torch.no_grad():
            action_probs = self.actor(state)
            return torch.argmax(action_probs).item()

    def update_epsilon(self) -> None:
        """Decay epsilon over time to shift from exploration to exploitation."""
        if self.use_epsilon_greedy:
            self.epsilon = max(self.epsilon * self.decay, self.epsilon_min)

    def soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float = 0.01) -> None:
        """Soft-update target network parameters.

        Args:
            target_net: Target network to update.
            source_net: Source network providing parameters.
            tau: Interpolation factor.
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
