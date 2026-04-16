"""MAPPO baseline implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class CentralizedValueCritic(nn.Module):
    """Evaluate the joint state to estimate the global value V(s)."""

    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 64):
        """Initialize the critic network.

        Args:
            state_dim: Per-agent state dimension.
            num_agents: Number of agents in the system.
            hidden_dim: Hidden layer width.
        """
        super(CentralizedValueCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim * num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, joint_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            joint_states: Flattened joint state tensor.

        Returns:
            Estimated state value tensor.
        """
        x = F.relu(self.fc1(joint_states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class StochasticActor(nn.Module):
    """Output a probability distribution over offloading locations."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize the actor network.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            hidden_dim: Hidden layer width.
        """
        super(StochasticActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
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
        return F.softmax(self.fc3(x), dim=-1)

class MAPPOAgent:
    """Multi-Agent PPO baseline."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        lr: float = 0.0001,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        ppo_epochs: int = 4,
    ):
        """Initialize the MAPPO agent.

        Args:
            state_dim: Per-agent state dimension.
            action_dim: Number of actions.
            num_agents: Number of agents in the system.
            lr: Learning rate.
            gamma: Discount factor.
            clip_param: PPO clipping parameter.
            ppo_epochs: PPO epochs per update.
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        
        self.actor = StochasticActor(state_dim, action_dim)
        self.critic = CentralizedValueCritic(state_dim, num_agents)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: torch.Tensor) -> int:
        """Sample an action from the policy distribution.

        Args:
            state: Per-agent state tensor.

        Returns:
            Selected action index.
        """
        with torch.no_grad():
            probs = self.actor(state)
            distribution = Categorical(probs)
            action = distribution.sample()
            return action.item()

    def update_agent(
        self,
        state_b: torch.Tensor,
        action_b: torch.Tensor,
        reward_b: torch.Tensor,
        next_state_b: torch.Tensor,
        agent_index: int,
    ) -> None:
        """Update agent parameters using a batch of experiences.

        Args:
            state_b: Batched joint states.
            action_b: Batched joint actions.
            reward_b: Batched joint rewards.
            next_state_b: Batched next joint states.
            agent_index: Index of the agent being updated.
        """
        
        # --- FIX 1: Dẹt (flatten) state_b và next_state_b ---
        batch_size = state_b.size(0)
        joint_state_b = state_b.view(batch_size, -1)
        joint_next_state_b = next_state_b.view(batch_size, -1)

        agent_rewards = reward_b[:, agent_index].unsqueeze(1)
        agent_states = state_b[:, agent_index, :]
        # --- FIX 2: Ép kiểu action sang .long() ---
        agent_actions = action_b[:, agent_index].long() 
        
        # 1. Calculate Advantages and Old Log Probs (Detached)
        with torch.no_grad():
            # Sử dụng joint_state cho critic
            next_v = self.critic(joint_next_state_b)
            target_v = agent_rewards + self.gamma * next_v
            current_v = self.critic(joint_state_b)
            
            advantages = target_v - current_v
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            probs = self.actor(agent_states)
            distribution = Categorical(probs)
            old_log_probs = distribution.log_prob(agent_actions)
            
        # 2. PPO Epochs
        for _ in range(self.ppo_epochs):
            probs = self.actor(agent_states)
            distribution = Categorical(probs)
            log_probs = distribution.log_prob(agent_actions)
            entropy = distribution.entropy().mean()
            
            current_v_epoch = self.critic(joint_state_b)
            
            # Calculate the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate Surrogate Losses
            surr1 = ratios * advantages.squeeze()
            surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.squeeze()
            
            # Actor Loss: Maximize surrogate objective -> minimize negative
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy 
            
            # Critic Loss: Standard MSE
            critic_loss = F.mse_loss(current_v_epoch, target_v)
            
            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Optimize Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
