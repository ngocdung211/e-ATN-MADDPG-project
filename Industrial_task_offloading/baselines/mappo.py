import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class CentralizedValueCritic(nn.Module):
    """
    Evaluates the joint state of all agents to estimate the global Value V(s).
    """
    def __init__(self, state_dim: int, num_agents: int, hidden_dim: int = 64):
        super(CentralizedValueCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim * num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, joint_states: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(joint_states))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class StochasticActor(nn.Module):
    """
    Outputs a probability distribution over the available offloading locations.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(StochasticActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class MAPPOAgent:
    """
    Multi-Agent PPO baseline.
    Uses clipped surrogate objective and multiple epochs per update.
    """
    def __init__(self, state_dim: int, action_dim: int, num_agents: int,
                 lr: float = 0.0001, gamma: float = 0.99, clip_param: float = 0.2, ppo_epochs: int = 4):
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        
        self.actor = StochasticActor(state_dim, action_dim)
        self.critic = CentralizedValueCritic(state_dim, num_agents)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state: torch.Tensor) -> int:
        """Samples an action from the policy distribution."""
        with torch.no_grad():
            probs = self.actor(state)
            m = Categorical(probs)
            action = m.sample()
            return action.item()

    def update_agent(self, state_b: torch.Tensor, action_b: torch.Tensor, 
                     reward_b: torch.Tensor, next_state_b: torch.Tensor, agent_index: int):
        
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
            m = Categorical(probs)
            old_log_probs = m.log_prob(agent_actions)
            
        # 2. PPO Epochs
        for _ in range(self.ppo_epochs):
            probs = self.actor(agent_states)
            m = Categorical(probs)
            log_probs = m.log_prob(agent_actions)
            entropy = m.entropy().mean()
            
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