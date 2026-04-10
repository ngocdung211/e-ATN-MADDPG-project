import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class CentralizedValueCritic(nn.Module):
    """
    Evaluates the joint state of all agents to estimate the global Value V(s).
    Used to calculate the Advantage for the Actor updates.
    """
    def __init__(self, joint_state_dim: int, hidden_dim: int = 64):
        super(CentralizedValueCritic, self).__init__()
        self.fc1 = nn.Linear(joint_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Outputs a single Value

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
        # Softmax creates the probability distribution
        return F.softmax(self.fc3(x), dim=-1)

class MAACAgent:
    """
    Multi-Agent Actor-Critic baseline.
    Uses categorical sampling for exploration and Advantage-based policy gradients.
    """
    def __init__(self, state_dim: int, action_dim: int, joint_state_dim: int, joint_action_dim: int, 
                 lr: float = 0.0001, gamma: float = 0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.actor = StochasticActor(state_dim, action_dim)
        self.critic = CentralizedValueCritic(joint_state_dim)
        
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
        """
        Self-contained update logic for the MAAC architecture.
        """
        # 1. Critic Update (Minimize TD Error)
        agent_rewards = reward_b[:, agent_index].unsqueeze(1)
        current_v = self.critic(state_b)
        
        with torch.no_grad():
            next_v = self.critic(next_state_b)
            target_v = agent_rewards + self.gamma * next_v
            
        critic_loss = F.mse_loss(current_v, target_v)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 2. Actor Update (Maximize Advantage * log_prob)
        advantage = (target_v - current_v).detach()
        
        agent_states = state_b[:, agent_index, :]
        agent_actions = action_b[:, agent_index]
        
        probs = self.actor(agent_states)
        m = Categorical(probs)
        log_probs = m.log_prob(agent_actions)
        
        # Policy gradient loss: -mean(log_pi * Advantage)
        actor_loss = -(log_probs * advantage.squeeze()).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()