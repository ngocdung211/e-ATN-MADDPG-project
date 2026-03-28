import torch
import torch.nn.functional as F
from utils.metrics import compute_reward
from models.replay_buffer import MultiAgentReplayBuffer
from models.maddpg import SelfAttention, CriticNetwork
from models.gcn import TaskPriorityGCN

def train_maddpg(agents, env, replay_buffer, gcn_model, num_episodes=1000, batch_size=64, gamma=0.99):
    """
    Implements Algorithm 1: e-ATN-MADDPG
    """
    num_agents = len(agents)
    
    # 1. Generate reasonable scheduling priorities of subtasks based on GCN
    # In a real run, you'd pass the DAG adjacency matrix and node features to the GCN here.
    # priorities = gcn_model(node_features, adjacency_matrix)
    
    for episode in range(num_episodes):
        # Initialize states of agents
        # state represents S = {s_{1,m}^t, ..., s_{D,m}^t}
        current_joint_state = env.reset() 
        
        # Loop over time slots (T) and subtasks (M) based on priority
        done = False
        while not done:
            joint_actions = []
            
            # Select actions based on epsilon-greedy strategy
            for i, agent in enumerate(agents):
                agent_state = torch.FloatTensor(current_joint_state[i])
                action = agent.select_action(agent_state)
                joint_actions.append(action)
                
            # Execute actions in the environment, observe reward and new state
            next_joint_state, joint_rewards, done, _ = env.step(joint_actions)
            
            # Store sample in replay buffer
            replay_buffer.push(current_joint_state, joint_actions, joint_rewards, next_joint_state)
            
            current_joint_state = next_joint_state
        
        # --- Network Updates ---
        if len(replay_buffer) >= batch_size:
            # Loop through each agent to update their individual Actor and Critic networks
            for i, agent in enumerate(agents):
                # Sample a random mini-batch from B
                state_b, action_b, reward_b, next_state_b = replay_buffer.sample(batch_size)
                
                # Get rewards specific to this agent
                agent_rewards = reward_b[:, i].unsqueeze(1)
                
                # --- Critic Network Update (Eq. 25 & 26) ---
                with torch.no_grad():
                    # Target actions from target actor networks for all agents
                    target_joint_actions = []
                    for j, a in enumerate(agents):
                        # Extract the state specifically for agent j
                        s_j = next_state_b[:, j, :]
                        t_action = a.target_actor(s_j)
                        target_joint_actions.append(t_action)
                    
                    target_joint_actions = torch.stack(target_joint_actions, dim=1)
                    
                    # Target Q value using target critic
                    target_q = agent.target_critic(next_state_b, target_joint_actions)
                    y_i = agent_rewards + gamma * target_q
                
                # Current Q value estimated by the critic network
                current_q = agent.critic(state_b, action_b)
                
                # Update critic network by minimizing the MSE loss
                critic_loss = F.mse_loss(current_q, y_i)
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()
                
                # --- Actor Network Update (Eq. 27) ---
                # Predict new actions for the current state using the current policy
                predicted_actions = agent.actor(state_b[:, i, :])
                
                # We need joint actions for the critic, replacing only this agent's action
                predicted_joint_actions = action_b.clone()
                predicted_joint_actions[:, i] = predicted_actions.argmax(dim=1).float() # simplified mapping
                
                # Update actor network using the policy gradient
                actor_loss = -agent.critic(state_b, predicted_joint_actions).mean()
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()
                
                # --- Soft Updates ---
                # Soft update target actor and target critic networks
                agent.soft_update(agent.target_actor, agent.actor)
                agent.soft_update(agent.target_critic, agent.critic)
                
                # Update exploration rate
                agent.update_epsilon()

        print(f"Episode {episode + 1}/{num_episodes} completed.")