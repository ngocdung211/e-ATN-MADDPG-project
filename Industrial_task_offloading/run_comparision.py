import numpy as np
import torch
import torch.nn.functional as F
import random
from baselines.maac import MAACAgent
from baselines.mappo import MAPPOAgent
# Environment & Models
from environment.network_env import NetworkEnvironment
from environment.system_model import IndustrialDevice, EdgeServer, TaskDAG, Subtask
from environment.diten_env import DITENEnv
from dataset.data_loader import KolektorSDDLoader
from models.replay_buffer import MultiAgentReplayBuffer
from models.gcn import TaskPriorityGCN
from models.maddpg import EpsilonATNMADDPGAgent
from ultils.graph_ultils import extract_gcn_inputs
from ultils.plotter import DITENPlotter

# TODO: Import your future baselines here once created
# from baselines.maac import MAACAgent
# from baselines.mappo import MAPPOAgent

def set_seed(seed=42):
    """Ensure reproducibility across different algorithm runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def generate_task_dags_for_episode(devices, data_loader):
    task_dags = {}
    for device in devices:
        task_params = data_loader.get_random_task_parameters()
        task_dag = TaskDAG(task_id=device.id, t_max=1.0, e_max=1.0) # e_max corrected to 1.0 J
        
        for i in range(1, 6):
            params = task_params[f"subtask_{i}"]
            task_dag.add_subtask(Subtask(i, params["cpu_cycles"], params["data_size"], params["result_size"]))
            
        task_dag.add_dependency(1, 2)
        task_dag.add_dependency(1, 3)
        task_dag.add_dependency(2, 4)
        task_dag.add_dependency(3, 4)
        task_dag.add_dependency(4, 5)
        
        task_dags[device.id] = task_dag
    return task_dags

def train_algorithm(algo_name, agent_config, devices, servers, network_env, data_loader, num_episodes=1000):
    """Trains a specific algorithm and returns histories for Reward, Delay, and Energy."""
    print(f"\n{'='*50}\nStarting Training for: {algo_name}\n{'='*50}")
    set_seed(42) # Reset seed for fair comparison
    
    env = DITENEnv(devices, servers, network_env)
    
    # State/Action dimensions
    STATE_DIM = 2 + len(servers) * 2
    ACTION_DIM = 1 + len(servers)
    JOINT_STATE_DIM = STATE_DIM * len(devices)
    JOINT_ACTION_DIM = len(devices)
    
    # Initialize Agents dynamically based on the config
    agents = []
    for _ in range(len(devices)):
        agent = agent_config["class"](
            state_dim=STATE_DIM, action_dim=ACTION_DIM, 
            joint_state_dim=JOINT_STATE_DIM, joint_action_dim=JOINT_ACTION_DIM,
            **agent_config.get("kwargs", {})
        )
        agents.append(agent)

    gcn_model = TaskPriorityGCN(num_features=3, hidden_dim=32)
    replay_buffer = MultiAgentReplayBuffer(capacity=100000)
    batch_size = 64
    gamma = 0.99
    
    # Track metrics
    history = {"reward": [], "delay": [], "energy": []}
    
    for episode in range(num_episodes):
        task_dags = generate_task_dags_for_episode(devices, data_loader)
        
        # 1. GCN Prioritization
        priorities = {}
        for device_id, task_dag in task_dags.items():
            X, A = extract_gcn_inputs(task_dag)
            with torch.no_grad():
                scores = gcn_model(X, A)
            sorted_indices = torch.argsort(scores.squeeze(), descending=True).tolist()
            priorities[device_id] = [idx + 1 for idx in sorted_indices]
            
        # 2. Environment Stepping
        current_joint_state = env.reset(task_dags, priorities) 
        done = False
        episode_reward = 0
        
        while not done:
            joint_actions = []
            for i, agent in enumerate(agents):
                agent_state = torch.FloatTensor(current_joint_state[i])
                action = agent.select_action(agent_state)
                joint_actions.append(action)
                
            next_joint_state, joint_rewards, done, _ = env.step(joint_actions)
            episode_reward += sum(joint_rewards) / len(agents)
            
            replay_buffer.push(current_joint_state, joint_actions, joint_rewards, next_joint_state)
            current_joint_state = next_joint_state
        
        # 3. Extract Delay and Energy from environment at the end of the episode
        avg_delay = np.mean(list(env.device_accumulated_delay.values()))
        avg_energy = np.mean(list(env.device_accumulated_energy.values()))
        
        history["reward"].append(episode_reward)
        history["delay"].append(avg_delay)
        history["energy"].append(avg_energy)
        
        # 4. Network Updates (Standard MADDPG logic)
        if len(replay_buffer) >= batch_size:
            state_b, action_b, reward_b, next_state_b = replay_buffer.sample(batch_size)
            
            for i, agent in enumerate(agents):
                # --> NEW: If it's MAAC or MAPPO, let it handle its own update
                if hasattr(agent, 'update_agent'):
                    agent.update_agent(state_b, action_b, reward_b, next_state_b, agent_index=i)
                    
                # --> EXISTING: Fallback to MADDPG logic
                elif hasattr(agent, 'target_actor'):
                    agent_rewards = reward_b[:, i].unsqueeze(1)
                    with torch.no_grad():
                        target_joint_actions = torch.stack([a.target_actor(next_state_b[:, j, :]) for j, a in enumerate(agents)], dim=1)
                        target_q = agent.target_critic(next_state_b, target_joint_actions)
                        y_i = agent_rewards + gamma * target_q
                        
                    current_q = agent.critic(state_b, action_b)
                    critic_loss = F.mse_loss(current_q, y_i)
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic_optimizer.step()
                    
                    predicted_actions = agent.actor(state_b[:, i, :])
                    predicted_joint_actions = action_b.clone()
                    predicted_joint_actions[:, i] = predicted_actions.argmax(dim=1).float()
                    
                    actor_loss = -agent.critic(state_b, predicted_joint_actions).mean()
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()
                    
                    agent.soft_update(agent.target_actor, agent.actor)
                    agent.soft_update(agent.target_critic, agent.critic)
                
                if hasattr(agent, 'update_epsilon'):
                    agent.update_epsilon()

        if (episode + 1) % 50 == 0:
            print(f"Ep {episode + 1}/{num_episodes} | Reward: {episode_reward:.3f} | Delay: {avg_delay:.3f}s | Energy: {avg_energy:.3f}J")
            
    return history

if __name__ == "__main__":
    # 1. Base Environment Setup
    BANDWIDTH, NOISE_POWER = 10e6, -43
    NUM_DEVICES, NUM_SERVERS = 10, 3
    
    network_env = NetworkEnvironment(bandwidth=BANDWIDTH, noise_power_dbm=NOISE_POWER)
    data_loader = KolektorSDDLoader(dataset_path="dataset/KolektorSDD/")
    
    # Fixed Edge Servers (Based on Paper Topology Fig 4)
    server_locations = [np.array([20.0, 30.0]), np.array([50.0, 45.0]), np.array([70.0, 20.0])]
    servers = [
        EdgeServer(j+1, loc, np.random.uniform(2.3, 2.5)*1e9, 1.2, 1e-27, coverage_radius=12.0)
        for j, loc in enumerate(server_locations)
    ]

    # Fixed starting locations for devices
    devices = [
        IndustrialDevice(i+1, np.array([np.random.uniform(0, 100), np.random.uniform(0, 100)]), 
                         np.random.uniform(0.8, 1.2)*1e9, 0.5, 1e-28)
        for i in range(NUM_DEVICES)
    ]

    # 2. Define Algorithms to Compare
    algorithms = {
        "MADDPG": {
            "class": EpsilonATNMADDPGAgent, 
            "kwargs": {"use_attention": False, "use_epsilon_greedy": False, "lr": 0.0001}
        },
        "GR-MADDPG": {
            "class": EpsilonATNMADDPGAgent, 
            "kwargs": {"use_attention": False, "use_epsilon_greedy": True, "lr": 0.0001}
        },
        "ATN-MADDPG": {
            "class": EpsilonATNMADDPGAgent, 
            "kwargs": {"use_attention": True, "use_epsilon_greedy": False, "lr": 0.0001}
        },
        "e-ATN-MADDPG": {
            "class": EpsilonATNMADDPGAgent, 
            "kwargs": {"use_attention": True, "use_epsilon_greedy": True, "lr": 0.0001}
        },
        # Uncomment when implemented:
        "MAAC": {"class": MAACAgent, "kwargs": {"lr": 0.0001}},

        "MAPPO": {"class": MAPPOAgent, "kwargs": {"lr": 0.0001}}
    }

    # 3. Run Comparisons
    results = {"reward": {}, "delay": {}, "energy": {}}
    EPISODES = 1000 # Use a smaller number (e.g., 100) for testing
    
    for algo_name, config in algorithms.items():
        history = train_algorithm(algo_name, config, devices, servers, network_env, data_loader, num_episodes=EPISODES)
        results["reward"][algo_name] = history["reward"]
        results["delay"][algo_name] = history["delay"]
        results["energy"][algo_name] = history["energy"]

    # 4. Plotting Results
    print("\nAll training complete! Generating comparison plots...")
    plotter = DITENPlotter()
    
    # Plot Reward (Like Fig. 8)
    plotter.plot_training_curve(
        data_dict=results["reward"], 
        title="Performance Comparison in Reward", 
        ylabel="Reward", 
        filename="comparison_reward.png"
    )
    
    # Plot Delay (Like Fig. 9)
    plotter.plot_training_curve(
        data_dict=results["delay"], 
        title="Performance Comparison in Task Processing Delay", 
        ylabel="Task Processing Delay (s)", 
        filename="comparison_delay.png"
    )
    
    # Plot Energy (Like Fig. 10)
    plotter.plot_training_curve(
        data_dict=results["energy"], 
        title="Performance Comparison in Energy Consumption", 
        ylabel="Energy Consumption (J)", 
        filename="comparison_energy.png"
    )
    
    print("Plots saved successfully! Check your directory.")