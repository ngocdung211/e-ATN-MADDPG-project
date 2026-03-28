import numpy as np
import torch
import torch.nn.functional as F
from ultils.graph_ultils import extract_gcn_inputs
from environment.system_model import TaskDAG, Subtask

# Sửa lại import theo đúng cấu trúc thư mục
from environment.network_env import NetworkEnvironment
from environment.system_model import IndustrialDevice, EdgeServer
from environment.diten_env import DITENEnv
from models.replay_buffer import MultiAgentReplayBuffer
from models.maddpg import EpsilonATNMADDPGAgent
from models.gcn import TaskPriorityGCN
from dataset.data_loader import KolektorSDDLoader
from ultils.plotter import DITENPlotter

def generate_task_dags_for_episode(devices, data_loader):
    """ Hàm phụ trợ: Sinh TaskDAG thực tế từ ảnh Dataset cho từng thiết bị """
    task_dags = {}
    for device in devices:
        task_params = data_loader.get_random_task_parameters()
        task_dag = TaskDAG(task_id=device.id, t_max=1.0, e_max=12.0)
        
        for i in range(1, 6):
            params = task_params[f"subtask_{i}"]
            task_dag.add_subtask(Subtask(i, params["cpu_cycles"], params["data_size"], params["result_size"]))
            
        # Thêm Edge phụ thuộc (DAG topology)
        task_dag.add_dependency(1, 2)
        task_dag.add_dependency(1, 3)
        task_dag.add_dependency(2, 4)
        task_dag.add_dependency(3, 4)
        task_dag.add_dependency(4, 5)
        
        task_dags[device.id] = task_dag
    return task_dags

def train_maddpg(agents, devices, env, replay_buffer, gcn_model, data_loader, num_episodes=1000, batch_size=64, gamma=0.99):
    num_agents = len(agents)
    rewards_history = [] 
    
    for episode in range(num_episodes):
        # 1. Sinh task cho episode này
        task_dags = generate_task_dags_for_episode(devices, data_loader)
        
        # 2. CHẠY GCN ĐỂ TÌM THỨ TỰ ƯU TIÊN (PRIORITIES)
        priorities = {}
        for device_id, task_dag in task_dags.items():
            X, A = extract_gcn_inputs(task_dag)
            with torch.no_grad():
                scores = gcn_model(X, A)
            
            # Sắp xếp index subtask (từ 1 đến 5) dựa trên điểm score giảm dần
            sorted_indices = torch.argsort(scores.squeeze(), descending=True).tolist()
            priorities[device_id] = [idx + 1 for idx in sorted_indices]
            
        # 3. Reset Env với Task và Priorities vừa sinh
        current_joint_state = env.reset(task_dags, priorities) 
        done = False
        episode_reward = 0
        
        # Vòng lặp này giờ sẽ lặp đúng M lần (M = số lượng subtask = 5)
        while not done:
            joint_actions = []
            for i, agent in enumerate(agents):
                agent_state = torch.FloatTensor(current_joint_state[i])
                action = agent.select_action(agent_state)
                joint_actions.append(action)
                
            next_joint_state, joint_rewards, done, _ = env.step(joint_actions)
            episode_reward += sum(joint_rewards) / num_agents
            
            replay_buffer.push(current_joint_state, joint_actions, joint_rewards, next_joint_state)
            current_joint_state = next_joint_state
        
        # --- Network Updates (Giữ nguyên logic cũ của bạn ở đây) ---
        if len(replay_buffer) >= batch_size:
            for i, agent in enumerate(agents):
                state_b, action_b, reward_b, next_state_b = replay_buffer.sample(batch_size)
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
                agent.update_epsilon()

        rewards_history.append(episode_reward)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - Avg Reward: {episode_reward:.4f}")
            
    return rewards_history
# ==========================================
# ENTRY POINT: Khởi tạo và chạy mô phỏng
# ==========================================
if __name__ == "__main__":
    print("Initializing Simulation Environment...")
    
    # 1. Cài đặt thông số mạng (Dựa theo Table II của bài báo)
    BANDWIDTH = 10e6        # 10 MHz
    NOISE_POWER = -43       # -43 dBm
    NUM_DEVICES = 10        # 10 thiết bị
    NUM_SERVERS = 3         # 3 Edge Servers
    TIME_SLOTS = 50         # Số time slot trong mỗi episode
    
    network_env = NetworkEnvironment(bandwidth=BANDWIDTH, noise_power_dbm=NOISE_POWER)
    
    # 2. Khởi tạo Edge Servers
    servers = []
    for j in range(NUM_SERVERS):
        f_edge = np.random.uniform(2.3, 2.5) * 1e9  # [2.3, 2.5] GHz
        server = EdgeServer(
            server_id=j+1,
            location=np.array([np.random.uniform(20, 80), np.random.uniform(20, 80)]),
            compute_power=f_edge,
            transmit_power=1.2,                     # 1.2 W
            energy_coeff=1e-27,                     # \tau_j^{edge} = 10^-27
            coverage_radius=20.0
        )
        servers.append(server)

    # 3. Khởi tạo Industrial Devices
    devices = []
    for i in range(NUM_DEVICES):
        f_loc = np.random.uniform(0.8, 1.2) * 1e9   # [0.8, 1.2] GHz
        device = IndustrialDevice(
            device_id=i+1,
            location=np.array([np.random.uniform(0, 100), np.random.uniform(0, 100)]),
            compute_power=f_loc,
            transmit_power=0.5,                     # 0.5 W
            energy_coeff=1e-28                      # \tau_i^{loc} = 10^-28
        )
        devices.append(device)
        
    # 4. Khởi tạo DITEN Environment
    env = DITENEnv(devices, servers, network_env, num_time_slots=TIME_SLOTS)
    
    # 5. Khởi tạo DRL Agents (\epsilon-ATN-MADDPG)
    # Xác định kích thước State và Action
    # State của mỗi agent: f_loc (1) + w_d (1) + edge_f (3) + edge_w (3) = 8 features
    STATE_DIM = 2 + NUM_SERVERS * 2
    # Action: 0 (Local) hoặc 1, 2, 3 (Các Server) -> 4 options
    ACTION_DIM = 1 + NUM_SERVERS
    
    JOINT_STATE_DIM = STATE_DIM * NUM_DEVICES
    JOINT_ACTION_DIM = NUM_DEVICES
    
    agents = []
    for _ in range(NUM_DEVICES):
        agent = EpsilonATNMADDPGAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            joint_state_dim=JOINT_STATE_DIM,
            joint_action_dim=JOINT_ACTION_DIM,
            lr=0.0001  # Learning rate tối ưu theo bài báo
        )
        agents.append(agent)

    # Khởi tạo GCN và Replay Buffer
    gcn_model = TaskPriorityGCN(num_features=3, hidden_dim=32)
    replay_buffer = MultiAgentReplayBuffer(capacity=100000)
    
    # Khởi tạo Data Loader
    data_loader = KolektorSDDLoader(dataset_path="dataset/kolektor_data/")
    
    # 6. Bắt đầu Huấn Luyện
    print("\nStarting Training (e-ATN-MADDPG)...")
    EPISODES = 1000 # Theo bảng II
    
    e_atn_rewards = train_maddpg(
        agents=agents, 
        env=env, 
        replay_buffer=replay_buffer, 
        gcn_model=gcn_model,
        num_episodes=EPISODES,
        batch_size=64
    )
    
    # 7. Vẽ biểu đồ kết quả sau khi train xong
    print("\nTraining complete! Generating plot...")
    plotter = DITENPlotter()
    
    # Tạo dictionary chứa dữ liệu để vẽ (Giả lập các đường baseline cho biểu đồ)
    data_to_plot = {
        "e-ATN-MADDPG": e_atn_rewards
    }
    
    plotter.plot_training_curve(
        data_dict=data_to_plot,
        title="Training Reward of e-ATN-MADDPG",
        ylabel="Average Reward",
        filename="training_reward_curve.png"
    )
    
    print("All processes completed successfully!")