import numpy as np
from typing import List, Tuple, Dict
from environment.network_env import NetworkEnvironment
from environment.system_model import IndustrialDevice, EdgeServer, TaskDAG

class DITENEnv:
    def __init__(self, devices: List[IndustrialDevice], servers: List[EdgeServer], network_env: NetworkEnvironment):
        self.devices = devices
        self.servers = servers
        self.network_env = network_env
        self.num_agents = len(devices)
        
        # Trạng thái cộng dồn
        self.device_accumulated_delay = {}
        self.device_accumulated_energy = {}
        
        # Dữ liệu task cho từng episode
        self.task_dags = {}
        self.priorities = {}
        self.current_step = 0 # Step qua từng subtask (0 đến 4)
        
        # Hướng di chuyển ngẫu nhiên của từng thiết bị (2D vector)
        self.device_directions = {}
        
    def reset(self, task_dags: Dict[int, TaskDAG], priorities: Dict[int, List[int]]) -> np.ndarray:
        self.task_dags = task_dags
        self.priorities = priorities
        self.current_step = 0
        
        self.device_accumulated_delay = {d.id: 0.0 for d in self.devices}
        self.device_accumulated_energy = {d.id: 0.0 for d in self.devices}
        
        for d in self.devices:
            # Khởi tạo hướng di chuyển ngẫu nhiên
            dir_vec = np.random.uniform(-1, 1, 2)
            self.device_directions[d.id] = dir_vec / np.linalg.norm(dir_vec)
            
        return self._get_joint_state()

    def step(self, joint_actions: List[int]) -> Tuple[np.ndarray, List[float], bool, dict]:
        rewards = []
        
        for i, action in enumerate(joint_actions):
            device = self.devices[i]
            task_dag = self.task_dags[device.id]
            
            # Lấy ID của subtask hiện tại dựa vào thứ tự ưu tiên (GCN cung cấp)
            current_subtask_id = self.priorities[device.id][self.current_step]
            subtask = task_dag.subtasks[current_subtask_id]
            
            # Lấy thông số thực tế từ Dataset
            cpu_cycles = subtask.cpu_cycles
            
            t_im = 0.0
            e_im = 0.0
            p_out = 0.0 
            
            if action == 0:
                # Tính toán Local
                t_im, e_im = self.network_env.calculate_local_computation(
                    cpu_cycles, device.energy_coeff, device.compute_power, device.compute_power
                )
            else:
                # Tính toán Edge
                server = self.servers[action - 1] 
                t_im, e_im = self.network_env.calculate_edge_computation(
                    cpu_cycles, server.energy_coeff, server.compute_power, server.compute_power
                )
                
                # Kiểm tra ngắt kết nối: Nếu thiết bị di chuyển ra khỏi vùng phủ sóng (Bán kính 20m)
                distance = np.linalg.norm(device.location - server.location)
                if distance > server.coverage_radius:
                    p_out = -10.0 # Bị phạt nặng nếu mất kết nối
            
            # Cộng dồn Cost (Cumulative)
            t_accm = self.device_accumulated_delay[device.id]
            e_accm = self.device_accumulated_energy[device.id]
            
            # Cập nhật state nội bộ
            self.device_accumulated_delay[device.id] += t_im
            self.device_accumulated_energy[device.id] += e_im
            
            # Mô phỏng Mobility: Thiết bị di chuyển 1m/s trong khoảng thời gian t_im (delay)
            # Vị trí mới = Vị trí cũ + (Vectơ hướng * 1.0 m/s * thời gian tính toán)
            device.update_location(device.location + self.device_directions[device.id] * 1.0 * t_im)

            # Tính Reward thực tế bằng Eq 24.
            reward = self._calculate_reward(t_im, e_im, t_accm, e_accm, p_out, task_dag)
            rewards.append(reward)

        # Chuyển sang subtask tiếp theo
        self.current_step += 1
        
        # Nếu đã xử lý hết toàn bộ subtask (thường là 5 subtask cho Image Recognition)
        done = self.current_step >= len(self.task_dags[self.devices[0].id].subtasks)
        
        next_joint_state = self._get_joint_state()
        return next_joint_state, rewards, done, {}

    def _get_joint_state(self) -> np.ndarray:
        # Tạm giữ logic get state của bạn (nên mở rộng thêm đặc trưng subtask hiện tại sau)
        joint_state = []
        for device in self.devices:
            f_loc = device.compute_power
            w_d = self.device_accumulated_delay[device.id]
            edge_f = [s.compute_power for s in self.servers]
            edge_w = [0.0 for s in self.servers] # Tạm bỏ qua server queue để tránh rối
            
            state_vector = [f_loc, w_d] + edge_f + edge_w 
            joint_state.append(state_vector)
            
        return np.array(joint_state, dtype=np.float32)

    def _calculate_reward(self, t_im, e_im, t_accm, e_accm, p_out, task_dag) -> float:
        # Lấy giới hạn từ TaskDAG
        t_max = task_dag.t_max
        e_max = task_dag.e_max
        m_total = len(task_dag.subtasks)
        
        r_delay_sub = 1.0 * ((t_max / m_total) - t_im)
        r_delay_accm = 1.0 * (t_max - t_accm)
        r_energy_sub = 1.0 * ((e_max / m_total) - e_im)
        r_energy_accm = 1.0 * (e_max - e_accm)
        
        return r_delay_sub + r_delay_accm + r_energy_sub + r_energy_accm + p_out