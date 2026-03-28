import numpy as np
from typing import List, Tuple, Dict
from network_env import NetworkEnvironment
from system_model import IndustrialDevice, EdgeServer
# Assuming we import our previously defined classes:
# from environment.system_models import IndustrialDevice, EdgeServer, TaskDAG
# from environment.network_env import NetworkEnvironment
# from utils.metrics import compute_reward

class DITENEnv:
    """
    The Digital Twin Edge Network (DITEN) Reinforcement Learning Environment.
    Manages the MDP for the multi-agent task offloading problem.
    """
    def __init__(self, devices: List[IndustrialDevice], servers: List[EdgeServer], network_env: NetworkEnvironment, num_time_slots: int):
        self.devices = devices
        self.servers = servers
        self.network_env = network_env
        self.num_time_slots = num_time_slots
        
        self.num_agents = len(devices)
        self.num_servers = len(servers)
        
        # Current time slot t
        self.current_time_slot = 0
        
        # Tracking states
        self.device_waiting_delays = {d.id: 0.0 for d in devices}
        self.server_waiting_delays = {s.id: 0.0 for s in servers}
        
    def reset(self) -> np.ndarray:
        """
        Resets the environment for a new episode.
        Generates new tasks and returns the initial joint state.
        """
        self.current_time_slot = 0
        self.device_waiting_delays = {d.id: 0.0 for d in self.devices}
        self.server_waiting_delays = {s.id: 0.0 for s in self.servers}
        
        # Reset device locations to starting points
        # Generate new TaskDAGs for each device for t=0
        
        return self._get_joint_state()

    def step(self, joint_actions: List[int]) -> Tuple[np.ndarray, List[float], bool, dict]:
        """
        Executes the agents' offloading decisions, calculates costs, 
        advances the environment, and returns the next state and rewards.
        
        joint_actions: List of offloading decisions (0 for local, 1..S for servers)[cite: 349].
        """
        rewards = []
        
        # 1. Process Actions & Calculate Costs
        for i, action in enumerate(joint_actions):
            device = self.devices[i]
            # In a full implementation, you'd extract the specific subtask v_{i,m}^t here[cite: 342].
            # For simplicity, we assume we are evaluating a generic subtask.
            cpu_cycles = 1000 # Placeholder for C_{i,m}^t [cite: 159]
            
            t_im = 0.0
            e_im = 0.0
            p_out = 0.0 # Penalty for signal disconnection [cite: 366]
            
            if action == 0:
                # Local execution
                t_im, e_im = self.network_env.calculate_local_computation(
                    cpu_cycles, device.energy_coeff, device.compute_power, device.compute_power
                )
                self.device_waiting_delays[device.id] += t_im
            else:
                # Edge execution (action maps to server index)
                server = self.servers[action - 1] 
                
                # Check connection window constraints (C1 and C2) [cite: 272, 273, 287, 288]
                # If the device moves out of range before finishing, apply penalty.
                # p_out = -10.0 if disconnected else 0.0
                
                t_im, e_im = self.network_env.calculate_edge_computation(
                    cpu_cycles, server.energy_coeff, server.compute_power, server.compute_power
                )
                self.server_waiting_delays[server.id] += t_im
            
            # 2. Calculate Reward using Eq. 24 [cite: 352]
            # Note: You'll need to track t_accm and e_accm (cumulative costs) per task.
            reward = self._calculate_reward(t_im, e_im, t_accm=0.0, e_accm=0.0, p_out=p_out)
            rewards.append(reward)

        # 3. Advance Environment State
        # Update device mobility (locations) for the next time slot[cite: 167, 168].
        self.current_time_slot += 1
        done = self.current_time_slot >= self.num_time_slots
        
        # 4. Observe New State
        next_joint_state = self._get_joint_state()
        
        info = {} # Can be used for debugging or tracking metrics
        
        return next_joint_state, rewards, done, info

    def _get_joint_state(self) -> np.ndarray:
        """
        Constructs the joint state vector for all agents based on Eq. 23[cite: 342, 347].
        State includes:
        - Local computing power and waiting delay [cite: 342]
        - Subtask info and execution priorities [cite: 342]
        - Edge server computing power and waiting delays [cite: 342]
        - Link connection times (start/end) [cite: 342]
        """
        joint_state = []
        for device in self.devices:
            # Gather state features for device d_i
            f_loc = device.compute_power
            w_d = self.device_waiting_delays[device.id]
            
            # Gather edge server features
            edge_f = [s.compute_power for s in self.servers]
            edge_w = [self.server_waiting_delays[s.id] for s in self.servers]
            
            # Combine into a single flat vector for the neural network
            # (In reality, you'd also append subtask features and connection windows here) [cite: 342, 347]
            state_vector = [f_loc, w_d] + edge_f + edge_w 
            joint_state.append(state_vector)
            
        return np.array(joint_state, dtype=np.float32)

    def _calculate_reward(self, t_im, e_im, t_accm, e_accm, p_out) -> float:
        """
        Calls the external metric utility to compute Eq. 24[cite: 352].
        """
        # Placeholder constants based on paper settings
        t_max = 1.0 # 1s [cite: 491]
        e_max = 12.0 # 12 J [cite: 491]
        m_total = 5 # Assume 5 subtasks for example
        
        # l1=1.0, l2=1.0, l3=1.0, l4=1.0, l5=1.0 [cite: 366]
        r_delay_sub = 1.0 * ((t_max / m_total) - t_im) #[cite: 352]
        r_delay_accm = 1.0 * (t_max - t_accm) #[cite: 352]
        r_energy_sub = 1.0 * ((e_max / m_total) - e_im) #[cite: 364]
        r_energy_accm = 1.0 * (e_max - e_accm) #[cite: 364]
        
        return r_delay_sub + r_delay_accm + r_energy_sub + r_energy_accm + p_out #[cite: 352, 364]