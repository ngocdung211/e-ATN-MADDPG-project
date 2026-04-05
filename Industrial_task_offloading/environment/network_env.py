import math
import numpy as np

class NetworkEnvironment:
    """
    Handles the communication and computation mathematical models.
    """
    # Công thức Python (10 ** (noise_power_dbm / 10)) / 1000: Đây chính là công thức toán học chuẩn bị ép kiểu từ dBm sang W
    def __init__(self, bandwidth: float, noise_power_dbm: float):
        self.B = bandwidth  # Subchannel bandwidth (e.g., 10 MHz)
        # Convert background noise from dBm to linear scale (Watts)
        self.N0 = (10 ** (noise_power_dbm / 10)) / 1000 
        
    def calculate_channel_gain(self, loc1: np.ndarray, loc2: np.ndarray) -> float:
        """Calculates simple path loss/channel gain between two locations."""
        distance = np.linalg.norm(loc1 - loc2)
        # Using a standard simplified path loss model; adjust exponent as needed
        return distance ** (-2) if distance > 0 else 1.0

    # --- COMMUNICATION MODEL ---
    
    def get_uplink_rate(self, p_i: float, g_ij: float) -> float:
        """Calculates uplink data transmission rate."""
        sinr = (p_i * g_ij) / self.N0
        return self.B * math.log2(1 + sinr)
        
    def get_downlink_rate(self, p_j: float, g_ij: float) -> float:
        """Calculates downlink data transmission rate."""
        sinr = (p_j * g_ij) / self.N0
        return self.B * math.log2(1 + sinr)

    # --- COMPUTATION MODEL ---

    def calculate_local_computation(self, cpu_cycles: float, tau_i: float, 
                                    f_est: float, f_actual: float) -> tuple:
        """
        Calculates delay and energy for Local Computing.
        Returns: (actual_delay, energy_consumption)
        """
        # Deviation of computing power
        f_deviation = f_actual - f_est
        
        # Estimated and actual execution delays
        t_est_loc = cpu_cycles / f_est
        delta_t_loc = cpu_cycles * f_deviation / (f_est * (f_est - f_deviation))
        t_actual_loc = t_est_loc + delta_t_loc
        
        # Energy consumption of local computation
        e_local_comp = tau_i * cpu_cycles * (f_deviation ** 2)
        
        return t_actual_loc, e_local_comp

    def calculate_edge_computation(self, cpu_cycles: float, tau_j: float, 
                                   f_est: float, f_actual: float) -> tuple:
        """
        Calculates delay and energy for Edge Computing.
        Returns: (actual_delay, energy_consumption)
        """
        # Deviation of computing power
        f_deviation = f_actual - f_est
        
        # Estimated and actual execution delays
        t_est_edge = cpu_cycles / f_est
        delta_t_edge = cpu_cycles * f_deviation / (f_est * (f_est - f_deviation))
        t_actual_edge = t_est_edge + delta_t_edge
        
        # Energy consumption of edge computation
        e_edge_comp = tau_j * cpu_cycles * (f_deviation ** 2)
        
        return t_actual_edge, e_edge_comp