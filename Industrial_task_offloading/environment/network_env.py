"""Communication and computation models for the industrial network."""

import math
from typing import Tuple

import numpy as np

class NetworkEnvironment:
    """Handle the communication and computation mathematical models."""

    def __init__(self, bandwidth: float, noise_power_dbm: float):
        """Initialize network parameters.

        Args:
            bandwidth: Subchannel bandwidth (Hz).
            noise_power_dbm: Background noise power in dBm.
        """
        # Convert background noise from dBm to linear scale (Watts).
        self.bandwidth_hz = bandwidth
        self.noise_power_w = (10 ** (noise_power_dbm / 10)) / 1000

    def calculate_channel_gain(self, loc1: np.ndarray, loc2: np.ndarray) -> float:
        """Calculate path-loss based channel gain between two locations.

        Args:
            loc1: First location (x, y).
            loc2: Second location (x, y).

        Returns:
            Channel gain value.
        """
        distance = np.linalg.norm(loc1 - loc2)
        # Using a standard simplified path loss model; adjust exponent as needed
        return distance ** (-2) if distance > 0 else 1.0

    # --- COMMUNICATION MODEL ---
    
    def get_uplink_rate(self, transmit_power: float, channel_gain: float) -> float:
        """Calculate uplink data transmission rate.

        Args:
            transmit_power: Device transmit power (W).
            channel_gain: Channel gain between device and server.

        Returns:
            Uplink data rate (bps).
        """
        sinr = (transmit_power * channel_gain) / self.noise_power_w
        return self.bandwidth_hz * math.log2(1 + sinr)
        
    def get_downlink_rate(self, transmit_power: float, channel_gain: float) -> float:
        """Calculate downlink data transmission rate.

        Args:
            transmit_power: Server transmit power (W).
            channel_gain: Channel gain between server and device.

        Returns:
            Downlink data rate (bps).
        """
        sinr = (transmit_power * channel_gain) / self.noise_power_w
        return self.bandwidth_hz * math.log2(1 + sinr)

    # --- COMPUTATION MODEL ---

    def calculate_local_computation(
        self, cpu_cycles: float, energy_coeff: float, f_est: float, f_actual: float
    ) -> Tuple[float, float]:
        """Calculate delay and energy for local computing.

        Args:
            cpu_cycles: CPU cycles required by the subtask.
            energy_coeff: Local computing energy coefficient.
            f_est: Estimated computing power.
            f_actual: Actual computing power.

        Returns:
            Tuple of (actual_delay, energy_consumption).
        """
        power_delta = f_actual - f_est
        
        # Estimated and actual execution delays
        # Guard against division by zero if deviation is extremely large (optional but recommended)
        # if f_est == f_deviation:
        #     delta_t_loc = 0
        # else:
        #     delta_t_loc = cpu_cycles * f_deviation / (f_est * (f_est - f_deviation))

        estimated_delay = cpu_cycles / f_est
        delta_delay = cpu_cycles * power_delta / (f_est * (f_est - power_delta))
        actual_delay = estimated_delay + delta_delay
        
        # Energy consumption of local computation
        energy_consumption = energy_coeff * cpu_cycles * (f_actual ** 2)
        
        return actual_delay, energy_consumption

    def calculate_edge_computation(
        self, cpu_cycles: float, energy_coeff: float, f_est: float, f_actual: float
    ) -> Tuple[float, float]:
        """Calculate delay and energy for edge computing.

        Args:
            cpu_cycles: CPU cycles required by the subtask.
            energy_coeff: Edge computing energy coefficient.
            f_est: Estimated computing power.
            f_actual: Actual computing power.

        Returns:
            Tuple of (actual_delay, energy_consumption).
        """
        power_delta = f_actual - f_est
        
        # Estimated and actual execution delays
        estimated_delay = cpu_cycles / f_est
        delta_delay = cpu_cycles * power_delta / (f_est * (f_est - power_delta))
        actual_delay = estimated_delay + delta_delay
        
        # Energy consumption of edge computation
        energy_consumption = energy_coeff * cpu_cycles * (f_actual ** 2)
        
        energy_consumption = 0 # --- IGNORE THIS LINE FOR NOW, SETTING EDGE COMPUTATION ENERGY TO ZERO FOR SIMPLICITY ---
        return actual_delay, energy_consumption
