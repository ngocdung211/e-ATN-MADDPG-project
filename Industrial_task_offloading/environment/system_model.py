import numpy as np
from typing import List, Tuple, Dict

class Subtask:
    """
    Represents a subtask v_{i,m}^t within a DAG.
    """
    def __init__(self, subtask_id: int, cpu_cycles: float, data_size: float, result_size: float):
        self.id = subtask_id
        # Number of CPU cycles required to process the subtask (C_{i,m}^t)
        self.cpu_cycles = cpu_cycles 
        # Data size of processing the subtask (D_{i,m}^t)
        self.data_size = data_size
        # Result data size of processing the subtask (R_{i,m}^t)
        self.result_size = result_size 

class TaskDAG:
    """
    Represents the image recognition task q_i^t modeled as a Directed Acyclic Graph (DAG).
    """
    def __init__(self, task_id: int, t_max: float, e_max: float):
        self.id = task_id
        # Maximum tolerable delay
        self.t_max = t_max
        # Maximum acceptable energy consumption
        self.e_max = e_max
        
        self.subtasks: Dict[int, Subtask] = {}
        self.edges: List[Tuple[int, int]] = [] # List of directed edges (predecessor_id, successor_id)
        
    def add_subtask(self, subtask: Subtask):
        self.subtasks[subtask.id] = subtask
        
    def add_dependency(self, pred_id: int, succ_id: int):
        """Adds a directed edge indicating succ_id depends on pred_id."""
        self.edges.append((pred_id, succ_id))

class IndustrialDevice:
    """
    Represents a mobile industrial device d_i.
    """
    def __init__(self, device_id: int, location: np.ndarray, compute_power: float, 
                 transmit_power: float, energy_coeff: float):
        self.id = device_id
        # Location L_i^t = [x, y]^T
        self.location = location
        # Actual local computing power f_i^{loc}
        self.compute_power = compute_power
        # Transmission power p_i
        self.transmit_power = transmit_power
        # Energy coefficient for local computing \tau_i^{loc}
        self.energy_coeff = energy_coeff
        
    def update_location(self, new_location: np.ndarray):
        """Updates the device's location based on the mobility model."""
        self.location = new_location

class EdgeServer:
    """
    Represents an edge server s_j.
    """
    def __init__(self, server_id: int, location: np.ndarray, compute_power: float, 
                 transmit_power: float, energy_coeff: float, coverage_radius: float):
        self.id = server_id
        # Fixed location L_j^s = [x, y]^T
        self.location = location
        # Actual computing power f_j^{edge}
        self.compute_power = compute_power
        # Downlink transmission power P_j
        self.transmit_power = transmit_power
        # Energy coefficient \tau_j^{edge}
        self.energy_coeff = energy_coeff
        # Signal coverage radius r
        self.coverage_radius = coverage_radius