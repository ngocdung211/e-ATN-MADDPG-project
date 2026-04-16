"""Core system entities for the DITEN environment."""

from typing import Dict, List, Optional, Tuple

import numpy as np

class Subtask:
    """Represent a subtask v_{i,m}^t within a DAG."""

    def __init__(self, subtask_id: int, cpu_cycles: float, data_size: float, result_size: float):
        """Initialize a subtask.

        Args:
            subtask_id: Unique subtask identifier.
            cpu_cycles: CPU cycles required for processing.
            data_size: Input data size (bytes).
            result_size: Output data size (bytes).
        """
        self.id: int = subtask_id
        # Number of CPU cycles required to process the subtask (C_{i,m}^t)
        self.cpu_cycles: float = cpu_cycles
        # Data size of processing the subtask (D_{i,m}^t)
        self.data_size: float = data_size
        # Result data size of processing the subtask (R_{i,m}^t)
        self.result_size: float = result_size

class TaskDAG:
    """Represent an image recognition task as a DAG."""

    def __init__(self, task_id: int, t_max: float, e_max: float):
        """Initialize the task DAG container.

        Args:
            task_id: Unique task identifier.
            t_max: Maximum tolerable delay.
            e_max: Maximum tolerable energy consumption.
        """
        self.id: int = task_id
        # Maximum tolerable delay
        self.t_max: float = t_max
        # Maximum acceptable energy consumption
        self.e_max: float = e_max
        
        self.subtasks: Dict[int, Subtask] = {}
        # List of directed edges (predecessor_id, successor_id).
        self.edges: List[Tuple[int, int]] = []
        
    def add_subtask(self, subtask: Subtask) -> None:
        """Add a subtask to the DAG."""
        self.subtasks[subtask.id] = subtask
        
    def add_dependency(self, pred_id: int, succ_id: int) -> None:
        """Add a directed edge indicating succ_id depends on pred_id."""
        self.edges.append((pred_id, succ_id))

class IndustrialDevice:
    """Represent a mobile industrial device d_i."""

    def __init__(
        self,
        device_id: int,
        location: np.ndarray,
        compute_power: float,
        transmit_power: float,
        energy_coeff: float,
        speed_mps: float = 1.0,
        direction: Optional[np.ndarray] = None,
    ):
        """Initialize the device model.

        Args:
            device_id: Unique device identifier.
            location: Initial location (x, y).
            compute_power: Local computing power (Hz).
            transmit_power: Device transmit power (W).
            energy_coeff: Local computing energy coefficient.
            speed_mps: Movement speed (m/s).
            direction: Initial movement direction vector.
        """
        self.id: int = device_id
        # Location L_i^t = [x, y]^T
        self.location: np.ndarray = location
        # Actual local computing power f_i^{loc}
        self.compute_power: float = compute_power
        # Transmission power p_i
        self.transmit_power: float = transmit_power
        # Energy coefficient for local computing \tau_i^{loc}
        self.energy_coeff: float = energy_coeff
        # Mobility configuration
        self.speed_mps: float = speed_mps
        if direction is None:
            direction = np.array([1.0, 0.0], dtype=float)
        norm = np.linalg.norm(direction)
        self.direction: np.ndarray = (
            direction / norm if norm > 0 else np.array([1.0, 0.0], dtype=float)
        )
        
    def update_location(self, new_location: np.ndarray) -> None:
        """Update the device's location based on the mobility model."""
        self.location = new_location

    def set_direction(self, direction: np.ndarray) -> None:
        """Set and normalize the device movement direction."""
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.direction = direction / norm

    def project_location(self, duration_s: float) -> np.ndarray:
        """Project device location after moving with fixed speed and direction."""
        return self.location + self.direction * self.speed_mps * duration_s

class EdgeServer:
    """Represent an edge server s_j."""

    def __init__(
        self,
        server_id: int,
        location: np.ndarray,
        compute_power: float,
        transmit_power: float,
        energy_coeff: float,
        coverage_radius: float,
    ):
        """Initialize the edge server model.

        Args:
            server_id: Unique server identifier.
            location: Fixed location (x, y).
            compute_power: Computing power (Hz).
            transmit_power: Downlink transmit power (W).
            energy_coeff: Edge computing energy coefficient.
            coverage_radius: Coverage radius (meters).
        """
        self.id: int = server_id
        # Fixed location L_j^s = [x, y]^T
        self.location: np.ndarray = location
        # Actual computing power f_j^{edge}
        self.compute_power: float = compute_power
        # Downlink transmission power P_j
        self.transmit_power: float = transmit_power
        # Energy coefficient \tau_j^{edge}
        self.energy_coeff: float = energy_coeff
        # Signal coverage radius r
        self.coverage_radius: float = coverage_radius
