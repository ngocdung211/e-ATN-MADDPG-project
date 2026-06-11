"""Tests for DITEN environment invariants from the paper MDP."""

import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment.diten_env import DITENEnv
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, Subtask, TaskDAG


def _build_task_dag() -> TaskDAG:
    """Build a small dependent task DAG."""
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(Subtask(1, cpu_cycles=1e6, data_size=1e5, result_size=1e4))
    task_dag.add_subtask(Subtask(2, cpu_cycles=2e6, data_size=2e5, result_size=2e4))
    task_dag.add_dependency(1, 2)
    return task_dag


def _build_env(server_location: np.ndarray) -> DITENEnv:
    """Build a one-device, one-server environment."""
    device = IndustrialDevice(
        device_id=1,
        location=np.array([0.0, 0.0]),
        compute_power=1e9,
        transmit_power=0.5,
        energy_coeff=1e-28,
        speed_mps=1.0,
    )
    server = EdgeServer(
        server_id=1,
        location=server_location,
        compute_power=2.4e9,
        transmit_power=1.2,
        energy_coeff=1e-27,
        coverage_radius=12.0,
    )
    network_env = NetworkEnvironment(bandwidth=10e6, noise_power_dbm=-43)
    return DITENEnv(
        [device],
        [server],
        network_env,
        slot_duration=1.0,
        subslot_count=10,
        time_slots=1,
        lambda1=1.0,
        lambda2=2.0,
        lambda3=3.0,
        lambda4=4.0,
        lambda5=5.0,
        p_out_value=-0.5,
        local_estimation_error=0.0,
        edge_estimation_error=0.0,
    )


def test_joint_state_matches_declared_dimension() -> None:
    """Eq. 23 state should match the declared state dimension."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = _build_task_dag()

    joint_state = env.start_time_slot({1: task_dag}, {1: [1, 2]})

    assert joint_state.shape == (1, env.get_state_dim())


def test_reward_matches_equation_24_terms() -> None:
    """Reward should follow Eq. 24 arithmetic exactly."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = _build_task_dag()

    reward = env._calculate_reward(
        t_im=0.1,
        e_im=0.2,
        t_accm=0.3,
        e_accm=0.4,
        p_out=-0.5,
        task_dag=task_dag,
    )

    expected = (
        1.0 * ((1.0 / 2) - 0.1)
        + 2.0 * (1.0 - 0.3)
        + 3.0 * ((1.0 / 2) - 0.2)
        + 4.0 * (1.0 - 0.4)
        + 5.0 * (-0.5)
    )
    assert reward == pytest.approx(expected)


def test_connection_window_violation_is_recorded() -> None:
    """Invalid edge offload should be visible in step metrics."""
    env = _build_env(server_location=np.array([100.0, 100.0]))
    task_dag = TaskDAG(task_id=1, t_max=1.0, e_max=1.0)
    task_dag.add_subtask(Subtask(1, cpu_cycles=1e6, data_size=1e5, result_size=1e4))
    env.reset({1: task_dag}, {1: [1]})

    env.step([1])

    assert env.last_step_metrics[0]["p_out"] == pytest.approx(-0.5)
    assert env.last_step_metrics[0]["fallback_local"] == pytest.approx(1.0)
    assert env.last_step_metrics[0]["requested_action"] == pytest.approx(1.0)
    assert env.last_step_metrics[0]["action"] == pytest.approx(0.0)
    assert env.last_step_metrics[0]["penalty_applied"] == pytest.approx(1.0)
    assert env.last_step_metrics[0]["penalty_time"] > 0.0
    assert env.last_step_metrics[0]["local_time"] > 0.0
    assert env.last_step_metrics[0]["server_time"] == pytest.approx(0.0)
    assert env.last_step_metrics[0]["attempted_server_time"] > 0.0
    assert env.last_step_metrics[0]["queue_or_wait_time"] >= 0.0
    assert env.last_step_metrics[0]["transfer_time"] >= 0.0


def test_invalid_priority_order_is_rejected() -> None:
    """A successor cannot be scheduled before its predecessor."""
    env = _build_env(server_location=np.array([0.0, 1.0]))
    task_dag = _build_task_dag()

    with pytest.raises(ValueError, match="predecessor"):
        env.start_time_slot({1: task_dag}, {1: [2, 1]})
