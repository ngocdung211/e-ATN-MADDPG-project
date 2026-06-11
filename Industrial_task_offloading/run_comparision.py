"""Run training comparisons across e-ATN-MADDPG and baselines.

This script trains multiple algorithms on the DITEN environment and
generates plots and JSON summaries for reward, delay, and energy.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import random
import json
import os
from tqdm import trange
from baselines.maac import MAACAgent
from baselines.mappo import MAPPOAgent
from baselines.offloading_baselines import (
    EdgeOnlyAgent,
    FeatureExtractionEdgeAgent,
    LocalOnlyAgent,
    RandomOffloadingAgent,
)
from baselines.scheduling_baselines import BaselineSchedulers
# Environment & Models
from environment.network_env import NetworkEnvironment
from environment.system_model import EdgeServer, IndustrialDevice, TaskDAG
from environment.diten_env import DITENEnv
from dataset.data_loader import KolektorSDDLoader
from models.replay_buffer import MultiAgentReplayBuffer
from models.gcn import TaskPriorityGCN
from models.maddpg import EpsilonATNMADDPGAgent
from ultils.plotter2 import DITENPlotter2
from ultils.gcn_training import load_or_train_gcn
from ultils.experiment_setup import build_priorities, generate_task_dags_for_episode, make_gcn_dag_sampler
from ultils.paper_config import PAPER_PARAMS
import time


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def build_algorithm_configs() -> Dict[str, Dict[str, object]]:
    """Build algorithm configurations for DRL and simple baselines.

    Returns:
        Mapping from display name to agent class and constructor kwargs.
    """
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    return {
        "Local Only": {"class": LocalOnlyAgent, "kwargs": {}},
        "Edge Only": {"class": EdgeOnlyAgent, "kwargs": {}},
        "Feature Extraction Edge": {"class": FeatureExtractionEdgeAgent, "kwargs": {}},
        # "Random Offloading": {"class": RandomOffloadingAgent, "kwargs": {}},
        "e-ATN-MADDPG": {
            "class": EpsilonATNMADDPGAgent,
            "kwargs": {
                "use_attention": True,
                "use_epsilon_greedy": True,
                "lr": confirmed["rl_lr"],
                "epsilon_init": provisional["epsilon_init"],
                "epsilon_min": provisional["epsilon_min"],
                "decay": provisional["epsilon_decay"],
            },
        },
        # "MAAC": {"class": MAACAgent, "kwargs": {"lr": confirmed["rl_lr"]}},
        "MAPPO": {"class": MAPPOAgent, "kwargs": {"lr": confirmed["rl_lr"]}},

    }


def build_priorities_by_mode(
    task_dags: Dict[int, TaskDAG],
    gcn_model: TaskPriorityGCN,
    mode: str = "gcn",
) -> Dict[int, List[int]]:
    """Build per-device subtask priorities based on the selected mode.

    Args:
        task_dags: Mapping of device IDs to TaskDAGs.
        gcn_model: TaskPriorityGCN used to infer priorities.
        mode: Scheduling mode ("gcn", "random", "greedy").

    Returns:
        Mapping of device IDs to ordered subtask IDs.
    """
    priorities = {}
    for device_id, task_dag in task_dags.items():
        if mode == "random":
            priorities[device_id] = BaselineSchedulers.random_scheduling(task_dag)
        elif mode == "greedy":
            priorities[device_id] = BaselineSchedulers.greedy_scheduling(task_dag)
        else:
            priorities[device_id] = build_priorities({device_id: task_dag}, gcn_model)[device_id]
    return priorities


def _collect_joint_actions(
    agents: Sequence[object], joint_state: np.ndarray, env: DITENEnv | None = None
) -> Tuple[List[int], int, int]:
    """Select joint actions and count local/edge choices.

    Args:
        agents: Agents selecting actions.
        joint_state: Joint state array shaped (num_agents, state_dim).
        env: Optional environment for rule-based baselines that need subtask id.

    Returns:
        Tuple of (joint_actions, local_count, edge_count).
    """
    joint_actions: List[int] = []
    local_count = 0
    edge_count = 0
    for agent_index, agent in enumerate(agents):
        agent_state = torch.FloatTensor(joint_state[agent_index])
        if env is not None and hasattr(agent, "select_action_for_subtask"):
            device = env.devices[agent_index]
            step_index = env.current_step[device.id]
            priority_order = env.priorities.get(device.id, [])
            subtask_id = priority_order[step_index] if step_index < len(priority_order) else -1
            action = agent.select_action_for_subtask(agent_state, subtask_id)
        else:
            action = agent.select_action(agent_state)
        joint_actions.append(action)
        if action == 0:
            local_count += 1
        else:
            edge_count += 1
    return joint_actions, local_count, edge_count


def _summarize_step_metrics(step_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Summarize timing diagnostics from environment step metrics.

    Args:
        step_metrics: Per-device metrics from `DITENEnv.last_step_metrics`.

    Returns:
        Aggregated timing diagnostics for the step.
    """
    summary = {
        "local_time": 0.0,
        "server_time": 0.0,
        "attempted_server_time": 0.0,
        "transfer_time": 0.0,
        "queue_or_wait_time": 0.0,
        "penalty_time": 0.0,
        "penalty_count": 0.0,
        "requested_local_count": 0.0,
        "requested_edge_count": 0.0,
        "resolved_local_count": 0.0,
        "resolved_edge_count": 0.0,
    }
    for metric in step_metrics:
        requested_action = int(metric.get("requested_action", -1.0))
        resolved_action = int(metric.get("action", -1.0))
        summary["local_time"] += float(metric.get("local_time", 0.0))
        summary["server_time"] += float(metric.get("server_time", 0.0))
        summary["attempted_server_time"] += float(metric.get("attempted_server_time", 0.0))
        summary["transfer_time"] += float(metric.get("transfer_time", 0.0))
        summary["queue_or_wait_time"] += float(metric.get("queue_or_wait_time", 0.0))
        summary["penalty_time"] += float(metric.get("penalty_time", 0.0))
        summary["penalty_count"] += float(metric.get("penalty_applied", 0.0))
        if requested_action == 0:
            summary["requested_local_count"] += 1.0
        elif requested_action > 0:
            summary["requested_edge_count"] += 1.0
        if resolved_action == 0:
            summary["resolved_local_count"] += 1.0
        elif resolved_action > 0:
            summary["resolved_edge_count"] += 1.0
    return summary


def _format_diagnostic_summary(algo_name: str, episode_number: int, history: Dict[str, List[float]]) -> str:
    """Format timing diagnostics as a readable multi-line block.

    Args:
        algo_name: Algorithm display name.
        episode_number: One-based episode number.
        history: Metric history populated by `train_algorithm`.

    Returns:
        Human-readable diagnostic summary.
    """
    requested_local = history["requested_local_count"][-1]
    requested_edge = history["requested_edge_count"][-1]
    resolved_local = history["resolved_local_count"][-1]
    resolved_edge = history["resolved_edge_count"][-1]
    penalty_count = history["penalty_count"][-1]
    penalty_time = history["penalty_time"][-1]
    local_time = history["local_time"][-1]
    server_time = history["server_time"][-1]
    transfer_time = history["transfer_time"][-1]
    wait_time = history["queue_or_wait_time"][-1]

    return (
        f"[{algo_name}] Episode {episode_number} diagnostics\n"
        f"  Requested actions: local={requested_local:.0f} edge={requested_edge:.0f}\n"
        f"  Actual execution:  local={resolved_local:.0f} edge={resolved_edge:.0f}\n"
        f"  Penalties:         count={penalty_count:.0f} time={penalty_time:.3f}s\n"
        f"  Timing avg/step:   local={local_time:.3f}s server={server_time:.3f}s "
        f"transfer={transfer_time:.3f}s wait={wait_time:.3f}s"
    )


def _should_print_diagnostics(episode_number: int, num_episodes: int) -> bool:
    """Return whether to print readable diagnostics for this episode."""
    return episode_number == num_episodes or episode_number % 50 == 0


def _update_agents_from_buffer(
    agents: Sequence[object],
    replay_buffer: MultiAgentReplayBuffer,
    batch_size: int,
    gamma: float,
) -> None:
    """Update agents from replay buffer samples when available.

    Args:
        agents: Agents to update.
        replay_buffer: Shared replay buffer.
        batch_size: Batch size for sampling.
        gamma: Discount factor.
    """
    if len(replay_buffer) < batch_size:
        return

    state_b, action_b, reward_b, next_state_b = replay_buffer.sample(batch_size)
    for agent_index, agent in enumerate(agents):
        action_dim = agent.action_dim
        action_b_onehot = F.one_hot(action_b.long(), num_classes=action_dim).float()
        if hasattr(agent, "update_agent"):
            agent.update_agent(state_b, action_b, reward_b, next_state_b, agent_index=agent_index)
            if hasattr(agent, "update_epsilon"):
                agent.update_epsilon()
            continue

        if hasattr(agent, "target_actor"):
            agent_rewards = reward_b[:, agent_index].unsqueeze(1)
            with torch.no_grad():
                target_joint_action_idx = torch.stack(
                    [
                        torch.argmax(a.target_actor(next_state_b[:, j, :]), dim=1)
                        for j, a in enumerate(agents)
                    ],
                    dim=1,
                )
                target_joint_actions = F.one_hot(
                    target_joint_action_idx.long(), num_classes=action_dim
                ).float()
                target_q = agent.target_critic(next_state_b, target_joint_actions)
                y_i = agent_rewards + gamma * target_q

            current_q = agent.critic(state_b, action_b_onehot)
            critic_loss = F.mse_loss(current_q, y_i)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()

            predicted_actions = agent.actor(state_b[:, agent_index, :])
            predicted_joint_actions = action_b_onehot.clone()
            predicted_joint_actions[:, agent_index] = predicted_actions

            actor_loss = -agent.critic(state_b, predicted_joint_actions).mean()
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()

            agent.soft_update(agent.target_actor, agent.actor)
            agent.soft_update(agent.target_critic, agent.critic)

        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon()


def train_algorithm(
    algo_name: str,
    agent_config: Dict[str, object],
    devices: Sequence[IndustrialDevice],
    servers: Sequence[EdgeServer],
    network_env: NetworkEnvironment,
    data_loader: KolektorSDDLoader,
    gcn_model: TaskPriorityGCN,
    num_episodes: int,
    priority_mode: str = "gcn",
) -> Dict[str, List[float]]:
    """Train one algorithm configuration and return metric histories.

    Args:
        algo_name: Display name for logging.
        agent_config: Dict with "class" and optional "kwargs" for agent init.
        devices: List of IndustrialDevice instances.
        servers: List of EdgeServer instances.
        network_env: NetworkEnvironment instance.
        data_loader: Dataset loader for DAG generation.
        gcn_model: TaskPriorityGCN used for priority extraction.
        num_episodes: Number of episodes to train.
        priority_mode: Scheduling mode ("gcn", "random", "greedy").

    Returns:
        Dict with reward, delay, energy, local_ratio, and edge_ratio histories.
    """
    print(f"\n{'='*50}\nStarting Training for: {algo_name}\n{'='*50}")
    set_seed(42)  # Reset seed for fair comparison
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]

    time_slots = int(confirmed["time_slots"])
    env = DITENEnv(
        devices,
        servers,
        network_env,
        slot_duration=confirmed["slot_duration_s"],
        subslot_count=200,
        time_slots=time_slots,
        lambda1=1,
        lambda2=2,
        lambda3=1,
        lambda4=2,
        lambda5=1,
        p_out_value=-0.5,
        local_estimation_error=0.2,
        edge_estimation_error=0.1,
    )
    
    # State/Action dimensions
    STATE_DIM = env.get_state_dim()
    ACTION_DIM = 1 + len(servers)
    # JOINT_STATE_DIM = STATE_DIM * len(devices)
    # JOINT_ACTION_DIM = len(devices)
    
    # Initialize Agents dynamically based on the config
    agents: List[object] = []
    for _ in range(len(devices)):
        agent = agent_config["class"](
            state_dim=STATE_DIM, action_dim=ACTION_DIM, 
            num_agents= len(devices),
            **agent_config.get("kwargs", {})
        )
        agents.append(agent)

    replay_buffer = MultiAgentReplayBuffer(capacity=int(provisional["replay_buffer_capacity"]))
    batch_size = int(provisional["batch_size"])
    gamma = provisional["gamma"]
    
    # Track metrics
    history = {
        "reward": [],
        "delay": [],
        "energy": [],
        "local_ratio": [],
        "edge_ratio": [],
        "local_time": [],
        "server_time": [],
        "attempted_server_time": [],
        "transfer_time": [],
        "queue_or_wait_time": [],
        "penalty_time": [],
        "penalty_count": [],
        "requested_local_count": [],
        "requested_edge_count": [],
        "resolved_local_count": [],
        "resolved_edge_count": [],
    }

    episode_iterator = trange(num_episodes, desc=f"Training {algo_name}", leave=True)
    for episode in episode_iterator:
        env.reset_episode()
        count_local = 0
        count_edge = 0
        slot_rewards = []
        slot_delays = []
        slot_energies = []
        slot_local_times = []
        slot_server_times = []
        slot_attempted_server_times = []
        slot_transfer_times = []
        slot_queue_or_wait_times = []
        slot_penalty_times = []
        slot_penalty_counts = []
        slot_requested_local_counts = []
        slot_requested_edge_counts = []
        slot_resolved_local_counts = []
        slot_resolved_edge_counts = []

        episode_done = False
        for _ in range(time_slots):
            if episode_done:
                break
            slot_reward = 0.0
            slot_local = 0
            slot_edge = 0
            slot_steps = 0
            prev_delay_mean = np.mean(list(env.device_accumulated_delay.values()))
            prev_energy_mean = np.mean(list(env.device_accumulated_energy.values()))
            task_dags = generate_task_dags_for_episode(devices, data_loader)
            priorities = build_priorities_by_mode(task_dags, gcn_model, priority_mode)

            current_joint_state = env.start_time_slot(task_dags, priorities)
            slot_done = False
            while not slot_done and not episode_done:
                joint_actions, local_count, edge_count = _collect_joint_actions(
                    agents, current_joint_state, env
                )
                count_local += local_count
                count_edge += edge_count
                slot_local += local_count
                slot_edge += edge_count

                next_joint_state, joint_rewards, step_episode_done, info = env.step(joint_actions)
                metric_summary = _summarize_step_metrics(env.last_step_metrics)
                slot_done = info.get("slot_done", False)
                episode_done = step_episode_done
                # Aggregate team reward as average per-agent immediate reward.
                step_reward = sum(joint_rewards) / len(agents)
                slot_reward += step_reward
                slot_steps += 1
                slot_local_times.append(metric_summary["local_time"])
                slot_server_times.append(metric_summary["server_time"])
                slot_attempted_server_times.append(metric_summary["attempted_server_time"])
                slot_transfer_times.append(metric_summary["transfer_time"])
                slot_queue_or_wait_times.append(metric_summary["queue_or_wait_time"])
                slot_penalty_times.append(metric_summary["penalty_time"])
                slot_penalty_counts.append(metric_summary["penalty_count"])
                slot_requested_local_counts.append(metric_summary["requested_local_count"])
                slot_requested_edge_counts.append(metric_summary["requested_edge_count"])
                slot_resolved_local_counts.append(metric_summary["resolved_local_count"])
                slot_resolved_edge_counts.append(metric_summary["resolved_edge_count"])
                replay_buffer.push(current_joint_state, joint_actions, joint_rewards, next_joint_state)
                current_joint_state = next_joint_state
        
            total_slot_actions = slot_local + slot_edge
            if total_slot_actions == 0:
                history["local_ratio"].append(0.0)
                history["edge_ratio"].append(0.0)
            else:
                history["local_ratio"].append(slot_local / total_slot_actions * 100)
                history["edge_ratio"].append(slot_edge / total_slot_actions * 100)

            current_delay_mean = np.mean(list(env.device_accumulated_delay.values()))
            current_energy_mean = np.mean(list(env.device_accumulated_energy.values()))
            slot_rewards.append(slot_reward / max(slot_steps, 1))
            slot_delays.append(max(0.0, current_delay_mean - prev_delay_mean))
            slot_energies.append(max(0.0, current_energy_mean - prev_energy_mean))

        episode_avg_reward = float(np.mean(slot_rewards)) if slot_rewards else 0.0
        episode_avg_delay = float(np.mean(slot_delays)) if slot_delays else 0.0
        episode_avg_energy = float(np.mean(slot_energies)) if slot_energies else 0.0
        episode_local_time = float(np.mean(slot_local_times)) if slot_local_times else 0.0
        episode_server_time = float(np.mean(slot_server_times)) if slot_server_times else 0.0
        episode_attempted_server_time = (
            float(np.mean(slot_attempted_server_times)) if slot_attempted_server_times else 0.0
        )
        episode_transfer_time = float(np.mean(slot_transfer_times)) if slot_transfer_times else 0.0
        episode_queue_or_wait_time = (
            float(np.mean(slot_queue_or_wait_times)) if slot_queue_or_wait_times else 0.0
        )
        episode_penalty_time = float(np.mean(slot_penalty_times)) if slot_penalty_times else 0.0
        episode_penalty_count = float(np.sum(slot_penalty_counts)) if slot_penalty_counts else 0.0
        episode_requested_local_count = (
            float(np.sum(slot_requested_local_counts)) if slot_requested_local_counts else 0.0
        )
        episode_requested_edge_count = (
            float(np.sum(slot_requested_edge_counts)) if slot_requested_edge_counts else 0.0
        )
        episode_resolved_local_count = (
            float(np.sum(slot_resolved_local_counts)) if slot_resolved_local_counts else 0.0
        )
        episode_resolved_edge_count = (
            float(np.sum(slot_resolved_edge_counts)) if slot_resolved_edge_counts else 0.0
        )
        total_actions = count_local + count_edge
        local_ratio = (count_local / total_actions * 100.0) if total_actions > 0 else 0.0
        edge_ratio = (count_edge / total_actions * 100.0) if total_actions > 0 else 0.0

        history["reward"].append(episode_avg_reward)
        history["delay"].append(episode_avg_delay)
        history["energy"].append(episode_avg_energy)
        history["local_ratio"].append(local_ratio)
        history["edge_ratio"].append(edge_ratio)
        history["local_time"].append(episode_local_time)
        history["server_time"].append(episode_server_time)
        history["attempted_server_time"].append(episode_attempted_server_time)
        history["transfer_time"].append(episode_transfer_time)
        history["queue_or_wait_time"].append(episode_queue_or_wait_time)
        history["penalty_time"].append(episode_penalty_time)
        history["penalty_count"].append(episode_penalty_count)
        history["requested_local_count"].append(episode_requested_local_count)
        history["requested_edge_count"].append(episode_requested_edge_count)
        history["resolved_local_count"].append(episode_resolved_local_count)
        history["resolved_edge_count"].append(episode_resolved_edge_count)

        episode_iterator.set_postfix(
            reward=f"{episode_avg_reward:.3f}",
            delay=f"{episode_avg_delay:.3f}s",
            energy=f"{episode_avg_energy:.3f}J",
            actual=f"L{episode_resolved_local_count:.0f}/E{episode_resolved_edge_count:.0f}",
            penalty=f"{episode_penalty_count:.0f}",
        )

        episode_number = episode + 1
        if _should_print_diagnostics(episode_number, num_episodes):
            episode_iterator.write(_format_diagnostic_summary(algo_name, episode_number, history))
         
        # 4. Network Updates (Standard MADDPG logic)
        _update_agents_from_buffer(agents, replay_buffer, batch_size, gamma)
    
        if episode_number % 50 == 0:
            print(
                f"[{algo_name}] Ep {episode_number}/{num_episodes} ----|--- Avg R/Slot: {history['reward'][-1]:.3f} "
                f"---|--- D: {history['delay'][-1]:.3f}s ---|--- E: {history['energy'][-1]:.3f}J"
            )
            print(
                f"[{algo_name}] Local ({count_local} / {history['local_ratio'][-1]:.1f}%) - "
                f"Edge ({count_edge} / {history['edge_ratio'][-1]:.1f}%)"
            )
            print(
                f"[{algo_name}] Timing | Local: {history['local_time'][-1]:.3f}s "
                f"| Server: {history['server_time'][-1]:.3f}s "
                f"| Transfer: {history['transfer_time'][-1]:.3f}s "
                f"| Wait: {history['queue_or_wait_time'][-1]:.3f}s "
                f"| Requested L/E: {history['requested_local_count'][-1]:.0f}/"
                f"{history['requested_edge_count'][-1]:.0f} "
                f"| Resolved L/E: {history['resolved_local_count'][-1]:.0f}/"
                f"{history['resolved_edge_count'][-1]:.0f} "
                f"| Penalty: {history['penalty_count'][-1]:.0f} / "
                f"{history['penalty_time'][-1]:.3f}s"
            )
            
    return history

if __name__ == "__main__":
    # 1. Base Environment Setup
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    BANDWIDTH, NOISE_POWER = confirmed["bandwidth_hz"], confirmed["noise_power_dbm"]
    NUM_DEVICES, NUM_SERVERS = int(confirmed["num_devices"]), int(confirmed["num_servers"])
    
    network_env = NetworkEnvironment(bandwidth=BANDWIDTH, noise_power_dbm=NOISE_POWER)
    data_loader = KolektorSDDLoader(dataset_path="dataset/KolektorSDD/")
    dataset_stats = data_loader.get_dataset_statistics()
    print(
        f"Dataset images found: {dataset_stats['total_images']} "
        f"(paper: 399; aligned={dataset_stats['is_paper_count_aligned']})"
    )

    gcn_model = TaskPriorityGCN(num_features=3, hidden_dim=int(confirmed["gcn_hidden_dim"]))
    gcn_ckpt_path = "models/checkpoints/gcn_priority.pt"
    sample_training_dag = make_gcn_dag_sampler(data_loader)

    gcn_model = load_or_train_gcn(
        gcn_model=gcn_model,
        dag_sampler=sample_training_dag,
        checkpoint_path=gcn_ckpt_path,
        epochs=200,
        samples_per_epoch=32,
        lr=confirmed["gcn_lr"],
    )
    
    # Fixed Edge Servers (User-confirmed Fig.4 topology)
    server_locations = [np.array([20.0, 30.0]), np.array([45.0, 50.0]), np.array([70.0, 20.0])]
    servers = [
        EdgeServer(
            j + 1,
            loc,
            np.random.uniform(3.3, 3.5) * 1e9,
            confirmed["server_tx_power_w"],
            confirmed["server_energy_coeff"],
            coverage_radius=confirmed["coverage_radius_m"],
        )
        for j, loc in enumerate(server_locations)
    ]

    # 10 devices: each pair shares one fixed rectangular route (starting at first waypoint)
    robot_starts = [
        np.array([10.0, 10.0]), np.array([30.0, 30.0]),
        np.array([70.0, 10.0]), np.array([40.0, 20.0]),
        np.array([40.0, 20.0]), np.array([60.0, 50.0]),
        np.array([70.0, 10.0]), np.array([90.0, 40.0]),
        np.array([65.0, 45.0]), np.array([90.0, 55.0]),
    ]
    devices = [
        IndustrialDevice(
            i + 1,
            robot_starts[i],
            np.random.uniform(0.8, 1.2) * 1e9,
            confirmed["device_tx_power_w"],
            confirmed["device_energy_coeff"],
            speed_mps=confirmed["device_speed_mps"],
        )
        for i in range(NUM_DEVICES)
    ]

    # 2. Define Algorithms to Compare
    algorithms = build_algorithm_configs()

    # 3. Run Comparisons
    results = {"reward": {}, "delay": {}, "energy": {}}
    priority_results = {"reward": {}, "delay": {}, "energy": {}}
    EPISODES = 10 # Use a smaller number (e.g., 100) for testing

    # Fig.6-style priority extraction comparison under fixed e-ATN-MADDPG.
    priority_modes = {"Random Scheduling": "random", "Greedy Scheduling": "greedy", "GCN Scheduling": "gcn"}
    # priority_modes = {"GCN Scheduling": "gcn"}
    # e_atn_cfg = {
    #     "class": EpsilonATNMADDPGAgent,
    #     "kwargs": {
    #         "use_attention": True,
    #         "use_epsilon_greedy": True,
    #         "lr": confirmed["rl_lr"],
    #         "epsilon_init": provisional["epsilon_init"],
    #         "epsilon_min": provisional["epsilon_min"],
    #         "decay": provisional["epsilon_decay"],
    #     },
    # }
    # for label, mode in priority_modes.items():
    #     history = train_algorithm(
    #         f"e-ATN-MADDPG ({label})",
    #         e_atn_cfg,
    #         devices,
    #         servers,
    #         network_env,
    #         data_loader,
    #         gcn_model=gcn_model,
    #         num_episodes=EPISODES,
    #         priority_mode=mode,
    #     )
    #     priority_results["reward"][label] = history["reward"]
    #     priority_results["delay"][label] = history["delay"]
    #     priority_results["energy"][label] = history["energy"]

    for algo_name, config in algorithms.items():
        history = train_algorithm(
            algo_name,
            config,
            devices,
            servers,
            network_env,
            data_loader,
            num_episodes=EPISODES,
            gcn_model=gcn_model,
            priority_mode="gcn",
        )
        results["reward"][algo_name] = history["reward"]
        results["delay"][algo_name] = history["delay"]
        results["energy"][algo_name] = history["energy"]

    # 4. Plotting Results
    print("\nAll training complete! Generating comparison plots...")
    plotter = DITENPlotter2()
    
    # Plot Reward (Like Fig. 8)
    date_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    plotter.plot_training_curve(
        data_dict=results["reward"], 
        title="Performance Comparison in Reward", 
        ylabel="Reward", 
        filename=f"{date_string}_comparison_reward.png"
    )
    
    # Plot Delay (Like Fig. 9)
    plotter.plot_training_curve(
        data_dict=results["delay"], 
        title="Performance Comparison in Task Processing Delay", 
        ylabel="Task Processing Delay (s)", 
        filename=f"{date_string}_comparison_delay.png"
    )
    
    # Plot Energy (Like Fig. 10)
    plotter.plot_training_curve(
        data_dict=results["energy"], 
        title="Performance Comparison in Energy Consumption", 
        ylabel="Energy Consumption (J)", 
        filename=f"{date_string}_comparison_energy.png"
    )

    # plotter.plot_training_curve(
    #     data_dict=priority_results["reward"],
    #     title="Rewards of Different Execution Priority Extracting Algorithms",
    #     ylabel="Reward",
    #     filename=f"{date_string}_priority_comparison_reward.png",
    # )

    os.makedirs("plots", exist_ok=True)
    results_path = os.path.join("plots", f"{date_string}_comparison_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "episodes": EPISODES,
                    "dataset_total_images": dataset_stats["total_images"],
                    "dataset_paper_expected_images": 399,
                    "dataset_paper_count_aligned": dataset_stats["is_paper_count_aligned"],
                },
                "algorithm_results": results,
                "priority_results": priority_results,
            },
            f,
            indent=2,
        )
    
    print(f"Plots and results saved successfully! JSON: {results_path}")
