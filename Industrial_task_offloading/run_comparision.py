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
from baselines.graph_gat_mappo import GraphGATMAPPOAgent, GraphGATRolloutBuffer
from baselines.maac import MAACAgent
from baselines.mappo import MAPPOAgent, MultiAgentRolloutBuffer
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
from ultils.topology_graph_state import TopologyGraphState, build_topology_graph_state
import time


FIXED_BASELINE_ALGORITHMS = frozenset(
    {
        "Local Only",
        "Edge Only",
        "Feature Extraction Edge",
        "Random Offloading",
    }
)


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
        # "Feature Extraction Edge": {"class": FeatureExtractionEdgeAgent, "kwargs": {}},
        # "Random Offloading": {"class": RandomOffloadingAgent, "kwargs": {}},
        "MAPPO": {
            "class": MAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "clip_param": provisional["mappo_clip_param"],
                "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
                "use_action_mask": provisional["mappo_use_action_mask"],
            },
        },
        "Graph-GAT MAPPO": {
            "class": GraphGATMAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                "clip_param": provisional["graph_gat_clip_param"],
                "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                "use_action_mask": provisional["graph_gat_use_action_mask"],
            },
        },
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
        "MADDPG": {
            "class": EpsilonATNMADDPGAgent,
            "kwargs": {
                "use_attention": False,
                "use_epsilon_greedy": False,
                "lr": confirmed["rl_lr"],
                "epsilon_init": provisional["epsilon_init"],
                "epsilon_min": provisional["epsilon_min"],
                "decay": provisional["epsilon_decay"],
            },
        },
        "MAAC": {"class": MAACAgent, "kwargs": {"lr": confirmed["rl_lr"]}},

    }


def _episodes_for_algorithm(algo_name: str, full_episodes: int, baseline_episodes: int) -> int:
    """Return the episode budget for a learning algorithm or fixed baseline."""
    if algo_name in FIXED_BASELINE_ALGORITHMS:
        return min(full_episodes, baseline_episodes)
    return full_episodes


def _mean_flat_history(history: Sequence[float], target_episodes: int) -> List[float]:
    """Convert a short fixed-baseline history into a mean-flat plot line."""
    if not history or target_episodes <= 0:
        return []
    mean_value = float(np.mean(history))
    return [mean_value for _ in range(target_episodes)]


def _build_plot_results(
    raw_results: Dict[str, Dict[str, List[float]]], target_episodes: int
) -> Dict[str, Dict[str, List[float]]]:
    """Build plot histories, extending fixed baselines as mean-flat lines."""
    plot_results: Dict[str, Dict[str, List[float]]] = {}
    for metric_name, algorithm_histories in raw_results.items():
        plot_results[metric_name] = {}
        for algo_name, history in algorithm_histories.items():
            if algo_name in FIXED_BASELINE_ALGORITHMS:
                plot_results[metric_name][algo_name] = _mean_flat_history(history, target_episodes)
            else:
                plot_results[metric_name][algo_name] = list(history)
    return plot_results


def _last_training_state_line(
    algo_name: str, history: Dict[str, List[float]], episode_count: int
) -> Dict[str, object]:
    """Build one flat JSONL row for the final training state of one model."""
    def last_value(metric_name: str) -> float:
        values = history.get(metric_name, [])
        return float(values[-1]) if values else 0.0

    requested_local = int(round(last_value("requested_local_count")))
    requested_edge = int(round(last_value("requested_edge_count")))
    resolved_local = int(round(last_value("resolved_local_count")))
    resolved_edge = int(round(last_value("resolved_edge_count")))

    return {
        "model": algo_name,
        "episode": int(episode_count),
        "reward": last_value("reward"),
        "delay_s": last_value("delay"),
        "energy_j": last_value("energy"),
        "local_count": requested_local,
        "edge_count": requested_edge,
        "local_ratio_percent": last_value("local_ratio"),
        "edge_ratio_percent": last_value("edge_ratio"),
        "requested_local_count": requested_local,
        "requested_edge_count": requested_edge,
        "resolved_local_count": resolved_local,
        "resolved_edge_count": resolved_edge,
        "local_time_s": last_value("local_time"),
        "server_time_s": last_value("server_time"),
        "transfer_time_s": last_value("transfer_time"),
        "wait_time_s": last_value("queue_or_wait_time"),
        "penalty_count": int(round(last_value("penalty_count"))),
        "penalty_time_s": last_value("penalty_time"),
    }


def _write_last_training_state_jsonl(
    output_path: str, rows: Sequence[Dict[str, object]]
) -> None:
    """Write one final-state JSON object per model line."""
    with open(output_path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True) + "\n")


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
        elif hasattr(agent, "select_action_with_log_prob"):
            action, log_prob = agent.select_action_with_log_prob(agent_state)
            agent.last_action_log_prob = log_prob
        else:
            action = agent.select_action(agent_state)
        joint_actions.append(action)
        if action == 0:
            local_count += 1
        else:
            edge_count += 1
    return joint_actions, local_count, edge_count


def _collect_graph_gat_actions(
    agent: GraphGATMAPPOAgent,
    joint_state: np.ndarray,
    num_devices: int,
    num_servers: int,
) -> Tuple[List[int], int, int, List[float], TopologyGraphState, float, float]:
    """Build graph state and select Graph-GAT MAPPO joint actions.

    Args:
        agent: Shared Graph-GAT MAPPO controller.
        joint_state: Joint flat state array shaped `(num_devices, state_dim)`.
        num_devices: Number of device agents.
        num_servers: Number of edge servers.

    Returns:
        Joint actions, local count, edge count, old log-probs, graph state, graph
        build time, and action selection time.
    """
    graph_build_start = time.perf_counter()
    graph_state = build_topology_graph_state(
        torch.as_tensor(joint_state, dtype=torch.float32),
        num_devices=num_devices,
        num_servers=num_servers,
    )
    graph_build_time = time.perf_counter() - graph_build_start

    graph_action_start = time.perf_counter()
    joint_actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
    graph_action_time = time.perf_counter() - graph_action_start
    local_count = sum(1 for action in joint_actions if action == 0)
    edge_count = len(joint_actions) - local_count
    return (
        joint_actions,
        local_count,
        edge_count,
        old_log_probs,
        graph_state,
        graph_build_time,
        graph_action_time,
    )


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

    summary = (
        f"[{algo_name}] Episode {episode_number} diagnostics\n"
        f"  Requested actions: local={requested_local:.0f} edge={requested_edge:.0f}\n"
        f"  Actual execution:  local={resolved_local:.0f} edge={resolved_edge:.0f}\n"
        f"  Penalties:         count={penalty_count:.0f} time={penalty_time:.3f}s\n"
        f"  Timing avg/step:   local={local_time:.3f}s server={server_time:.3f}s "
        f"transfer={transfer_time:.3f}s wait={wait_time:.3f}s"
    )
    graph_transition_count = history.get("graph_transition_count", [0.0])[-1]
    if graph_transition_count > 0:
        graph_build_time = history["graph_build_time"][-1]
        graph_action_time = history["graph_action_time"][-1]
        graph_update_time = history["graph_update_time"][-1]
        summary += (
            f"\n  Graph-GAT cost:    build={graph_build_time:.3f}s "
            f"action={graph_action_time:.3f}s update={graph_update_time:.3f}s "
            f"transitions={graph_transition_count:.0f}"
        )
    return summary


def _should_print_diagnostics(episode_number: int, num_episodes: int) -> bool:
    """Return whether to print readable diagnostics for this episode."""
    return episode_number == num_episodes or episode_number % 500 == 0


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


def _update_agents_from_rollout(
    agents: Sequence[object],
    rollout_buffer: MultiAgentRolloutBuffer,
    gamma: float,
) -> None:
    """Update on-policy MAPPO agents from a collected rollout and clear it."""
    if len(rollout_buffer) == 0:
        return

    state_b, action_b, reward_b, next_state_b, old_log_prob_b, done_b = (
        rollout_buffer.as_tensors()
    )
    for agent_index, agent in enumerate(agents):
        agent.update_agent(
            state_b,
            action_b,
            reward_b,
            next_state_b,
            agent_index=agent_index,
            old_log_prob_b=old_log_prob_b,
            done_b=done_b,
        )
    rollout_buffer.clear()


def _update_graph_gat_mappo_from_rollout(
    agent: GraphGATMAPPOAgent,
    rollout_buffer: GraphGATRolloutBuffer,
    gamma: float,
) -> float:
    """Update Graph-GAT MAPPO from graph rollouts and return update time."""
    if len(rollout_buffer) == 0:
        return 0.0

    agent.gamma = gamma
    update_start = time.perf_counter()
    agent.update_from_rollout(rollout_buffer)
    return time.perf_counter() - update_start


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
        subslot_count=int(provisional["subslot_count"]),
        time_slots=time_slots,
        lambda1=provisional["lambda1"],
        lambda2=provisional["lambda2"],
        lambda3=provisional["lambda3"],
        lambda4=provisional["lambda4"],
        lambda5=provisional["lambda5"],
        p_out_value=provisional["p_out_value"],
        local_estimation_error=provisional["local_estimation_error"],
        edge_estimation_error=provisional["edge_estimation_error"],
    )
    # State/Action dimensions
    STATE_DIM = env.get_state_dim()
    ACTION_DIM = 1 + len(servers)
    graph_priority_width = STATE_DIM - (5 + 4 * len(servers))
    graph_node_feature_dim = 9 + graph_priority_width
    graph_edge_feature_dim = 7
    # JOINT_STATE_DIM = STATE_DIM * len(devices)
    # JOINT_ACTION_DIM = len(devices)
    
    # Initialize Agents dynamically based on the config
    agent_class = agent_config["class"]
    uses_graph_gat_mappo = agent_class is GraphGATMAPPOAgent
    agents: List[object] = []
    if uses_graph_gat_mappo:
        agents.append(
            agent_class(
                num_devices=len(devices),
                num_servers=len(servers),
                node_feature_dim=graph_node_feature_dim,
                edge_feature_dim=graph_edge_feature_dim,
                **agent_config.get("kwargs", {}),
            )
        )
    else:
        for _ in range(len(devices)):
            agent = agent_class(
                state_dim=STATE_DIM, action_dim=ACTION_DIM,
                num_agents=len(devices),
                **agent_config.get("kwargs", {})
            )
            agents.append(agent)

    replay_buffer = MultiAgentReplayBuffer(capacity=int(provisional["replay_buffer_capacity"]))
    rollout_buffer = MultiAgentRolloutBuffer()
    graph_rollout_buffer = GraphGATRolloutBuffer()
    uses_rollout_buffer = (
        not uses_graph_gat_mappo
        and all(hasattr(agent, "select_action_with_log_prob") for agent in agents)
    )
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
        "graph_build_time": [],
        "graph_action_time": [],
        "graph_update_time": [],
        "graph_transition_count": [],
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
        episode_graph_build_time = 0.0
        episode_graph_action_time = 0.0
        episode_graph_update_time = 0.0
        episode_graph_transition_count = 0.0

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
            task_dags = generate_task_dags_for_episode(
                devices,
                data_loader,
                t_max=provisional["t_max"],
                e_max=provisional["e_max"],
            )
            priorities = build_priorities_by_mode(task_dags, gcn_model, priority_mode)

            current_joint_state = env.start_time_slot(task_dags, priorities)
            slot_done = False
            while not slot_done and not episode_done:
                if uses_graph_gat_mappo:
                    (
                        joint_actions,
                        local_count,
                        edge_count,
                        old_log_probs,
                        graph_state,
                        graph_build_time,
                        graph_action_time,
                    ) = _collect_graph_gat_actions(
                        agent=agents[0],
                        joint_state=current_joint_state,
                        num_devices=len(devices),
                        num_servers=len(servers),
                    )
                    episode_graph_build_time += graph_build_time
                    episode_graph_action_time += graph_action_time
                else:
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
                step_reward = sum(joint_rewards) / len(devices)
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
                if uses_graph_gat_mappo:
                    next_graph_build_start = time.perf_counter()
                    next_graph_state = build_topology_graph_state(
                        torch.as_tensor(next_joint_state, dtype=torch.float32),
                        num_devices=len(devices),
                        num_servers=len(servers),
                    )
                    episode_graph_build_time += (
                        time.perf_counter() - next_graph_build_start
                    )
                    graph_rollout_buffer.push(
                        graph_state=graph_state,
                        actions=joint_actions,
                        rewards=joint_rewards,
                        next_graph_state=next_graph_state,
                        old_log_probs=old_log_probs,
                        done=step_episode_done,
                    )
                    episode_graph_transition_count += 1.0
                elif uses_rollout_buffer:
                    old_log_probs = [
                        float(getattr(agent, "last_action_log_prob", 0.0))
                        for agent in agents
                    ]
                    rollout_buffer.push(
                        current_joint_state,
                        joint_actions,
                        joint_rewards,
                        next_joint_state,
                        old_log_probs,
                        done=step_episode_done,
                    )
                else:
                    replay_buffer.push(
                        current_joint_state, joint_actions, joint_rewards, next_joint_state
                    )
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

        if uses_graph_gat_mappo:
            episode_graph_update_time = _update_graph_gat_mappo_from_rollout(
                agents[0], graph_rollout_buffer, gamma
            )
        history["graph_build_time"].append(episode_graph_build_time)
        history["graph_action_time"].append(episode_graph_action_time)
        history["graph_update_time"].append(episode_graph_update_time)
        history["graph_transition_count"].append(episode_graph_transition_count)

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
         
        # 4. Network Updates
        if uses_rollout_buffer:
            _update_agents_from_rollout(agents, rollout_buffer, gamma)
        elif not uses_graph_gat_mappo:
            _update_agents_from_buffer(agents, replay_buffer, batch_size, gamma)
    
        if episode_number % 500 == 0 or episode_number == 1:
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
            if uses_graph_gat_mappo:
                print(
                    f"[{algo_name}] Graph-GAT Cost | Build: "
                    f"{history['graph_build_time'][-1]:.3f}s "
                    f"| Action: {history['graph_action_time'][-1]:.3f}s "
                    f"| Update: {history['graph_update_time'][-1]:.3f}s "
                    f"| Transitions: {history['graph_transition_count'][-1]:.0f}"
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
    sample_training_dag = make_gcn_dag_sampler(
        data_loader,
        t_max=provisional["t_max"],
        e_max=provisional["e_max"],
    )

    gcn_model = load_or_train_gcn(
        gcn_model=gcn_model,
        dag_sampler=sample_training_dag,
        checkpoint_path=gcn_ckpt_path,
        epochs=int(provisional["gcn_pretrain_epochs"]),
        samples_per_epoch=int(provisional["gcn_samples_per_epoch"]),
        lr=confirmed["gcn_lr"],
    )
    
    # Fixed Edge Servers (User-confirmed Fig.4 topology)
    server_locations = [np.array([20.0, 30.0]), np.array([45.0, 50.0]), np.array([70.0, 20.0])]
    servers = [
        EdgeServer(
            j + 1,
            loc,
            np.random.uniform(
                provisional["server_compute_power_min_ghz"],
                provisional["server_compute_power_max_ghz"],
            )
            * 1e9,
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
            np.random.uniform(
                provisional["device_compute_power_min_ghz"],
                provisional["device_compute_power_max_ghz"],
            )
            * 1e9,
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
    FULL_EPISODES = int(provisional["comparison_full_episodes"])
    BASELINE_EVALUATION_EPISODES = int(provisional["baseline_evaluation_episodes"])
    algorithm_episode_counts = {}
    last_training_state_rows = []

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
    #         num_episodes=FULL_EPISODES,
    #         priority_mode=mode,
    #     )
    #     priority_results["reward"][label] = history["reward"]
    #     priority_results["delay"][label] = history["delay"]
    #     priority_results["energy"][label] = history["energy"]

    for algo_name, config in algorithms.items():
        run_episodes = _episodes_for_algorithm(
            algo_name,
            full_episodes=FULL_EPISODES,
            baseline_episodes=BASELINE_EVALUATION_EPISODES,
        )
        algorithm_episode_counts[algo_name] = run_episodes
        history = train_algorithm(
            algo_name,
            config,
            devices,
            servers,
            network_env,
            data_loader,
            num_episodes=run_episodes,
            gcn_model=gcn_model,
            priority_mode="gcn",
        )
        results["reward"][algo_name] = history["reward"]
        results["delay"][algo_name] = history["delay"]
        results["energy"][algo_name] = history["energy"]
        last_training_state_rows.append(
            _last_training_state_line(algo_name, history, episode_count=run_episodes)
        )

    # 4. Plotting Results
    print("\nAll training complete! Generating comparison plots...")
    plotter = DITENPlotter2()
    plot_results = _build_plot_results(results, target_episodes=FULL_EPISODES)
    
    # Plot Reward (Like Fig. 8)
    date_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    plotter.plot_training_curve(
        data_dict=plot_results["reward"], 
        title="Performance Comparison in Reward", 
        ylabel="Reward", 
        filename=f"{date_string}_comparison_reward.png"
    )
    
    # Plot Delay (Like Fig. 9)
    plotter.plot_training_curve(
        data_dict=plot_results["delay"], 
        title="Performance Comparison in Task Processing Delay", 
        ylabel="Task Processing Delay (s)", 
        filename=f"{date_string}_comparison_delay.png"
    )
    
    # Plot Energy (Like Fig. 10)
    plotter.plot_training_curve(
        data_dict=plot_results["energy"], 
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

    last_state_path = os.path.join(
        plotter.save_dir, f"{date_string}_last_training_state.jsonl"
    )
    _write_last_training_state_jsonl(last_state_path, last_training_state_rows)
    
    print(f"Plots and last training states saved successfully! JSONL: {last_state_path}")
