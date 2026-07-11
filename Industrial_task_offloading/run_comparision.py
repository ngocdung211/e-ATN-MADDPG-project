"""Run training comparisons across e-ATN-MADDPG and baselines.

This script trains multiple algorithms on the DITEN environment and
generates plots and JSON summaries for reward, delay, and energy.
"""

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import random
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
from models.maddpg import EpsilonATNMADDPGAgent
from utils.comparison_outputs import (
    build_last_training_state_line,
    build_model_checkpoint,
    flatten_topology_metrics,
    save_comparison_outputs,
)
from utils.gcn_training import load_or_train_priority_model
from utils.experiment_setup import (
    build_priorities,
    build_task_priority_model,
    generate_task_dags_for_episode,
    get_priority_checkpoint_path,
    make_gcn_dag_sampler,
)
from utils.paper_config import PAPER_PARAMS
from utils.topology_graph_state import TopologyGraphState, build_topology_graph_state
from utils.topology_scenarios import (
    TopologyScenario,
    available_topology_scenario_names,
    compute_topology_metrics,
    device_start_points,
    get_topology_scenario,
)
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


def parse_args() -> argparse.Namespace:
    """Parse comparison-runner CLI arguments."""
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    parser = argparse.ArgumentParser(description="Run DITEN comparison experiments.")
    parser.add_argument(
        "--topology-scenario",
        default=str(provisional["topology_scenario"]),
        choices=available_topology_scenario_names(),
        help="Named topology scenario to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override comparison_full_episodes for smoke or short runs.",
    )
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=None,
        help="Override baseline_evaluation_episodes.",
    )
    return parser.parse_args()


def build_servers_for_scenario(
    scenario: TopologyScenario,
    confirmed: Dict[str, float],
    provisional: Dict[str, float],
) -> List[EdgeServer]:
    """Build edge servers from a named topology scenario."""
    servers = []
    for server_index, location in enumerate(scenario.server_locations, start=1):
        servers.append(
            EdgeServer(
                server_index,
                np.asarray(location, dtype=float),
                np.random.uniform(
                    provisional["server_compute_power_min_ghz"],
                    provisional["server_compute_power_max_ghz"],
                )
                * 1e9,
                confirmed["server_tx_power_w"],
                confirmed["server_energy_coeff"],
                coverage_radius=scenario.coverage_radius,
            )
        )
    return servers


def build_devices_for_scenario(
    scenario: TopologyScenario,
    confirmed: Dict[str, float],
    provisional: Dict[str, float],
) -> List[IndustrialDevice]:
    """Build mobile devices from a named topology scenario."""
    starts = device_start_points(scenario)
    devices = []
    for device_index, start_location in enumerate(starts, start=1):
        devices.append(
            IndustrialDevice(
                device_index,
                start_location.copy(),
                np.random.uniform(
                    provisional["device_compute_power_min_ghz"],
                    provisional["device_compute_power_max_ghz"],
                )
                * 1e9,
                confirmed["device_tx_power_w"],
                confirmed["device_energy_coeff"],
                speed_mps=confirmed["device_speed_mps"],
            )
        )
    return devices


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
        # "e-ATN-MADDPG": {
        #     "class": EpsilonATNMADDPGAgent,
        #     "kwargs": {
        #         "use_attention": True,
        #         "use_epsilon_greedy": True,
        #         "lr": confirmed["rl_lr"],
        #         "epsilon_init": provisional["epsilon_init"],
        #         "epsilon_min": provisional["epsilon_min"],
        #         "decay": provisional["epsilon_decay"],
        #     },
        # "MADDPG": {
        #     "class": EpsilonATNMADDPGAgent,
        #     "kwargs": {
        #         "use_attention": False,
        #         "use_epsilon_greedy": False,
        #         "lr": confirmed["rl_lr"],
        #         "epsilon_init": provisional["epsilon_init"],
        #         "epsilon_min": provisional["epsilon_min"],
        #         "decay": provisional["epsilon_decay"],
        #     },
        # },
        # "MAAC": {"class": MAACAgent, "kwargs": {"lr": confirmed["rl_lr"]}},
        "MAPPO": {
            "class": MAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "clip_param": provisional["mappo_clip_param"],
                "ppo_epochs": int(provisional["mappo_ppo_epochs"]),
                "use_action_mask": False,
            },
        },
        "Mask-MAPPO": {
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
                "use_action_mask": False,
            },
        },
        "Graph-GAT Warmup MAPPO": {
            "class": GraphGATMAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                "clip_param": provisional["graph_gat_clip_param"],
                "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                "use_action_mask": False,
                "topology_warmup_episodes": int(
                    provisional["graph_gat_topology_warmup_episodes"]
                ),
                "topology_warmup_updates_per_step": int(
                    provisional["graph_gat_topology_warmup_updates_per_step"]
                ),
                "topology_warmup_lr": provisional["graph_gat_topology_warmup_lr"],
            },
        },
        "Graph-GAT Mask MAPPO": {
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
        "Graph-GAT Warmup Mask MAPPO": {
            "class": GraphGATMAPPOAgent,
            "kwargs": {
                "lr": confirmed["rl_lr"],
                "gamma": provisional["gamma"],
                "hidden_dim": int(provisional["graph_gat_hidden_dim"]),
                "embedding_dim": int(provisional["graph_gat_embedding_dim"]),
                "clip_param": provisional["graph_gat_clip_param"],
                "ppo_epochs": int(provisional["graph_gat_ppo_epochs"]),
                "use_action_mask": provisional["graph_gat_use_action_mask"],
                "topology_warmup_episodes": int(
                    provisional["graph_gat_topology_warmup_episodes"]
                ),
                "topology_warmup_updates_per_step": int(
                    provisional["graph_gat_topology_warmup_updates_per_step"]
                ),
                "topology_warmup_lr": provisional["graph_gat_topology_warmup_lr"],
            },
        },

    }


def _episodes_for_algorithm(algo_name: str, full_episodes: int, baseline_episodes: int) -> int:
    """Return the episode budget for a learning algorithm or fixed baseline."""
    if algo_name in FIXED_BASELINE_ALGORITHMS:
        return min(full_episodes, baseline_episodes)
    return full_episodes


def build_priorities_by_mode(
    task_dags: Dict[int, TaskDAG],
    priority_model: torch.nn.Module,
    mode: str = "gcn",
) -> Dict[int, List[int]]:
    """Build per-device subtask priorities based on the selected mode.

    Args:
        task_dags: Mapping of device IDs to TaskDAGs.
        priority_model: GCN/GAT priority model used to infer priorities.
        mode: Scheduling mode ("gcn", "gat", "random", "greedy").

    Returns:
        Mapping of device IDs to ordered subtask IDs.
    """
    priorities = {}
    for device_id, task_dag in task_dags.items():
        if mode == "random":
            priorities[device_id] = BaselineSchedulers.random_scheduling(task_dag)
        elif mode == "greedy":
            priorities[device_id] = BaselineSchedulers.greedy_scheduling(task_dag)
        elif mode in {"gcn", "gat"}:
            priorities[device_id] = build_priorities(
                {device_id: task_dag}, priority_model
            )[device_id]
        else:
            raise ValueError(
                "priority mode must be one of: gcn, gat, random, greedy"
            )
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
    episode_index: int,
) -> Tuple[List[int], int, int, List[float], TopologyGraphState, float, float, float, float]:
    """Build graph state and select Graph-GAT MAPPO joint actions.

    Args:
        agent: Shared Graph-GAT MAPPO controller.
        joint_state: Joint flat state array shaped `(num_devices, state_dim)`.
        num_devices: Number of device agents.
        num_servers: Number of edge servers.

    Returns:
        Joint actions, local count, edge count, old log-probs, graph state, graph
        build time, topology warmup time/loss, and action selection time.
    """
    graph_build_start = time.perf_counter()
    graph_state = build_topology_graph_state(
        torch.as_tensor(joint_state, dtype=torch.float32),
        num_devices=num_devices,
        num_servers=num_servers,
    )
    graph_build_time = time.perf_counter() - graph_build_start

    graph_warmup_time = 0.0
    graph_warmup_loss = 0.0
    if agent.should_warmup_topology(episode_index):
        graph_warmup_start = time.perf_counter()
        graph_warmup_loss = agent.warmup_topology_encoder(
            graph_state, agent.topology_warmup_updates_per_step
        )
        graph_warmup_time = time.perf_counter() - graph_warmup_start

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
        graph_warmup_time,
        graph_warmup_loss,
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
        graph_warmup_time = history.get("graph_warmup_time", [0.0])[-1]
        graph_warmup_loss = history.get("graph_warmup_loss", [0.0])[-1]
        graph_warmup_count = history.get("graph_warmup_count", [0.0])[-1]
        graph_action_time = history["graph_action_time"][-1]
        graph_update_time = history["graph_update_time"][-1]
        summary += (
            f"\n  Graph-GAT cost:    build={graph_build_time:.3f}s "
            f"warmup={graph_warmup_time:.3f}s/{graph_warmup_count:.0f} "
            f"warmup_loss={graph_warmup_loss:.4f} "
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
    priority_model: torch.nn.Module,
    num_episodes: int,
    priority_mode: str = "gcn",
    topology_scenario: Optional[TopologyScenario] = None,
    topology_metrics: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, Any]]]:
    """Train one algorithm configuration and return metrics and checkpoint.

    Args:
        algo_name: Display name for logging.
        agent_config: Dict with "class" and optional "kwargs" for agent init.
        devices: List of IndustrialDevice instances.
        servers: List of EdgeServer instances.
        network_env: NetworkEnvironment instance.
        data_loader: Dataset loader for DAG generation.
        priority_model: GCN/GAT model used for priority extraction.
        num_episodes: Number of episodes to train.
        priority_mode: Scheduling mode ("gcn", "gat", "random", "greedy").
        topology_scenario: Optional named topology scenario for mobility routes.
        topology_metrics: Optional topology metrics stored in checkpoints.

    Returns:
        Tuple of metric history and optional trainable model checkpoint payload.
    """
    print(f"\n{'='*50}\nStarting Training for: {algo_name}\n{'='*50}")
    set_seed(75)  # Reset seed for fair comparison
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
        route_rectangles=(
            topology_scenario.route_rectangles if topology_scenario is not None else None
        ),
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
        "graph_warmup_time": [],
        "graph_warmup_loss": [],
        "graph_warmup_count": [],
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
        episode_graph_warmup_time = 0.0
        episode_graph_warmup_loss_total = 0.0
        episode_graph_warmup_count = 0.0
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
            priorities = build_priorities_by_mode(task_dags, priority_model, priority_mode)

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
                        graph_warmup_time,
                        graph_warmup_loss,
                        graph_action_time,
                    ) = _collect_graph_gat_actions(
                        agent=agents[0],
                        joint_state=current_joint_state,
                        num_devices=len(devices),
                        num_servers=len(servers),
                        episode_index=episode,
                    )
                    episode_graph_build_time += graph_build_time
                    episode_graph_warmup_time += graph_warmup_time
                    if graph_warmup_time > 0.0:
                        episode_graph_warmup_loss_total += graph_warmup_loss
                        episode_graph_warmup_count += 1.0
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
        history["graph_warmup_time"].append(episode_graph_warmup_time)
        history["graph_warmup_loss"].append(
            episode_graph_warmup_loss_total / max(episode_graph_warmup_count, 1.0)
        )
        history["graph_warmup_count"].append(episode_graph_warmup_count)
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
                    f"| Warmup: {history['graph_warmup_time'][-1]:.3f}s "
                    f"/ {history['graph_warmup_count'][-1]:.0f} "
                    f"| Warmup Loss: {history['graph_warmup_loss'][-1]:.4f} "
                    f"| Action: {history['graph_action_time'][-1]:.3f}s "
                    f"| Update: {history['graph_update_time'][-1]:.3f}s "
                    f"| Transitions: {history['graph_transition_count'][-1]:.0f}"
                )
            
    checkpoint = build_model_checkpoint(
        algo_name=algo_name,
        agents=agents,
        agent_config=agent_config,
        history=history,
        episode_count=num_episodes,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        num_devices=len(devices),
        num_servers=len(servers),
        graph_node_feature_dim=graph_node_feature_dim if uses_graph_gat_mappo else None,
        graph_edge_feature_dim=graph_edge_feature_dim if uses_graph_gat_mappo else None,
        topology_metrics=topology_metrics,
    )
    return history, checkpoint

if __name__ == "__main__":
    # 1. Base Environment Setup
    args = parse_args()
    confirmed = PAPER_PARAMS["confirmed"]
    provisional = PAPER_PARAMS["provisional_table2_needed"]
    BANDWIDTH, NOISE_POWER = confirmed["bandwidth_hz"], confirmed["noise_power_dbm"]
    topology_scenario = get_topology_scenario(args.topology_scenario)
    topology_metrics = compute_topology_metrics(topology_scenario)
    flat_topology_metrics = flatten_topology_metrics(topology_metrics)

    print(
        "Topology scenario: "
        f"{topology_scenario.name} "
        f"({topology_scenario.device_count} devices / "
        f"{len(topology_scenario.server_locations)} servers, "
        f"radius={topology_scenario.coverage_radius}m)"
    )
    print(
        "Topology metrics: "
        f"avg_links={flat_topology_metrics['topology_avg_feasible_servers']:.2f}, "
        f"density={flat_topology_metrics['topology_density']:.2f}, "
        f"zero={flat_topology_metrics['topology_zero_link_ratio']:.2f}, "
        f"multi={flat_topology_metrics['topology_multi_link_ratio']:.2f}"
    )
    
    network_env = NetworkEnvironment(bandwidth=BANDWIDTH, noise_power_dbm=NOISE_POWER)
    data_loader = KolektorSDDLoader(dataset_path="dataset/KolektorSDD/")
    dataset_stats = data_loader.get_dataset_statistics()
    print(
        f"Dataset images found: {dataset_stats['total_images']} "
        f"(paper: 399; aligned={dataset_stats['is_paper_count_aligned']})"
    )

    priority_model_name = str(provisional["priority_model"]).lower()
    priority_model = build_task_priority_model(
        priority_model_name,
        num_features=3,
        hidden_dim=int(confirmed["gcn_hidden_dim"]),
    )
    priority_ckpt_path = get_priority_checkpoint_path(priority_model_name)
    sample_training_dag = make_gcn_dag_sampler(
        data_loader,
        t_max=provisional["t_max"],
        e_max=provisional["e_max"],
    )

    priority_model = load_or_train_priority_model(
        priority_model=priority_model,
        dag_sampler=sample_training_dag,
        checkpoint_path=priority_ckpt_path,
        epochs=int(provisional["gcn_pretrain_epochs"]),
        samples_per_epoch=int(provisional["gcn_samples_per_epoch"]),
        lr=confirmed["gcn_lr"],
        model_label=priority_model_name.upper(),
    )
    
    servers = build_servers_for_scenario(topology_scenario, confirmed, provisional)
    devices = build_devices_for_scenario(topology_scenario, confirmed, provisional)

    # 2. Define Algorithms to Compare
    algorithms = build_algorithm_configs()

    # 3. Run Comparisons
    results = {"reward": {}, "delay": {}, "energy": {}}
    priority_results = {"reward": {}, "delay": {}, "energy": {}}
    FULL_EPISODES = (
        int(args.episodes)
        if args.episodes is not None
        else int(provisional["comparison_full_episodes"])
    )
    BASELINE_EVALUATION_EPISODES = (
        int(args.baseline_episodes)
        if args.baseline_episodes is not None
        else int(provisional["baseline_evaluation_episodes"])
    )
    algorithm_episode_counts = {}
    last_training_state_rows = []
    model_checkpoints = []

    # Fig.6-style priority extraction comparison under fixed e-ATN-MADDPG.
    priority_modes = {"Random Scheduling": "random", "Greedy Scheduling": "greedy", "GCN Scheduling": "gcn", "GAT Scheduling": "gat"}
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
    #         priority_model=priority_model,
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
        history, checkpoint = train_algorithm(
            algo_name,
            config,
            devices,
            servers,
            network_env,
            data_loader,
            num_episodes=run_episodes,
            priority_model=priority_model,
            priority_mode=priority_model_name,
            topology_scenario=topology_scenario,
            topology_metrics=topology_metrics,
        )
        results["reward"][algo_name] = history["reward"]
        results["delay"][algo_name] = history["delay"]
        results["energy"][algo_name] = history["energy"]
        last_training_state_rows.append(
            build_last_training_state_line(
                algo_name,
                history,
                episode_count=run_episodes,
                topology_metrics=topology_metrics,
            )
        )
        if checkpoint is not None:
            model_checkpoints.append(checkpoint)

    # 4. Plotting Results
    print("\nAll training complete! Generating comparison plots...")
    output_paths = save_comparison_outputs(
        raw_results=results,
        full_episodes=FULL_EPISODES,
        last_training_state_rows=last_training_state_rows,
        model_checkpoints=model_checkpoints,
        fixed_baseline_algorithms=FIXED_BASELINE_ALGORITHMS,
    )
    last_state_path = output_paths["last_state_path"]
    checkpoint_paths = output_paths["checkpoint_paths"]
    
    print(f"Plots and last training states saved successfully! JSONL: {last_state_path}")
    if checkpoint_paths:
        print("Model checkpoints saved:")
        for checkpoint_path in checkpoint_paths:
            print(f"  {checkpoint_path}")
