"""Tests for Graph-GAT MAPPO ablation behavior."""

import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.graph_gat_mappo import GraphGATMAPPOAgent, GraphGATRolloutBuffer
from run_comparision import (
    _collect_graph_gat_actions,
    _update_graph_gat_mappo_from_rollout,
    build_algorithm_configs,
)
from ultils.topology_graph_state import build_topology_graph_state


def _make_joint_state(offset: float = 0.0) -> torch.Tensor:
    """Create a two-device, two-server flat state with valid graph links."""
    return torch.tensor(
        [
            [
                1.0 + offset,
                0.1,
                0.5,
                0.2,
                0.05,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                2.3,
                2.5,
                0.1,
                0.2,
                0.0,
                1.0,
                0.8,
                1.0,
            ],
            [
                0.9 + offset,
                0.3,
                0.7,
                0.4,
                0.06,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
                2.3,
                2.5,
                0.1,
                0.2,
                1.0,
                0.25,
                1.0,
                0.75,
            ],
        ],
        dtype=torch.float32,
    )


def _make_graph_state(offset: float = 0.0):
    """Create a topology graph state fixture."""
    return build_topology_graph_state(
        _make_joint_state(offset), num_devices=2, num_servers=2
    )


def _clone_module_parameters(module: torch.nn.Module):
    """Return detached copies of module parameters."""
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before, after) -> bool:
    """Return whether any parameter tensor changed."""
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_graph_gat_mappo_samples_valid_actions_from_graph_state() -> None:
    """Graph-GAT MAPPO should sample one valid offloading action per device."""
    torch.manual_seed(31)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )

    actions, log_probs = agent.select_actions_with_log_probs(_make_graph_state())

    assert len(actions) == 2
    assert len(log_probs) == 2
    assert all(0 <= action < agent.action_dim for action in actions)
    assert all(isinstance(log_prob, float) for log_prob in log_probs)


def test_graph_gat_mappo_masks_disconnected_server_actions() -> None:
    """Graph-GAT MAPPO should assign zero probability to disconnected servers."""
    torch.manual_seed(33)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )
    graph_state = _make_graph_state()
    device_embeddings = agent._encode_graph(graph_state)
    probabilities = agent.actor(device_embeddings)

    mask = agent._action_mask_for_graph_state(graph_state)
    masked_probabilities = agent._masked_action_probabilities(
        probabilities, graph_state
    )

    assert mask.tolist() == [[True, True, False], [True, False, True]]
    assert masked_probabilities[0, 2].item() == 0.0
    assert masked_probabilities[1, 1].item() == 0.0
    assert torch.allclose(masked_probabilities.sum(dim=1), torch.ones(2))


def test_graph_gat_mappo_action_mask_can_be_disabled() -> None:
    """Graph-GAT MAPPO mask toggle should preserve raw probabilities."""
    torch.manual_seed(34)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        use_action_mask=False,
    )
    graph_state = _make_graph_state()
    device_embeddings = agent._encode_graph(graph_state)
    probabilities = agent.actor(device_embeddings)

    masked_probabilities = agent._masked_action_probabilities(
        probabilities, graph_state
    )

    assert torch.allclose(masked_probabilities, probabilities)


def test_graph_gat_rollout_buffer_keeps_graph_transitions() -> None:
    """Graph-GAT rollout buffer should keep graph states and PPO metadata."""
    buffer = GraphGATRolloutBuffer()

    buffer.push(
        graph_state=_make_graph_state(),
        actions=[0, 1],
        rewards=[1.0, 0.5],
        next_graph_state=_make_graph_state(offset=0.1),
        old_log_probs=[-0.2, -0.4],
        done=True,
    )

    assert len(buffer) == 1
    transitions = buffer.as_transitions()
    assert transitions[0].actions == [0, 1]
    assert transitions[0].rewards == [1.0, 0.5]
    assert transitions[0].old_log_probs == [-0.2, -0.4]
    assert transitions[0].done is True


def test_graph_gat_mappo_update_changes_actor_critic_and_gat_parameters() -> None:
    """One PPO update should optimize actor, critic, and GAT parameters."""
    torch.manual_seed(37)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=2,
    )
    buffer = GraphGATRolloutBuffer()
    for step_index in range(4):
        graph_state = _make_graph_state(offset=0.05 * step_index)
        actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
        buffer.push(
            graph_state=graph_state,
            actions=actions,
            rewards=[1.0 - 0.1 * step_index, 0.6 + 0.05 * step_index],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=old_log_probs,
            done=step_index == 3,
        )
    encoder_before = _clone_module_parameters(agent.encoder)
    actor_before = _clone_module_parameters(agent.actor)
    critic_before = _clone_module_parameters(agent.critic)

    agent.update_from_rollout(buffer)

    assert _parameters_changed(encoder_before, _clone_module_parameters(agent.encoder))
    assert _parameters_changed(actor_before, _clone_module_parameters(agent.actor))
    assert _parameters_changed(critic_before, _clone_module_parameters(agent.critic))
    assert len(buffer) == 0


def test_graph_gat_mappo_is_registered_as_separate_comparison_model() -> None:
    """Comparison config should expose Graph-GAT MAPPO without replacing MAPPO."""
    configs = build_algorithm_configs()

    assert configs["Graph-GAT MAPPO"]["class"] is GraphGATMAPPOAgent
    if "MAPPO" in configs:
        assert configs["MAPPO"]["class"] is not GraphGATMAPPOAgent


def test_collect_graph_gat_actions_builds_graph_from_joint_state() -> None:
    """Comparison loop should collect Graph-GAT actions from topology graph state."""
    torch.manual_seed(41)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
    )

    (
        actions,
        local_count,
        edge_count,
        old_log_probs,
        graph_state,
        graph_build_time,
        graph_action_time,
    ) = (
        _collect_graph_gat_actions(
            agent=agent,
            joint_state=_make_joint_state(),
            num_devices=2,
            num_servers=2,
        )
    )

    assert len(actions) == 2
    assert len(old_log_probs) == 2
    assert local_count + edge_count == 2
    assert graph_state.node_features.shape == (4, 14)
    assert all(0 <= action < agent.action_dim for action in actions)
    assert graph_build_time >= 0.0
    assert graph_action_time >= 0.0


def test_graph_gat_rollout_update_helper_updates_agent_and_clears_buffer() -> None:
    """Comparison loop helper should train Graph-GAT MAPPO from graph rollouts."""
    torch.manual_seed(43)
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=2,
    )
    buffer = GraphGATRolloutBuffer()
    for step_index in range(4):
        graph_state = _make_graph_state(offset=0.05 * step_index)
        actions, old_log_probs = agent.select_actions_with_log_probs(graph_state)
        buffer.push(
            graph_state=graph_state,
            actions=actions,
            rewards=[1.0 - 0.1 * step_index, 0.6 + 0.05 * step_index],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=old_log_probs,
            done=step_index == 3,
        )
    actor_before = _clone_module_parameters(agent.actor)

    update_time = _update_graph_gat_mappo_from_rollout(agent, buffer, gamma=0.95)

    assert _parameters_changed(actor_before, _clone_module_parameters(agent.actor))
    assert len(buffer) == 0
    assert update_time >= 0.0


def test_graph_gat_mappo_reuses_epoch_embeddings_for_actor_and_critic() -> None:
    """PPO update should not encode the same graph twice in one epoch."""
    agent = GraphGATMAPPOAgent(
        num_devices=2,
        num_servers=2,
        node_feature_dim=14,
        edge_feature_dim=7,
        embedding_dim=8,
        ppo_epochs=2,
    )
    buffer = GraphGATRolloutBuffer()
    transition_count = 4
    for step_index in range(transition_count):
        buffer.push(
            graph_state=_make_graph_state(offset=0.05 * step_index),
            actions=[0, 1],
            rewards=[1.0 - 0.1 * step_index, 0.6 + 0.05 * step_index],
            next_graph_state=_make_graph_state(offset=0.05 * (step_index + 1)),
            old_log_probs=[-0.5, -0.5],
            done=step_index == transition_count - 1,
        )
    original_encode_graph = agent._encode_graph
    encode_call_count = 0

    def counted_encode_graph(graph_state):
        nonlocal encode_call_count
        encode_call_count += 1
        return original_encode_graph(graph_state)

    agent._encode_graph = counted_encode_graph

    agent.update_from_rollout(buffer)

    expected_encode_calls = transition_count * (2 + agent.ppo_epochs)
    assert encode_call_count == expected_encode_calls
