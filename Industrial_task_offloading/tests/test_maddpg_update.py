"""Tests for MADDPG training update behavior."""

import numpy as np
import pathlib
import sys
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import _update_agents_from_buffer
from models.maddpg import EpsilonATNMADDPGAgent
from models.replay_buffer import MultiAgentReplayBuffer


def _clone_actor_parameters(agent: EpsilonATNMADDPGAgent):
    """Return detached copies of actor parameters."""
    return [parameter.detach().clone() for parameter in agent.actor.parameters()]


def _parameters_changed(before, after) -> bool:
    """Return whether any parameter tensor changed."""
    return any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_actor_parameters_change_after_replay_update() -> None:
    """Actor update should propagate gradients into actor parameters."""
    torch.manual_seed(7)
    np.random.seed(7)

    num_agents = 2
    state_dim = 4
    action_dim = 3
    batch_size = 4
    agents = [
        EpsilonATNMADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            use_attention=False,
            use_epsilon_greedy=False,
        )
        for _ in range(num_agents)
    ]
    replay_buffer = MultiAgentReplayBuffer(capacity=16)

    for sample_index in range(batch_size):
        state = np.full((num_agents, state_dim), 0.1 * (sample_index + 1), dtype=np.float32)
        next_state = state + 0.05
        actions = [sample_index % action_dim, (sample_index + 1) % action_dim]
        rewards = [1.0 - 0.1 * sample_index, 0.5 + 0.1 * sample_index]
        replay_buffer.push(state, actions, rewards, next_state)

    before = _clone_actor_parameters(agents[0])

    _update_agents_from_buffer(agents, replay_buffer, batch_size=batch_size, gamma=0.95)

    after = _clone_actor_parameters(agents[0])
    assert _parameters_changed(before, after)
