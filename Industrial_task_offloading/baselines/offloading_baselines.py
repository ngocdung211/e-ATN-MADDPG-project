"""Simple offloading-location baseline policies."""

import random
from typing import Sequence


class LocalOnlyAgent:
    """Baseline policy that always executes subtasks locally."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        """Initialize the policy with the common agent constructor shape."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

    def select_action(self, state: Sequence[float]) -> int:
        """Return the local execution action."""
        return 0


class RandomOffloadingAgent:
    """Baseline policy that samples an execution location uniformly."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        """Initialize the policy with the common agent constructor shape."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

    def select_action(self, state: Sequence[float]) -> int:
        """Return a random valid action index."""
        return random.randint(0, self.action_dim - 1)


class EdgeOnlyAgent:
    """Baseline policy that always attempts edge execution."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        """Initialize the policy with the common agent constructor shape."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents

    def select_action(self, state: Sequence[float]) -> int:
        """Return the first edge with a non-empty connection window."""
        server_count = self.action_dim - 1
        state_values = self._to_list(state)
        window_start_offset = len(state_values) - (2 * server_count)
        window_end_offset = len(state_values) - server_count

        for server_index in range(server_count):
            window_start = state_values[window_start_offset + server_index]
            window_end = state_values[window_end_offset + server_index]
            if window_end > window_start:
                return server_index + 1
        return random.randint(1, 3)  # Fallback to random edge if all windows are empty

    def _to_list(self, state: Sequence[float]) -> Sequence[float]:
        """Convert tensor-like state values to a simple sequence."""
        if hasattr(state, "detach"):
            return state.detach().cpu().tolist()
        if hasattr(state, "tolist"):
            return state.tolist()
        return state


class FeatureExtractionEdgeAgent(EdgeOnlyAgent):
    """Baseline that offloads only the feature extraction subtask."""

    FEATURE_EXTRACTION_SUBTASK_ID = 4

    def select_action(self, state: Sequence[float]) -> int:
        """Return local execution when subtask identity is not supplied."""
        return 0

    def select_action_for_subtask(self, state: Sequence[float], subtask_id: int) -> int:
        """Return edge execution only for the feature extraction subtask.

        Args:
            state: Per-agent environment state.
            subtask_id: Current subtask identifier from the DAG priority order.

        Returns:
            Edge action for subtask 4, otherwise local action.
        """
        if subtask_id == self.FEATURE_EXTRACTION_SUBTASK_ID:
            return super().select_action(state)
        return 0
