"""Graph-GAT MAPPO ablation agent."""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from models.topology_gat import TopologyGATEncoder
from ultils.topology_graph_state import TopologyGraphState


class GraphGATActor(nn.Module):
    """Shared actor head that maps device embeddings to action probabilities."""

    def __init__(self, embedding_dim: int, action_dim: int, hidden_dim: int = 64):
        """Initialize the actor head.

        Args:
            embedding_dim: Device embedding dimension from the GAT encoder.
            action_dim: Number of offloading actions.
            hidden_dim: Hidden layer width.
        """
        super(GraphGATActor, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, device_embeddings: torch.Tensor) -> torch.Tensor:
        """Return action probabilities for each device embedding."""
        x = F.relu(self.fc1(device_embeddings))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class GraphGATValueCritic(nn.Module):
    """Centralized value critic over all device embeddings."""

    def __init__(self, embedding_dim: int, num_devices: int, hidden_dim: int = 64):
        """Initialize the centralized graph critic.

        Args:
            embedding_dim: Device embedding dimension from the GAT encoder.
            num_devices: Number of device agents.
            hidden_dim: Hidden layer width.
        """
        super(GraphGATValueCritic, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * num_devices, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, device_embeddings: torch.Tensor) -> torch.Tensor:
        """Return one scalar value for the joint graph state."""
        joint_embedding = device_embeddings.reshape(1, -1)
        x = F.relu(self.fc1(joint_embedding))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass(frozen=True)
class GraphGATTransition:
    """One on-policy Graph-GAT MAPPO transition."""

    graph_state: TopologyGraphState
    actions: List[int]
    rewards: List[float]
    next_graph_state: TopologyGraphState
    old_log_probs: List[float]
    done: bool


class GraphGATRolloutBuffer:
    """Store on-policy graph transitions for Graph-GAT MAPPO."""

    def __init__(self):
        """Initialize an empty graph rollout buffer."""
        self.transitions: List[GraphGATTransition] = []

    def push(
        self,
        graph_state: TopologyGraphState,
        actions: List[int],
        rewards: List[float],
        next_graph_state: TopologyGraphState,
        old_log_probs: List[float],
        done: bool,
    ) -> None:
        """Store one graph transition."""
        self.transitions.append(
            GraphGATTransition(
                graph_state=graph_state,
                actions=list(actions),
                rewards=list(rewards),
                next_graph_state=next_graph_state,
                old_log_probs=list(old_log_probs),
                done=bool(done),
            )
        )

    def as_transitions(self) -> List[GraphGATTransition]:
        """Return stored transitions in insertion order."""
        return list(self.transitions)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self.transitions.clear()

    def __len__(self) -> int:
        """Return number of stored graph transitions."""
        return len(self.transitions)


class GraphGATMAPPOAgent:
    """Shared Graph-GAT MAPPO controller for all device agents."""

    def __init__(
        self,
        num_devices: int,
        num_servers: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        lr: float = 0.0001,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        ppo_epochs: int = 4,
        use_action_mask: bool = True,
    ):
        """Initialize Graph-GAT MAPPO.

        Args:
            num_devices: Number of device agents.
            num_servers: Number of edge servers.
            node_feature_dim: Input topology node feature dimension.
            edge_feature_dim: Input topology edge feature dimension.
            embedding_dim: GAT device embedding dimension.
            hidden_dim: Hidden layer width.
            lr: Learning rate.
            gamma: Discount factor.
            clip_param: PPO clipping parameter.
            ppo_epochs: PPO epochs per rollout update.
            use_action_mask: Whether to mask disconnected edge-server actions.
        """
        self.num_devices = num_devices
        self.action_dim = num_servers + 1
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.use_action_mask = use_action_mask

        self.encoder = TopologyGATEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.actor = GraphGATActor(
            embedding_dim=embedding_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )
        self.critic = GraphGATValueCritic(
            embedding_dim=embedding_dim,
            num_devices=num_devices,
            hidden_dim=hidden_dim,
        )
        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )

    def select_actions_with_log_probs(
        self, graph_state: TopologyGraphState
    ) -> Tuple[List[int], List[float]]:
        """Sample one offloading action per device from graph embeddings."""
        with torch.no_grad():
            device_embeddings = self._encode_graph(graph_state)
            probabilities = self._masked_action_probabilities(
                self.actor(device_embeddings), graph_state
            )
            distribution = Categorical(probabilities)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            return actions.tolist(), [float(value) for value in log_probs.tolist()]

    def update_from_rollout(self, rollout_buffer: GraphGATRolloutBuffer) -> None:
        """Update encoder, actor, and critic from on-policy graph rollouts."""
        transitions = rollout_buffer.as_transitions()
        if not transitions:
            return

        actions = torch.tensor(
            [transition.actions for transition in transitions], dtype=torch.long
        )
        rewards = torch.tensor(
            [transition.rewards for transition in transitions], dtype=torch.float32
        )
        old_log_probs = torch.tensor(
            [transition.old_log_probs for transition in transitions],
            dtype=torch.float32,
        )
        dones = torch.tensor(
            [[float(transition.done)] for transition in transitions],
            dtype=torch.float32,
        )
        team_rewards = rewards.mean(dim=1, keepdim=True)
        graph_states = [transition.graph_state for transition in transitions]
        next_graph_states = [transition.next_graph_state for transition in transitions]

        with torch.no_grad():
            current_values = self._values_for_graphs(graph_states)
            next_values = self._values_for_graphs(next_graph_states)
            target_values = team_rewards + self.gamma * next_values * (1.0 - dones)
            advantages = target_values - current_values
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        for _ in range(self.ppo_epochs):
            log_probs, entropy, values = self._policy_and_values_for_graphs(
                graph_states, actions
            )

            ratios = torch.exp(log_probs - old_log_probs)
            expanded_advantages = advantages.expand_as(ratios)
            unclipped = ratios * expanded_advantages
            clipped = torch.clamp(
                ratios, 1.0 - self.clip_param, 1.0 + self.clip_param
            ) * expanded_advantages
            actor_loss = -torch.min(unclipped, clipped).mean() - 0.01 * entropy
            critic_loss = F.mse_loss(values, target_values)

            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self.optimizer.step()

        rollout_buffer.clear()

    def _encode_graph(self, graph_state: TopologyGraphState) -> torch.Tensor:
        """Encode one topology graph into device embeddings."""
        return self.encoder(
            graph_state.node_features,
            graph_state.edge_index,
            graph_state.edge_features,
            graph_state.device_node_indices,
        )

    def _action_mask_for_graph_state(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Return valid local/server actions for each device."""
        mask = torch.zeros(
            (self.num_devices, self.action_dim),
            dtype=torch.bool,
            device=graph_state.node_features.device,
        )
        mask[:, 0] = True

        server_to_action = {
            int(server_node): action_index + 1
            for action_index, server_node in enumerate(
                graph_state.server_node_indices.tolist()
            )
        }
        for edge_index, (source_node, target_node) in enumerate(
            graph_state.edge_index.transpose(0, 1).tolist()
        ):
            if source_node >= self.num_devices or target_node not in server_to_action:
                continue

            is_connected = graph_state.edge_features[edge_index, 2] > 0.5
            if is_connected:
                mask[source_node, server_to_action[target_node]] = True

        return mask

    def _masked_action_probabilities(
        self, probabilities: torch.Tensor, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Zero disconnected server probabilities and renormalize actions."""
        if not self.use_action_mask:
            return probabilities

        action_mask = self._action_mask_for_graph_state(graph_state).to(
            device=probabilities.device
        )
        masked_probabilities = probabilities * action_mask.float()
        probability_sum = masked_probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return masked_probabilities / probability_sum

    def _values_for_graphs(
        self, graph_states: List[TopologyGraphState]
    ) -> torch.Tensor:
        """Return centralized values for a batch of graph states."""
        values = [
            self.critic(self._encode_graph(graph_state))
            for graph_state in graph_states
        ]
        return torch.cat(values, dim=0)

    def _policy_and_values_for_graphs(
        self,
        graph_states: List[TopologyGraphState],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return action log-probs and values while reusing graph embeddings."""
        log_prob_rows = []
        entropy_rows = []
        value_rows = []
        for graph_index, graph_state in enumerate(graph_states):
            device_embeddings = self._encode_graph(graph_state)
            probabilities = self._masked_action_probabilities(
                self.actor(device_embeddings), graph_state
            )
            distribution = Categorical(probabilities)
            log_prob_rows.append(distribution.log_prob(actions[graph_index]))
            entropy_rows.append(distribution.entropy())
            value_rows.append(self.critic(device_embeddings))

        return (
            torch.stack(log_prob_rows),
            torch.stack(entropy_rows).mean(),
            torch.cat(value_rows, dim=0),
        )
