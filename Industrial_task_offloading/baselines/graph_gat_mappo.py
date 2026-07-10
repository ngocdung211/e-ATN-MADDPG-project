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


class TopologyWarmupHead(nn.Module):
    """Predict link feasibility and window quality from local device embeddings."""

    def __init__(self, embedding_dim: int, num_servers: int, hidden_dim: int = 64):
        """Initialize topology warmup prediction head.

        Args:
            embedding_dim: Device embedding dimension from local GAT subgraph.
            num_servers: Number of edge servers.
            hidden_dim: Hidden layer width.
        """
        super(TopologyWarmupHead, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.feasible_head = nn.Linear(hidden_dim, num_servers)
        self.window_head = nn.Linear(hidden_dim, num_servers)

    def forward(self, local_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return feasible logits and normalized window estimates per server."""
        x = F.relu(self.fc1(local_embeddings))
        x = F.relu(self.fc2(x))
        return self.feasible_head(x), torch.sigmoid(self.window_head(x))


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
        topology_warmup_episodes: int = 0,
        topology_warmup_updates_per_step: int = 0,
        topology_warmup_lr: float = 0.001,
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
            topology_warmup_episodes: Number of first episodes using online GAT warmup.
            topology_warmup_updates_per_step: Auxiliary GAT updates before action selection.
            topology_warmup_lr: Learning rate for topology warmup optimizer.
        """
        self.num_devices = num_devices
        self.num_servers = num_servers
        self.action_dim = num_servers + 1
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.use_action_mask = use_action_mask
        self.topology_warmup_episodes = topology_warmup_episodes
        self.topology_warmup_updates_per_step = topology_warmup_updates_per_step

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
        self.topology_warmup_head = TopologyWarmupHead(
            embedding_dim=embedding_dim,
            num_servers=num_servers,
            hidden_dim=hidden_dim,
        )
        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )
        self.topology_warmup_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.topology_warmup_head.parameters()),
            lr=topology_warmup_lr,
        )

    def select_actions_with_log_probs(
        self, graph_state: TopologyGraphState
    ) -> Tuple[List[int], List[float]]:
        """Sample one offloading action per device from graph embeddings."""
        with torch.no_grad():
            probabilities = self._actor_probabilities_for_graph_state(graph_state)
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

    def should_warmup_topology(self, episode_index: int) -> bool:
        """Return whether online topology warmup is active for this episode."""
        return (
            self.topology_warmup_episodes > 0
            and self.topology_warmup_updates_per_step > 0
            and episode_index < self.topology_warmup_episodes
        )

    def warmup_topology_encoder(
        self, graph_state: TopologyGraphState, update_count: int
    ) -> float:
        """Train topology encoder on link feasibility/window labels."""
        if update_count <= 0:
            return 0.0

        feasible_targets, window_targets = self._topology_warmup_targets(graph_state)
        last_loss = 0.0
        for _ in range(update_count):
            local_embeddings = self._encode_local_actor_embeddings(graph_state)
            feasible_logits, window_estimates = self.topology_warmup_head(
                local_embeddings
            )
            feasible_loss = F.binary_cross_entropy_with_logits(
                feasible_logits, feasible_targets
            )
            window_loss = F.mse_loss(window_estimates, window_targets)
            loss = feasible_loss + 0.5 * window_loss

            self.topology_warmup_optimizer.zero_grad()
            loss.backward()
            self.topology_warmup_optimizer.step()
            last_loss = float(loss.detach().item())

        return last_loss

    def _topology_warmup_targets(
        self, graph_state: TopologyGraphState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build feasible-link and window-length labels from graph edges."""
        device = graph_state.node_features.device
        feasible_targets = torch.zeros(
            (self.num_devices, self.num_servers),
            dtype=torch.float32,
            device=device,
        )
        window_targets = torch.zeros_like(feasible_targets)
        server_to_index = {
            int(server_node): server_index
            for server_index, server_node in enumerate(
                graph_state.server_node_indices.tolist()
            )
        }

        for edge_position, (source_node, target_node) in enumerate(
            graph_state.edge_index.transpose(0, 1).tolist()
        ):
            if source_node >= self.num_devices or target_node not in server_to_index:
                continue
            server_index = server_to_index[target_node]
            edge_features = graph_state.edge_features[edge_position]
            feasible_targets[source_node, server_index] = edge_features[2]
            window_targets[source_node, server_index] = edge_features[6]

        return feasible_targets, window_targets

    def _actor_probabilities_for_graph_state(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Return actor probabilities from per-device local subgraph embeddings."""
        local_embeddings = self._encode_local_actor_embeddings(graph_state)
        probabilities = self.actor(local_embeddings)
        return self._masked_action_probabilities(probabilities, graph_state)

    def _encode_local_actor_embeddings(
        self, graph_state: TopologyGraphState
    ) -> torch.Tensor:
        """Encode each device using only its device-server local subgraph."""
        local_embeddings = []
        for device_index in range(self.num_devices):
            local_graph_state = self._local_subgraph_for_device(
                graph_state, device_index
            )
            local_embeddings.append(self._encode_graph(local_graph_state)[0])
        return torch.stack(local_embeddings, dim=0)

    def _local_subgraph_for_device(
        self, graph_state: TopologyGraphState, device_index: int
    ) -> TopologyGraphState:
        """Build one actor subgraph with one device node and all server nodes."""
        device = graph_state.node_features.device
        source_device_node = int(graph_state.device_node_indices[device_index])
        source_server_nodes = [
            int(server_node) for server_node in graph_state.server_node_indices.tolist()
        ]
        source_nodes = [source_device_node] + source_server_nodes
        local_index_by_source = {
            source_node: local_index
            for local_index, source_node in enumerate(source_nodes)
        }

        node_features = graph_state.node_features[
            torch.tensor(source_nodes, dtype=torch.long, device=device)
        ]
        edge_pairs: List[Tuple[int, int]] = []
        edge_features = []
        for edge_position, (source_node, target_node) in enumerate(
            graph_state.edge_index.transpose(0, 1).tolist()
        ):
            if source_node not in local_index_by_source:
                continue
            if target_node not in local_index_by_source:
                continue
            if source_device_node not in (source_node, target_node):
                continue
            edge_pairs.append(
                (
                    local_index_by_source[source_node],
                    local_index_by_source[target_node],
                )
            )
            edge_features.append(graph_state.edge_features[edge_position])

        if edge_pairs:
            edge_index = torch.tensor(
                edge_pairs, dtype=torch.long, device=device
            ).transpose(0, 1)
            edge_feature_tensor = torch.stack(edge_features, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_feature_tensor = graph_state.edge_features.new_empty(
                (0, graph_state.edge_features.shape[1])
            )

        return TopologyGraphState(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_feature_tensor,
            device_node_indices=torch.tensor([0], dtype=torch.long, device=device),
            server_node_indices=torch.arange(
                1, self.num_servers + 1, dtype=torch.long, device=device
            ),
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
            probabilities = self._actor_probabilities_for_graph_state(graph_state)
            distribution = Categorical(probabilities)
            log_prob_rows.append(distribution.log_prob(actions[graph_index]))
            entropy_rows.append(distribution.entropy())
            device_embeddings = self._encode_graph(graph_state)
            value_rows.append(self.critic(device_embeddings))

        return (
            torch.stack(log_prob_rows),
            torch.stack(entropy_rows).mean(),
            torch.cat(value_rows, dim=0),
        )
