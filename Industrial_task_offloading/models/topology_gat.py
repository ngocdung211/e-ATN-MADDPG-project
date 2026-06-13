"""Graph attention encoder for device-server topology states."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyGraphAttentionLayer(nn.Module):
    """Single graph-attention layer over directed topology edges."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, output_dim: int):
        """Initialize the graph-attention layer.

        Args:
            node_feature_dim: Input node feature dimension.
            edge_feature_dim: Input edge feature dimension.
            output_dim: Output node embedding dimension.
        """
        super(TopologyGraphAttentionLayer, self).__init__()
        self.edge_feature_dim = edge_feature_dim
        self.node_projection = nn.Linear(node_feature_dim, output_dim, bias=False)
        self.edge_projection = nn.Linear(edge_feature_dim, output_dim, bias=False)
        self.attention_projection = nn.Linear(output_dim * 3, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features with attention over incoming edges.

        Args:
            node_features: Tensor shaped `(num_nodes, node_feature_dim)`.
            edge_index: Directed edges shaped `(2, num_edges)`.
            edge_features: Edge features shaped `(num_edges, edge_feature_dim)`.

        Returns:
            Updated node embeddings shaped `(num_nodes, output_dim)`.
        """
        num_nodes = node_features.shape[0]
        edge_index, edge_features = self._add_self_edges(
            edge_index, edge_features, num_nodes, node_features.device
        )

        projected_nodes = self.node_projection(node_features)
        projected_edges = self.edge_projection(edge_features)
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        source_embeddings = projected_nodes[source_nodes]
        target_embeddings = projected_nodes[target_nodes]
        attention_input = torch.cat(
            [source_embeddings, target_embeddings, projected_edges], dim=-1
        )
        attention_logits = self.leaky_relu(
            self.attention_projection(attention_input)
        ).squeeze(-1)
        messages = source_embeddings + projected_edges

        node_outputs = []
        for node_index in range(num_nodes):
            incoming_mask = target_nodes == node_index
            incoming_logits = attention_logits[incoming_mask]
            incoming_messages = messages[incoming_mask]
            attention_weights = F.softmax(incoming_logits, dim=0).unsqueeze(-1)
            node_outputs.append(torch.sum(attention_weights * incoming_messages, dim=0))

        return torch.stack(node_outputs, dim=0)

    def _add_self_edges(
        self,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add self edges so every node keeps its own feature context."""
        edge_index = edge_index.to(device=device, dtype=torch.long)
        edge_features = edge_features.to(device=device, dtype=torch.float32)
        node_indices = torch.arange(num_nodes, device=device, dtype=torch.long)
        self_edge_index = torch.stack([node_indices, node_indices], dim=0)
        self_edge_features = torch.zeros(
            (num_nodes, self.edge_feature_dim), device=device, dtype=torch.float32
        )

        if edge_index.numel() == 0:
            return self_edge_index, self_edge_features

        return (
            torch.cat([edge_index, self_edge_index], dim=1),
            torch.cat([edge_features, self_edge_features], dim=0),
        )


class TopologyGATEncoder(nn.Module):
    """Encode topology graph tensors into one embedding per device node."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
    ):
        """Initialize the topology GAT encoder.

        Args:
            node_feature_dim: Input node feature dimension.
            edge_feature_dim: Input edge feature dimension.
            hidden_dim: Hidden graph-attention dimension.
            embedding_dim: Final device embedding dimension.
        """
        super(TopologyGATEncoder, self).__init__()
        self.gat1 = TopologyGraphAttentionLayer(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            output_dim=hidden_dim,
        )
        self.gat2 = TopologyGraphAttentionLayer(
            node_feature_dim=hidden_dim,
            edge_feature_dim=edge_feature_dim,
            output_dim=embedding_dim,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        device_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return device-node embeddings from graph tensors.

        Args:
            node_features: Tensor shaped `(num_nodes, node_feature_dim)`.
            edge_index: Directed edges shaped `(2, num_edges)`.
            edge_features: Edge features shaped `(num_edges, edge_feature_dim)`.
            device_node_indices: Indices of device nodes in `node_features`.

        Returns:
            Tensor shaped `(num_devices, embedding_dim)`.
        """
        hidden_nodes = F.elu(self.gat1(node_features, edge_index, edge_features))
        encoded_nodes = self.gat2(hidden_nodes, edge_index, edge_features)
        return encoded_nodes[device_node_indices.to(encoded_nodes.device)]
