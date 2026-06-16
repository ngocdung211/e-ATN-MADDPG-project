"""Graph attention network for task priority prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskPriorityGATLayer(nn.Module):
    """Dense graph-attention layer over a task DAG adjacency matrix."""

    def __init__(self, in_features: int, out_features: int):
        """Initialize the graph-attention layer.

        Args:
            in_features: Input node feature dimension.
            out_features: Output node feature dimension.
        """
        super(TaskPriorityGATLayer, self).__init__()
        self.node_projection = nn.Linear(in_features, out_features, bias=False)
        self.attention_projection = nn.Linear(out_features * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply masked attention over adjacent task nodes.

        Args:
            features: Node feature matrix shaped `(num_nodes, in_features)`.
            adjacency: DAG adjacency matrix shaped `(num_nodes, num_nodes)`.

        Returns:
            Updated node features shaped `(num_nodes, out_features)`.
        """
        projected = self.node_projection(features)
        num_nodes = projected.shape[0]
        attention_mask = self._attention_mask(adjacency, num_nodes, projected.device)

        source_features = projected.unsqueeze(1).expand(-1, num_nodes, -1)
        neighbor_features = projected.unsqueeze(0).expand(num_nodes, -1, -1)
        attention_input = torch.cat([source_features, neighbor_features], dim=-1)
        attention_logits = self.leaky_relu(
            self.attention_projection(attention_input).squeeze(-1)
        )
        attention_logits = attention_logits.masked_fill(~attention_mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        return torch.matmul(attention_weights, projected)

    def _attention_mask(
        self, adjacency: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        """Build adjacency mask with self loops."""
        adjacency = adjacency.to(device=device, dtype=torch.float32)
        self_loops = torch.eye(num_nodes, device=device, dtype=torch.float32)
        return (adjacency + self_loops) > 0.0


class TaskPriorityGAT(nn.Module):
    """3-layer GAT architecture for subtask priority prediction."""

    def __init__(self, num_features: int, hidden_dim: int = 32):
        """Initialize the task-priority GAT model.

        Args:
            num_features: Input node feature dimension.
            hidden_dim: Hidden layer width.
        """
        super(TaskPriorityGAT, self).__init__()
        self.gat1 = TaskPriorityGATLayer(num_features, hidden_dim)
        self.gat2 = TaskPriorityGATLayer(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Predict one priority score per subtask node.

        Args:
            features: Node feature matrix.
            adjacency: DAG adjacency matrix.

        Returns:
            Priority score tensor shaped `(num_nodes, 1)`.
        """
        x = F.elu(self.gat1(features, adjacency))
        x = F.elu(self.gat2(x, adjacency))
        return self.output_layer(x)
