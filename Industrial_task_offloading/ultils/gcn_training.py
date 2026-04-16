"""Train or load GCN models for task priority extraction."""

from __future__ import annotations

import os
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from environment.system_model import TaskDAG


def _minmax_normalize(values: List[float]) -> List[float]:
    """Normalize a list of values into [0, 1] with min-max scaling.

    Args:
        values: Input values.

    Returns:
        Normalized values in the range [0, 1].
    """
    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-9:
        return [0.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def _compute_levels(task_dag: TaskDAG) -> Dict[int, int]:
    """Compute hierarchy levels for DAG nodes.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Mapping of subtask IDs to hierarchy levels.
    """
    indegree = {sid: 0 for sid in task_dag.subtasks.keys()}
    succs: Dict[int, List[int]] = {sid: [] for sid in task_dag.subtasks.keys()}
    for pred, succ in task_dag.edges:
        indegree[succ] += 1
        succs[pred].append(succ)

    queue = [sid for sid, deg in indegree.items() if deg == 0]
    levels = {sid: 1 for sid in queue}
    while queue:
        cur = queue.pop(0)
        cur_level = levels[cur]
        for nxt in succs[cur]:
            levels[nxt] = max(levels.get(nxt, 1), cur_level + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    for sid in task_dag.subtasks.keys():
        levels.setdefault(sid, 1)
    return levels


def _compute_successor_cpu(task_dag: TaskDAG) -> Dict[int, float]:
    """Compute cumulative successor CPU cycles for each subtask.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Mapping of subtask IDs to cumulative successor CPU cycles.
    """
    succs: Dict[int, List[int]] = {sid: [] for sid in task_dag.subtasks.keys()}
    for pred, succ in task_dag.edges:
        succs[pred].append(succ)

    memo: Dict[int, float] = {}

    def dfs(node_id: int) -> float:
        if node_id in memo:
            return memo[node_id]
        total = 0.0
        for nxt in succs[node_id]:
            total += float(task_dag.subtasks[nxt].cpu_cycles)
            total += dfs(nxt)
        memo[node_id] = total
        return total

    for sid in task_dag.subtasks.keys():
        dfs(sid)
    return memo


def build_priority_targets(task_dag: TaskDAG) -> torch.Tensor:
    """Build supervised priority targets for GCN pretraining.

    Args:
        task_dag: Task DAG definition.

    Returns:
        Tensor of priority targets shaped (num_nodes, 1).
    """
    subtask_ids = sorted(task_dag.subtasks.keys())
    levels = _compute_levels(task_dag)
    max_level = max(levels.values()) if levels else 1
    out_degree = {
        sid: sum(1 for pred, _ in task_dag.edges if pred == sid) for sid in subtask_ids
    }
    successor_cpu = _compute_successor_cpu(task_dag)

    hierarchy_score = [float(max_level - levels[sid] + 1) for sid in subtask_ids]
    out_degree_score = [float(out_degree[sid]) for sid in subtask_ids]
    successor_cpu_score = [float(successor_cpu[sid]) for sid in subtask_ids]

    h_norm = _minmax_normalize(hierarchy_score)
    o_norm = _minmax_normalize(out_degree_score)
    c_norm = _minmax_normalize(successor_cpu_score)

    # Paper-inspired label composition: hierarchy + out-degree + successor compute volume.
    targets = [
        0.3 * h_norm[i] + 0.3 * o_norm[i] + 0.4 * c_norm[i] for i in range(len(subtask_ids))
    ]
    return torch.tensor(targets, dtype=torch.float32).unsqueeze(1)


def load_or_train_gcn(
    gcn_model: torch.nn.Module,
    dag_sampler: Callable[[], TaskDAG],
    checkpoint_path: str,
    epochs: int = 200,
    samples_per_epoch: int = 32,
    lr: float = 1e-3,
) -> torch.nn.Module:
    """Load a pretrained GCN model or train one if missing.

    Args:
        gcn_model: GCN model instance.
        dag_sampler: Callable that returns TaskDAG samples.
        checkpoint_path: Path to the model checkpoint.
        epochs: Number of training epochs.
        samples_per_epoch: DAG samples per epoch.
        lr: Learning rate.

    Returns:
        Trained or loaded GCN model.
    """
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        gcn_model.load_state_dict(state_dict)
        gcn_model.eval()
        print(f"Loaded pretrained GCN weights from: {checkpoint_path}")
        return gcn_model

    print("No pretrained GCN weights found. Training GCN priority model...")
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)
    gcn_model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(samples_per_epoch):
            task_dag = dag_sampler()
            from ultils.graph_ultils import extract_gcn_inputs

            features, adjacency = extract_gcn_inputs(task_dag)
            y = build_priority_targets(task_dag)
            pred = gcn_model(features, adjacency)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / float(samples_per_epoch)
            print(f"GCN pretrain epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(gcn_model.state_dict(), checkpoint_path)
    print(f"Saved GCN weights to: {checkpoint_path}")
    gcn_model.eval()
    return gcn_model
