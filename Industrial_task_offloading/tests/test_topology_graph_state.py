"""Tests for topology graph construction from flat DITEN state."""

import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultils.topology_graph_state import build_topology_graph_state


def _make_joint_state() -> torch.Tensor:
    """Create a two-device, two-server flat state with one disconnected link."""
    # Layout: 5 device/task features, 5 priority values, 2 edge powers,
    # 2 edge waits, 2 window starts, 2 window ends.
    return torch.tensor(
        [
            [
                1.0,
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
                0.9,
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


def test_topology_graph_state_shapes_are_stable() -> None:
    """Graph tensors should have stable node and edge feature shapes."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )

    assert graph_state.node_features.shape == (4, 9)
    assert graph_state.edge_index.shape == (2, 4)
    assert graph_state.edge_features.shape == (4, 5)
    assert graph_state.device_node_indices.tolist() == [0, 1]
    assert graph_state.server_node_indices.tolist() == [2, 3]


def test_topology_graph_state_encodes_device_and_server_nodes() -> None:
    """Device and server nodes should use the shared padded feature schema."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )

    first_device = graph_state.node_features[0]
    first_server = graph_state.node_features[2]

    assert first_device.tolist() == pytest.approx(
        [1.0, 0.0, 1.0, 0.1, 0.5, 0.2, 0.05, 0.0, 0.0]
    )
    assert first_server.tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3, 0.1]
    )


def test_topology_graph_state_excludes_disconnected_links() -> None:
    """Links with non-positive connection windows should not create edges."""
    graph_state = build_topology_graph_state(
        _make_joint_state(), num_devices=2, num_servers=2
    )
    edges = {
        (int(src), int(dst))
        for src, dst in graph_state.edge_index.transpose(0, 1).tolist()
    }

    assert (0, 2) in edges
    assert (2, 0) in edges
    assert (0, 3) not in edges
    assert (3, 0) not in edges
    assert (1, 2) not in edges
    assert (2, 1) not in edges
    assert (1, 3) in edges
    assert (3, 1) in edges


def test_topology_graph_state_rejects_incompatible_flat_state_shape() -> None:
    """Invalid flat state widths should fail before graph construction."""
    bad_state = torch.zeros((2, 12), dtype=torch.float32)

    with pytest.raises(ValueError, match="flat state width"):
        build_topology_graph_state(bad_state, num_devices=2, num_servers=2)
