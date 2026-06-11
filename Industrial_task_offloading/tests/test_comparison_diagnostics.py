"""Tests for comparison metric diagnostics."""

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_comparision import (
    _format_diagnostic_summary,
    _should_print_diagnostics,
    _summarize_step_metrics,
)


def test_summarize_step_metrics_counts_penalties_and_times() -> None:
    """Step summary should expose penalty count and local/server time."""
    metrics = [
        {
            "requested_action": 1.0,
            "action": 0.0,
            "local_time": 0.4,
            "server_time": 0.0,
            "attempted_server_time": 0.2,
            "transfer_time": 0.1,
            "queue_or_wait_time": 0.3,
            "penalty_time": 0.05,
            "penalty_applied": 1.0,
        },
        {
            "requested_action": 1.0,
            "action": 1.0,
            "local_time": 0.0,
            "server_time": 0.7,
            "attempted_server_time": 0.7,
            "transfer_time": 0.2,
            "queue_or_wait_time": 0.1,
            "penalty_time": 0.0,
            "penalty_applied": 0.0,
        },
    ]

    summary = _summarize_step_metrics(metrics)

    assert summary["penalty_count"] == pytest.approx(1.0)
    assert summary["requested_local_count"] == pytest.approx(0.0)
    assert summary["requested_edge_count"] == pytest.approx(2.0)
    assert summary["resolved_local_count"] == pytest.approx(1.0)
    assert summary["resolved_edge_count"] == pytest.approx(1.0)
    assert summary["local_time"] == pytest.approx(0.4)
    assert summary["server_time"] == pytest.approx(0.7)
    assert summary["attempted_server_time"] == pytest.approx(0.9)
    assert summary["transfer_time"] == pytest.approx(0.3)
    assert summary["queue_or_wait_time"] == pytest.approx(0.4)
    assert summary["penalty_time"] == pytest.approx(0.05)


def test_format_diagnostic_summary_is_readable() -> None:
    """Diagnostic formatter should make counts and timings easy to scan."""
    history = {
        "requested_local_count": [2329.0],
        "requested_edge_count": [171.0],
        "resolved_local_count": [2480.0],
        "resolved_edge_count": [20.0],
        "penalty_count": [151.0],
        "local_time": [2.75],
        "server_time": [0.081],
        "transfer_time": [0.21],
        "queue_or_wait_time": [0.44],
        "penalty_time": [0.856],
    }

    summary = _format_diagnostic_summary("Random Offloading", 10, history)

    assert "[Random Offloading] Episode 10 diagnostics" in summary
    assert "Requested actions: local=2329 edge=171" in summary
    assert "Actual execution:  local=2480 edge=20" in summary
    assert "Penalties:         count=151 time=0.856s" in summary
    assert "Timing avg/step:   local=2.750s server=0.081s transfer=0.210s wait=0.440s" in summary


def test_short_run_diagnostics_print_only_on_final_episode() -> None:
    """Short comparison runs should avoid printing diagnostics every epoch."""
    assert not _should_print_diagnostics(1, 10)
    assert not _should_print_diagnostics(9, 10)
    assert _should_print_diagnostics(10, 10)
    assert _should_print_diagnostics(50, 1000)
