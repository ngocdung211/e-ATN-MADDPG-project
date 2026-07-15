"""Tests for CUDA readiness and device selection."""

import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.gpu_readiness import build_gpu_readiness_report, resolve_torch_device


def test_resolve_cpu_device() -> None:
    """Explicit CPU selection should work on every machine."""
    assert resolve_torch_device("cpu") == torch.device("cpu")


def test_resolve_auto_device_matches_cuda_availability() -> None:
    """Auto selection should prefer CUDA and otherwise retain CPU."""
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_torch_device("auto") == expected_device


def test_explicit_cuda_fails_clearly_when_unavailable() -> None:
    """A required CUDA device must not silently fall back to CPU."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this test machine")

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_torch_device("cuda")


def test_gpu_readiness_report_is_serializable() -> None:
    """Readiness report should expose machine and selection status."""
    report = build_gpu_readiness_report("auto")

    assert report["cuda_available"] == torch.cuda.is_available()
    assert report["preferred_device"] == "auto"
    assert report["selected_device"] in {"cpu", "cuda"}
    assert isinstance(report["gpu_names"], list)
