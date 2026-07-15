"""Report CUDA readiness and resolve the requested PyTorch device."""

import argparse
import json
from typing import Dict, Union

import torch


def resolve_torch_device(
    preferred_device: Union[str, torch.device] = "auto",
) -> torch.device:
    """Resolve ``auto``, ``cpu``, or a CUDA device request.

    Args:
        preferred_device: Requested PyTorch device. ``auto`` selects CUDA when
            available and otherwise selects CPU.

    Returns:
        Resolved PyTorch device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable, or the requested
            CUDA index does not exist.
        ValueError: If the requested device is not CPU or CUDA.
    """
    if isinstance(preferred_device, torch.device):
        requested_device = preferred_device
    else:
        normalized_device = str(preferred_device).strip().lower()
        if normalized_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        requested_device = torch.device(normalized_device)

    if requested_device.type == "cpu":
        return requested_device
    if requested_device.type != "cuda":
        raise ValueError("preferred_device must be 'auto', 'cpu', or a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and verify the GPU driver."
        )
    if (
        requested_device.index is not None
        and requested_device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device index {requested_device.index} was requested, but only "
            f"{torch.cuda.device_count()} device(s) are available."
        )
    return requested_device


def build_gpu_readiness_report(preferred_device: str = "auto") -> Dict[str, object]:
    """Build a serializable CUDA readiness report."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    selection_error = ""
    try:
        selected_device = str(resolve_torch_device(preferred_device))
    except (RuntimeError, ValueError) as error:
        selected_device = "unavailable"
        selection_error = str(error)

    return {
        "pytorch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "gpu_names": gpu_names,
        "preferred_device": preferred_device,
        "selected_device": selected_device,
        "selection_error": selection_error,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check PyTorch CUDA readiness.")
    parser.add_argument(
        "--preferred-device",
        default="auto",
        help="Device request: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the readiness report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    """Print the CUDA readiness report and return a process exit code."""
    args = _parse_args()
    report = build_gpu_readiness_report(args.preferred_device)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        gpu_name = ", ".join(report["gpu_names"]) or "None"
        print("GPU readiness report")
        print(f"  PyTorch version: {report['pytorch_version']}")
        print(f"  PyTorch CUDA build: {report['pytorch_cuda_build']}")
        print(f"  CUDA available: {report['cuda_available']}")
        print(f"  CUDA device count: {report['cuda_device_count']}")
        print(f"  GPU name: {gpu_name}")
        print(f"  Preferred device: {report['preferred_device']}")
        print(f"  Selected device: {report['selected_device']}")
        if report["selection_error"]:
            print(f"  Selection error: {report['selection_error']}")
    return 1 if report["selection_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
