# Windows GPU Readiness

This project currently keeps training logic CPU-safe by default. GPU work should
start with a readiness check before any model code is moved to CUDA.

## Scope

- This is only a CUDA environment check.
- It does not change `run_comparision.py`, `main.py`, or any agent training
  logic.
- The Mac demo path should keep running as before.
- Full CUDA tensor/model migration should wait until the Windows machine proves
  that PyTorch can see the RTX 4080.

## Install PyTorch on Windows

Install project dependencies as usual, then install the Windows CUDA build of
PyTorch separately. Use the official PyTorch selector:

https://pytorch.org/get-started/locally/

Recommended selector choices:

- OS: Windows
- Package: Pip or Conda, matching the environment you use
- Language: Python
- Compute Platform: CUDA, using the newest stable option that matches your
  driver

Do not copy a CUDA wheel command from this repository long-term. PyTorch updates
the recommended command over time, so the official selector is the source of
truth.

## Check the Machine

From the project root on Windows:

```bash
python -m utils.gpu_readiness --preferred-device cuda
```

Expected successful RTX 4080 shape:

```text
GPU readiness report
  PyTorch version: ...
  PyTorch CUDA build: ...
  CUDA available: True
  CUDA device count: 1
  GPU name: NVIDIA GeForce RTX 4080
  Preferred device: cuda
  Selected device: cuda
```

If `CUDA available` is `False`, do not change model code yet. Fix the Windows
driver or PyTorch installation first.

For machine-readable output:

```bash
python -m utils.gpu_readiness --preferred-device cuda --json
```

## Next Decision

Only after the readiness check passes should one training path move to CUDA.
The first selected target is `Graph-GAT MAPPO`, because it is the slowest active
training path.

## Graph-GAT MAPPO Device Setting

Graph-GAT MAPPO can now choose CPU or CUDA through `utils/paper_config.py`:

```python
"graph_gat_device": "auto",
```

Use these values:

- `"auto"`: CUDA on the Windows GPU machine, CPU on Mac.
- `"cuda"`: require CUDA and fail clearly if PyTorch cannot see the GPU.
- `"cpu"`: force CPU for Mac demos and debugging.

Recommended server setting after readiness passes:

```python
"graph_gat_device": "cuda",
```

This setting only affects `Graph-GAT MAPPO`. The environment simulation and
other algorithms remain unchanged.
