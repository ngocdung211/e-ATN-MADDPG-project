# Paper Reimplementation Plan and Checklist

**Goal:** Reimplement the paper environment and MADDPG path with minimum code changes, then compare trained DRL models against the currently enabled comparison baselines.

**Success criteria:**
- The environment follows the paper's task, mobility, communication, computation, MDP state, action, reward, and constraint definitions closely enough that training results are meaningful.
- MADDPG is verified first in isolation before comparing against baselines.
- Baseline comparisons are only trusted after environment invariants pass.
- If MADDPG cannot beat simple baselines after the model update path is verified, treat that as evidence to re-audit the environment against the paper before tuning the model.

**Working rule:** Every small task updates this checklist. Code changes should be surgical and limited to functions needed for the current verified goal.

---

## Active Research Plan - 2026-07-08

**Status:** This section is the active plan. Older open queues below are historical context unless a task is explicitly moved here.

**Current research questions:**
- Does `Graph-GAT Mask MAPPO` fail because the topology encoder is trained from scratch inside PPO?
- Does `Graph-GAT Mask MAPPO` help more when the topology is larger, denser, and harder than the current 10-device/3-server map?
- How can we prove `GAT Scheduling` is better than `GCN Scheduling` for task priority, instead of only showing one lucky training curve?

**Success criteria:**
- Pretrained Topology-GAT experiments compare at least three variants under the same seeds and episode budget: scratch, pretrained-frozen, and pretrained-finetuned.
- Larger-topology experiments report topology complexity metrics, not only reward/delay/energy.
- GAT-vs-GCN task-priority evidence is separated into supervised priority quality, fixed-policy scheduling quality, and end-to-end DRL impact.
- Every result includes the exact config, checkpoint path, seed list, final JSONL row, and plot path.

### Phase 0: Close the Previous Plan and Fix Drift

Goal: make the plan match the code before adding new experiments.

- [x] Treat the old `Next Task Queue`, old `Ablation Study Queue`, and old `Implementation Checklist` as historical context.
- [x] Move the remaining active work into this 2026-07-08 section.
- [ ] Repair the local CPU PyTorch runtime before trusting tests on this Mac.
  - Current blocker: importing `torch` fails because `libtorch_cpu.dylib` is missing from the checked Conda environments.
  - Verify: `python -c "import torch; print(torch.__version__)"` works in the selected project environment.
- [x] Align model names across plan, config, tests, plots, and JSONL.
  - Standard names: `Graph-GAT MAPPO` for the unmasked ablation and `Graph-GAT Mask MAPPO` for the masked proposal.
  - Reason: these names make clear that the GAT is the topology encoder, not the task-priority GAT.
  - Verify: `tests/test_graph_gat_mappo.py` expects the same names returned by `run_comparision.build_algorithm_configs`.
- [x] Drop GPU-device work from the active plan.
  - Decision: keep `GraphGATMAPPOAgent` as CPU/default-device code for now.
  - Impact: do not add a `device` argument, do not pass `graph_gat_device`, and do not plan Windows GPU smoke runs in the active queue.

### Phase 1: Pretrained Topology-GAT Experiment

Goal: test whether the topology encoder helps more when it starts from useful topology knowledge instead of random weights.

Assumption: pretraining should learn connection feasibility and topology quality from graph states before PPO begins. PPO then either freezes that encoder or fine-tunes it.

1. [ ] **Define the Topology-GAT pretraining target**
   - Input: `TopologyGraphState` from `utils/topology_graph_state.py`.
   - Encoder: `models/topology_gat.py::TopologyGATEncoder`.
   - Minimal supervised target:
     - per-device feasible server mask from edge feature `is_connected`;
     - optional per-device best-server target using lowest edge wait and longest valid window.
   - Verify:
     - target can be computed from graph tensors only;
     - no environment physics or reward function changes.

2. [ ] **Add topology pretraining dataset generation**
   - Goal: collect graph states from random or scripted rollouts without training MAPPO.
   - Candidate file: `utils/topology_gat_pretraining.py`.
   - Output:
     - train/validation graph samples;
     - feasible action mask targets;
     - optional best-server labels.
   - Verify:
     - dataset contains connected and disconnected links;
     - report average feasible servers per device.

3. [ ] **Train and save a pretrained Topology-GAT checkpoint**
   - Candidate checkpoint: `models/checkpoints/topology_gat_pretrained.pt`.
   - Metrics:
     - feasible-mask binary accuracy or F1;
     - best-server accuracy if the best-server head is used;
     - validation loss.
   - Verify:
     - checkpoint saves encoder weights separately from the pretraining head;
     - loading the encoder into `GraphGATMAPPOAgent` changes initial encoder weights.

4. [ ] **Add three Graph-GAT MAPPO topology modes**
   - `scratch`: current behavior, encoder trains from random initialization during PPO.
   - `pretrained_frozen`: load pretrained encoder, freeze encoder, train actor and critic only.
   - `pretrained_finetune`: load pretrained encoder and continue PPO updates through encoder.
   - Verify:
     - frozen mode excludes encoder parameters from the optimizer;
     - finetune mode includes encoder parameters in the optimizer;
     - all three modes produce one final JSONL row with mode name and checkpoint path.

5. [ ] **Run smoke tests for all topology modes**
   - Run: 1-3 episodes each.
   - Compare:
     - `Graph-GAT Mask MAPPO scratch`;
     - `Graph-GAT Mask MAPPO pretrained_frozen`;
     - `Graph-GAT Mask MAPPO pretrained_finetune`.
   - Verify:
     - no crash;
     - reward, delay, and energy are finite;
     - requested edge actions do not target disconnected servers when mask is enabled;
     - graph build/action/update cost is printed.

6. [ ] **Run short learning comparison**
   - Run: 10-30 episodes first, same seeds and same episode budget.
   - Expected interpretation:
     - frozen better than scratch means topology features are useful but PPO damages the encoder;
     - finetune better than frozen means pretraining is useful and PPO can improve it;
     - scratch matching both means current topology task may be too easy or features are already sufficient.

### Phase 2: Larger and Denser Topology Stress Test

Goal: test whether Graph-GAT Mask MAPPO only shows value when topology is complex enough.

1. [x] **Add named topology scenario configs**
   - Keep current paper-like scenario as `paper_10d_3s`.
   - Add `medium_20d_6s`.
   - Add `large_30d_10s`.
   - Config controls:
     - map size;
     - number of devices;
     - number of servers;
     - server locations;
     - coverage radius;
     - device route templates.
   - Verify:
     - old `paper_10d_3s` results are reproducible with the new config path.
     - Scenario source: `utils/topology_scenarios.py`.
     - Preview source: `utils/topology_scenario_preview.py`.
     - Runner selection: `run_comparision.py --topology-scenario <name>`.

2. [x] **Add topology complexity diagnostics**
   - Metrics:
     - average feasible servers per device;
     - percent of devices with zero feasible servers;
     - graph edge count;
     - graph density;
     - edge request success rate;
     - mask rejection count or masked-action count;
     - graph build/action/update time.
   - Verify:
     - static scenario diagnostics are saved in final JSONL/checkpoints and printed before training.
     - Static scenario diagnostics are saved in final JSONL/checkpoints:
       `topology_avg_feasible_servers`, `topology_density`,
       `topology_zero_link_ratio`, `topology_multi_link_ratio`, and
       `topology_device_server_ratio`.

3. [ ] **Run topology scale smoke tests**
   - Run 1-3 episodes for:
     - `Mask-MAPPO`;
     - `Graph-GAT Mask MAPPO scratch`;
     - best pretrained mode from Phase 1, if available.
   - Scenarios:
     - `paper_10d_3s`;
     - `medium_20d_5s`;
     - `large_30d_8s`.
   - Verify:
     - no scenario has impossible all-disconnected or all-connected topology unless intentionally configured;
     - reward/delay/energy are finite.

4. [ ] **Run controlled scale comparison**
   - Run: same seeds, same episode count, same priority model, same reward config.
   - Compare:
     - `Mask-MAPPO`;
     - `Graph-GAT Mask MAPPO scratch`;
     - `Graph-GAT Mask MAPPO pretrained_frozen`;
     - `Graph-GAT Mask MAPPO pretrained_finetune`.
   - Success signal:
     - GAT advantage should increase when average feasible server count and graph density increase.
   - If GAT still equals Mask-MAPPO:
     - inspect attention weights or edge-feature sensitivity;
     - check whether flat state already gives MAPPO all topology information in an easy-to-learn order;
     - do not tune reward before checking those explanations.

### Phase 3: Prove GAT Task Priority Beats GCN Task Priority

Goal: separate task-priority quality from offloading-policy quality.

1. [ ] **Freeze the comparison protocol**
   - Use the same DAG generator, train/validation split, model hidden size policy, optimizer, epochs, and seeds for GCN and GAT.
   - Compare checkpoints:
     - `models/checkpoints/gcn_priority.pt`;
     - `models/checkpoints/gat_priority.pt`.
   - Verify:
     - both models train from scratch when running the priority-only comparison;
     - old checkpoints do not leak into the comparison unless explicitly requested.

2. [ ] **Run supervised priority-quality comparison**
   - Metrics:
     - validation loss against the current priority targets;
     - topological-order validity rate;
     - Kendall or Spearman rank agreement with target order if added.
   - Success signal:
     - GAT has lower validation loss or better ranking quality across multiple seeds.
   - Caveat:
     - if targets are simple hand-built labels, GAT beating GCN here only proves better fit to the current target, not better scheduling.

3. [ ] **Run fixed-policy scheduling comparison**
   - Goal: isolate priority order from RL learning.
   - Keep the offloading policy fixed, for example:
     - local-only;
     - edge-only first-valid;
     - a simple mask-aware heuristic.
   - Compare priority modes:
     - random;
     - greedy;
     - GCN;
     - GAT.
   - Metrics:
     - reward;
     - delay;
     - energy;
     - deadline/energy-constraint violation count if available.
   - Success signal:
     - GAT priority improves scheduling metrics under at least one fixed offloading policy without changing the DRL agent.

4. [ ] **Run end-to-end DRL priority comparison**
   - Keep the DRL algorithm fixed.
   - Recommended first algorithm: `Graph-GAT Mask MAPPO pretrained_frozen` or current best stable MAPPO variant.
   - Compare only the priority extractor:
     - `priority_model = "gcn"`;
     - `priority_model = "gat"`.
   - Verify:
     - same seeds;
     - same topology scenario;
     - same episode budget;
     - same pretrained topology checkpoint mode if used.

5. [ ] **Report statistical evidence**
   - Use at least 3 seeds for smoke-level evidence and 5 seeds for final evidence.
   - Report mean and standard deviation.
   - Prefer paired comparison by seed.
   - Claim “GAT priority is better” only if it wins in at least two evidence layers:
     - supervised priority validation;
     - fixed-policy scheduling;
     - end-to-end DRL.

### Recommended Execution Order

1. Fix local CPU PyTorch so tests can run.
2. Use `Graph-GAT MAPPO` / `Graph-GAT Mask MAPPO` names consistently between tests, config, plots, and result JSONL.
3. Add pretrained Topology-GAT infrastructure and run 1-3 episode smoke tests.
4. Run the default 10-device/3-server short comparison.
5. Add larger topology configs and diagnostics.
6. Run medium/large topology comparisons.
7. Run GCN-vs-GAT priority evidence in supervised, fixed-policy, and end-to-end layers.

### Optional Waiting List

- [ ] **Graph-GAT MADDPG / GATMA-style CTDE**
  - Actor: local graph/subgraph embedding to local action.
  - Critic: global graph plus joint action to cooperative Q-value.
  - Buffer/update: replay buffer, TD loss, target networks, soft update.
  - Status: optional later ablation. Do not mix into current Graph-GAT MAPPO path.

- [ ] **C-4 Graph-GAT topology warmup decay with hybrid link-quality loss**
  - Current code stays on the simpler warmup design for now.
  - Future schedule:
    - episodes 0-4: train Topology-GAT 10 times per environment step before action selection;
    - episodes 5-20: reduce to 1-2 topology updates per environment step;
    - after episode 20: stop auxiliary topology warmup and let PPO continue normally.
  - Future loss: `BCE(feasible_link) + MSE(edge_delay_estimate) + ranking_loss(server_quality)`.
  - Purpose: avoid training Topology-GAT from scratch only through PPO, then test whether smoother warmup decay and richer link-quality supervision improve convergence.
  - Status: optional future ablation, not part of the current implemented experiment.

---

## Current Checkpoint - 2026-06-11

**Status:** Environment is frozen for the first e-ATN-MADDPG sanity phase. The user manually fixed the environment and reported it is good enough for the model to run.

What is now implemented:
- [x] Local-only baseline.
- [x] Edge-only baseline.
- [x] Random offloading baseline implemented, but not required/enabled for the current long-run comparison.
- [x] Feature-extraction-only edge baseline implemented, but not required/enabled for the current long-run comparison.
- [x] Timing diagnostics for local compute time, successful server compute time, attempted server compute time, transfer time, wait time, penalty count, and penalty time.
- [x] Action diagnostics for requested local/edge counts and actual resolved local/edge counts.
- [x] Readable comparison output for short runs, printed only on the final episode.
- [x] Environment logic was manually adjusted by the user and is reported as good enough for the model to run.
- [x] Environment freeze decision completed for first model sanity phase.
- [x] Paper-style delay fix: `environment/diten_env.py` now accumulates DAG makespan delay instead of summing overlapping branch subtask latencies.
- [x] Efficient 100-episode comparison plotting: train learning models for 100 episodes, run fixed baselines for a short evaluation, and plot fixed baselines as mean-flat reference lines.
- [x] Last training state JSONL: save one flat line per model beside the generated plot images for easy comparison.
- [x] Experiment parameters are centralized in `utils/paper_config.py` for the active comparison runner, including reward weights, penalties, estimation errors, GCN pretraining size, compute-power ranges, episode budgets, MAPPO settings, and Graph-GAT settings.
- [x] `t_max` and `e_max` from `PAPER_PARAMS` are passed into GCN DAG sampling and per-episode task DAG generation in `run_comparision.py`.
- [x] MAPPO and Graph-GAT MAPPO action masks can be turned on/off from `utils/paper_config.py`.
- [x] Trainable model checkpoints are saved into the same plot folder as images and last-state JSONL for later evaluation without retraining.
- [x] GPU readiness stage added without changing training logic: `utils/gpu_readiness.py` reports PyTorch/CUDA/GPU status, and `docs/windows_gpu_readiness.md` documents the Windows RTX 4080 setup check.
- [x] GAT task-priority extractor added as an option beside GCN: `models/task_priority_gat.py` implements `TaskPriorityGAT`, `priority_model` in `utils/paper_config.py` selects `gcn` or `gat`, and GAT uses its own `models/checkpoints/gat_priority.pt`.

Current output should now look like:

```text
[Random Offloading] Episode 10 diagnostics
  Requested actions: local=2329 edge=171
  Actual execution:  local=2480 edge=20
  Penalties:         count=151 time=0.856s
  Timing avg/step:   local=2.750s server=0.081s transfer=0.210s wait=0.440s
```

Interpretation:
- `Requested actions` means what the policy asked for.
- `Actual execution` means what really ran after fallback.
- `Penalties` means edge requests that violated the connection window.
- `Timing avg/step` helps identify whether high delay comes from local compute, edge compute, transfer, or queue/wait.

---

## Historical Task Queue - Superseded by Active Research Plan

Historical note: this queue recorded the earlier paper-reimplementation path. Use the 2026-07-08 active research plan above for current work.

1. [x] **Verify e-ATN-MADDPG agent mechanics**
   - Goal: Confirm current `EpsilonATNMADDPGAgent` can act and update before running longer training.
   - Files to inspect/test only:
     - `models/maddpg.py`
     - `models/replay_buffer.py`
     - `run_comparision.py::_update_agents_from_buffer`
   - Verify:
     - `select_action` with epsilon `1.0` samples valid random actions. `DONE`
     - `select_action` with epsilon disabled uses actor argmax. `DONE`
     - replay buffer samples shape correctly for multi-agent training. `DONE`
     - actor parameters change after one update. `DONE`
     - critic parameters change after one update. `DONE`
     - target networks move after soft update. `DONE`
   - Result: mechanics tests pass without production model changes.
   - Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_maddpg_update.py -q -p no:cacheprovider`

2. [x] **Run e-ATN-MADDPG smoke training on frozen env**
   - Goal: Verify model execution path runs on the fixed environment.
   - Run: very small episode count first, for example 1-3 episodes.
   - Verify:
     - no crash. `DONE by user`
     - replay buffer receives samples. `DONE by user`
     - actor/critic update path runs. `DONE by user`
     - reward, delay, and energy are finite. `DONE by user`
     - epsilon decays. `DONE by user`
     - final diagnostics show meaningful requested/actual local-edge counts. `DONE by user`

3. [x] **Verify MAPPO baseline mechanics**
   - Goal: Confirm current `MAPPOAgent` can act and update before trusting it as a comparison baseline.
   - Files to inspect/test only:
     - `baselines/mappo.py`
     - `run_comparision.py::_update_agents_from_buffer`
   - Verify:
     - stochastic actor outputs a valid probability distribution. `DONE`
     - `select_action` samples valid discrete actions. `DONE`
     - centralized value critic accepts flattened joint state and returns one value per sample. `DONE`
     - actor parameters change after one MAPPO update. `DONE`
     - critic parameters change after one MAPPO update. `DONE`
     - comparison update loop routes MAPPO agents through `update_agent`. `DONE`
   - Result: MAPPO mechanics tests pass without production code changes.
   - Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_mappo_update.py -q -p no:cacheprovider`

4. [x] **Fix MAPPO rollout buffer correctness**
   - Goal: Make MAPPO use on-policy rollout data instead of replay memory samples.
   - Files touched:
     - `baselines/mappo.py`
     - `run_comparision.py`
     - `tests/test_mappo_update.py`
   - Verify:
     - MAPPO stores action log-probability when sampling action. `DONE`
     - MAPPO rollout buffer stores state, action, reward, next state, old log-probability, and done. `DONE`
     - PPO update uses rollout old log-probabilities instead of recomputing them after collection. `DONE`
     - PPO target value masks terminal transitions with `done`. `DONE`
     - comparison loop routes MAPPO through rollout update and clears the rollout after update. `DONE`
   - Verification: `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_mappo_update.py -q -p no:cacheprovider`

5. [ ] **Add e-ATN-MADDPG training diagnostics if smoke output is insufficient** `HISTORICAL`
   - Goal: Make model learning observable without changing env.
   - Candidate diagnostics:
     - actor loss.
     - critic loss.
     - epsilon.
     - replay buffer size.
     - action distribution.
   - Rule: add diagnostics only if the smoke run does not already show enough information to judge model health.

6. [ ] **Run short e-ATN-MADDPG learning check**
   - Goal: Check whether reward trend and action behavior move in a plausible direction.
   - Run: 10-30 episodes.
   - Verify:
     - reward does not become NaN or explode.
     - delay/energy are finite.
     - action distribution does not collapse immediately without explanation.
     - if reward is flat, inspect actor/critic losses before changing reward/env.

7. [ ] **Compare e-ATN-MADDPG against enabled baselines only after smoke passes**
   - Goal: Check trend direction, not exact paper numbers.
   - Currently enabled comparison set:
     - Local Only
     - Edge Only
     - MAPPO
     - Graph-GAT MAPPO
     - e-ATN-MADDPG
     - MADDPG
     - MAAC
   - Implemented but currently disabled/not required:
     - Feature Extraction Edge
     - Random Offloading
   - Verify:
     - Learning algorithms run for the requested full episode count. `DONE`
     - Fixed baselines run for a shorter episode count and are extended only for plotting. `DONE`
     - e-ATN-MADDPG should improve over episodes.
     - If e-ATN-MADDPG cannot beat simple baselines, first inspect model training/update path before changing the frozen environment.

8. [ ] **Only then move toward paper-style long training**
   - Goal: Produce a training trend similar to the paper where the MADDPG-based method is best.
   - Verify:
     - Do not tune reward weights until baseline and smoke model runs are understood.
     - Do not claim paper reproduction from short runs.

9. [x] **Add GPU readiness check without training rewiring**
   - Goal: Prepare for Windows RTX 4080 training while keeping the Mac demo stable.
   - Files touched:
     - `utils/gpu_readiness.py`
     - `docs/windows_gpu_readiness.md`
     - `tests/test_gpu_readiness.py`
   - Verify:
     - readiness helper runs on CPU-only machines. `DONE`
     - Windows guide says to verify CUDA before changing training code. `DONE`
     - no agent, replay-buffer, environment, or comparison training logic is moved to GPU yet. `DONE`

10. [ ] **Only move one training path to CUDA after Windows readiness passes**
    - Goal: Avoid messy GPU bugs by proving the machine setup first.
    - First target selected by user:
      - `Graph-GAT MAPPO`
    - Verify before/after code changes:
      - `python -m utils.gpu_readiness --preferred-device cuda` reports `CUDA available: True`.
      - GPU name shows the server GPU.
      - Mac CPU demo still remains the control path.
    - Status:
      - [x] `GraphGATMAPPOAgent` accepts a `device` setting and moves encoder/actor/critic modules to that torch device.
      - [x] Graph-GAT rollout update tensors are created on the agent device.
      - [x] Graph tensors remain built from CPU environment state, then move to the agent device at encode/mask boundaries.
      - [x] `run_comparision.py` passes `graph_gat_device` from `utils/paper_config.py` only to `Graph-GAT MAPPO`.
      - [ ] Run 1-3 Graph-GAT MAPPO smoke episodes on the Windows GPU server.

11. [x] **Add GAT as task-priority option beside GCN**
    - Goal: Compare GCN priority extraction against a GAT priority extractor without removing the original GCN baseline.
    - Files touched:
      - `models/task_priority_gat.py`
      - `utils/experiment_setup.py`
      - `utils/gcn_training.py`
      - `run_comparision.py`
      - `main.py`
      - `tests/test_task_priority_gat.py`
    - Verify:
      - GAT returns one priority score per subtask. `DONE`
      - GAT can train against the current supervised priority targets. `DONE`
      - `build_priorities` accepts a GAT model through the existing `(features, adjacency)` interface. `DONE`
      - `priority_model = "gat"` uses a separate `gat_priority.pt` checkpoint. `DONE`
      - GCN remains the default unless `priority_model` is changed. `DONE`

---

## Historical Ablation Study Queue - Graph-GAT State Encoder

Historical purpose: test whether converting the flat environment state into a topology graph and encoding it with GAT gives DRL models a better state representation for task offloading. The current active version of this work is the pretrained Topology-GAT and topology stress-test plan above.

Working assumption:
- "Typology" means topology: device nodes, edge-server nodes, task/subtask features, connection-window features, queue/wait features, and feasible device-server links.
- This is a new ablation/proposal path, not a paper-faithful baseline.
- GAT is a reusable state encoder idea, not tied to only one DRL algorithm.
- First integration target is MAPPO because its rollout/PPO update path is now corrected and easier to extend with graph-state rollouts.
- Later integration targets can include e-ATN-MADDPG and MAAC after the MAPPO ablation is verified.
- Implementation choice: use Option A first, a separate `GraphGATMAPPOAgent`, while keeping reusable graph/GAT utilities. Do not refactor all DRL models or put trainable GAT only in the training loop yet.
- Rationale: a separate agent is safer for the first ablation because it does not risk breaking flat-state MAPPO, and it keeps GAT inside the model/update path so gradients can update GAT correctly.
- The first ablation comparison is:
  - flat-state MAPPO.
  - Graph-GAT MAPPO.

Historical note: this queue is preserved for traceability. Use the 2026-07-08 active research plan above for current work.

1. [x] **Define graph-state representation**
   - Goal: Decide exactly how to convert current flat state into graph data.
   - Candidate node types:
     - device node: local compute power, local wait, current subtask CPU/data/result, accumulated local context.
     - server node: edge compute power, edge wait.
   - Candidate edge types:
     - device-to-server edge when the connection window is valid.
     - edge features: normalized `l_start`, `l_end`, window length, optional transfer-rate estimate.
   - Verify:
     - [x] graph can be built from information already available in `DITENEnv._get_joint_state`.
     - [x] no environment physics changes.
     - [x] graph shape is stable from `joint_state`, `num_devices`, and `num_servers`.

2. [x] **Add graph builder tests before implementation**
   - Goal: Lock the graph API before adding model code.
   - Candidate file:
     - `tests/test_topology_graph_state.py`
   - Verify:
     - [x] node feature tensor shape is `(num_devices + num_servers, node_feature_dim)`.
     - [x] edge index shape is `(2, num_edges)`.
     - [x] edge feature tensor length matches `num_edges`.
     - [x] disconnected server windows are represented with explicit edge flags.

3. [x] **Implement minimal topology graph builder**
   - Goal: Convert existing flat joint state into graph tensors without touching env internals.
   - Candidate file:
     - `utils/topology_graph_state.py`
   - Rule:
     - read only the current `joint_state`, `num_devices`, and `num_servers`.
     - do not change `DITENEnv._get_joint_state`.
   - Verify:
     - [x] graph builder tests pass with `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_topology_graph_state.py -q -p no:cacheprovider`.

4. [x] **Add a small GAT encoder unit test**
   - Goal: Verify the encoder can consume graph tensors and produce one embedding per device.
   - Candidate files:
     - `models/topology_gat.py`
     - `tests/test_topology_gat.py`
   - Verify:
     - [x] output shape is `(num_devices, embedding_dim)`.
     - [x] gradients flow through the encoder.
     - [x] encoder works without `torch_geometric` unless the repo already depends on it.

5. [x] **Implement minimal GAT state encoder**
   - Goal: Create a small PyTorch-only GAT encoder to avoid new dependencies.
   - Candidate architecture:
     - one or two graph-attention layers.
     - masked attention over graph edges.
     - output only device-node embeddings for actor input.
   - Verify:
     - [x] unit tests pass with `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_topology_gat.py -q -p no:cacheprovider`.
     - [x] no training script changes yet.

6. [x] **Create Graph-GAT MAPPO ablation agent tests**
   - Goal: Define the first graph-state DRL integration around MAPPO before implementation.
   - Candidate files:
     - `tests/test_graph_gat_mappo.py`
     - `baselines/graph_gat_mappo.py`
   - Verify:
     - [x] action dimension remains `0=local, 1..S=edge server`.
     - [x] actor samples valid actions from graph-derived device embeddings.
     - [x] rollout buffer can store graph states, next graph states, actions, rewards, old log-probs, and done flags.
     - [x] GAT parameters change after one PPO rollout update.
     - [x] flat-state `MAPPOAgent` tests still pass unchanged.

7. [x] **Implement Graph-GAT MAPPO agent**
   - Goal: Add a separate MAPPO variant that consumes graph states through a GAT encoder.
   - Candidate name:
     - `GraphGATMAPPOAgent`
   - Candidate file:
     - `baselines/graph_gat_mappo.py`
   - Verify:
     - [x] uses `TopologyGATEncoder` to create one embedding per device.
     - [x] MAPPO actor uses each device embedding to sample an action.
     - [x] centralized critic uses graph/device embeddings.
     - [x] PPO update backpropagates through MAPPO and GAT.
     - [x] no changes are required in `baselines/mappo.py` for flat MAPPO behavior.
     - [x] related tests pass with `PYTHONDONTWRITEBYTECODE=1 /Users/admin/miniconda3/bin/pytest tests/test_topology_graph_state.py tests/test_topology_gat.py tests/test_graph_gat_mappo.py tests/test_mappo_update.py -q -p no:cacheprovider`.

8. [x] **Wire Graph-GAT MAPPO into comparison config behind a separate name**
   - Goal: Add the model as an optional comparison line, not as a replacement for flat MAPPO.
   - Candidate display name:
     - `Graph-GAT MAPPO`
   - Candidate file:
     - `run_comparision.py`
   - Verify:
     - [x] original flat-state `MAPPO` tests still pass.
     - [x] `Graph-GAT MAPPO` is registered as a separate comparison model.
     - [x] comparison loop can collect graph-state actions and update graph rollout.
     - [x] Graph-GAT timing diagnostics report graph build, action selection, update time, and transition count.
     - [x] Graph-GAT PPO update reuses each epoch embedding for both actor log-probabilities and critic value.
     - [x] Graph state includes task priority vector in device nodes and explicit connected/disconnected edge flags.
     - [x] Graph-GAT MAPPO masks disconnected server actions during sampling and PPO log-prob recomputation.
     - [ ] `Graph-GAT MAPPO` can run 1-3 smoke episodes.
     - [ ] JSONL saves a separate one-line final state for the ablation model.

9. [ ] **Run Graph-GAT MAPPO smoke test** (next)
   - Goal: Check integration before long training.
   - Run:
     - 1-3 episodes for `Graph-GAT MAPPO`.
   - Verify:
     - no crash.
     - reward/delay/energy finite.
     - graph rollout buffer fills.
     - PPO actor/critic/GAT update path runs.
     - Graph-GAT cost line shows whether graph build, action, or update is the bottleneck.
     - requested edge actions should no longer target disconnected servers.

10. [ ] **Run short MAPPO ablation comparison**
   - Goal: Compare trend, not final paper-quality numbers.
   - Compare:
     - flat-state MAPPO.
     - Graph-GAT MAPPO.
   - Run:
     - 10-30 episodes first.
   - Verify:
     - Graph-GAT MAPPO does not collapse to all-local/all-edge immediately.
     - Graph-GAT MAPPO has plausible action distribution.
     - if Graph-GAT MAPPO is worse, inspect graph features and attention before tuning reward.

11. [ ] **Attach reusable GAT encoder to other models only after MAPPO ablation works**
   - Goal: Extend the same graph-state encoder idea after the MAPPO path is stable.
    - Candidate next targets:
      - Graph-GAT e-ATN-MADDPG.
      - Graph-GAT MAAC.
    - Verify:
      - each new model is added as a separate comparison line.
      - the flat-state version remains as the control.
      - only consider a shared encoder abstraction after at least one graph model shows useful behavior.

12. [ ] **Run final ablation only after short comparison is stable**
    - Goal: Produce a clean ablation result.
    - Compare:
      - flat-state MAPPO control.
      - Graph-GAT MAPPO proposal.
      - optional Graph-GAT e-ATN-MADDPG only after separate smoke tests.
    - Verify:
      - plots, checkpoints, and JSONL include each enabled model as separate outputs.
      - result summary clearly states this is a proposed graph-state encoder ablation, not the original paper model.

---

## Current Status

- [x] Read the `AGENTS.md` rules supplied by the user; skip `CLAUDE.md` per user request.
- [x] Inspect repo structure and identify main code paths.
- [x] Read current environment code in `environment/diten_env.py`.
- [x] Read system model code in `environment/system_model.py`.
- [x] Read network model code in `environment/network_env.py`.
- [x] Read MADDPG code in `models/maddpg.py` and training update loops in `main.py` / `run_comparision.py`.
- [x] Read baseline scheduler code in `baselines/scheduling_baselines.py`.
- [x] Extract and inspect `baseline_paper.pdf`.
- [x] Confirm Table II experiment parameters from the rendered paper page.
- [x] Draft this plan/checklist.
- [x] Confirm baseline interpretation with user before writing code.
- [x] Verify and fix MADDPG actor update so actor parameters change after replay-buffer training.
- [x] Implement and test local-only, edge-only, and random offloading baseline policies.
- [x] Implement and test feature-extraction-only edge baseline.
- [x] Add timing diagnostics for local time, server time, transfer time, wait time, action counts, and penalty count/time.
- [x] Make comparison diagnostics readable with short progress postfix plus multi-line episode summaries.

---

## Code Understanding

### Main Training Path

- `main.py`
  - Function `train_maddpg` trains `EpsilonATNMADDPGAgent` over episodes, time slots, and subtask steps.
  - It generates one `TaskDAG` per device at each time slot, builds priorities using GCN, calls `env.start_time_slot`, collects joint actions, calls `env.step`, stores replay samples, and updates agents from replay.
  - State dimension is already dynamic through `env.get_state_dim()`.

- `run_comparision.py`
  - Function `train_algorithm` trains multiple algorithms on the same `DITENEnv` style.
  - Current comparisons include `e-ATN-MADDPG`, `MAAC`, and `MAPPO`.
  - Random and greedy currently mean priority-generation modes, not offloading-location baselines.

### Environment Path

- `environment/diten_env.py`
  - Class `DITENEnv` owns slot state, connection windows, queues, accumulated delay/energy, and reward.
  - `_get_joint_state` already attempts Eq. 23: local compute/wait, current subtask `(C, D, R)`, priority sequence, edge compute/wait, connection start/end windows.
  - `_calculate_reward` already follows Eq. 24 form.
  - `_update_connection_windows` uses subslots to estimate `l_start` and `l_end`.
  - `_schedule_edge_items` currently falls back to local execution when an offload violates the connection window. This may conflict with paper constraints C1/C2, which define feasibility of offloaded execution.
  - `_calculate_result_transfer` handles local-to-edge and edge-to-local transfer, but not edge-to-different-edge transfer.

### MADDPG Path

- `models/maddpg.py`
  - `ActorNetwork` outputs a discrete action probability distribution.
  - `CriticNetwork` consumes joint states and one-hot joint actions and can use self-attention.
  - `EpsilonATNMADDPGAgent.select_action` uses epsilon-greedy exploration.

- `main.py::_update_agents_from_buffer` and `run_comparision.py::_update_agents_from_buffer`
  - Current critic update is structurally MADDPG-like.
  - High-risk issue: actor loss uses `argmax(agent.actor(...))` and one-hot conversion. `argmax` is non-differentiable, so the actor may not receive a useful gradient. This must be verified before trusting any training trend.

### Existing Tests

- `tests/` currently contains only `__pycache__` bytecode files, not source `.py` tests.
- `python3 -m pytest --version` failed because pytest is not installed in the active system Python.
- We should add small source tests before touching model/environment behavior.

---

## Paper Understanding

### Environment and MDP

- The scenario has `D` mobile industrial devices and `S` fixed edge servers.
- Each device generates an image-recognition task at the beginning of each time slot.
- Each task is a DAG with subtasks. Each subtask has CPU cycles `C`, data size `D`, and result size `R`.
- Offloading action is discrete:
  - `0`: process locally.
  - `j`: offload to edge server `s_j`.
- Mobility matters. Each device moves on a predetermined path. A device can offload to an edge server only when execution starts after `l_start` and finishes before `l_end`.
- State Eq. 23 contains local compute/wait, subtask features, execution priorities, edge compute/wait, and connection windows.
- Reward Eq. 24 rewards lower immediate and accumulated delay/energy, plus a penalty for offloading outside the link window.
- Algorithm 1 loops over episode, time slot, and subtask index, then stores joint transition samples `(S, A, R, S')`.

### Confirmed Table II Parameters

- `training_episodes`: 1000
- `batch_size`: 64
- `t`: `{0, 1, ..., 49}`
- `E_max`: `1 J`
- `r`: `12 m`
- `B`: `10 MHz`
- `p_i`: `0.5 W`
- `P_j`: `1.2 W`
- `N_0`: `-43 dBm`
- `T_max`: `1 s`
- `f_i^loc`: `[0.8, 1.2] GHz`
- `f_j^edge`: `[2.3, 2.5] GHz`
- `tau_i^local`: `10^-28`
- `tau_j^edge`: `10^-27`

### Paper Baselines and Expected Trend

- Priority comparison: GCN scheduling should beat greedy scheduling, which should beat random scheduling.
- DRL comparison: epsilon + attention MADDPG should have better reward, delay, and energy than MAAC, MAPPO, MADDPG, GR-MADDPG, and ATN-MADDPG.
- For our requested minimum scope, first compare MADDPG/e-ATN-MADDPG against simple offloading baselines:
  - local-only
  - edge-only
  - random offloading

---

## Assumptions and Tradeoffs

- The paper does not fully specify reward weights `lambda1` through `lambda5` or `p_out` magnitude. We should keep these configurable and avoid tuning them until invariants pass.
- The paper does not specify the exact channel gain model. Current code uses distance to the power of `-2`; keep it unless it blocks trend reproduction.
- The paper does not fully specify how KolektorSDD image properties map to subtask `C`, `D`, and `R`. Current code uses proportional heuristics; keep them unless environment checks show impossible scales.
- The paper says random scheduling ignores dependencies, but the environment still should not silently execute a subtask before an unfinished predecessor without recording that as invalid or blocked.
- Edge-only baseline needs clarification because there are three edge servers and a device may not be connected long enough to execute on every server.

---

## Historical Implementation Checklist

Historical note: this checklist documents the original paper-reimplementation audit. Open boxes here are not the current execution queue unless they are moved into the active research plan above.

### Phase 1: Verification Scaffolding Before Behavior Changes

- [ ] Add source tests under `tests/` for environment invariants.
  - Verify state dimension and state contents from `DITENEnv.get_state_dim` and `_get_joint_state`.
  - Verify Eq. 24 reward arithmetic in `_calculate_reward`.
  - Verify connection-window violation is observable in `last_step_metrics`.
  - Verify predecessor dependencies do not silently disappear when priority order is invalid.
- [x] Add a small gradient test for MADDPG actor update.
  - Verify actor parameters change after one replay-buffer update with a deterministic batch.
  - Fixed the actor-loss path in `main.py::_update_agents_from_buffer` and `run_comparision.py::_update_agents_from_buffer` by passing differentiable actor probabilities into the critic instead of non-differentiable `argmax` one-hot actions.
- [ ] Add a smoke script or test fixture that runs one tiny episode without dataset files.
  - Verify no crash.
  - Verify finite reward, delay, energy.
  - Verify `last_step_metrics` has one row per device per subtask step.

### Phase 2: Baseline Policies

- [x] Implement local-only baseline.
  - Scope: action policy only, not a new environment.
  - Expected behavior: every device always returns action `0`.
  - Verification: all `last_step_metrics[*]["action"]` values are `0`.

- [x] Implement edge-only baseline after confirming server-selection rule.
  - Option A: always use nearest edge server.
  - Option B: always use first valid edge server by connection window.
  - Option C: always use fixed server `1`.
  - Implemented default: first valid edge server by connection-window state, otherwise server `1`, because the current state vector does not include distance for nearest-server selection.
  - Verification: no local action unless the chosen edge action is explicitly rejected by the environment policy we decide.

- [x] Implement random offloading baseline after confirming interpretation.
  - Implemented default: random execution location from `{local, edge1, edge2, edge3}` with the same priority source as MADDPG, because this isolates offloading-location quality.
  - Verification: observed action histogram includes both local and edge actions over a short run.

- [x] Implement feature-extraction-only edge baseline.
  - Scope: comparison-path baseline only, not a DRL state/model change.
  - Implemented default: subtask `4` uses the same first-connected-edge rule as `EdgeOnlyAgent`; all other subtasks use local action `0`.
  - Function impact: `FeatureExtractionEdgeAgent.select_action_for_subtask` chooses edge only for the feature extraction subtask; `run_comparision._collect_joint_actions` passes current subtask id from `env.current_step` and `env.priorities` only to agents that expose `select_action_for_subtask`.
  - Diagnostic meaning: if this baseline loses to all-local, inspect reward weights, communication rate/energy, and transfer accounting before tuning DRL.

### Phase 3: MADDPG Correctness First

- [ ] Verify `EpsilonATNMADDPGAgent.select_action`.
  - Epsilon `1.0` should sample random actions.
  - Epsilon `0.0` should choose actor argmax.

- [ ] Verify critic update.
  - Critic loss should be finite.
  - Critic parameters should change after one update.

- [x] Verify actor update.
  - Actor parameters should change after one update.
  - Replaced the non-differentiable argmax actor-loss path with a minimal differentiable action-probability path.

- [ ] Verify replay buffer shapes.
  - State shape: `(batch, num_agents, state_dim)`.
  - Action shape after one-hot: `(batch, num_agents, action_dim)`.
  - Reward shape: `(batch, num_agents)`.

### Phase 4: Environment Audit and Minimal Fixes

- [ ] Verify mobility and connection windows.
  - `l_start` and `l_end` should be within the current slot.
  - Offloaded execution should start after `l_start` and finish before `l_end`.

- [ ] Verify waiting delay Eq. 7 and Eq. 12 behavior.
  - Local queue waits for previous local subtask on the same device.
  - Edge queue waits for previous subtask assigned to the same edge server.

- [ ] Verify predecessor-result transfer Eq. 8, Eq. 13, Eq. 15, Eq. 16, Eq. 17.
  - Local-to-edge upload is counted.
  - Edge-to-local download is counted.
  - Edge-to-different-edge transfer is either counted or explicitly justified as ignored according to paper assumptions.

- [ ] Decide whether invalid offload should reject, fallback to local, or execute with penalty.
  - Paper constraints C1/C2 say offloaded execution must satisfy the connection window.
  - Current code falls back to local with penalty.
  - Minimum-change path: keep fallback only if we document it as an environment repair action; otherwise change to explicit invalid action handling.

- [ ] Verify task constraints C3 and C6.
  - `T_max` and `E_max` should be both reward terms and feasibility checks.
  - Current code uses them in reward but not as explicit constraints.

### Phase 5: Comparison Run

- [x] Add diagnostic counters for baseline comparison runs.
  - `DITENEnv.last_step_metrics` now records requested action, resolved action, start/finish/current slot, execution time, local compute time, successful server compute time, attempted server compute time, transfer time, wait time, penalty flag, and penalty time.
  - `run_comparision._summarize_step_metrics` aggregates local time, server time, attempted server time, transfer time, wait time, requested local/edge counts, resolved local/edge counts, penalty count, and penalty time.
  - Training progress now shows compact reward/delay/energy, actual local/edge counts, and penalty count in the progress bar postfix.
  - Short runs now print one readable diagnostic block on the final episode, with requested actions, actual execution counts, penalties, and timing averages.

- [ ] Run a small deterministic comparison first.
  - Episodes: 3 to 10.
  - Purpose: catch crashes and impossible metric scales.
  - Expected: all methods produce finite reward, delay, and energy.

- [ ] Run a medium comparison.
  - Episodes: enough to see trend direction, not final paper reproduction.
  - Expected: MADDPG reward should improve over time if actor update is correct.

- [ ] Compare against local-only, edge-only, and random.
  - If MADDPG is worse than all simple baselines, stop tuning and re-audit environment.
  - If local-only or edge-only dominates unrealistically, inspect delay/energy scale, connection-window handling, and transfer-energy accounting.

- [ ] Only after correctness passes, run longer training toward paper-like trend.
  - Target trend: MADDPG/e-ATN-MADDPG best or improving toward best.
  - Exact numeric match is not required.

---

## Historical Immediate Next Step

Historical note: this was the immediate next step for the older paper-reimplementation phase, not the current active research plan.

Run the first frozen-env DRL task: verify the current `EpsilonATNMADDPGAgent` mechanics before any longer training run.

Scope:
- inspect and test `models/maddpg.py`, `models/replay_buffer.py`, and `_update_agents_from_buffer`;
- do not change `environment/diten_env.py`, `environment/network_env.py`, or `dataset/data_loader.py`;
- stop after the mechanics check so the user can verify before continuing.
