# Paper Reimplementation Plan and Checklist

**Goal:** Reimplement the paper environment and MADDPG path with minimum code changes, then compare MADDPG against local-only, edge-only, and random baselines.

**Success criteria:**
- The environment follows the paper's task, mobility, communication, computation, MDP state, action, reward, and constraint definitions closely enough that training results are meaningful.
- MADDPG is verified first in isolation before comparing against baselines.
- Baseline comparisons are only trusted after environment invariants pass.
- If MADDPG cannot beat simple baselines after the model update path is verified, treat that as evidence to re-audit the environment against the paper before tuning the model.

**Working rule:** Every small task updates this checklist. Code changes should be surgical and limited to functions needed for the current verified goal.

---

## Current Checkpoint - 2026-06-11

**Status:** User manually fixed the environment, and the baseline instrumentation is ready to verify whether the environment is now good enough for model training.

What is now implemented:
- [x] Local-only baseline.
- [x] Edge-only baseline.
- [x] Random offloading baseline.
- [x] Feature-extraction-only edge baseline.
- [x] Timing diagnostics for local compute time, successful server compute time, attempted server compute time, transfer time, wait time, penalty count, and penalty time.
- [x] Action diagnostics for requested local/edge counts and actual resolved local/edge counts.
- [x] Readable comparison output for short runs, printed only on the final episode.
- [x] Environment logic was manually adjusted by the user and is reported as good enough for the model to run.

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

## Next Task Queue

Stop after each task and wait for user verification.

1. [ ] **Verify the user-fixed environment with a short baseline run**
   - Goal: Confirm the updated environment behaves plausibly before touching DRL again.
   - Run: 10 episodes for Local Only, Edge Only, Feature Extraction Edge, and Random Offloading.
   - Verify:
     - All methods produce finite reward, delay, and energy.
     - Local Only has `Requested actions: edge=0`.
     - Feature Extraction Edge should not be worse than Local Only if offloading feature extraction is physically beneficial under the fixed env.
     - If Feature Extraction Edge still loses to Local Only, inspect the final diagnostics before changing reward weights: penalty count/time, transfer time, server time, and wait time.

2. [ ] **Freeze the environment for the first model sanity run**
   - Goal: Stop changing environment physics while checking whether MADDPG can learn.
   - Verify:
     - No more env changes unless the baseline diagnostics show a concrete environment bug.
     - Keep reward weights and communication model unchanged for the first sanity run.

3. [ ] **Run MADDPG/e-ATN-MADDPG smoke training**
   - Goal: Verify model execution path runs on the fixed environment.
   - Run: very small episode count first, then stop.
   - Verify:
     - No crash.
     - Replay buffer receives samples.
     - Actor update path is active.
     - Reward, delay, energy are finite.
     - Final diagnostics show meaningful requested/actual local-edge counts.

4. [ ] **Compare MADDPG against simple baselines only after smoke passes**
   - Goal: Check trend direction, not exact paper numbers.
   - Compare against:
     - Local Only
     - Edge Only
     - Feature Extraction Edge
     - Random Offloading
   - Verify:
     - MADDPG should improve over episodes.
     - If MADDPG cannot beat simple baselines, first inspect model training/update path before changing the environment again.

5. [ ] **Only then move toward paper-style long training**
   - Goal: Produce a training trend similar to the paper where the MADDPG-based method is best.
   - Verify:
     - Do not tune reward weights until baseline and smoke model runs are understood.
     - Do not claim paper reproduction from short runs.

---

## Current Status

- [x] Read `AGENTS.md` and `CLAUDE.md`.
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

## Implementation Checklist

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

## Immediate Next Question

Before writing baseline code, confirm what "random" means:

1. Random offloading location only, while keeping the same GCN priority order as MADDPG.
2. Random subtask priority only, while keeping the same model/policy.
3. Both random priority and random offloading location.

Recommended choice: option 1, because it isolates whether MADDPG learns better offloading decisions than a random action policy.
