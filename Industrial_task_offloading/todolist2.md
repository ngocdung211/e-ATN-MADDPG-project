Mapped each formula‑driven TODO to exact file/function locations and organized a prioritized fix plan below.

**Formula‑driven TODO map**

| TODO (paper ref) | Exact file/function location | What to change |
|---|---|---|
| Add **edge↔edge** transfer and **result‑down** energy/time (Eq.(15)–(17)) | `environment/diten_env.py` → `_calculate_result_transfer`, `_resolve_predecessor_ready_time`; `environment/network_env.py` → `get_uplink_rate`, `get_downlink_rate` | Extend transfer logic to handle edge↔edge paths and include result‑down energy/time per Eq.(16)–(17). |
| Apply **result‑down only when required** (Eq.(17)) | `environment/diten_env.py` → `_calculate_result_transfer` | Gate result‑down energy/time on successor placement (local or different edge). |
| Enforce **queueing delay definitions** based on scheduling orders (Eq.(7), Eq.(12)) | `environment/diten_env.py` → `start_time_slot`, `step` (server grouping/sort), `_compute_waiting_delay_local`, `_compute_waiting_delay_server` | Build and use \(R_{d_i}^t, R_{s_j}^t\) rather than ad‑hoc ordering by readiness. |
| Enforce **C1 link window constraint** strictly | `environment/diten_env.py` → `step` (in_window check), `strict_connection_window` | Reject infeasible offloads instead of fallback‑to‑local; only apply penalty if defined by paper. |
| Enforce **C3/C6** (delay/energy constraints) | `environment/diten_env.py` → `step`, `_calculate_reward` | Add feasibility checks for \(T_{max}, E_{max}\) at task/slot level as constraints, not only in reward. |
| Inject **Table II** parameters (r, B, \(p_i\), \(P_j\), \(N_0\), \(f_{loc}\), \(f_{edge}\), \(\tau\), \(T_s\), episodes, batch size) | `main.py`, `run_comparision.py`, `environment/system_model.py` (device/server init), `environment/network_env.py` | Replace guessed constants with Table II values. |
| Set **\(E_{max}\) / \(T_{max}\)** from paper | `ultils/experiment_setup.py` → `build_task_dag`, `generate_task_dags_for_episode`; `main.py`, `run_comparision.py` | Pass paper values into each TaskDAG instead of defaults 1.0. |

**Assumptions to confirm (paper doesn’t specify numbers):**

| Assumption | File/function | Why it matters |
|---|---|---|
| Channel gain model \(g_{i,j}\) (distance‑power law) | `environment/network_env.py` → `calculate_channel_gain` | Paper only defines rate formula; path‑loss model is unspecified. |
| Subtask DAG structure + mapping from image properties to \(C,D,R\) | `dataset/data_loader.py` → `get_random_task_parameters`, `ultils/experiment_setup.py` → `DEFAULT_DAG_EDGES` | Paper defines \(C,D,R\) but not the concrete mapping from images. |
| \(R_{d_i}^t\) encoding in state (priority vector) | `environment/diten_env.py` → `_get_joint_state` | Eq.(23) includes \(R_{d_i}^t\) but not its numeric encoding. |
| Estimation error model for \(\tilde f^{loc}, \tilde f^{edge}\) | `environment/diten_env.py` → `_sample_estimated_power` | Paper states estimation but not distribution. |
| \(\lambda_1…\lambda_5\), \(p_{out}\) | `DITENEnv` init in `main.py`, `run_comparision.py` | Eq.(24) doesn’t give numeric weights. |
| \(\epsilon\)-decay schedule | `models/maddpg.py` → `update_epsilon` | Algorithm describes \(\epsilon\)-greedy but not exact decay parameters. |

**Prioritized fix plan (no code):**
1. **Correct physics and feasibility**: implement Eq.(15)–(17) transfers, enforce C1 link window and C3/C6 constraints, and align queueing to \(R_{d_i}^t, R_{s_j}^t\).
2. **Parameter alignment**: wire Table II values everywhere (network, device/server, slot length, episodes, batch size) and inject \(T_{max}, E_{max}\) into TaskDAG construction.
3. **Assumption validation**: decide/justify the channel‑gain model, DAG structure + \(C,D,R\) mapping, \(R_{d_i}^t\) encoding, estimation‑error distribution, and \(\lambda\)/\(p_{out}\)/\(\epsilon\) schedules.