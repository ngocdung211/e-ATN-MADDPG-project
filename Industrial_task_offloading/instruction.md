# Copilot Implementation Guide: Environment Reimplementation for e-ATN-MADDPG Project

This document is a natural-language instruction contract for Copilot/Cursor. Do not generate code until you have read every section. Follow the sections in order.

## 1. Core Principles - You Must Follow These
1. **Think Before Coding**: Don't assume. Don't hide confusion. Surface tradeoffs. State assumptions explicitly. Present multiple interpretations. Push back when warranted. Stop when confused and ask for clarification.
2. **Simplicity First**: Minimum code that solves the problem. Nothing speculative. No features beyond what was asked. No abstractions for single-use code. No error handling for impossible scenarios. If 200 lines could be 50, rewrite it.
3. **Surgical Changes**: Touch only what you must. Clean up only your own mess. Don't "improve" adjacent code, comments, or formatting. Match existing style. Remove unused imports/variables/functions created by your changes.
4. **Goal-Driven Execution**: Define success criteria. Loop until verified. Transform imperative tasks into verifiable goals (e.g., 1. [Step] -> verify: [check]).

## 2. What You Are Building
The goal is to strictly reimplement the digital twin environment (`DITENEnv`) and system models to match the mathematical formulations described in the author's paper. 

Currently, the environment has "TODOs" and lacks accurate state space representations (missing connection times, subtask features), realistic slotted time mechanics, and strict adherence to the paper's reward function. You will refactor the environment and training scripts to faithfully simulate the author's exact setup without breaking the existing GCN and MADDPG model architectures.

## 3. Stage 1 — System Model and Environment Physics (Prerequisite)
The physical mechanics of the environment must be fixed to align with the paper before updating the state space.
**What Copilot should do:**
- **Files:** `environment/system_model.py` & `environment/diten_env.py`
- **Time Slots & Mobility:** The simulation operates over $T$ time slots (e.g., 50 slots). In each slot, industrial devices move along predetermined paths at 1 m/s. Implement sub-time slot tracking to calculate the exact start ($l_{start}$) and end ($l_{end}$) moments a device is within an edge server's coverage radius ($r=12m$) as described in Section III.B.
- **Waiting Delays:** Accurately implement the waiting delay calculation for both local ($W_{d_i}$) and edge servers ($W_{s_j}$) based on equations (7) and (12). A subtask cannot start until its predecessors finish AND the previous subtask assigned to that specific computing node finishes.
- **Constraints:** Enforce the rule that if a subtask is offloaded, its start and end execution times must fall strictly within the valid $l_{start}$ and $l_{end}$ connection window.

## 4. Stage 2 — State Space Expansion
The current state representation `[f_loc, w_d] + edge_f + edge_w` is incomplete and must be mapped to the paper's MDP formulation.
**What Copilot should do:**
- **File:** `environment/diten_env.py`
- **Action:** Rewrite `_get_joint_state()` to strictly match Equation (23).
- The state vector $s_{i,m}^t$ for an agent observing subtask $m$ must include:
  1. Local computing power ($f_{loc}$) and local waiting delay ($W_d$).
  2. Subtask features: CPU cycles ($C$), data size ($D$), result size ($R$).
  3. Execution priority sequence of the task DAG.
  4. Edge computing powers ($F_{edge}$) and edge waiting delays ($W_s$).
  5. Connection start times ($l_{start}$) and end times ($l_{end}$) for all servers.
- **Constraint:** Ensure all components are flattened into a 1D vector and reasonably normalized to prevent gradient explosion. 

## 5. Stage 3 — Reward Function Correction
The current reward function uses arbitrary hardcoded lambdas and inaccurate penalty logic.
**What Copilot should do:**
- **File:** `environment/diten_env.py`
- **Action:** Rewrite `_calculate_reward()` to strictly implement Equation (24).
- $r = \lambda_1 (\frac{T_{max}}{M} - T_{im}) + \lambda_2 (T_{max} - T_{accm}) + \lambda_3 (\frac{E_{max}}{M} - E_{im}) + \lambda_4 (E_{max} - E_{accm}) + \lambda_5 p_{out}$
- Remove the hardcoded magic numbers (like `l3=2`). Make $\lambda_1$ through $\lambda_5$ configurable parameters initialized in the environment.
- Implement $p_{out}$ strictly as a penalty applied *only* when the subtask execution time violates the connection window constraint established in Stage 1.

## 6. Stage 4 — Training Loop Alignment
Integrate the fixed environment into the main execution scripts.
**What Copilot should do:**
- **Files:** `main.py` and `run_comparision.py`
- Dynamically calculate the new `STATE_DIM` based on the expanded state vector from Stage 2. Update the agent initializations to use this new dimension.
- Review the step logic. Ensure that the environment iterates correctly through the subtasks ordered by the GCN priority list.
- Remove any unused dead code created by your modifications. Ensure the loops correctly record Delay, Energy, and Reward for the plotter.

## 7. File Summary
| File | Action | Purpose |
|------|--------|---------|
| `environment/system_model.py` | Edit | Add mobility sub-time slot and waiting queue tracking. |
| `environment/diten_env.py` | Edit | Rewrite step logic, state vector (`_get_joint_state`), and reward function (Eq 24). |
| `main.py` | Edit | Update `STATE_DIM` and ensure training loop handles new env physics. |
| `run_comparision.py` | Edit | Update `STATE_DIM` and align evaluation logic with the fixed env. |