TODOs tied to formulas / Algorithm 1 (actual mismatches or gaps):

Eq.(15)–(17) transmission modeling: The paper includes result‑down transmission energy/time and notes it may include edge‑to‑edge transmission. The current env only models device↔edge transfers and never models edge↔edge transfer; this undercounts both delay and energy.
Eq.(17) conditional result‑down energy: Result‑down is only needed when a successor subtask executes locally (or on a different edge). The code doesn’t check successor placement to decide whether to add result‑down energy/time.
Eq.(7) & Eq.(12) queueing delay definitions: Waiting delays are defined over scheduling orders (R_{d_i}^t, R_{s_j}^t). The current server queue order is an ad‑hoc sort by readiness; it is not derived from the scheduling order defined in the paper.
Constraint C1 (link window): If a subtask is offloaded, start and end must lie within ([l_{start}, l_{end}]). The current code falls back to local execution (with penalty) rather than enforcing the constraint as stated.
Constraints C3/C6 (delay/energy limits): The formulation explicitly constrains task delay/energy by (T_{max}, E_{max}); the code only uses these in the reward, not as feasibility constraints.
Algorithm 1 interpretation that must be kept exact:

Nested loops are episode → time slot (t) → subtask (m) with epsilon‑greedy action selection and replay‑buffer updates. The implementation should preserve this ordering and define (S = {s_{1,m}^t,\dots,s_{D,m}^t}) at each subtask step.
Parameters explicitly given by Table II (OCR‑extracted):

Parameter	Paper value
training episodes	1000
batch size	64
time slots (t)	(t \in {0,\dots,49}) (i.e., (T=50))
(E_{max})	1 J
coverage radius (r)	12 m
bandwidth (B)	10 MHz
device power (p_i)	0.5 W
server power (P_j)	1.2 W
noise power (N_0)	−43 dBm
local CPU (f_i^{loc})	[0.8, 1.2] GHz
edge CPU (f_j^{edge})	[2.3, 2.5] GHz
(\tau_i^{loc})	(10^{-28})
(\tau_j^{edge})	(10^{-27})
slot duration (T_s)	1 s
Not specified in the paper (currently guessed in code):

(\lambda_1\ldots\lambda_5) reward weights and (p_{out}) magnitude (Eq.(24) gives form but no numbers).
Channel gain model (g_{i,j}) (code uses distance(^{-2})); the paper does not specify the path‑loss formula.
Exact DAG topology for the 5 subtasks and how image properties map to (C,D,R).
Estimation‑error model for (\tilde f^{loc}) / (\tilde f^{edge}) (distribution and scale).
Sub‑time‑slot discretization (subslot_count) used to compute (l_{start}, l_{end}).
Server scheduling order (R_{s_j}^t) when multiple devices offload to the same server in one slot.
Network architecture hyperparameters (hidden sizes, attention dimension, epsilon decay schedule), except for the learning rate explicitly discussed in the text.
If you want, I can now map each of these TODOs to the exact file/function locations and produce a prioritized fix plan without writing code.