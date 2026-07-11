# Paper Review - 2026-06-15

Review target:
- User PDF: `/Users/admin/Downloads/Graph_Attention_and_DRL_for_Task_Offloading_in_Industrial_Image_Recognition_over_Dynamic_Digital_Twin_Edge_Networks (1).pdf`
- Repo plan: `paper_reimplementation_plan.md`
- Main experiment code: `run_comparision.py`, `utils/paper_config.py`
- Result artifact matching the PDF table: `results/runs/2026-06-14_12/2026-06-14_12-48-54_last_training_state.jsonl`

## Summary

The paper direction is coherent: it frames the project as a GAT-enhanced MAPPO offloading method with coverage-aware action masking, and the reported Table III values are traceable to an existing run artifact.

The main risk is provenance drift. The PDF, repo LaTeX source, current plan, and current comparison code are not all describing the same experiment state. Before submission, make the paper source and experiment claim match the exact run you want to defend.

## Must Fix Before Submission

1. The repo paper source is stale.
   - `Graph_Attention_and_DRL_for_Task_Offloading_in_Industrial_Image_Recognition_over_Dynamic_Digital_Twin_Edge_Networks/Final_report.tex` does not match the PDF in Downloads.
   - The repo source still has the older title and no experimental results section.
   - The in-repo `conference_101719.pdf` is still the IEEE template, not the project paper.
   - Fix: put the real current `.tex` source in the repo and rebuild the repo PDF from that source.

2. Table III must cite its exact result source.
   - The PDF Table III matches `results/runs/2026-06-14_12/2026-06-14_12-48-54_last_training_state.jsonl`.
   - That run reports learning models at episode 500 and fixed baselines at episode 5.
   - Current `utils/paper_config.py` now has `comparison_full_episodes = 1000`, and `results/runs/2026-06-15_05/2026-06-15_05-01-24_last_training_state.jsonl` reports different final values.
   - Fix: either keep the paper as a 500-episode report and mention the exact `2026-06-14_12` run, or update the paper tables/figures to the current 1000-episode run.

3. The plan and current code disagree about enabled algorithms.
   - `paper_reimplementation_plan.md` says the enabled comparison includes `e-ATN-MADDPG` and `Graph-GAT MAPPO`.
   - `run_comparision.py::build_algorithm_configs()` currently comments out `e-ATN-MADDPG`.
   - Current active graph methods are named `GAT-MAPPO` and `GAT-Mask MAPPO`.
   - Fix: update the plan to reflect the actual paper scope, or re-enable and verify `e-ATN-MADDPG` before claiming it is part of the current comparison.

4. Do not overclaim the proposed GAT components.
   - Code supports a task-priority GAT through `models/task_priority_gat.py` and `priority_model = "gat"`.
   - Code supports topology graph encoding through `baselines/graph_gat_mappo.py` and `models/topology_gat.py`.
   - The paper should describe this as task-priority GAT plus topology-state GAT for GAT-Mask MAPPO.
   - Avoid implying a fully integrated dual-GAT architecture for every DRL baseline.

5. Clean `references.bib`.
   - `references.bib` contains pasted `@font-face` browser-extension blocks.
   - These are not BibTeX entries and should be removed before rebuilding.

## Paper Text Corrections

Suggested corrections:
- Replace broad claims such as "Experimental results demonstrate superior performance compared with existing methods" with "In the simulated comparison, GAT-Mask MAPPO outperforms MAPPO and the included baselines under the selected configuration."
- In Section V-D, say "learning-based methods are reported at episode 500" only if you keep the `2026-06-14_12` results.
- Add a short sentence near Table III: "Results are taken from the final state log of the `2026-06-14_12` run; fixed policies use 5 evaluation episodes and learning methods use 500 training episodes."
- Add Local Only and Edge Only to Table II, since the result table includes them.
- Rename "GAT-Mask MAPPO" consistently everywhere. Avoid mixing `Graph-GAT MAPPO`, `GAT-MAPPO`, and `GAT-Mask MAPPO` unless they are separate ablations.

## Visual/Layout Notes

- The PDF renders cleanly and is readable.
- Page 8 has a large algorithm block and two tables. It is readable, but the algorithm caption appears as `[t] Masked GAT-MAPPO...`, which looks like a LaTeX float option leaked into the caption. Fix the algorithm environment/caption.
- Figure 1 is readable, but the x-axis shows 0-500 episodes. This must match the table and text.
- The final references page is acceptable, but the last page is visually dense. Cleaning the bibliography source may improve rebuild stability.

## Recommended Next Step

First decide the canonical experiment:

- Option A: submit the 500-episode paper using `results/runs/2026-06-14_12` as the source of truth.
- Option B: update the paper to the newer 1000-episode run from `results/runs/2026-06-15_05`.

After that, sync these four things together: PDF, `.tex`, `paper_reimplementation_plan.md`, and the result artifact path used by Table III.
