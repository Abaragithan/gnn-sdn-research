# Lightweight GNN delay models for SDN (RouteNet-Fermi teacher → Queue-Link Fusion student)

**Current work order: [`docs/dec1_execution_plan.md`](docs/dec1_execution_plan.md).** Read it first and follow its task order.

| Path | Contents |
|---|---|
| `delay_model.py` | RouteNet-Fermi (teacher and baseline), reference code |
| `student_delay_model.py` | Queue-Link Fusion student |
| `all_mixed/` | Data pipeline, baseline/teacher training and evaluation, v0 result JSONs |
| `all_mixed/student/` | Student training (scratch, KD) and evaluation |
| `all_mixed/evaluation/` | Parameter count, latency benchmark, other evaluations |
| `mininet_experiment/` | Mininet/Ryu emulation (see its README) |
| `report/` | LaTeX report (`main.tex`), edited on the report laptop only |
| `docs/research_progress_summary.md` | Experiment history and results |
| `logs/experiment_dataset_checkpoint_metadata.md` | Dataset splits, checkpoints and provenance |
| `runs/`, `results/dec1/` | Dec 1 runs and results (weights are git-ignored) |

The data (`data/all_mixed`, `data/all_mixed_v1`) and the checkpoints are not in Git. They live on the Fedora training machine.
