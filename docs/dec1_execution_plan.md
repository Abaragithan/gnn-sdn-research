# Dec 1 Execution Plan (Fedora machine)

**Written:** 22 Nov 2026 · **Supersedes:** `docs/strengthening_plan_dec2026.md`, `docs/research_plan_to_dec1.md`
**Deadlines:** presentation video uploaded **before 08:00 on 1 Dec** · printed report by 3 Dec
**Number lock:** **28 Nov, 08:00.** Results that are not finished by then do not go into the slides or report.

This file is the work order for the agent on the Fedora machine, which holds the data (`data/`), the checkpoints and the conda environment. Do the tasks **in order**. Every task has a "done when" check. Do not start a task until the check of the previous one passes, unless the task says it is independent.

---

## 0. Context: why these tasks

A review of the work against the report (22 Nov) found these problems. The tasks below fix the ones that can be fixed in time.

| # | Problem | Fixed by |
|---|---|---|
| P1 | The KD soft loss is MSE in seconds² (≈0.17), while the hard loss is MAPE in percent (≈7). The soft term was only ≈2.3% of the total loss, so student-KD was in effect a second scratch run. | T5, T6 |
| P2 | The scratch and KD students started from different random weights (`train_kd.py` builds and calls the teacher first). KD also selected its checkpoint with an unweighted validation mean; every other model used the flow-weighted Keras mean. | T5, T6 |
| P3 | Every model was trained once. The two identical baseline runs differed by ≈1 MAPE point, about the size of the KD gain (2.1 points). | T6 (2 paired seeds) |
| P4 | The 5.2× speed-up comes from one validation scenario whose size the report says was not recorded (an older doc says "9,900 flows", but no file supports it). Student-KD was never timed. "Most time is spent preparing inputs" was never measured. | T2, T3 |
| P5 | Hop-order bug in the reference converter. In `hypergraph_to_input_data`, `link_to_path` and `queue_to_path` come from in-edges after `nx.relabel_nodes`, so they follow global node order. `path_to_link`/`path_to_queue` positions come from out-edges, which follow true hop order. All models were trained with the scrambled order; the emulation uses true hop order. Measured on only one validation sample. | T2 (full test set), T4 (emulation) |
| P6 | Only one aggregate MAPE. Test delays are ≈12× smaller than training delays, so "larger topology" and "different delay regime" are mixed up. | T2 (per-size breakdown) |
| P7 | No ablation, so the accuracy loss can't be attributed to queue-link fusion versus making the model smaller. | T7 (optional) |
| P8 | No offline evidence that the models rank configurations correctly, only the single six-switch emulation. | T2 (ranking metrics) |

**Out of scope until after Dec 1:** matched-protocol baseline, Mininet re-runs, pruning/quantisation, jitter/loss, `routing_decision_eval.py`.

---

## 1. Ground rules

1. **Do not edit anything under `report/`.** The report is edited on the other laptop. Put every number the report needs into `results/dec1/SUMMARY.md` (T8).
2. **Do not change historical scripts or checkpoints** (`train_scratch.py`, `train_kd.py`, `train_teacher.py`, `few_shot_all.py`, `validation_all.py`, `model_worker.py`, any existing `ckpt*`). Write new scripts. The historical numbers must stay reproducible.
3. **The test split is for final evaluation only.** Never choose epochs, seeds, α or settings by looking at test results. Select by validation only.
4. **Nothing may be lost.**
   - Every run writes to its own `runs/<name>/` folder with `config.json`, `history.json` (updated every epoch) and `train.log` (via `tee`).
   - Never reuse a log file. The scratch student's log was lost this way before.
5. **Commit code, JSON and logs. Never commit checkpoints** (see the `.gitignore` entries for `runs/`). Commit after each task with a short message, then `git push` so the other laptop can pull.
6. **Python:** `/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python` (TF 2.6). Repo root: `~/gnn-sdn-research`. Below, `PY` means that interpreter.
7. **If a "done when" check fails, stop and write down what happened** in `results/dec1/SUMMARY.md` under "Problems". Do not work around it silently. Report honestly if any result goes against the expected direction.

8. **Review stops.** After writing the code for **T2, T3, T4, T5 and T7**, stop before running the full job. Commit and push the new files plus the output of the small check run (`--limit 5` or smoke test), then wait for the user to confirm.

| Stop | Before which run | Push for review |
|---|---|---|
| R1 (**required**) | T2 full overnight run | `eval_multi.py`, `hop_order.py`, `aggregate_eval.py`, and the `--limit 5` output next to the Keras `evaluate` values |
| R2 (recommended) | T3 benchmark | `latency_v2.py` and the output for one model on one scenario |
| R3 (recommended) | T4 re-score | `rescore_hop_order.py` and its JSON output (it runs in minutes, so run it, then wait for review before writing it into `SUMMARY.md`) |
| R4 (**required**) | T6 training queue | `train_student_v2.py`, the full smoke-test output (checks 1–3), and `config.json` from both smoke runs |
| R5 (**required**, if T7 is done) | T7 ablation run | the `delay_model.py` diff and the check that reloaded `ckpt-26` reproduces the teacher's T2 predictions |

9. **`runs/multiseed/` (seed-42 runs from 19 Nov) is archived, not used.** Those runs came from code changes (hop-ordered and data-ordered training) that were later reverted and never committed, so they can't be reproduced.
   - `git mv runs/multiseed runs/archive/multiseed_ordered_unreproducible`
   - Add a `NOTE.md` there explaining this. Results: scratch val 9.44 / test 10.30; KD (MSE) val 9.99 / test 11.00.
   - If the original code still exists anywhere on this machine, copy it into that folder and say so in `NOTE.md`.
   - Do **not** reuse its initial weights, checkpoints or settings in T5–T8. Do not include it in `SUMMARY.md` tables. Mention it only under "Problems and deviations".

### Fixed checkpoints (v0)

| Name | Class | Checkpoint (relative to `all_mixed/`) | Test MAPE in report |
|---|---|---|---|
| baseline | `RouteNet_Fermi` | `ckpt_dirs/ckpt_dir_0_10000/15-6.41` | 8.4011 |
| teacher | `RouteNet_Fermi` | `ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26` | 5.3734 |
| scratch | `StudentRouteNet(16, 4)` | `student/ckpt_dirs_scratch/ckpt-08` | 11.9244 |
| kd | `StudentRouteNet(16, 4)` | `student/ckpt_dirs_kd_alpha0.5/ckpt-19` | 9.8280 |

The test split is `data/all_mixed_v1/test` (478 scenarios, 200–300 nodes). The validation split is `data/all_mixed_v1/validation` (2,191 scenarios).

---

## 2. Schedule

| When | Task | Machine time |
|---|---|---|
| 22 Nov, evening | T1 preflight, T2 code + 5-scenario check, **start T2 full run overnight** | ~4–5 h |
| 23 Nov | T3, T4 (while T2 finishes or after it); T5 code + smoke test | ~2 h |
| 23 Nov, evening → 27 Nov | T6 training queue (≈41 h sequential), then T7 if time allows | ~41–60 h |
| 27 Nov, by 20:00 at the latest | **Start T8 final evaluation** of every finished run | ~4 h |
| 28 Nov, 08:00 | T8 summary committed and pushed. **Lock.** | — |

**Parallelism:** while the first T6 job runs, watch `top`. If one job uses less than ~50% of total CPU, run two T6 jobs at once (e.g. scratch_s2 alongside kdrel_s1). Otherwise run them one at a time. Note the decision in `SUMMARY.md`. **Timing benchmarks (T3) must run with nothing else on the machine.**

---

## T1. Preflight (≈30 min)

- [x] `git pull`. Check that you are on `main` and that this plan is present.
- [x] Write machine specs to `results/dec1/machine_specs.txt`: `lscpu`, `free -h`, `nproc`, `df -h .`, and the TF and networkx versions printed by `PY`.
- [x] Count samples, using `all_mixed/check_teacher_counts.py` or an equivalent loop over `DatanetAPI`. Expect train 9,972, validation 2,191, test 478. Write the counts to `machine_specs.txt`.
- [x] Create `results/dec1/` and `runs/`.

**Done when:** the counts match. If they don't, stop.

**Completed: 22 Nov 2026.** On `main`; `git pull` reported already up to date. Counts from `DatanetAPI` on `data/all_mixed_v1`: train **9,972**, validation **2,191**, test **478** — all match. Machine specifications and counts are recorded in `results/dec1/machine_specs.txt` (TensorFlow **2.6.5**, NetworkX **2.6.3**). T1 passed.

---

## T2. Single-pass multi-model evaluation of the test set (fixes P4, P5, P6, P8)

New file: `all_mixed/evaluation/eval_multi.py`.

### What it does
Reads each test scenario **once**, builds its input tensors, and runs every model on it. It writes one JSON line per scenario, so the run can resume after a crash.

### Arguments
- `--split` (path; default the test split)
- `--models` (a list of `name:class:checkpoint`, where class is `fermi` or `student`). Default: the four v0 checkpoints above. Load checkpoints by **exact path**, never with `latest_checkpoint`.
- `--orders` (`ref`, `hop` or `ref,hop`; default `ref,hop`)
- `--limit N` (for the smoke test)
- `--out` (a `.jsonl` file; on restart, skip scenarios already written)

### How to build the inputs
- Iterate `DatanetAPI(split, shuffle=False)` directly instead of `input_fn`, so that you have the sample object. Per sample:
  - `G = nx.DiGraph(sample.get_topology_object())`
  - `n_nodes = G.number_of_nodes()`
  - a scenario id that includes the topology folder or graph file (see `sample._get_data_set_file_name()` and the sample's graph attributes in `datanetAPI.py`)
  - `HG = network_to_hypergraph(G, R, T, P)`
- **`ref` order:** `hypergraph_to_input_data(HG)`, exactly as training did.
- **`hop` order:** the same `raw` dict, but with `link_to_path` and `queue_to_path` rebuilt from `HG.out_edges(p)`. Copy the logic of `convert_inputs()` in `mininet_experiment/model_worker.py` (lines ~70–100), including its consistency check. **Do not import `model_worker.py`**: it sets TF to one thread and imports the Mininet settings. Put the copied function in `all_mixed/evaluation/hop_order.py` with a comment saying where it came from.
- Convert to tensors with exactly the dtypes and ragged ranks of `input_fn`'s `output_signature` in `all_mixed/data_generator.py`. Drop `delay` from the inputs; it is the label.
- **Time the construction:** wall time of `network_to_hypergraph` plus `hypergraph_to_input_data` plus the tensor conversion, as `construct_ms`.
- **Retracing:** the models' `call` is a `tf.function`, and inputs vary in shape. If per-call time keeps growing or tracing warnings appear, wrap the call with `tf.function(model.call, experimental_relax_shapes=True)`, or feed the tensors through a one-element `tf.data` dataset with the same signature as `input_fn`. Check that the predictions are unchanged.

### Per scenario, per model, per order, record
`n_flows`, `sum_ape` (Σ|y−ŷ|/|y|), `sum_ae`, `sum_se`, `mean_true`, `mean_pred`, `forward_ms`, `n_nonfinite`, `n_negative`.
Per scenario also record `scenario_id`, `topology_id`, `n_nodes`, `n_links`, `n_queues` and `construct_ms`.

### Aggregation
A separate script, `all_mixed/evaluation/aggregate_eval.py`, writes `results/dec1/eval_v0_test_summary.json`. Per model and order:
- **pooled MAPE** = 100·Σsum_ape / Σn_flows (this should equal the Keras flow-weighted number)
- unweighted mean of per-scenario MAPE
- pooled MAE and RMSE (in seconds and ms)
- pooled MAPE **per topology size** (200, 240, 260, 280, 300)
- **ranking within each topology:** across that topology's scenarios, the Spearman and Kendall correlation between `mean_pred` and `mean_true`, and the pairwise ordering accuracy (fraction of scenario pairs ordered correctly). Report the mean over topologies and each topology's values.
- **timing:** median and mean of `construct_ms` and `forward_ms`, and `forward / (construct + forward)`.

### Steps
1. [x] Run with `--limit 5`. Compare with Keras: `model.evaluate(input_fn(test).take(5))` for teacher and kd. The pooled MAPE must match to within 0.01.
2. [x] Full run overnight: `nohup PY eval_multi.py --out ../../results/dec1/eval_v0_test.jsonl > ../../results/dec1/eval_v0_test.log 2>&1 &`
3. [x] Aggregate.

**Done when (gate for everything that follows):** the pooled MAPE with `ref` order reproduces **8.4011 / 5.3734 / 11.9244 / 9.8280** to within 0.01, and the file has 478 scenarios. If any value is off, find out why before T6, because every later number depends on this evaluator.

**R1 review stop: 22 Nov 2026.** Implemented `eval_multi.py`, `hop_order.py` and `aggregate_eval.py`. The five-scenario smoke run completed for all four fixed checkpoints under both orders. A fixed input signature prevents graph-size retracing; its predictions matched the original model call for all four models under both orders (`rtol=1e-6`, `atol=1e-9`). The independent historical Keras pipeline reproduced the pooled reference-order metrics:

| Model | Five-scenario pooled MAPE (%) | Keras evaluate MAPE (%) | Absolute difference (points) | Check |
|---|---|---|---|---|
| teacher | 2.088734561 | 2.088734388 | 0.000000172 | PASS |
| kd | 3.237940309 | 3.237940073 | 0.000000236 | PASS |

Evidence: `results/dec1/eval_v0_test_smoke.jsonl`, `.meta.json`, `.keras_check.json` and `.log`; aggregation: `eval_v0_test_smoke_summary.json` and `eval_v0_test_smoke_aggregate.log`. `results/dec1/eval_t2_checks.log` records passing checks for flow weighting, ranking ties, undefined correlations, timing fractions and recovery of a partial JSONL write. Ranking metrics use NumPy; no SciPy installation is needed. Historical scripts and checkpoints are unchanged.

**R1 status: approved. The subsequent v2 smoke check and full evaluation are complete; see the T2 completion hand-off below.**

**R1 review result (22 Nov): approved. The three fixes below are already in the code, made during the review on the other laptop.** Do not redo them; read the diff so you know what changed.
1. **No crash on labels ≤ 0** (`eval_multi.py`). The old code raised an error, which would have stopped the overnight run, and resuming would have failed on the same scenario. APE now uses the Keras MAPE rule, `|y−ŷ| / max(|y|, 1e-7)` (constant `KERAS_EPSILON`). For positive labels the values are unchanged. Each record has `n_nonpositive_labels`, and a warning is printed. Only empty or non-finite labels still raise an error.
2. **Parsing is timed** (`eval_multi.py`). `parse_ms` = time of the `DatanetAPI` iterator's `next()` (tar read plus sample parsing). `construct_ms` now also includes `nx.DiGraph(...)`. Metadata is `schema_version: 2`, so old smoke files can't be resumed into a new run; use new output files.
3. **Ranking by topology folder** (`aggregate_eval.py`). `topology_id` is a graph file, i.e. topology plus scheduling configuration (`graph_topo-200-40-0.txt`, `-3`, `-4`), which gives tiny groups. The main ranking keys are now per topology folder (`ranking_by_topology`, `ranking_mean_over_topologies`, for example `200/ch22-testing-scheduling-topo-200-40`). The graph-file view is kept as `ranking_by_graph_config` / `ranking_mean_over_graph_configs`. Timing adds `parse` and `forward_fraction_of_end_to_end`. Tested on the existing smoke JSONL: pooled MAPE is unchanged and the 5 scenarios form one topology group. **Caveat for `SUMMARY.md`:** within a group, scenarios differ mostly in traffic intensity, so high ranking scores may be easy to reach. Report them with that caveat.

### T2 completion hand-off (22 Nov 2026)

**T2 COMPLETE — full-test reproduction gate PASS.** After R1 approval, the user ran the fresh schema-v2 smoke check, full evaluation and aggregation on the Fedora machine. No T2 rerun is needed.

- **Fresh v2 smoke check:** teacher and KD both passed the independent Keras comparison (absolute differences 0.000000172 and 0.000000236 MAPE points). `parse_ms` was present. Evidence: `results/dec1/eval_v0_test_smoke_v2.jsonl`, `eval_v0_test_smoke_v2.meta.json`, `eval_v0_test_smoke_v2.keras_check.json`, `eval_v0_test_smoke_v2.log` and `eval_v0_test_smoke_v2_summary.json`.
- **Full coverage:** 478 records with 478 unique scenario IDs; all four fixed v0 checkpoints under both `ref` and `hop` orders. Scenario counts by node count: 200 → 80; 240 → 160; 260 → 79; 280 → 79; 300 → 80. Each model/order covers **31,028,940 flows**.
- **Validity:** zero nonfinite predictions, zero negative predictions and zero nonpositive labels across the full run. The log ends with `Finished: 478 scenarios`.
- **Runtime:** approximately **6.8 hours**, measured from the metadata and final log modification times, longer than the planned 4–5 hours. Account for this when scheduling later evaluations.

| Model | Expected ref MAPE (%) | Obtained ref MAPE (%) | Hop MAPE (%) | Hop − ref (points) | Reproduction |
|---|---:|---:|---:|---:|---|
| baseline | 8.4011 | 8.401104 | 8.301324 | −0.099780 | PASS |
| teacher | 5.3734 | 5.373394 | 5.328621 | −0.044773 | PASS |
| scratch | 11.9244 | 11.924413 | 12.118968 | +0.194555 | PASS |
| kd | 9.8280 | 9.827972 | 9.922264 | +0.094292 | PASS |

All four reference-order results match their expected values within **0.01 MAPE points**. Correcting hop order slightly improves baseline and teacher MAPE and slightly worsens both students on this test set. These are aggregate effects; the per-size results are in the JSON summary.

**Full-run artifacts for the other agent:**

- `results/dec1/eval_v0_test.jsonl` — per-scenario results, both orders, including parsing/construction/forward timing.
- `results/dec1/eval_v0_test.meta.json` — schema v2, exact checkpoint paths and hashes, split and metric/timing definitions.
- `results/dec1/eval_v0_test.log` — full execution log.
- `results/dec1/eval_v0_test_summary.json` — final pooled MAPE, MAE/RMSE, per-size errors, both ranking views and timing summaries. Use this as the source for T8 and the report hand-off.

**Interpretation for T8:** main ranking metrics cover 12 topology-folder groups. Retain the review caveat that scenarios differ mostly in traffic intensity; high correlations do not establish routing-decision accuracy. Mean parsing time was about **2.520 s/scenario** and construction **16.995 s/scenario**; reference-order mean forward times were baseline **1.789 s**, teacher **1.812 s**, scratch **0.287 s**, KD **0.292 s**. These single-pass T2 timings do not replace the dedicated repeated T3 benchmark.

### T2 review findings (user/reviewer, 22 Nov): copy these into `SUMMARY.md` in T8
T2 results reviewed and accepted. These interpretations are fixed; use the same wording in `SUMMARY.md`:
1. **Hop order has a negligible effect.** True hop order changes test MAPE by at most 0.2 points for any model (baseline −0.10, teacher −0.04, scratch +0.19, KD +0.09). The "checked on one sample only" threat is closed: the effect was measured on all 478 test scenarios and is small.
2. **Error grows only slightly with topology size, and the model order is the same at every size.** From 200 to 300 nodes: baseline 8.25 → 8.87, teacher 5.32 → 5.58, scratch 11.83 → 12.57, KD 9.79 → 10.38. 240 nodes (160 scenarios, the largest group) is the easiest size for all models.
3. **The ranking metrics do not separate the models.** All four, including the baseline, reach Spearman ≈ 0.999–1.0 and pairwise accuracy ≈ 0.996–0.999. Scenarios in a topology differ mostly in traffic intensity. Report the metrics only as "all models track load correctly", never as evidence that one model ranks better or that routing decisions are accurate.
4. **Graph construction, not the model, dominates end-to-end time.** On test-size graphs the student's forward pass is ≈ 6.3× faster than the teacher's (0.287 s vs 1.812 s, mean). Each scenario also needs ≈ 2.52 s of parsing and ≈ 17.0 s of graph construction. End to end, the student is only ≈ 8% faster (≈ 19.8 s vs ≈ 21.3 s per scenario). The forward pass is ≈ 8.5% of the teacher's total time and ≈ 1.5% of the student's. The 5.2× figure must always be described as forward-pass only.
5. **Runtime note:** T2 took 6.8 h. T8 evaluates only the new students (cheap forward passes) with `--orders ref`, so expect ≈ 3–3.5 h on test plus ≈ 15–20 min on validation. Plan T8's start accordingly.

**T3/T4 accepted; T5/R4 approved. All four required T6 runs are complete. Validation gains are inconsistent across the two pairs; the user requested review before further work. See the T6 hand-off dated 25 Nov.**

---

## T3. Latency benchmark v2 (fixes P4; independent of T2, but nothing else may be running)

New file: `all_mixed/evaluation/latency_v2.py`. One **process per (model, thread setting)**, so that peak memory is per model. Drive it with a small shell loop.

- **Arguments:** `--model name:class:ckpt`, `--threads {default,1}`, `--out`. If `--threads 1`, set `TF_NUM_INTRAOP_THREADS=1` and `TF_NUM_INTEROP_THREADS=1` **before importing TensorFlow**.
- **Scenarios:**
  - (a) the **first scenario of the validation split**, as in the old benchmark: this reproduces 178.5 / 34.2 ms and finally records its size;
  - (b) the first test scenario of each size 200, 240, 260, 280 and 300.
  - Record `n_nodes`, `n_links`, `n_queues` and `n_flows` for each.
- **Models:** baseline, teacher, scratch, **kd**.
- **Per scenario:** construction time (3 repeats, median); forward pass (5 warm-up, 30 timed; mean, SD, median, p95); peak RSS via `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`.
- **Output:** `results/dec1/latency_v2.json`.

**Done when:** on scenario (a), default threads, the teacher and scratch means are within ~20% of 178.5 ms and 34.2 ms. If they aren't, record the difference; it may come from machine load. The JSON contains all 4 models × 2 thread settings × 6 scenarios.

### T3 implementation and checks

- Added `all_mixed/evaluation/latency_v2.py`: exact checkpoint loading; thread settings applied before TensorFlow imports; reference-order inputs; three construction repeats; five forward warm-ups and 30 timed, synchronized calls; raw timing samples, summary statistics, scenario identity/size and peak RSS. A fixed input signature is checked against the original model call before timing. Historical comparisons record deviations above 20% without silently discarding results.
- Added `all_mixed/evaluation/run_latency_v2.sh`: sequentially launches eight fresh worker processes (four models × two thread settings), each with its own JSON and log in a timestamped folder, then validates and merges all 48 combinations into `results/dec1/latency_v2.json`. Existing output files are refused. Peak RSS is explicitly the process-wide high-water mark, including earlier scenarios, not independent memory usage per scenario.
- Test scenario selection preserves DatanetAPI archive order and checks graph-file metadata to skip folders of other topology sizes; it does not depend on T2 prediction files. Construction timing excludes sample parsing; selection/parsing time is recorded separately.
- **Checks passed:** Python and Bash syntax; timing-statistic calculations; historical comparison thresholds; 48-combination merge; rejection of missing workers and inconsistent scenarios. These checks did not import TensorFlow or run model inference.
- **R2 small check completed:** teacher/default/validation passed (see review below). The subsequent full benchmark completed after the selector fix, with a scratch historical-timing deviation documented below. The user ran the benchmark commands. No commits or pushes made by the execution agent.

**Original R2 small-check command (already completed; preserve its outputs):**

```bash
set -o pipefail
python -u all_mixed/evaluation/latency_v2.py \
  --model teacher:fermi:all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26 \
  --threads default --scenarios validation \
  --out results/dec1/latency_v2_smoke.json \
  2>&1 | tee results/dec1/latency_v2_smoke.log
```

Share both new scripts, the smoke JSON and its log for R2 review. After review, the full queue command is `bash all_mixed/evaluation/run_latency_v2.sh`. Do not start that queue before the review.

### R2 review result (22 Nov): approved, start the full queue
- **Code accepted:**
  - thread settings applied before TensorFlow starts, with the actual values recorded;
  - one fresh process per model and thread setting;
  - the same fixed-signature check as T2;
  - `set -euo pipefail`, so the queue stops if any worker fails;
  - the merge refuses incomplete or mismatched workers.
- **Smoke check passed:** teacher, default threads, validation scenario: 184.2 ms vs 178.5 ms historically (+3.2%, within 20%).
- **The old benchmark scenario's size is now known:** `ch22-testing-scheduling-topo-100-20` (validation split), **100 nodes, 560 links, 1,484 queues, 9,900 flows**. Put this in `SUMMARY.md`. It replaces the report's "the size of the benchmark scenario was not recorded", and it confirms the old "9,900-flow" note in `docs/research_progress_summary.md`.
- **Run now, with no other job on the machine** (expected ≈ 1 h; the teacher on one thread with 300-node graphs is the slowest part):
  ```bash
  bash all_mixed/evaluation/run_latency_v2.sh
  ```
  Done when `results/dec1/latency_v2.json` exists with `n_benchmark_combinations: 48`, and the scratch validation mean (default threads) is within 20% of 34.2 ms. If a worker fails, report the error from its log. Don't rerun blindly.
- **In `SUMMARY.md`,** report forward and construction times per scenario size for both thread settings, and the end-to-end comparison (construction + forward) between teacher and student, consistent with T2 finding 4.

### T3 first full-queue attempt: selector failure fixed (22 Nov)

The queue started in `results/dec1/latency_v2_runs_20261122_205401/`. Baseline/default completed validation (mean forward **180.059 ms**), then stopped at `test200` with `DatanetException` from `set_files_to_process`. The bundled API shadows the built-in `tuple` in its validation loop and rejects valid tuples. The original R2 smoke covered only validation, so it did not exercise this branch.

Fixed only `latency_v2.py`: assign the selection copied from `get_available_files()` directly to the same `_selected_tuple_files` field used by the broken API method. Archive order is unchanged; historical `datanetAPI.py`, checkpoints and timing code are untouched. Read-only regression checks selected the first scenario at **all five sizes (200–300)** and matched each scenario ID against T2. Evidence: `results/dec1/latency_v2_selector_check.log`; no TensorFlow imports or inference in those checks. The failure and resolution are also recorded in `results/dec1/SUMMARY.md`.

**Subsequent action completed:** the user reran `bash all_mixed/evaluation/run_latency_v2.sh` in a fresh timestamped folder. The failed attempt and its logs were preserved. See the completed-run hand-off below. The execution agent did not restart the benchmark or commit changes.

### T3 completed-run hand-off (22 Nov 2026)

**All benchmark measurements completed; historical scratch timing has a recorded deviation.** `results/dec1/latency_v2.json` has `status: complete`, **8 complete workers and 48 combinations** (4 models × 2 thread settings × 6 scenarios). Per-worker JSON/logs are in `results/dec1/latency_v2_runs_20261122_212858/`. Elapsed wall time across the successful workers was approximately **1.54 h (92 min)**. No automatic rerun was performed after checking the results.

| Model | Validation forward mean, default threads (ms) | Historical mean (ms) | Difference | Within 20% |
|---|---:|---:|---:|---|
| teacher | 181.634 | 178.5 | +1.76% | PASS |
| scratch | 41.180 | 34.2 | +20.41% | NO — record deviation |
| kd | 41.383 | not previously measured | — | — |
| baseline | 181.191 | not part of this reproduction check | — | — |

The scratch upper 20% bound is **41.04 ms**; the measured mean is **41.1798 ms**, about **0.14 ms** above that bound. Do not label both historical checks as passing. The cause was not established; do not attribute it to machine load without evidence. Use the new measurements in the report and retain the original numbers as historical evidence. This deviation is recorded under Problems in `results/dec1/SUMMARY.md`.

| Model | 300-node mean forward, default threads (ms) | 300-node mean forward, one thread (ms) | Peak process RSS, default / one thread (MiB) |
|---|---:|---:|---:|
| baseline | 2693.997 | 7717.250 | 3522.8 / 2452.9 |
| teacher | 2748.769 | 7995.186 | 3587.6 / 2452.2 |
| scratch | 406.212 | 1059.903 | 3144.8 / 2266.3 |
| kd | 404.973 | 1085.581 | 3227.6 / 2268.0 |

Use the merged JSON for all six scenarios' construction medians, mean/SD/median/p95 forward times and raw repeats. Construction plus forward excludes dataset parsing; `selection_and_parse_ms` is separate and includes selection overhead. RSS values are cumulative process peaks including TensorFlow, construction and earlier scenarios, not model-weight size or isolated per-scenario allocations. Carry the scratch deviation and the resolved selector failure into T8's Problems and deviations. T4 subsequently completed its diagnostic; see its R3 hand-off below.

### T3 review findings (user/reviewer, 23 Nov): accepted; copy into `SUMMARY.md` in T8
- **Accepted.** All 48 combinations are complete. Parameter counts are 27,025 and 4,505. Construction times agree across the four model processes (within ≈ 10%), so the scenario selection is consistent. The selector workaround is justified: `DatanetAPI.set_files_to_process` really does shadow `tuple` with its loop variable, so its type check rejects every entry. The read-only selector check matched T2's first scenario for every size.
- **Scratch deviation (+20.4% vs the historical 34.2 ms) is accepted as a reported deviation, not a failure.** Use the new values everywhere. The historical 5.2× was a single earlier measurement; this benchmark is the reproducible one.
- **Forward-pass speed-up, teacher/student, default threads:** **4.4×** on the 100-node validation scenario (181.6 vs 41.2 ms); **5.7–7.9×** on the 200–300-node test scenarios. One thread: 5.6× and 6.4–7.5×. Student-KD is now measured and equals scratch (same architecture; differences ≤ 3%).
- **End to end (construction + forward):** the student is only **1.08–1.10× faster with default threads** and **1.20–1.28× with one thread**, because construction (1.5–24.7 s, growing with graph size) dominates. Consistent with T2 finding 4.
- **Memory:** peak process RSS is ≈ 3.1–3.6 GiB with default threads and ≈ 2.3–2.5 GiB with one thread. The student saves only ≈ 0.2–0.4 GiB, because input tensors and the TensorFlow runtime dominate. The 105.6 KB vs 17.6 KB weight size is not the memory that matters. Report RSS as a cumulative process peak.
- Forward time does not grow strictly with flow count (240 nodes is slower than 260); graph structure such as path lengths also matters. Report per-size values, not a fitted trend.
- **Report impact:** the abstract, §5.2 and the conclusion must replace "about 5.2 times faster" with the forward-pass range above, and add the end-to-end figure. Table 5.2's "not measured" for student-KD and the "size not recorded" sentence are now resolved (100 nodes, 9,900 flows).

### Order of work from here
1. **T3 full queue:** measurements completed; preserve the scratch historical-timing deviation above in the review/report hand-off.
2. **T4** emulation re-score (minutes, no sudo). It can run right after T3, or meanwhile if it is only scored after T3 finishes. It must not run while T3 is timing.
3. **T5 code + smoke tests, then stop at R4** for review. This is the critical path: **T6 training must start no later than 23 Nov, evening** (≈ 41 h queue, and T8 must start by 27 Nov, 20:00).
4. **T6** right after R4 approval, then T7 only if time allows, then T8.

---

## T4. Emulation re-scored with the training hop order (fixes P5 for the emulation; no sudo, no Mininet)

New file: `mininet_experiment/rescore_hop_order.py`. This is a diagnostic, not part of the five runtime files.

- Import `scenario`, `build_inputs`, `load_model`, `prediction` and `physical_checks` from `model_worker.py` (no changes to that file).
- For each of teacher, scratch and kd, and for each candidate k = 0, 1, 2, under both `preserve_hop_order=True` and `preserve_hop_order=False`:
  - the foreground predicted delay (ms),
  - the physical check result,
  - the predicted ranking and the selected candidate.
- **Measured trial means**, for the APE: take them from `mininet_experiment/outputs/six_validation_summary.json` if present. Otherwise use the report's table: 123.785 / 112.929 / 196.331 ms for candidates 0 / 1 / 2.
- Check that the `True` order reproduces the frozen predictions in `outputs/six_validation_predictions.json`, if present.
- **Output:** `results/dec1/emulation_hop_order.json`, plus a line in `SUMMARY.md` saying whether the ranking 1, 0, 2 and the choice of candidate 1 survive under the training order.

**Done when:** both orders are scored for all 3 models, and the `True` order matches the frozen predictions.

### T4 diagnostic completed (22 Nov); R3 approved (23 Nov 2026)

Added and ran `mininet_experiment/rescore_hop_order.py`, importing the unchanged worker's `scenario`, `build_inputs`, `load_model`, `prediction` and `physical_checks`. No Mininet, sudo, servers, network trials or training were run. All **18 candidate/model/order combinations** were scored (3 candidates × 3 models × 2 orders).

**Preflight issue and resolution:** the first attempt stopped on a scenario hash mismatch. The frozen scenario includes `measurement_s: 60.0`; `settings.scenario()` omits it. Every other field matched. The script now verifies all model-input fields against the current scenario, ignores only `measurement_s` for that comparison, then uses the exact frozen scenario and its hash. It also checks checkpoint hashes and the measurement summary's prediction-source hash. No historical files were edited. Preserve `results/dec1/emulation_hop_order_preflight_failed.json` and `.log`; the problem and resolution are recorded in `SUMMARY.md` without adding T4 conclusions before R3.

**Gate checks PASS:** both orders scored for every model; all 18 physical checks valid; all nine true-hop-order foreground predictions exactly reproduce the frozen values (maximum absolute difference **0.0 ms** for each model). Under both orders, every model ranks the candidates **[1, 0, 2]** and selects **candidate 1**.

| Model | APE vs 15 measured foreground trial means, true hop order (%) | APE vs the same trial means, training/ref order (%) | Ranking / selected candidate, both orders |
|---|---:|---:|---|
| teacher | 7.198059 | 7.246069 | [1, 0, 2] / 1 |
| scratch | 5.280694 | 5.283797 | [1, 0, 2] / 1 |
| kd | 6.770033 | 6.739159 | [1, 0, 2] / 1 |

Measured candidate means came from `mininet_experiment/outputs/six_validation_summary.json`: **123.785039 / 112.929213 / 196.331203 ms** for candidates 0 / 1 / 2, with five trial means per candidate. These APEs use the 15 foreground trial means, not all flows or training seeds. This is one fixed six-switch scenario and does not establish general routing-decision accuracy.

**Review artifacts:**

- `mininet_experiment/rescore_hop_order.py` — diagnostic code.
- `results/dec1/emulation_hop_order.json` — exact per-candidate delays, APEs, physical checks, rankings/selections, source hashes, frozen checks and `review_status: pending_R3`.
- `results/dec1/emulation_hop_order.log` — successful execution log.
- `results/dec1/emulation_hop_order_preflight_failed.json` and `.log` — preserved preflight failure.

**R3 complete:** review approved on 23 Nov (see below), and the approved T4 conclusion and APE table have been added to `results/dec1/SUMMARY.md`. The diagnostic JSON retains its original generation-time `review_status: pending_R3`; this plan records the subsequent approval. No commits or pushes made by the execution agent. The reviewer supplied T5's script; its smoke tests subsequently passed and are awaiting R4 (see T5 hand-off).

### R3 review result (user/reviewer, 23 Nov): approved; add to `SUMMARY.md` now
- **Accepted.**
  - The preflight fix is justified: `build_inputs` never reads `measurement_s`, and every other field is verified.
  - True-hop predictions reproduce the frozen predictions exactly (max difference 0.0 ms) for all three models.
  - Measured means come from the real 15-trial summary (5 per candidate), not the fallback.
- **Conclusion for `SUMMARY.md`:** with the training-time (reference) hop order, all three models still rank the candidates 1, 0, 2, still select candidate 1, and pass all physical checks. Foreground predictions change by at most 0.21 ms. Mean APE is essentially unchanged: teacher 7.20 → 7.25%, scratch 5.28 → 5.28%, KD 6.77 → 6.74%. The emulation result does not depend on the input-order convention.
- **Report impact:** §3.7's "The effect … on the emulation's predictions was not measured" and the matching threat and future-work items are resolved. With T2 finding 1, the hop-order issue can be described as measured and negligible offline and in the emulation.

**Next task: T5** (code and smoke tests), then **stop at R4**. This is now the critical path: T6 must start by **23 Nov, evening**.

---

## T5. Training script v2 for students (fixes P1, P2)

> **Update 23 Nov: the script has already been written by the reviewer**: `all_mixed/student/train_student_v2.py` (≈ 215 lines). **Do not rewrite or restructure it.** Your task in T5 is only to run the smoke tests below and report the output at R4.
> - If it fails on this machine, make the **smallest fix** that makes it work, and list every changed line and the reason in `SUMMARY.md` under "Problems and deviations".
> - No refactoring, no extra features, no style changes.
> - The specification below is kept for reference. Deviations already in the script, all intended:
>   - **Soft MAPE uses the Keras rule** (`max(|t|, 1e-7)`): it reuses `MeanAbsolutePercentageError` with the teacher as the target.
>   - **No `restore_best_weights`:** the best student weights are saved on every improvement and recorded in `best.json`, which is the checkpoint to use.
>   - **Proof of identical starting weights:** `config.json` records the SHA-256 of the init files, and the log prints the first-step hard loss. A separate `np.allclose` script is not needed.
>   - **Crash restart:** use `--resume-from <ckpt>` with a **new** `--out-dir`, because the script refuses to reuse a folder that already has a `config.json`.
>   - `PYTHONHASHSEED` is not set: setting it inside a running process has no effect.

New file: `all_mixed/student/train_student_v2.py`. **Scratch and KD use the same script and the same `train_step`.** Scratch is simply α = 1 with no teacher.

### Arguments
`--mode {scratch,kd}`, `--soft-loss {mse,mape}`, `--alpha` (default 0.5; forced to 1.0 for scratch), `--seed`, `--data-seed` (default = seed), `--out-dir`, `--epochs 30`, `--patience 5`, `--steps-per-epoch 9972`, and for smoke tests only `--max-steps` and `--val-take`.

### Seeds and identical starting weights
- Parse the arguments, then set `PYTHONHASHSEED`, `random`, `np.random` and `tf.random` from `--seed` **before** building any model.
- Starting weights:
  1. Build `StudentRouteNet(16, 4)` and call it once on the first validation sample.
  2. If `runs/init/student_init_s{seed}.index` exists, **load it**. Otherwise save the current weights there.
  3. Only after this, build and load the teacher (kd mode), from the exact path `all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26`, with `teacher.trainable = False`.
- This makes scratch_sN and kd_sN start from identical weights whichever runs first.
- Training data: `input_fn(train, seed=data_seed, shuffle=True).repeat()`, so that paired runs also see the same data order.

### Loss (custom `train_step`)
- `hard = MAPE(y, s)`, using `tf.keras.losses.MeanAbsolutePercentageError()`.
- `soft`:
  - `mse` → `MSE(t, s)` (the original loss, for the paired replication);
  - `mape` → `100 * mean(|t − s| / max(|t|, 1e-9))`, so that both terms are on the same percent scale.
- `loss = α·hard + (1−α)·soft`. In scratch mode, skip the teacher call entirely.
- Track `loss`, `hard` and `soft`.

### Validation and selection (same for every run)
- In `test_step`, accumulate `Sum(|y−s|/|y|)` and `Sum(n_flows)` with two `tf.keras.metrics.Sum` trackers, and return `mape_flow = 100·sum/count`.
- Also return the old unweighted mean as `mape_unweighted`, for comparison only.
- Put **all** trackers in the `metrics` property so that they reset each epoch.
- Keras will log the values as `val_mape_flow` and `val_mape_unweighted`.
- **Checkpoint and early stopping:**
  - Monitor `val_mape_flow` (mode `min`, patience 5, max 30 epochs, `restore_best_weights=True`).
  - Save the student weights only on improvement, to `<out-dir>/ckpt-XX`.
  - Write `<out-dir>/best.json` with the epoch and value.
- **Each epoch:** append to `<out-dir>/history.json`. At the start, write `<out-dir>/config.json` with all arguments, the git commit hash and the teacher checkpoint path.

### Smoke test (all must pass before T6)
Run from the repo root (`PY` = the conda Python). Use **seed 99** so that the smoke runs never create the init files of the real seeds 1 and 2:
```bash
mkdir -p runs/smoke_scratch runs/smoke_kdrel runs/smoke_kdmse
PY all_mixed/student/train_student_v2.py --mode scratch --seed 99 --max-steps 50 --epochs 1 --val-take 5 --out-dir runs/smoke_scratch 2>&1 | tee runs/smoke_scratch/train.log
PY all_mixed/student/train_student_v2.py --mode kd --soft-loss mape --seed 99 --max-steps 50 --epochs 1 --val-take 5 --out-dir runs/smoke_kdrel 2>&1 | tee runs/smoke_kdrel/train.log
PY all_mixed/student/train_student_v2.py --mode kd --soft-loss mse --seed 99 --max-steps 50 --epochs 1 --val-take 5 --out-dir runs/smoke_kdmse 2>&1 | tee runs/smoke_kdmse/train.log
```
1. [x] **Same starting weights.** The first run says `init_status: created`; the two kd runs say `loaded`. All three `config.json` files have the **same `init_sha256`**, and all three logs print the **same "First-step hard loss"** (same init and same first batch). If the loss differs, stop and report.
2. [x] **Validation metric is correct.** For `runs/smoke_scratch`, the `val_mape_flow` in `history.json` must equal the pooled MAPE from T2's evaluator on the same checkpoint, to within 0.01:
   `PY all_mixed/evaluation/eval_multi.py --split data/all_mixed_v1/validation --limit 5 --orders ref --models smoke:student:runs/smoke_scratch/ckpt-01 --out results/dec1/smoke_t5_val.jsonl`, then aggregate it with `aggregate_eval.py --input … --out …`.
3. [x] **The soft loss is meaningful.** In `runs/smoke_kdrel/history.json`, `soft` is of the same order as `hard` (for example 2–30), while in `runs/smoke_kdmse` it is small (≈ 0.1–2, as in the original). In `runs/smoke_scratch`, `soft` is 0.
4. [x] Delete `runs/smoke_*`, `runs/init/student_init_s99*` and `results/dec1/smoke_t5_val*` after recording the results in `SUMMARY.md`.

**Done when:** checks 1–3 pass. **Then stop at R4:** push the smoke logs, the three `config.json` and `history.json` files (copy them to `results/dec1/t5_smoke/` before deleting), and the check-2 numbers. Wait for review.

### T5 smoke gates passed — R4 approved (23 Nov 2026)

The user ran all three seed-99 smoke jobs (50 steps, one epoch, five validation scenarios) and the independent validation evaluator. **The reviewer-provided training script ran unchanged; no fixes or refactoring were needed.** Scratch created the init, both KD jobs loaded it, every init-file SHA-256 matched, and all first-step hard losses were **31.252909**.

| Run | Epoch hard loss | Epoch soft loss | Validation flow-weighted MAPE (%) |
|---|---:|---:|---:|
| scratch | 23.478565216 | 0.000000000 | 18.998365520 |
| KD relative/MAPE | 23.478540421 | 23.514001846 | 18.998277313 |
| KD MSE | 23.478630066 | 0.316480219 | 18.998422465 |

**Independent metric check PASS:** scratch `ckpt-01`, first five validation scenarios: training `val_mape_flow` = **18.998365520112273%**; evaluator pooled MAPE = **18.998365520112273%**; difference **0.0 points**, within 0.01. Relative soft loss is on the hard-loss scale, MSE soft loss is much smaller, and scratch soft loss is zero. These functional smoke results are not final accuracy evidence.

**R4 evidence preserved before cleanup:** `results/dec1/t5_smoke/` contains three per-run folders (`smoke_scratch`, `smoke_kdrel`, `smoke_kdmse`), each with `config.json`, `history.json`, `best.json` and `train.log`; `validation/` contains the evaluator JSONL, metadata, log and aggregate summary; `checks.json` records all gate values, init hashes, training-source hash and a SHA-256 manifest of the 16 preserved evidence files. Copies were checksum-verified. Config/best checkpoint paths retain their original run provenance; smoke checkpoint binaries are intentionally not retained.

Results were recorded in `SUMMARY.md`, then only the three temporary smoke folders, seed-99 init files and four original validation smoke files were deleted as specified above. The smoke tests did not use real seeds 1 and 2. **R4 approved on 23 Nov (see below); logging-only changes and their source-hash difference are recorded in `SUMMARY.md`. T6's seed-1 pair subsequently completed; see the 24 Nov hand-off.** No commits or pushes made by the execution agent.

**Warning/logging note for R4 (user request):** each smoke run emitted 12 `UserWarning: Converting sparse IndexedSlices ... to a dense Tensor of unknown shape` messages. The three complete training logs total only about **42 KiB**, with approximately **23 KiB** of warning text; these warnings were not a disk-space problem and did not prevent the smoke runs from completing. Preserve the unfiltered smoke logs and their manifest. For future runs, this exact Python warning can be suppressed without changing the training script by prefixing its command with `PYTHONWARNINGS='ignore:Converting sparse IndexedSlices:UserWarning:tensorflow.python.framework.indexed_slices'`. A synthetic filter check confirmed that this message is suppressed while a different UserWarning in the same module remains visible. This filter has not been applied to a training run; it does not affect CUDA/C++ logs or reduce memory usage. The warning describes actual dense-gradient conversion, so suppression must not be presented as a memory fix.

### R4 review result (user/reviewer, 23 Nov): approved; start T6 now
- **All three gates pass.**
  - Identical init hashes; `created` then `loaded` twice; the first-step hard loss is identical (31.252909) in all three runs.
  - Training `val_mape_flow` equals T2's evaluator exactly (difference 0.0).
  - Relative soft loss ≈ hard loss (23.51 vs 23.48), MSE soft loss small (0.32), scratch soft = 0.
  - The script ran unchanged.
- **Expected, not a bug:** after 50 steps the three runs have almost identical hard loss and validation MAPE. Early in training the student's error (≈ 20–30%) is far larger than the teacher's (≈ 5%), so the teacher's predictions and the true delays pull in almost the same direction; Adam also ignores the overall loss scale. Any KD effect can only appear later, once the student's error approaches the teacher's level. Do not read anything into the smoke-run differences.
- **Logging-only change after the smoke test (reviewer, 23 Nov):** `train_student_v2.py` now:
  1. filters exactly the `Converting sparse IndexedSlices` UserWarning inside the script, so no `PYTHONWARNINGS` prefix is needed;
  2. calls `fit(..., verbose=2)`: one line per epoch instead of a progress bar (the old KD log was 16 MB).

  Nothing else changed: training, loss, validation, initialisation and saving are identical, so **no new smoke test is needed**. `config.json` records the new git commit. Because the source hash differs from the one in `t5_smoke/checks.json`, note this change under "Problems and deviations" in `SUMMARY.md`.
- **Start T6 now**, in the table's order, from the repo root (commands in T6). After `scratch_s1`'s first epoch, check that `train.log` shows one line per epoch plus the "First-step hard loss" line. When `kdrel_s1` starts, confirm its `init_status` is `loaded`, its `init_sha256` equals `scratch_s1`'s, and its first-step hard loss equals `scratch_s1`'s.

---

## T6. Training queue (fixes P1–P3)

Run from the **repo root**, one run at a time, for each run: `mkdir -p runs/<name> && nohup PY all_mixed/student/train_student_v2.py <arguments> > runs/<name>/train.log 2>&1 &`. Start the next run when the previous one has finished; a small shell loop over the table is fine. Run them in this order, because the most important comparison comes first:

| # | Name | Arguments | Est. time |
|---|---|---|---|
| 1 | `scratch_s1` | `--mode scratch --seed 1 --out-dir runs/scratch_s1` | ~5.5 h |
| 2 | `kdrel_s1` | `--mode kd --soft-loss mape --alpha 0.5 --seed 1 --out-dir runs/kdrel_s1` | ~10 h |
| 3 | `scratch_s2` | `--mode scratch --seed 2 --out-dir runs/scratch_s2` | ~5.5 h |
| 4 | `kdrel_s2` | `--mode kd --soft-loss mape --alpha 0.5 --seed 2 --out-dir runs/kdrel_s2` | ~10 h |
| 5 | `kdmse_s1` | `--mode kd --soft-loss mse --alpha 0.5 --seed 1 --out-dir runs/kdmse_s1` | ~10 h |

- After each run's first epoch, check that its `config.json` `init_sha256` matches its partner's (same seed), and that the "First-step hard loss" in `train.log` is identical within each seed pair.
- Check each run every morning and evening: is it alive, is `history.json` growing, is the loss finite? If a run crashes, restart it with the same arguments plus `--resume-from <its latest ckpt-XX>` and a **new** `--out-dir` (e.g. `runs/kdrel_s1_resume`). Note the restart in `SUMMARY.md`, because the restart resets the early-stopping counter and the optimizer state.
- **Do not evaluate on test between runs.** That happens in T8.

**Done when:** runs 1–4 have finished (early stop or 30 epochs), each with `best.json`. Run 5 is next in priority; if it can't finish before the T8 start, stop it and record that.

**Required-run gate MET, 25 Nov:** all four required runs finished and have valid selected checkpoints. `kdmse_s1` remains pending. The user requested a review of the current results before starting anything else; see the two-pair hand-off below.

### T6 interim hand-off for reviewer — seed 1 complete (24 Nov 2026)

**Status: 2 of 4 required runs complete; T6 is not complete.** The user ran `scratch_s1` and `kdrel_s1` sequentially. Both finished normally, all recorded losses/metrics are finite, and both selected checkpoint files are present. **No new test evaluation has been run.** Results below are full-validation, flow-weighted MAPE from the agreed checkpoint-selection procedure.

| Run | Epochs completed | Stop reason | Selected checkpoint | Best epoch | Best validation MAPE (%) | Actual training + validation time |
|---|---:|---|---|---:|---:|---:|
| scratch_s1 | 26 | Early stopping, patience 5 | `runs/scratch_s1/ckpt-21` | 21 | 10.153210530 | 10.17 h |
| kdrel_s1 | 30 | Maximum epoch limit | `runs/kdrel_s1/ckpt-30` | 30 | 9.851046374 | 12.27 h |

**Paired validation result:** KD − scratch = **−0.302164157 percentage points**, a **2.976% relative error reduction**. This is an observed validation gain for one paired seed, not evidence of significance, consistency across seeds or final test improvement. KD's best epoch is the last permitted epoch; do not extend its limit or alter patience/alpha to chase additional improvement. The historical runs used different training protocols and are context, not matched controls.

**Pair/provenance checks PASS:** both runs use seed/data-seed **1**, 9,972 steps per epoch, a 30-epoch cap, patience 5 and code commit `94eacee`. Scratch created the init and KD loaded it. Both init-file SHA-256 hashes match, and both first-step hard losses are **33.599453**. KD uses the fixed teacher `ckpt-26`; its hashes match the teacher recorded in T2. Both use `val_mape_flow` for selection. Scratch soft loss remains zero; KD's final hard/soft losses are **7.143393517 / 6.770048141**, so both KD terms remain active.

**Local times (Asia/Colombo):** scratch ran 23 Nov **06:36–16:47**; KD ran 23 Nov **17:19** to 24 Nov **05:35**. The successful pair consumed **22.44 h**, versus the original **15.5 h** estimate. Observed epochs were roughly 23.4 min for scratch and 24.5 min for KD. For scheduling, allow approximately **22–24 h for the remaining required seed-2 pair** and **12–13 h for optional `kdmse_s1`** if epoch counts and machine load are similar; early stopping may change these. The original 41-hour total is optimistic (roughly 57–60 hours for all five under similar conditions). Keep T8's latest start at **27 Nov, 20:00** and the number lock at **28 Nov, 08:00**.

**Evidence to commit/push for the other agent:** `runs/scratch_s1/` and `runs/kdrel_s1/` each contain `config.json`, `history.json`, `best.json` and `train.log`. `results/dec1/t6_seed1_review.json` records exact comparison values, matching-setting checks, local times and selected-checkpoint hashes. Commit those JSON/log files and this plan/summary update; **do not commit checkpoint binaries or shared init files**. Checkpoints remain on this machine for T8.

**Requested interim review:** confirm the seed-1 interpretation and revised schedule, then continue in the existing order with `scratch_s2`, followed by `kdrel_s2`; keep the second pair ahead of `kdmse_s1` if time is constrained. One positive or negative seed does not establish repeatability. This hand-off does not change the training protocol or authorize test-based selection. **The execution agent has not started seed 2 or any additional run and has made no commits/pushes.**

### T6 two-pair hand-off for reviewer — required training complete (25 Nov 2026)

This supersedes the 24 Nov status snapshot above. The user subsequently completed `scratch_s2` and `kdrel_s2`. **All four required runs finished normally; no new test evaluation has been run.** Results are each run's best flow-weighted validation MAPE, selected using the unchanged protocol.

| Run | Best validation MAPE (%) | Best epoch / epochs completed | Selected checkpoint | Stop reason | Actual hours |
|---|---:|---:|---|---|---:|
| scratch_s1 | 10.153210530 | 21 / 26 | `runs/scratch_s1/ckpt-21` | Early stopping | 10.17 |
| kdrel_s1 | 9.851046374 | 30 / 30 | `runs/kdrel_s1/ckpt-30` | Epoch limit | 12.27 |
| scratch_s2 | 10.014834048 | 15 / 20 | `runs/scratch_s2/ckpt-15` | Early stopping | 7.77 |
| kdrel_s2 | 11.918671162 | 4 / 9 | `runs/kdrel_s2/ckpt-04` | Early stopping | 3.65 |

| Paired seed | KD − scratch validation MAPE (points) | Direction |
|---|---:|---|
| 1 | −0.302164157 | KD better |
| 2 | +1.903837114 | KD worse |
| Mean paired difference | **+0.800836479** | KD worse on average |

Mean best-validation MAPE across the two seeds: scratch **10.084022289%**, KD **10.884858768%**. **KD improves validation in only 1 of 2 pairs; there is no consistent validation benefit under this protocol.** Do not omit seed 2, describe the study as a repeatable gain, or infer statistical significance from two seeds. Test generalization remains unknown until T8.

**Validity/provenance checks PASS:** all history values are finite; each `best.json` matches its minimum validation epoch and its checkpoint index/data files exist. Initial-weight hashes and first-step hard losses match within each pair: seed 1 **33.599453**, seed 2 **34.021622**. Within each pair, seeds, data seeds, epoch cap, patience, steps per epoch and code commit match. The training-source hashes are also identical across both runs' source commits. Both KD jobs used the same fixed teacher `ckpt-26` and its hashes match T2. The poor seed-2 result is a completed valid run, not a crash or initialization mismatch. No training-code or stopping-rule changes were made.

**Seed-2 progression:** scratch's best was epoch 15, followed by five non-improving epochs; KD's best was epoch 4, followed by five non-improving epochs. KD stopped at epoch 9 under the predefined patience. Training loss continued decreasing, but validation did not improve; this alone does not identify the cause. Do not selectively extend KD seed 2 or replace it with a favorable seed.

**Scheduling:** the four required runs consumed **33.85 h** of training/validation, excluding gaps. `scratch_s2` finished **24 Nov, 14:06**, and `kdrel_s2` finished **25 Nov, 00:37**, Asia/Colombo. The planned fifth run, `kdmse_s1`, remains pending; allow up to roughly **12–13 h** if it uses 30 epochs at the observed KD rate, with earlier stopping possible. T7 remains optional and subject to its time/review gate; T8 must still start by **27 Nov, 20:00**, with number lock **28 Nov, 08:00**.

**Review evidence:** `results/dec1/t6_paired_validation_review.json` contains exact values for all four runs, pair checks, cross-seed source hashes, checkpoint hashes, timestamps and comparison results. `runs/scratch_s2/` and `runs/kdrel_s2/` each contain `config.json`, `history.json`, `best.json` and `train.log`; seed-1 evidence is already committed. Keep checkpoint binaries and shared init files local and uncommitted.

**Requested review before further work:** assess the mixed validation finding and the value/schedule of the already-planned MSE replication. Do not use the unfavorable seed-2 result to change settings based on test data, discard a run, or skip reporting it. T8 must include both completed pairs, regardless of direction. No `kdmse_s1`, T7 run or T8 evaluation has been launched by the execution agent. The user requested this pause for review; no additional approval gate is being inferred from the original plan. No commits or pushes made by the execution agent.

### Two-pair review result (user/reviewer, 25 Nov): accepted; run seed 3 first, then `kdmse_s1`, chained
- **Accepted as valid evidence.** Pair checks pass, nothing is excluded, and the protocol stays unchanged (epoch cap, patience, α, seeds). Do not extend or rerun `kdrel_s2`.
- **Interpretation (fixed wording for `SUMMARY.md`):** KD was better in seed 1 (−0.30 points) and worse in seed 2 (+1.90). The main driver is how noisy validation MAPE is from epoch to epoch. For example, `scratch_s2` jumped 12.58 → 10.15 between epochs 8 and 9, and `kdrel_s2` stopped at epoch 9 after five rising epochs, while scratch survived a three-epoch rise. Under this protocol the paired difference is dominated by training and selection noise, and **no consistent KD benefit is shown**. Report it like that, with no significance claims. Final test numbers come from T8.
- **Next: a third pair is now the most valuable run**, because the two pairs disagree. Then `kdmse_s1`.
- **The seed-2 runs had a ≈ 7 h idle gap.** Start the remaining runs **now, as one chained queue**, from the repo root, with nothing else running:
  ```bash
  nohup bash -c '
  set -e
  PY=/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python
  run() { name=$1; shift; mkdir -p runs/$name; $PY all_mixed/student/train_student_v2.py "$@" --out-dir runs/$name > runs/$name/train.log 2>&1; }
  run scratch_s3 --mode scratch --seed 3
  run kdrel_s3   --mode kd --soft-loss mape --alpha 0.5 --seed 3
  run kdmse_s1   --mode kd --soft-loss mse  --alpha 0.5 --seed 1
  ' > runs/t6_queue.log 2>&1 &
  ```
  Expected ≈ 8–10 h + 4–12 h + up to 12 h, so roughly 26 Nov morning to evening.
- **Hard cut-off for T8:** at **27 Nov, 12:00**, stop whatever is still running and start T8 with all finished runs. Record the stopped run under "Problems and deviations". T7 (ablation) is dropped.
- **T8 must evaluate on test:** all finished new students (`scratch_s1/s2/s3`, `kdrel_s1/s2/s3`, `kdmse_s1` if finished), using each run's `best.json` checkpoint, `--orders ref`, on the test split; then the same on the validation split. Report every run, favourable or not.

### T6 three-pair hand-off — request T8 before MSE (26 Nov 2026)

**Current status: all six scratch/MAPE-KD runs finished; review of the requested order change is pending.** This supersedes earlier completion/status snapshots. The user ran seed 3 one job at a time, rather than using the proposed chained queue. No MSE training or T8 evaluation was launched during this check.

| Run | Best flow-weighted validation MAPE (%) | Best epoch / epochs completed | Selected checkpoint | Hours |
|---|---:|---:|---|---:|
| scratch_s1 | 10.153210530 | 21 / 26 |`runs/scratch_s1/ckpt-21` | 10.17 |
| kdrel_s1 | 9.851046374 | 30 / 30 |`runs/kdrel_s1/ckpt-30` | 12.27 |
| scratch_s2 | 10.014834048 | 15 / 20 |`runs/scratch_s2/ckpt-15` | 7.77 |
| kdrel_s2 | 11.918671162 | 4 / 9 |`runs/kdrel_s2/ckpt-04` | 3.65 |
| scratch_s3 | 10.162595209 | 28 / 30 |`runs/scratch_s3/ckpt-28` | 11.66 |
| kdrel_s3 | 9.958502915 | 30 / 30 |`runs/kdrel_s3/ckpt-30` | 12.22 |

Paired KD − scratch validation differences: seed 1 **−0.302164157**, seed 2 **+1.903837114**, seed 3 **-0.204092294** percentage points. KD is better in **2 of 3** seeds, but mean best-validation MAPE is **10.110213262% scratch** versus **10.576073483% KD**, with mean paired difference **+0.465860221 points** (KD worse on average). Results remain mixed; no consistent KD benefit, statistical significance or final test improvement is established.

**Seed-3 checks PASS:** both runs completed 30 epochs normally; every recorded history metric is finite; `best.json` matches each validation minimum; selected index/data files exist and were hashed. Initial-weight hashes, first-step hard losses (**32.87072**), data seeds and stopping settings match within the pair. Training source matches the earlier pairs, and KD uses the same fixed teacher hashes. Scratch's best is epoch 28; KD's best is epoch 30. Do not extend the training cap. Seed-3 scratch/KD took **11.66 / 12.22 h**, respectively. Exact values, timestamps, hashes and checks: `results/dec1/t6_three_pair_validation_review.json`. Original two-pair evidence is preserved unchanged.

**Request to the reviewing agent:** may we start **T8 now for all six completed students**, then run the already-planned **`kdmse_s1` afterward**, using its fixed settings (`--mode kd --soft-loss mse --alpha 0.5 --seed 1`, 30 epochs, patience 5, existing seed-1 init and teacher)? After MSE finishes, evaluate its validation-selected `best.json` checkpoint with the same T8 test/ref and validation/ref protocol and add its results; existing completed evaluations need not be repeated. Please confirm this order and whether MSE can fit the final deadline or must be labeled follow-up.

If approved, freeze MSE settings before T8 and run it regardless of whether the first T8 results favor KD. Do not tune settings or decide whether to include MSE using test outcomes. Report all six completed runs, including unfavorable seed 2. **This request is not yet reviewer approval.** MSE remains planned, not silently removed. T7 ablation stays dropped. The latest approved T8 cutoff remains **27 Nov, 12:00 Asia/Colombo**, with number lock **28 Nov, 08:00**, unless explicitly revised by review. If MSE misses the agreed deadline, mark it as a deferred follow-up and report the limitation.

**Evidence for user commit/push:** this plan, `results/dec1/SUMMARY.md`, `results/dec1/t6_three_pair_validation_review.json`, and the seed-3 run folders' `config.json`, `history.json`, `best.json`, `train.log`. Keep checkpoint binaries and shared init files local and uncommitted. No commit/push was made by the execution agent.

### Three-pair review result (user/reviewer, 26 Nov): approved; start T8 now, run `kdmse_s1` in parallel
- **Accepted.** All pair checks pass; nothing is excluded.
- **Interpretation (fixed wording for `SUMMARY.md`):**
  - On validation, KD was better in 2 of 3 pairs by small margins (−0.30, −0.20 points) and worse in 1 (+1.90; that run stopped early at epoch 9). The mean is +0.47 points.
  - Both KD runs that trained the full 30 epochs reached their best at epoch 30, the cap.
  - The differences are within the run-to-run spread of scratch-like runs (≈ 1.6 points). **No consistent KD benefit is shown.** No significance claims.
- **T8 now**, from the repo root. Test first, then validation, in one background job:
  ```bash
  PY=/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python
  M="scratch_s1:student:runs/scratch_s1/ckpt-21 kdrel_s1:student:runs/kdrel_s1/ckpt-30 scratch_s2:student:runs/scratch_s2/ckpt-15 kdrel_s2:student:runs/kdrel_s2/ckpt-04 scratch_s3:student:runs/scratch_s3/ckpt-28 kdrel_s3:student:runs/kdrel_s3/ckpt-30"
  nohup sh -c "$PY all_mixed/evaluation/eval_multi.py --orders ref --models $M --out results/dec1/eval_v2_test.jsonl && $PY all_mixed/evaluation/eval_multi.py --split data/all_mixed_v1/validation --orders ref --models $M --out results/dec1/eval_v2_val.jsonl" > results/dec1/eval_v2.log 2>&1 &
  ```
  Then aggregate both: `aggregate_eval.py --input results/dec1/eval_v2_test.jsonl --out results/dec1/eval_v2_test_summary.json`, and the same for `eval_v2_val`.
- **Gate:** each model's pooled validation MAPE from `eval_v2_val` must equal its `best.json` `val_mape_flow` within 0.01. That proves the right checkpoint was evaluated. Expected ≈ 3–4 h on test plus ≈ 20–30 min on validation.
- **`kdmse_s1` in parallel, starting now** (settings frozen; T8's timing fields are not reported, so sharing the CPU is fine):
  `mkdir -p runs/kdmse_s1 && nohup $PY all_mixed/student/train_student_v2.py --mode kd --soft-loss mse --alpha 0.5 --seed 1 --out-dir runs/kdmse_s1 > runs/kdmse_s1/train.log 2>&1 &`
  - Check that its first-step hard loss equals `scratch_s1`'s (33.599453).
  - If it finishes by **27 Nov, 20:00**, evaluate its `best.json` checkpoint the same way (test and validation, `--orders ref`) into `results/dec1/eval_v2_kdmse_test.jsonl` and `_val.jsonl`, and add it to the summary. Otherwise report it as a deferred follow-up. Include it whatever the result.
- **`SUMMARY.md` for T8:** a table of test MAPE, MAE and validation MAPE for all six runs (plus `kdmse_s1` if done); the paired KD − scratch test difference per seed, with the mean; per-size test MAPE; and the fixed interpretation above, updated with the test numbers. Report every run, favourable or not. Push as soon as the six-run results are in; don't wait for `kdmse_s1`.

---

## T7. Dropped by 25 Nov review (original instructions retained below)

Historical instructions only: the 25 Nov review dropped this ablation; do not execute it. The seed-3 pair listed below has now completed.

1. **Ablation, small RouteNet-Fermi with queues kept** (fixes P7).
   - Add constructor arguments to `RouteNet_Fermi` in `delay_model.py`, with **defaults equal to the current values** (`state_dim=32`, `iterations=8`). The teacher, baseline and emulation must behave exactly as before. Check this by reloading `ckpt-26` and reproducing the T2 teacher prediction on 2 test scenarios.
   - Add `--arch {student,fermi_small}` to `train_student_v2.py` (state 16, T = 4, random init). Run `--mode scratch --arch fermi_small --seed 1 --out-dir ../../runs/fermismall_s1`.
   - Record its parameter count.
2. `scratch_s3`, then `kdrel_s3`.

---

## T8. Final evaluation and summary (start by 27 Nov, 12:00 at the latest)

1. [ ] Run `eval_multi.py` on the test split, `--orders ref`, with every **finished** run's `best.json` checkpoint: scratch_s1/s2/s3 and kdrel_s1/s2/s3; include kdmse_s1 if finished. The requested T8-before-MSE order is pending review above. Output `results/dec1/eval_v2_test.jsonl`, then aggregate it.
2. [ ] Also run it on the validation split for the same checkpoints, so that the flow-weighted validation MAPE comes from one evaluator.
3. [ ] Write **`results/dec1/SUMMARY.md`**, the hand-off to the report agent, with these sections:
   - **Gate checks:** the T2 reproduction table (expected vs obtained).
   - **Offline accuracy, v0 models:** pooled MAPE, MAE, per-size MAPE, ranking metrics; `ref` vs `hop` order.
   - **Latency v2:** the table from T3, including the size of the old benchmark scenario and the construction/forward split.
   - **Emulation under the training hop order:** from T4.
   - **Controlled KD study:** one row per run (val MAPE, test MAPE, best epoch, epochs run, hours). Then the **paired differences** kd − scratch for each seed and loss type.
     - Use plain language, e.g. "kdrel was better than scratch in 2 of 2 seeds, by X and Y points". No significance claims with n = 2.
     - Compare the differences with the ≈1-point run-to-run variation of the baseline.
     - If KD did not help, say so directly.
   - **Ablation** (if run).
   - **Problems and deviations:** crashes, restarts, parallel runs, anything that differs from this plan.
   - **Report impact:** a list of the report sentences or tables whose numbers or claims change (see Section 3).
4. [ ] Update `docs/research_progress_summary.md`: add a dated section pointing to `results/dec1/SUMMARY.md`.
5. [ ] Commit and push by **28 Nov, 08:00**.

---

## 3. Report items that depend on these results (for the report agent; listed here so T8 covers them)

| Report location | Current text | Changes with |
|---|---|---|
| Ch4 §4.3.2 and Table 5.2 | "The size of the benchmark scenario was not recorded"; student-KD "not measured" | T3 |
| Ch5 §5.2 | "most of the end-to-end time … is spent preparing inputs" (inferred) | T2 timing, T3 |
| Ch3 §3.7, §5.5 threats, Ch6 future work | hop-order effect "checked on one sample only" | T2 `ref` vs `hop`, T4 |
| Ch5 §5.1, §5.5 "The effect of distillation", Abstract, Ch6 Objective 3 | KD gain from a single run; cause not isolated | T6/T8 KD study |
| Ch5 §5.5 "Why the student loses accuracy" | the four causes are not separated | T7 ablation, if run |
| Ch5 threats "Aggregate metrics" | "not broken down by topology size" | T2 per-size |
| Ch5 §5.3 "How strong is this evidence?" | emulation is the only ranking evidence | T2 ranking metrics |

Two text fixes that don't depend on new results (report agent):
- Ch3 §3.4, "What the fusion gives up": the mean weight over a link's remaining queues is exactly 1/|Q(l)| only when every queue is used. Otherwise it can be any value (for example 0.1 if only the weight-0.1 queue remains), so "close to" is wrong.
- Ch5: the KD-to-teacher gap is "4.45 points" in §5.1 and "4.46 points" in §5.5 (unrounded: 9.828 − 5.373 = 4.455). Use one value.
