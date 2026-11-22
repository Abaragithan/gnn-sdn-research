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
2. [ ] Full run overnight: `nohup PY eval_multi.py --out ../../results/dec1/eval_v0_test.jsonl > ../../results/dec1/eval_v0_test.log 2>&1 &`
3. [ ] Aggregate.

**Done when (gate for everything that follows):** the pooled MAPE with `ref` order reproduces **8.4011 / 5.3734 / 11.9244 / 9.8280** to within 0.01, and the file has 478 scenarios. If any value is off, find out why before T6, because every later number depends on this evaluator.

**R1 review stop: 22 Nov 2026.** Implemented `eval_multi.py`, `hop_order.py` and `aggregate_eval.py`. The five-scenario smoke run completed for all four fixed checkpoints under both orders. A fixed input signature prevents graph-size retracing; its predictions matched the original model call for all four models under both orders (`rtol=1e-6`, `atol=1e-9`). The independent historical Keras pipeline reproduced the pooled reference-order metrics:

| Model | Five-scenario pooled MAPE (%) | Keras evaluate MAPE (%) | Absolute difference (points) | Check |
|---|---|---|---|---|
| teacher | 2.088734561 | 2.088734388 | 0.000000172 | PASS |
| kd | 3.237940309 | 3.237940073 | 0.000000236 | PASS |

Evidence: `results/dec1/eval_v0_test_smoke.jsonl`, `.meta.json`, `.keras_check.json` and `.log`; aggregation: `eval_v0_test_smoke_summary.json` and `eval_v0_test_smoke_aggregate.log`. `results/dec1/eval_t2_checks.log` records passing checks for flow weighting, ranking ties, undefined correlations, timing fractions and recovery of a partial JSONL write. Ranking metrics use NumPy; no SciPy installation is needed. Historical scripts and checkpoints are unchanged.

**Awaiting user review at R1. The full 478-scenario run has not started; the full-test reproduction gate remains pending.**

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

---

## T5. Training script v2 for students (fixes P1, P2)

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
1. [ ] `--mode scratch --seed 1 --max-steps 50 --epochs 1 --val-take 5 --out-dir runs/smoke_scratch`, then the same with `--mode kd --soft-loss mape --out-dir runs/smoke_kd`. Both runs must load the **same** `runs/init/student_init_s1` file. Check with `np.allclose` on every variable of a freshly built student that loads that file, against the first saved weights of each run. Also check that both runs print the same first-step `hard` loss.
2. [ ] The `val_mape_flow` value after the smoke epoch equals the pooled MAPE from `eval_multi.py --split validation --limit 5` on the same saved checkpoint, to within 0.01.
3. [ ] In the kd smoke run, `soft` is of the same order as `hard` (for example 2–20, not 0.2).
4. [ ] Delete `runs/smoke_*` and `runs/init/student_init_s1*` afterwards, so that the real runs create a fresh init. Alternatively keep the init, but then use it for both real s1 runs.

**Done when:** checks 1–3 pass.

---

## T6. Training queue (fixes P1–P3)

Run from `all_mixed/student/`, with `nohup PY train_student_v2.py ... 2>&1 | tee ../../runs/<name>/train.log`. Run them in this order, because the most important comparison comes first:

| # | Name | Command arguments | Est. time |
|---|---|---|---|
| 1 | `scratch_s1` | `--mode scratch --seed 1 --out-dir ../../runs/scratch_s1` | ~5.5 h |
| 2 | `kdrel_s1` | `--mode kd --soft-loss mape --alpha 0.5 --seed 1 --out-dir ../../runs/kdrel_s1` | ~10 h |
| 3 | `scratch_s2` | `--mode scratch --seed 2 --out-dir ../../runs/scratch_s2` | ~5.5 h |
| 4 | `kdrel_s2` | `--mode kd --soft-loss mape --alpha 0.5 --seed 2 --out-dir ../../runs/kdrel_s2` | ~10 h |
| 5 | `kdmse_s1` | `--mode kd --soft-loss mse --alpha 0.5 --seed 1 --out-dir ../../runs/kdmse_s1` | ~10 h |

- Check each run every morning and evening: is it alive, is `history.json` growing, is the loss finite? If a run crashes, restart it from its latest checkpoint with the same arguments. Note the restart in `SUMMARY.md`, because the restart resets the early-stopping counter.
- **Do not evaluate on test between runs.** That happens in T8.

**Done when:** runs 1–4 have finished (early stop or 30 epochs), each with `best.json`. Run 5 is next in priority; if it can't finish before the T8 start, stop it and record that.

---

## T7. Optional, only if T6 finishes before ~26 Nov, 12:00

In priority order. Stop at the T8 deadline whatever happens.

1. **Ablation, small RouteNet-Fermi with queues kept** (fixes P7).
   - Add constructor arguments to `RouteNet_Fermi` in `delay_model.py`, with **defaults equal to the current values** (`state_dim=32`, `iterations=8`). The teacher, baseline and emulation must behave exactly as before. Check this by reloading `ckpt-26` and reproducing the T2 teacher prediction on 2 test scenarios.
   - Add `--arch {student,fermi_small}` to `train_student_v2.py` (state 16, T = 4, random init). Run `--mode scratch --arch fermi_small --seed 1 --out-dir ../../runs/fermismall_s1`.
   - Record its parameter count.
2. `scratch_s3`, then `kdrel_s3`.

---

## T8. Final evaluation and summary (start by 27 Nov, 20:00 at the latest)

1. [ ] Run `eval_multi.py` on the test split, `--orders ref`, with every **finished** run's `best.json` checkpoint: scratch_s1/s2, kdrel_s1/s2, kdmse_s1, and T7 runs if any. Output `results/dec1/eval_v2_test.jsonl`, then aggregate it.
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
