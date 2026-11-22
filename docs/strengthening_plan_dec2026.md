# Plan: Strengthening the Research Before Submission

> **Superseded on 22 Nov 2026 by [`dec1_execution_plan.md`](dec1_execution_plan.md).** Kept for history; do not execute this plan.


**Period:** 18 Nov – 3 Dec 2026
**Hard deadlines:** video upload before 08:00 on 1 Dec · printed report by 3 Dec
**Machine:** the Fedora CPU-only box used for all earlier runs

This plan fixes the weaknesses an examiner is most likely to raise, in priority order. Each task has a "done when" check. If time runs out, stop at the end of any priority level. The report stays valid at every stopping point as long as the claims match what was actually run.

---

## 1. Weaknesses this plan addresses

| # | Weakness | Why it matters at the viva | Fix (section) |
|---|----------|----------------------------|---------------|
| W1 | Every model was trained once (one seed) | The KD gain (11.92% → 9.83%) could be seed noise. The two baseline runs differed by about 1 point (9.17% vs 8.22%). | 4.2, 4.3 |
| W2 | Scratch and KD students started from different random weights | `train_kd.py` builds the teacher first, which changes the random generator's state when the student is built | 3.1 |
| W3 | KD selected its checkpoint with a different metric | The KD `test_step` uses an unweighted mean over scenarios; all other runs use the flow-weighted Keras mean | 3.2 |
| W4 | Baseline vs teacher is confounded | 37,500 vs ~299,000 steps, different validation data, different protocol | 4.4 |
| W5 | Only one aggregate MAPE | Unknown where errors concentrate; MAPE alone is sensitive to the small test delays | 3.4, 5 |
| W6 | No decision-level evaluation | "Is the model useful to a controller?" goes unanswered | 3.6, 4.5 |
| W7 | Latency counts only the forward pass, on unconstrained hardware | The deployment claim is weak, and graph building dominates end-to-end time | 3.7, 4.6 |
| W8 | Each test evaluation takes ~2.5 h | Makes W1, W5 and W6 too expensive unless fixed | 3.3 |

**Explicitly out of scope** (stays as future work in the report): a Mininet/RYU deployment, pruning/quantisation, jitter/loss models. A Mininet testbed cannot give valid accuracy results in two weeks. Its traffic and queues differ from the simulator data the models were trained on, so the models would need new ground truth and retraining.

---

## 2. Step 0: Preparation (Nov 18, ~1 h)

- [ ] Create a branch: `git checkout -b strengthen-dec2026`.
- [ ] Freeze the current results as the "v0" reference. Copy `results_*.json`, `results_student.json` and `inference_benchmark_results.json` into `results/v0/`, unchanged.
- [ ] Record machine specs into `logs/machine_specs.txt`:
  ```bash
  lscpu > logs/machine_specs.txt; free -h >> logs/machine_specs.txt; nproc >> logs/machine_specs.txt
  python -c "import tensorflow as tf, networkx; print(tf.__version__, networkx.__version__)" >> logs/machine_specs.txt
  ```
- [ ] Measure free disk space (`df -h`); the graph cache needs to fit (see 3.3).
- [ ] Decide parallelism. Run one short training job with `top` open. If TensorFlow keeps all cores busy, run jobs **one at a time**. If CPU use stays below ~50%, two jobs in parallel are fine.

**Done when:** the branch exists, the v0 results are copied, and the specs file is committed.

---

## 3. Code changes (Nov 18–20)

### 3.1 Seeds and identical student initialisation (fixes W1, W2)

Files: `all_mixed/student/train_scratch.py`, `all_mixed/student/train_kd.py`

- [ ] Add CLI arguments `--seed` (model initialisation), `--data-seed` (training shuffle), `--out-dir` (checkpoints and results).
- [ ] In **both** scripts, build the student like this, immediately before use:
  ```python
  tf.random.set_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
  student = StudentRouteNet(state_dim=16, iterations=4)
  _ = student(sample_x)              # create weights now
  student.save_weights(os.path.join(out_dir, "init"))   # keep for proof
  ```
  In `train_kd.py`, this block must come **after** the teacher is built and loaded.
- [ ] Save `history.history` (all per-epoch values) to `<out-dir>/history.json`. The scratch console log was lost last time; this prevents a repeat.
- [ ] Write each run's console output to `<out-dir>/train.log` (with `tee`), never to a shared file.

**Done when:** a 50-step smoke run of scratch and KD with the same `--seed` produces `init` files with identical weights. Check with `np.allclose` on every variable.

### 3.2 Same validation metric for KD (fixes W3)

File: `all_mixed/student/train_kd.py`

- [ ] In `test_step`, also accumulate the **flow-weighted** MAPE: the sum over flows of |y−ŷ|/|y|, and the total flow count, using two `tf.keras.metrics.Sum` trackers. Report `val_mape_flow = 100 * sum / count`.
- [ ] Monitor `val_mape_flow` for checkpointing and early stopping, so that every model uses the same selection metric. Keep logging the old unweighted value too.

**Done when:** in a smoke run, `val_mape_flow` equals `model.evaluate()` on the same weights to within 0.01.

### 3.3 Cache preprocessed graphs (fixes W8)

New file: `all_mixed/cache_dataset.py`

- [ ] For `data/all_mixed_v1/validation` and `data/all_mixed_v1/test`, run the existing generator once and save the outputs.
- [ ] Alongside the graphs, save a metadata list in the same order: graph file, number of nodes, routing file, traffic-matrix/results-file identifier, number of flows, and global delay.
- [ ] **Step 1: test the format on 5 scenarios first.** Try `tf.data.experimental.save` / `load` (available in TF 2.6) and confirm that RaggedTensors round-trip correctly. If they don't, fall back to pickling the plain Python dict from `hypergraph_to_input_data` (before the `tf.ragged.constant` conversion) and rebuild the ragged tensors at load time.
- [ ] Then cache the full validation set (2,191) and test set (478). Expect ~3 h, so run it overnight.

**Done when:** the loader yields tensors identical to `input_fn` for the first 5 scenarios, and the counts are 2,191 and 478.

### 3.4 A single evaluator with per-scenario output (fixes W5)

New file: `all_mixed/evaluation/evaluate_cached.py`

- [ ] Arguments: `--model {fermi,student}`, `--ckpt <path>`, `--split {validation,test}`, `--out <json>`.
- [ ] For each scenario, store: n_flows, n_nodes, sum APE, sum AE, sum SE, mean true delay, mean predicted delay, and the predicted per-flow mean.
- [ ] Compute pooled MAPE (flow-weighted, the same as Keras), the unweighted per-scenario MAPE, MAE, RMSE, and MAPE per topology size (200, 240, 260, 280, 300).
- [ ] Also save the raw per-flow predictions for the test set as `.npy` (needed by 3.6), if disk allows.

**Done when (critical check):** the cached evaluator reproduces the v0 numbers on the four existing checkpoints:

| Model | Checkpoint | Expected test MAPE |
|---|---|---|
| Baseline | `ckpt_dirs/ckpt_dir_0_10000/15-6.41` | 8.4011 |
| Teacher | `ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26` | 5.3734 |
| Student-scratch | `student/ckpt_dirs_scratch/ckpt-08` | 11.9244 |
| Student-KD | `student/ckpt_dirs_kd_alpha0.5/ckpt-19` | 9.8280 |

If any value differs by more than 0.01, stop and find out why before running new experiments. This check also confirms the flow-weighting explanation given in the report (Section 5.3.4): compute both aggregates for the KD checkpoint on validation. Expect ≈ 9.04 flow-weighted and ≈ 9.64 unweighted.

### 3.5 Matched-protocol baseline (fixes W4)

- [ ] Create `data/all_mixed_v1_10only/` with symlinks:
  - `train/` → the six 10-node topology folders of `all_mixed_v1/train`
  - `validation/` → `ch22-training-scheduling-topo-10-7` (the 10-node folder in `all_mixed_v1/validation`)
- [ ] Copy `train_teacher.py` to `train_baseline_matched.py`. The only changes are the data paths, the checkpoint directory, and `NUM_TRAIN_SAMPLES` (count it with `check_teacher_counts.py` first).

This is exactly the teacher's protocol (same initial weights, epochs, early stopping, one pass per epoch) **minus the 50–170-node topologies**. Any remaining difference from the teacher can then be attributed to topology diversity.

### 3.6 Fix and run the decision-level evaluation (fixes W6)

File: `all_mixed/evaluation/routing_decision_eval.py`

Current flaws:
1. It groups candidates by topology file only, which mixes different traffic matrices.
2. It runs on the validation split, which was used for checkpoint selection.
3. It compares the mean predicted flow delay with `global_delay`, which may be defined differently.

- [ ] **Step 1: inspect the dataset (≤1 h).** Using the metadata from 3.3, list for each test topology which routing files and which traffic-matrix/results files occur. Answer: *are there scenarios with the same topology and the same traffic matrix but different routings?*
- [ ] **If yes:** group by `(graph_file, traffic-matrix id)`. Each group is a decision; the candidates are its routings.
- [ ] **If no:** a routing decision cannot be evaluated validly on this dataset. Use the fallback instead: **ranking evaluation**. Within each test topology, compute the Spearman and Kendall rank correlation between predicted and true scenario-mean delay, plus the pairwise ordering accuracy (the fraction of scenario pairs ordered correctly). This answers "can the model tell which configuration is worse?", which is what a controller needs. Report it under that name and do not call it routing-decision accuracy.
- [ ] In both cases: use the **test** split, use the cached predictions from 3.4 (no extra inference needed), and use the mean of the true per-flow labels as the ground truth, so that both sides are computed the same way.
- [ ] Metrics: decision accuracy, top-2 accuracy, mean/median regret (or the ranking metrics), for all four models and all seeds.

### 3.7 Constrained-resource benchmark (fixes W7)

New file: `all_mixed/evaluation/deployment_benchmark.py`

- [ ] Use 2 test scenarios per topology size (200–300 nodes), so the benchmark uses deployment-sized networks.
- [ ] For each model (baseline, teacher, scratch, **KD**), measure:
  - graph-construction time (parse + NetworkX + tensors)
  - forward-pass time (5 warm-up runs, 30 timed)
  - end-to-end time (construction + forward)
  - peak memory (`resource.getrusage(...).ru_maxrss`, one process per model)
- [ ] Run under three CPU budgets:
  1. all cores (current behaviour)
  2. `tf.config.threading.set_intra_op_parallelism_threads(2)` and `set_inter_op_parallelism_threads(1)`
  3. 1 thread, or inside Docker with `--cpus=1 --memory=2g`
- [ ] Optional: if a Raspberry Pi 4/5 or a similar ARM board is available, run the forward-pass benchmark there (TensorFlow for aarch64). Skip this if setting it up takes more than half a day.

**Done when:** a JSON file with all three budgets × four models exists, and the scenario sizes (nodes, flows) are recorded.

---

## 4. Experiment queue (Nov 20–26)

Seeds: **1, 2, 3** (model initialisation). Use `--data-seed` equal to the seed. Run commands from `all_mixed/student/` (or `all_mixed/` for the baseline), with `nohup ... > <out-dir>/train.log 2>&1 &`.

Estimated times come from the earlier logs; evaluation time assumes the cache from 3.3.

### 4.1 Evaluation of existing checkpoints on the cache: about 1 h
Runs the 3.4 check. Gives per-size errors and MAE/RMSE for the v0 models.

### 4.2 Student-scratch, seeds 1–3: about 16 h
```bash
python train_scratch.py --seed 1 --data-seed 1 --out-dir runs/scratch_s1
python train_scratch.py --seed 2 --data-seed 2 --out-dir runs/scratch_s2
python train_scratch.py --seed 3 --data-seed 3 --out-dir runs/scratch_s3
```

### 4.3 Student-KD (α = 0.5), seeds 1–3: about 30 h
```bash
python train_kd.py --seed 1 --data-seed 1 --out-dir runs/kd_s1
python train_kd.py --seed 2 --data-seed 2 --out-dir runs/kd_s2
python train_kd.py --seed 3 --data-seed 3 --out-dir runs/kd_s3
```
Same seed ⇒ same initial weights as the matching scratch run (check against the saved `init` files). This makes each seed a **paired** comparison.

### 4.4 Matched baseline: about 13 h
```bash
python train_baseline_matched.py   # from all_mixed/
```

### 4.5 Decision or ranking evaluation: under 1 h
Runs on the cached predictions of every checkpoint.

### 4.6 Deployment benchmark: about 2–3 h

### 4.7 Optional, only if the machine is idle before Nov 26
- Seeds 4–5 for scratch and KD (~23 h). Five paired seeds are much more convincing than three.
- Teacher seeds 2–3 (~26 h).

**Compute total:** essentials ≈ 60–65 h. Running jobs back to back, overnight and during the day, this finishes by about Nov 23–24, leaving room for one failed run.

**Rule for all runs:** evaluate on the test set only **after** a checkpoint has been selected on validation. Never choose seeds, epochs or settings by looking at test results.

---

## 5. Analysis (Nov 24–27)

New file: `all_mixed/evaluation/aggregate_results.py`, which reads every run JSON and writes the tables and figures.

- [ ] **Main table:** each model as mean ± standard deviation over seeds, with each seed's value listed in an appendix. Test MAPE, test MAE, validation MAPE.
- [ ] **Paired KD effect:** for each seed, KD − scratch on test MAPE. Report all three (or five) differences and their mean. State results plainly, e.g. "KD was better in 3 of 3 seeds, by 1.6–2.4 points". Do not claim statistical significance with n = 3. With n = 5, a paired t-test or a sign test may be reported with that caveat.
- [ ] **Topology-diversity effect:** matched baseline vs teacher, with the same protocol and validation choice.
- [ ] **Per-size figure:** MAPE against topology size (200–300) for every model.
- [ ] **Decision/ranking table** from 4.5.
- [ ] **Deployment table:** latency (construction, forward, end-to-end) and peak memory under each CPU budget.
- [ ] Update the accuracy-vs-size figure to show error bars across seeds.

---

## 6. Report and presentation updates (Nov 26–30)

Report sections to revise (the files are in `report/chapters/`):

| Section | Change |
|---|---|
| Abstract | New mean ± std numbers; mention seeds, decision/ranking and deployment results |
| 1.2 Objectives | Objective 3 can add "decision-level and resource-constrained evaluation" |
| 4.3 Settings | Seeds, identical initialisation, the matched baseline, the new KD selection metric |
| 4.4 Metrics | Add MAE/RMSE, per-size MAPE, decision/ranking metrics, deployment protocol |
| 5.1 Overall | Replace single-run numbers with mean ± std; keep v0 as the "initial run" |
| 5.3 Analyses | Add per-size, decision/ranking and deployment subsections; update the KD discrepancy section with the measured check from 3.4 |
| 5.4 Discussion | Rewrite the KD caveats and the baseline confound according to what the new runs show |
| Threats to validity | Remove the fixed items; keep what still applies (simulated data, no live SDN) |
| 6 Conclusion and future work | Remove completed items |
| Appendix | Per-seed tables; new reproducibility commands |

Presentation (video, 6–8 min):
- Results slide: mean ± std bars, plus the paired KD-effect chart.
- One slide: "Is it useful to a controller?" (decision/ranking and deployment latency).
- Prepare viva answers for:
  - Why not Mininet?
  - Why MAPE?
  - Why does the student lose accuracy? (the ToS/queue information lost in the fusion)
  - How do you know the test set was never used for selection?
  - What does n = 3 seeds allow you to claim?

---

## 7. Schedule

| Date | Work | Machine (overnight) |
|---|---|---|
| Nov 18 (Wed) | Step 0; start 3.1–3.2 | Cache run (3.3) |
| Nov 19 (Thu) | Finish 3.1–3.4; smoke tests; reproduce v0 numbers | scratch_s1 |
| Nov 20 (Fri) | 3.5, 3.6 inspection step | scratch_s2 → scratch_s3 |
| Nov 21 (Sat) | 3.6 implementation, 3.7 script | kd_s1 |
| Nov 22 (Sun) | Check runs; start report edits for the methods sections | kd_s2 |
| Nov 23 (Mon) | Decision/ranking evaluation on finished runs | kd_s3 |
| Nov 24 (Tue) | Deployment benchmark; aggregation script | Matched baseline |
| Nov 25 (Wed) | Analysis; figures | Optional seeds 4–5 |
| Nov 26 (Thu) | Report: Chapters 4–5 | Optional runs continue |
| Nov 27 (Fri) | Report: abstract, Chapters 1 and 6, appendix; final numbers locked | — |
| Nov 28 (Sat) | Slides | — |
| Nov 29 (Sun) | Record video; rehearse viva questions | — |
| Nov 30 (Mon) | Re-record if needed; check every number against the JSON files | — |
| **Dec 1 (Tue)** | **Upload video before 08:00**; presentation and viva | — |
| Dec 2 (Wed) | Final report proofreading; compile in Overleaf | — |
| **Dec 3 (Thu)** | **Print and submit report** | — |

**Lock rule:** no new experiments enter the report after **Nov 27**. Later runs may be mentioned at the viva, not in the submitted numbers.

---

## 8. Risks and fallbacks

| Risk | Fallback |
|---|---|
| The cache format fails with ragged tensors | Use the pickle-based cache (3.3). If that also fails, run seed evaluations on the test set only, skip validation re-evaluation, and accept 2.5 h per evaluation (this limits the work to 3 seeds) |
| The cached evaluator does not reproduce the v0 numbers | Do not continue until fixed; it would invalidate every new number |
| A training run crashes or the machine reboots | Scripts resume from the latest checkpoint; check logs every morning and evening |
| Not enough time for all seeds | Minimum acceptable: 3 paired seeds for scratch and KD. Next to drop: matched baseline, then the deployment benchmark's Docker variant |
| The dataset has no same-traffic routing groups | Use the ranking evaluation (3.6) and describe it accurately |
| The KD gain disappears across seeds | Report it honestly: "the single-run gain was not reproduced". This is still a valid, defensible finding, and it changes the conclusion, not the quality of the work |
| The matched baseline turns out as good as the teacher | Then the claim becomes "the gain came from training length, not topology diversity". Rewrite 5.4 accordingly |

---

## 9. Definition of done

The research is defensible when all of these hold:

- [ ] The cached evaluator reproduces all four v0 test numbers to within 0.01.
- [ ] Scratch and KD each have ≥3 seeds with **identical initial weights per seed** (proven by the saved `init` files).
- [ ] KD selects its checkpoint with the same flow-weighted metric as all other models.
- [ ] A matched-protocol baseline exists, or its absence is stated as a limitation.
- [ ] Test results include per-size MAPE and MAE, not only an aggregate.
- [ ] A decision-level or ranking evaluation exists on the test split, named accurately.
- [ ] Latency and memory are reported end-to-end under a constrained CPU budget, including student-KD.
- [ ] Every number in the report and slides traces to a JSON file in `runs/` or `results/`.
- [ ] No claim exceeds the evidence. Live SDN deployment, pruning/quantisation and jitter/loss remain future work.
