# RouteNet-Fermi Research — Full Progress Summary

> Last updated: 2026-11-22 (report now includes the emulation; Dec 1 work order in [`dec1_execution_plan.md`](dec1_execution_plan.md)) | Historical offline results retained separately from new deployment evidence

---

## 1. Project Overview

**Goal:** Investigate GNN-based network performance prediction using RouteNet-Fermi, with a focus on **delay prediction** and **out-of-distribution (OOD) generalization** to unseen topology sizes. The research explores knowledge distillation / teacher-student approaches to improve generalization.

**Current scope:** offline accuracy/compression evaluation plus a completed
controlled SDN emulation proof of concept. Production deployment, arbitrary
real-traffic accuracy and guaranteed real-time operation have not been established.

**Reference Papers:**
- RouteNet: Leveraging GNNs for network modeling and optimization in SDN
- RouteNet-Fermi: Network Modeling with Graph Neural Networks
- T2-GNN: GNNs for Graphs with Incomplete Features and Structure via Teacher-Student Distillation
- On Self-Distilling Graph Neural Network
- Learning to Distill Graph Neural Networks
- LightGNN: Simple Graph Neural Network for Recommendation

---

## 2. Dataset: `all_mixed` (ch22 Challenge Dataset)

### Current Dataset Roots

```text
data/
├── all_mixed/       # Original baseline split: train, validation, broad test
└── all_mixed_v1/    # Teacher/student split: train, validation, held-out OOD test
    └── validation_extra/  # Preserved additional split; not used in recorded runs
```

The original baseline scripts use `data/all_mixed/{train,validation,test}`.
Teacher/student scripts use `data/all_mixed_v1/{train,validation,test}`.
The baseline test spans 50–300-node topologies; the v1 test is restricted to
200–300-node OOD topologies.

### Original Split

| Split | Samples | Topology Sizes | Notes |
|-------|---------|----------------|-------|
| **Train** | 11,761 usable (11,763 raw, 2 filtered) | 10-node (×7 topologies) | Small topologies only |
| **Validation** | 25 | 10-node | Small topologies only |
| **Test** | 877 usable (878 raw, 1 filtered) | 50, 75, 100, 130, 170, 200, 240, 260, 280, 300 | Wide OOD range |

### Sample Structure (1 sample = 1 network scenario)
- **852 links** — `capacity` shape `(852, 1)`
- **2,124 queues** — `queue_size` shape `(2124, 1)`
- **57,360 paths/flows** — label `y` shape `(57,360,)` = per-flow average delay
- Features include: `traffic`, `packets`, `eq_lambda`, `model` (traffic model type), `policy` (scheduling), `priority`, `weight`, link/queue/path connectivity via `RaggedTensors`

---

## 3. Experiment 0: Original RouteNet (v1) Evaluation

> [!NOTE]
> This was an early evaluation using the **original RouteNet** (TF1-based Estimator API), not RouteNet-Fermi.

- **Dataset:** `geant2bw` (GEANT topology, 9/12/15 node variants, SP and weighted routings)
- **Checkpoint:** `model.ckpt-260380` (pre-trained)
- **Result:**

| Metric | Value |
|--------|-------|
| MSE Loss | 0.0211 |
| MAE | 0.0600 |
| MRE | 2.99% |
| Pearson ρ | **0.9894** |

This confirmed the pre-trained RouteNet model works well on in-distribution data.

---

## 4. Experiment 1: RouteNet-Fermi Baseline Reproduction

### Training — Run 1 (Jan 25, 2026)
- **Script:** `few_shot_all.py`
- **Config:** 15 epochs, 2,500 steps/epoch, batch size = 1 (1 graph per step)
- **Train data:** 10,000 samples from `all_mixed/train` (10-node topologies)
- **Best checkpoint:** `03-6.50` (val_loss = 6.499, early stopping)
- **Training time:** ~1h 40m

| Epoch | Train Loss | Val Loss | Saved? |
|-------|-----------|----------|--------|
| 1 | 11.649 | 7.371 | ✅ Best |
| 3 | 6.135 | **6.499** | ✅ Best |
| 15 | 3.865 | 7.068 | ❌ Overfitting |

### Validation Results — Run 1
| Test Set | Samples | Loss (MAPE) |
|----------|---------|-------------|
| Validation (50 samples) | 50 | **7.474** |
| Full Test (878 samples) | 878 | **9.171** |

### Training — Run 2 (Feb 5, 2026)
- Same config, but **saving all checkpoints** (not just best)
- **Best checkpoint:** `15-6.41` (last epoch, val_loss = 6.410)
- **Training time:** ~1h 45m

### Validation Results — Run 2
| Test Set | Samples | Loss (MAPE) |
|----------|---------|-------------|
| Validation (50 samples) | 50 | **6.851** |
| Full Test (878 samples) | 878 | **8.225** |
| Shared OOD test (200–300 nodes) | 478 | **8.4011** |

The shared-test result reuses checkpoint `15-6.41` and evaluates it on the exact
200–300-node test split used for teacher and student evaluation. It was run
after training, without changing the checkpoint or training data.

### Training — Run 3 (Apr 12, 2026)
- Same checkpoint `15-6.41`, re-evaluation
- Full test loss: **8.225** (consistent)

> [!IMPORTANT]
> **Key observation:** The baseline model trained only on 10-node topologies generalizes **poorly** to larger OOD topologies (50–300 nodes). Val loss ~6.5–7.4 vs. test loss ~8.2–9.2.

---

## 5. Experiment 2: Teacher Model for Delay Prediction

### Motivation
To improve OOD generalization, a **teacher model** was trained on a wider range of topology sizes (10–170 nodes), then tested on truly unseen large topologies (200–300 nodes).

### Teacher Data Split

**Created by:**
1. Moving test topologies of sizes 50, 75, 100, 130, 170 from `test/` into the teacher training split
2. Moving all original training data (10-node) into the teacher training split
3. Holding out 1 topology per size for the teacher validation split
4. Leaving only 200–300 node topologies in `test/`

| Current split | Historical name in result artifacts | Topologies | Samples | Delay Range |
|---------------|----------------------------------------|-----------|---------|-------------|
| **train** | `train_teacher` | 6×10, 1×50, 1×75, 1×100, 1×130, 1×170 | 9,972 | 0.010–2.099, mean 0.217 |
| **validation** | `val_teacher` | 1×10, 1×50, 1×75, 1×100, 1×130, 1×170 | 2,191 | 0.010–1.874, mean 0.252 |
| **test (OOD)** | `test` | 2×200, 4×240, 2×260, 2×280, 2×300 | 478 | 0.006–0.085, mean 0.017 |

> [!WARNING]
> The test set delay values (mean 0.017) are an order of magnitude smaller than training delays (mean 0.217). This distribution shift in the target variable itself is a significant challenge.

### Teacher Training (Feb 5–14, 2026)
- **Script:** Modified `few_shot_all.py` for teacher configuration
- **Config:** 30 epochs, steps_per_epoch = 9,972 (full dataset)
- **Training time:** **~13 hours** (CPU-only, no GPU)
- **Best checkpoint:** `ckpt-26` (val_loss = 4.714)

| Epoch | Train Loss | Val Loss | Saved? |
|-------|-----------|----------|--------|
| 1 | 7.617 | 7.716 | ✅ |
| 5 | 5.079 | 6.083 | ✅ |
| 10 | 4.397 | 5.487 | ✅ |
| 12 | 4.225 | 5.193 | ✅ |
| 17 | 4.090 | 5.086 | ✅ |
| 21 | 3.964 | 4.733 | ✅ |
| **26** | **3.804** | **4.714** | ✅ **Best** |
| 30 | 3.657 | 5.215 | ❌ |

### Teacher Evaluation Results

| Test Set | Samples | Loss (MAPE) | Time |
|----------|---------|-------------|------|
| val_teacher | 2,191 | **4.714** | 11 min |
| test (OOD: 200–300 nodes) | 478 | **5.373** | 2h 42m |

---

## 6. Results Comparison

| Model | Train Topos | Val Loss | OOD Test Loss | Test Topologies |
|-------|-------------|----------|---------------|-----------------|
| **Baseline (Run 1)** | 10 only | 6.499 | 9.171 | 50–300 nodes |
| **Baseline (Run 2; historical broad test)** | 10 only | 6.410 | 8.225 | 50–300 nodes |
| **Baseline (Run 2; shared OOD test)** | 10 only | 6.410 | **8.4011** | **478 samples; 200–300 nodes** |
| **Teacher** | 10–170 | **4.714** | **5.373** | 200–300 nodes |

> [!IMPORTANT]
> On the common 478-sample 200–300-node test set, the teacher's MAPE is **3.0277 percentage points lower** than the baseline (5.3734% vs 8.4011%), a **36.0% relative reduction**. The earlier broad-test baseline score (8.2249%) is on a different test distribution and is not used for this direct comparison.

---

## 7. Experiment 3: Student GNN — Architecture Design

### Motivation
With the teacher model established, a **lightweight student model** (`student_delay_model.py`) was designed to explore whether a significantly smaller GNN can achieve competitive accuracy — first trained from scratch (baseline), then via knowledge distillation.

### Student Architecture (Queue-Link Fusion)

| | Teacher (RouteNet-Fermi) | Student |
|---|---|---|
| Graph type | Tripartite (Link ↔ Queue ↔ Path) | **Bipartite (Interface ↔ Path)** |
| Key innovation | — | Queue-Link Fusion: mean-pool queues into link embeddings |
| GRU cells | 3 | **2** |
| State dimension | 32 | **16** |
| Iterations | 8 | **4** |
| MP stages / iter | 3 | **2** |
| **Total MP stages** | **24** | **8 (67% reduction)** |
| **Trainable parameters** | **27,025** | **4,505 (83.3% fewer)** |
| Readout formula | queue_delay + trans_delay | same (preserved) |

- **Key idea:** Instead of maintaining separate queue embeddings (requiring a 3rd GRU), queues are mean-pooled into their parent link's state before message passing — reducing the graph to a bipartite structure at a fraction of the compute cost.
- **Files:** `student_delay_model.py`, `all_mixed/student/train_scratch.py`, `all_mixed/student/train_kd.py`

---

## 8. Experiment 4: Student — Train from Scratch (Baseline)

### Setup
- **Script:** `all_mixed/student/train_scratch.py`
- **Dataset:** Same teacher split — 9,972 train samples (topologies 10–170 nodes)
- **Loss function:** MAPE (student predictions vs ground truth only — **no teacher involvement**)
- **Config:** max 30 epochs, EarlyStopping patience=5, ~24 min/epoch
- **Training time:** ~5.4 hours (overnight run, Aug 11–12, 2026)
- **Best checkpoint:** `ckpt_dirs_scratch/ckpt-08`

### Per-Epoch Val Loss

| Epoch | Val Loss (MAPE) | Saved? |
|-------|----------------|--------|
| 1 | 13.19% | ✅ |
| 2 | 12.89% | ✅ |
| 3 | 12.40% | ✅ |
| 4 | 11.68% | ✅ |
| 5 | 11.76% | ❌ |
| 6 | 11.60% | ✅ |
| 7 | 11.23% | ✅ |
| **8** | **10.65%** | ✅ **← BEST** |
| 9 | 11.26% | ❌ |
| 10 | 11.07% | ❌ |
| 11 | 11.10% | ❌ |
| 12 | 10.86% | ❌ |
| 13 | 11.05% | ❌ → EarlyStopping triggered |

> Model weights automatically restored to epoch 8 on exit.

### Results

| Metric | Value |
|--------|-------|
| **Best Val MAPE** | **10.65%** |
| **OOD Test MAPE** | **11.9244%** |
| **OOD test set** | **478 samples, 200–300-node topologies** |
| **Evaluated checkpoint** | `ckpt_dirs_scratch/ckpt-08` |
| **OOD evaluation time** | **8,938 seconds (~2h 29m)** |

> [!NOTE]
> On the same 200–300-node OOD set, scratch has **2.22× the teacher's MAPE** (11.9244% vs 5.3734%). This measures the accuracy/efficiency trade-off for the compressed model without knowledge distillation.

---

## 9. Experiment 5: Student — Knowledge Distillation (KD)

### Setup
- **Script:** `all_mixed/student/train_kd.py`
- **Config:** max 30 epochs, EarlyStopping patience=5, α=0.5 (50% ground truth, 50% teacher predictions)
- **Best checkpoint:** `ckpt_dirs_kd_alpha0.5/ckpt-19`

### Results
- The training successfully completed via Early Stopping at epoch 24.
- Best weights were restored from **epoch 19**.
- **Training-reported best Val MAPE:** **9.6439%** (`results_kd_alpha0.5.json`)
- **Evaluated checkpoint:** `ckpt_dirs_kd_alpha0.5/ckpt-19`
- **Evaluation Val MAPE:** **9.0444%**
- **OOD Test MAPE:** **9.8280%** on the same **478-sample, 200–300-node** set as the teacher
- **OOD evaluation time:** **9,032 seconds (~2h 30m)**

> [!NOTE]
> The evaluated KD checkpoint's validation MAPE is 9.0444%, versus 10.6455% for scratch. This differs from the 9.6439% training-reported best validation value; retain both values with their provenance rather than treating them as interchangeable.

---

## 10. Results Comparison (All Experiments)

### Accuracy

| Model | Train Topos | Val MAPE | OOD Test MAPE | Test Set |
|-------|-------------|----------|---------------|----------|
| Baseline (Run 1) | 10-node only | 6.50% | 9.17% | 50–300 nodes; prior evaluation |
| Baseline (Run 2; broad test) | 10-node only | 6.41% | 8.2249% | 878 raw / 877 usable; 50–300 nodes |
| Baseline (Run 2; shared OOD test) | 10-node only | 6.41% | **8.4011%** | **478 samples; 200–300 nodes** |
| **Teacher** | 10–170 nodes | **4.7139%** | **5.3734%** | **478 samples; 200–300 nodes** |
| **Student-Scratch** | 10–170 nodes | **10.6455%** | **11.9244%** | **478 samples; 200–300 nodes** |
| **Student-KD (α=0.5)** | 10–170 nodes | **9.0444%** | **9.8280%** | **478 samples; 200–300 nodes** |

> [!IMPORTANT]
> The earlier baseline OOD results (8.2249% and 9.1707%) used a broader **50–300-node** test set and are not directly comparable with teacher/student results. The baseline has now been evaluated on the same 478-sample 200–300-node test set: it scores 8.4011%, compared with 5.3734% for the teacher, 11.9244% for scratch, and 9.8280% for KD. KD improves over scratch on this common set, while the teacher remains most accurate.

### Inference Speed & Model Size (Measured — 30 runs, single sample)

| Model | Params | Size | MP Stages | Latency (mean) | Speedup vs Teacher |
|-------|--------|------|-----------|----------------|--------------------|
| Baseline (10-node only) | 27,025 | 105.6 KB | 24 | 177.5 ms | 1.0× (same arch) |
| Teacher (RouteNet-Fermi) | 27,025 | 105.6 KB | 24 | 178.5 ms | — |
| **Student-Scratch** | **4,505** | **17.6 KB** | **8** | **34.2 ms** | **5.2× faster** |

> [!NOTE]
> Baseline and Teacher have identical architectures and parameter counts, with similar measured inference times. Their training data differ.
> The historical 5.2× speedup is a model-only CPU benchmark on one validation sample. (An earlier note said "9,900 flows", but no saved file records the size; the report states it was not recorded. Task T3 of `dec1_execution_plan.md` re-measures it.) It is not total SDN controller latency or a real-time guarantee. Separate six-switch deployment-input timings are reported below; the workloads must not be mixed.

### The Research Narrative

```
Baseline (10-node training)  →  8.40% OOD error on shared test, ~178 ms inference
        ↓  train on more topology sizes
Teacher  (10-170 training)   →  5.37% OOD error,  ~178 ms inference
        ↓  compress the model
Student  (scratch, no KD)    → 11.92% OOD error,   ~34 ms inference  (5.2× faster, accuracy cost)
        ↓  add Knowledge Distillation
Student-KD                   →  9.83% OOD error  (historical KD result; separate deployment timings below)
```

- **Script:** `evaluation/inference_benchmark.py`
- **Results saved:** `evaluation/inference_benchmark_results.json`
- **Shared-test baseline evaluation:** `all_mixed/results_baseline_ood.json`
- **Shared-test baseline console log:** `all_mixed/baseline_ood.log`

---

## 11. Infrastructure Notes

- **Hardware:** CPU-only (Fedora Linux, no NVIDIA GPU detected)
- **Framework:** TensorFlow 2.x with `tf.keras` (RouteNet-Fermi) / TF 1.x Estimator API (original RouteNet)
- **Training throughput:** ~143–150 ms/step for mixed topologies (10–170 nodes)
- **Loss metric:** MAPE (Mean Absolute Percentage Error) via custom loss
- **Inference benchmark:** `time.perf_counter()`, 5 warm-up runs + 30 timed runs per model, single-graph sample (9,900 flows)

---

## 12. Controlled Mininet Routing Validation — Completed

### Setup and what was actually measured

- **Implementation:** `mininet_experiment/`; separate Mininet runner, Ryu
  OpenFlow 1.3 controller and TensorFlow model worker. Earlier model-driven
  pilots demonstrated selection and installation of explicit forwarding rules.
- **Network:** six switches, one host each, 30 ordered host-pair flows; foreground
  host 0 to host 5. Background routes/demands remained fixed.
- **Traffic:** deterministic fixed-size CBR UDP, 125-byte frames, 20,000-bit/s
  inter-switch links, 32,000-bit FIFO queues, no configured propagation delay.
  Reference busiest-link offered utilization is 60%; all components share a laptop.
- **Frozen checkpoints:** teacher `ckpt-26`, scratch `ckpt-08`, KD `ckpt-19`.
  No retraining or checkpoint selection using Mininet outcomes was performed.
- **Validation:** force each of three candidate paths, then measure for 60 seconds
  after two seconds warm-up, with drainage afterward. Five repetitions per path,
  **15 accepted trials**, using a fixed, rotated execution order.
- Scenario hashes matched the frozen predictions. Rules, path counters, queue
  settings, offered rates and packet pacing passed the recorded checks.

### Network outcomes

| Candidate | Path | Mean delay, ms | SD of five trial means, ms | Mean of trial p95 values, ms |
|---|---|---:|---:|---:|
| 0 | 0 → 1 → 5 | 123.785 | 0.016 | 169.144 |
| **1** | **0 → 4 → 5** | **112.929** | **0.019** | **148.144** |
| 2 | 0 → 2 → 3 → 5 | 196.331 | 0.099 | 296.375 |

All three models predicted ranking **1, 0, 2**, matching measured mean-delay
ranking in all five repetitions. Candidate 1 reduced mean delay by **8.770%**
relative to candidate 0 (paired-reduction SD **0.023 percentage points**), about
**10.856 ms**. Both have two hops: this compares one deterministic shortest-path
tie choice with delay-based selection, not every shortest-path routing policy.

Each trial received all 180 foreground packets: **900 per path, 2,700 total,
zero observed foreground loss**. The p95 column averages trial percentiles;
it is not a pooled percentile. SD describes trial means, not individual packets.

### Prediction and inference cost

| Model | Mean APE across 15 foreground trial means | Three-candidate inference mean, ms | Sample SD, ms | p95, ms |
|---|---:|---:|---:|---:|
| Teacher | 7.20% | 7.55 | 0.45 | 8.42 |
| Scratch | 5.28% | 4.08 | 0.51 | 5.35 |
| KD | 6.77% | 3.62 | 0.23 | 4.05 |

The errors compare frozen predictions against **foreground route trial means**;
they are not full-dataset MAPEs or replacements for historical OOD scores.
Model timing used 30 passes after five warm-ups, with all three inputs prepared
before timing. Student inference cost is approximately **1.85×/2.09× lower**
than teacher on this input. This excludes preprocessing, communication, rule
installation and preflight. Repeated total model decision-to-install latency
was not measured by these fixed-path trials.

All models choose the same candidate: **no KD routing advantage is demonstrated**.
The 5/5 agreement repeats one frozen choice against five network measurements;
it is not five independent traffic scenarios or five new model-driven decisions.
Scratch's lower error here does not establish general superiority on other data.

### Evidence and limitations

- Calculations, trial sources/hashes and checkpoint identities:
  `mininet_experiment/outputs/six_validation_summary.json`.
- Frozen predictions and raw inference timings:
  `mininet_experiment/outputs/six_validation_predictions.json`.
- Raw trials: `six_validation_path_0/1/2` and
  `six_validation_rep2/3/4/5_path_<candidate>` under `outputs/`.
- This supports a **repeatable controlled routing pilot**. One topology, load,
  FIFO configuration and deterministic traffic schedule do not demonstrate
  realistic traffic robustness, physical edge performance or real-time readiness.

---

## 13. Abilene Investigation — Limitation Remains

The 12-switch Abilene pilot used published topology/capacities and adapted traffic
demands from the authors' real-traces dataset, but generated fixed-size CBR UDP.
It did **not** replay original packet traces. Initial runs passed operational
checks while producing inaccurate and sometimes physically impossible predictions.

- Our teacher predicted **1,842.831 ms** for a route measured at **0.804 ms**.
  Internal tracing found large embeddings, saturated GRU gates and approximately
  **3.07 million queued bits per hop**, versus 32,000-bit configured queues.
- Scratch produced negative queue contributions. KD also had invalid queue
  contributions on some background flows despite positive foreground totals.
- Full training-archive scanning covered 404 archives / 9,972 raw sample lines:
  positive traffic ranged **74.626–4,367.03 bits/s**, versus up to **344,561.89
  bits/s** in our Abilene scenario. These are observed ranges, not guarantees of
  distribution coverage. No uniform scale fits all demand/capacity ranges while
  preserving their ratios. Diagnostic scaling was not adopted as a validated fix.
- The deployment adapter changes `link_to_path`/`queue_to_path` hop order to match
  physical routes and reverse connectivity. A known validation sample gives
  teacher 4.1395% MAPE with historical ordering and 4.0373% with deployment order.
  These are single-sample checks, not full evaluation. Restoring historical order
  leaves the Abilene failure at about 1.843 seconds, so ordering alone does not
  explain it. Original training/evaluation scripts remain unchanged.

### Direct comparison with the authors' real-traffic checkpoint

On ten Abilene samples selected before evaluation (**1,320 flows**), both
checkpoints received identical inputs through the authors' unchanged pipeline:

| Checkpoint | MAPE | MAE |
|---|---:|---:|
| Authors' real-traffic `01-5.20` | **3.81%** | **0.158 ms** |
| Our all-mixed teacher `ckpt-26` | **12,636.82%** | **74.621 ms** |

The model definitions and normalization constants match; learned weights differ.
The authors' weights still predicted **38.56 ms** on our changed CBR input for
the route measured at 0.804 ms, and failed some physical checks. Their trace
pipeline encodes traffic features differently. Diagnostic feature interventions
were not applied to deployment inputs as fixes.

The supported finding is **checkpoint/input-domain incompatibility**, not that
RouteNet-Fermi universally fails on real traffic or that historical offline
results are invalid. The [authors' real-traffic experiment](https://arxiv.org/html/2212.12070#S5.SS3.SSS2)
adapted a previous checkpoint using 200 GEANT samples. Our ten-sample comparison
is a diagnostic, not a reproduction of their full reported test score.

### Safeguards and stored evidence

Prediction checks now stop invalid model runs **before starting Mininet**.
They detect non-finite values, negative/overfull queues and impossible total
delays for the single-FIFO configuration. Nothing is clipped. The guards are
tested; **Abilene prediction accuracy itself is not fixed**. All nine six-switch
model/candidate combinations pass these checks and were subsequently compared
with measurements in Section 12.

Evidence: `mininet_experiment/outputs/adapter_audit.json`,
`teacher_mechanism_audit.json`, `six_switch_physical_audit.json`, and
`author_real_traffic/{comparison.json,provenance.json}`. Author source is pinned
to commit `ef72a37bb6f677e149f354f088f5836f29c862b5`. Diagnostic/preparation
scripts and complete prior notes are preserved under
`outputs/source_archive_before_cleanup/`. All `outputs/` artifacts are local
and Git-ignored; an agent using only GitHub cannot claim to have inspected them.

---

## 14. Current State & Next Steps

### Completed ✅
- [x] Original RouteNet evaluation on geant2bw (ρ=0.989)
- [x] RouteNet-Fermi baseline reproduction on `all_mixed`
- [x] Teacher model training with extended topology coverage (10–170 nodes)
- [x] OOD evaluation — teacher achieves 5.37% on 200–300 node topologies
- [x] Deep-dive understanding of GNN architecture, data pipeline, and embedding mechanics
- [x] Student model design (Queue-Link Fusion, bipartite graph, 83.3% fewer parameters than teacher)
- [x] Student trained from scratch — best val MAPE: **10.65%** (epoch 8, ckpt-08)
- [x] Inference benchmark — student is **5.2× faster** (34 ms vs 178 ms), **83.3% fewer parameters** (4,505 vs 27,025)
- [x] **Exp 5 — Student KD training:** Completed with α=0.5, training-reported best val MAPE: **9.6439%** (epoch 19, ckpt-19)
- [x] **Exp 6 — Student OOD evaluation:** On the same 478-sample, 200–300-node set as teacher: scratch **11.9244%**, KD **9.8280%**; evaluation results saved in `all_mixed/student/results_student.json`.
- [x] **Exp 7 — Fair baseline OOD evaluation:** Baseline checkpoint `15-6.41` scored **8.4011% MAPE** on the same 478 samples as teacher and students; result and console log saved separately.
- [x] Mininet/Ryu/model-worker integration with the frozen historical checkpoints.
- [x] Six-switch fixed-path validation: **15 accepted trials**, five per candidate;
  all models rank the measured candidates correctly in this controlled scenario.
- [x] Repeated three-candidate inference timing for teacher, scratch and KD on the
  matching six-switch inputs; raw timings saved separately from network results.
- [x] Abilene failure investigated, including original author-checkpoint comparison;
  negative findings retained and pre-network physical prediction guards added.
- [x] Active experiment simplified to five runtime Python files; diagnostic sources
  preserved with result artifacts. Existing report files were left unchanged.
- [x] Report-agent handoff prepared in
  [`mininet_experiment/README.md`, Section 9](../mininet_experiment/README.md#9-handoff-for-the-agent-updating-the-report).

### Immediate Next Steps
- [x] **Report update (22 Nov):** the LaTeX report now includes the controlled
  emulation, the Abilene diagnostic, the physical checks and their limitations
  (Ch3 §3.7, Ch4 §4.3.3, Ch5 §5.3–5.4, Appendix A).
- [ ] **Dec 1 strengthening work:** controlled KD study, per-size and hop-order
  evaluation, latency v2. See [`dec1_execution_plan.md`](dec1_execution_plan.md).
- [ ] **Optional further validation:** predeclare additional traffic/load conditions
  and check predictions before collecting measurements. These experiments are
  proposed, not completed; the planned five-repetition fixed scenario is finished.
- [ ] **Total processing cost:** repeat end-to-end decision timings separately if
  making broader controller-latency claims. Inference-only timings are insufficient.
- [ ] **Historical metric provenance:** retain the distinct KD training/evaluation
  validation values and resolve their aggregation/protocol explanation separately.

### Future work, not required to describe the completed pilot

- Real-traffic adaptation with appropriate labelled training data and independent
  evaluation for every compared model; no new training has been performed here.
- Broader traffic/topology tests, jitter/loss prediction and resource profiling.
- Stronger attribution of fusion benefits beyond width/iteration reduction.
