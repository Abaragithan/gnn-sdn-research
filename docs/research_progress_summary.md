# RouteNet-Fermi Research — Full Progress Summary

> Last updated: 2026-08-12 (inference benchmark added) | Synthesized from all experiment logs

---

## 1. Project Overview

**Goal:** Investigate GNN-based network performance prediction using RouteNet-Fermi, with a focus on **delay prediction** and **out-of-distribution (OOD) generalization** to unseen topology sizes. The research explores knowledge distillation / teacher-student approaches to improve generalization.

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
| **Baseline (Run 2)** | 10 only | 6.410 | 8.225 | 50–300 nodes |
| **Teacher** | 10–170 | **4.714** | **5.373** | 200–300 nodes |

> [!IMPORTANT]
> The teacher model achieved a **35% reduction in OOD test loss** (8.225 → 5.373), despite testing on only the hardest, largest topologies (200–300 nodes). Exposing the model to intermediate topology sizes dramatically improved generalization.

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
| Baseline (Run 2) | 10-node only | 6.41% | 8.23% | 50–300 nodes; prior evaluation |
| **Teacher** | 10–170 nodes | **4.7139%** | **5.3734%** | **478 samples; 200–300 nodes** |
| **Student-Scratch** | 10–170 nodes | **10.6455%** | **11.9244%** | **478 samples; 200–300 nodes** |
| **Student-KD (α=0.5)** | 10–170 nodes | **9.0444%** | **9.8280%** | **478 samples; 200–300 nodes** |

> [!IMPORTANT]
> The earlier baseline OOD results (8.23% and 9.17%) used a broader **50–300-node** test set and are not directly comparable with the teacher/student results on the 200–300-node set. Re-evaluate the baseline on the exact 478-sample test set before making a direct comparison. On the common 200–300-node set, KD improves over scratch (9.8280% vs 11.9244%), while the teacher remains more accurate (5.3734%).

### Inference Speed & Model Size (Measured — 30 runs, single sample)

| Model | Params | Size | MP Stages | Latency (mean) | Speedup vs Teacher |
|-------|--------|------|-----------|----------------|--------------------|
| Baseline (10-node only) | 27,025 | 105.6 KB | 24 | 177.5 ms | 1.0× (same arch) |
| Teacher (RouteNet-Fermi) | 27,025 | 105.6 KB | 24 | 178.5 ms | — |
| **Student-Scratch** | **4,505** | **17.6 KB** | **8** | **34.2 ms** | **5.2× faster** |

> [!NOTE]
> Baseline and Teacher are **identical architectures** — same params, same inference speed. The only difference is training data diversity.
> The student's 5.2× inference speedup is the key deployment advantage, measured on CPU (the real SDN controller environment).

### The Research Narrative

```
Baseline (10-node training)  →  8.23% OOD error,  ~178 ms inference
        ↓  train on more topology sizes
Teacher  (10-170 training)   →  5.37% OOD error,  ~178 ms inference
        ↓  compress the model
Student  (scratch, no KD)    → 11.92% OOD error,   ~34 ms inference  (5.2× faster, accuracy cost)
        ↓  add Knowledge Distillation
Student-KD                   →  9.83% OOD error,   ~34 ms inference  (KD improves over scratch)
```

- **Script:** `evaluation/inference_benchmark.py`
- **Results saved:** `evaluation/inference_benchmark_results.json`

---

## 11. Infrastructure Notes

- **Hardware:** CPU-only (Fedora Linux, no NVIDIA GPU detected)
- **Framework:** TensorFlow 2.x with `tf.keras` (RouteNet-Fermi) / TF 1.x Estimator API (original RouteNet)
- **Training throughput:** ~143–150 ms/step for mixed topologies (10–170 nodes)
- **Loss metric:** MAPE (Mean Absolute Percentage Error) via custom loss
- **Inference benchmark:** `time.perf_counter()`, 5 warm-up runs + 30 timed runs per model, single-graph sample (9,900 flows)

---

## 12. Current State & Next Steps

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

### Immediate Next Steps
- [ ] **Re-evaluate baseline:** Measure the 10-node-trained baseline on the same 478-sample, 200–300-node test set before comparing it directly with teacher/students.
- [ ] **Investigate KD validation discrepancy:** Compare the training-reported best val MAPE (9.6439%) with the checkpoint evaluation result (9.0444%) and document the metric provenance.
- [ ] **Exp 7 — Repeat inference benchmark** after KD to confirm the same latency advantage.
- [ ] **Jitter/Loss models:** Extend beyond delay prediction
- [ ] **Larger-scale testing:** Generate custom datasets with even larger topologies
