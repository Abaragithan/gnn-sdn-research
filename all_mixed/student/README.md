# Student GNN Training — Command Reference

## Project Structure
```
gnn-sdn-research/
├── delay_model.py               # Teacher model (RouteNet-Fermi, 32-dim, 8 iters, 3 stages)
├── student_delay_model.py       # Student model (Queue-Link Fusion, 16-dim, 4 iters, 2 stages)
└── all_mixed/
    ├── data_generator.py        # Shared data pipeline
    ├── teacher/
    │   ├── train.py             # Teacher training
    │   └── evaluate.py          # Teacher evaluation
    ├── student/
    │   ├── train_scratch.py     # Student: train from scratch (baseline)
    │   ├── train_kd.py          # Student: train with knowledge distillation
    │   └── evaluate.py          # Student evaluation
    └── evaluation/
        ├── evaluate_all.py      # Compare teacher vs student (all models)
        └── count_parameters.py  # Count & compare model parameters
```

## Dataset
- **Location:** `data/all_mixed_v1/` (symlinked as `data/all_mixed/`)
- **Train:** `train_teacher/` — topologies with 10–170 nodes (9972 samples)
- **Val:**   `val_teacher/`   — topologies with 10–170 nodes
- **Test:**  `test/`          — topologies with 50–300 nodes (OOD: 200–300 unseen)

---

## Step 1 — Train Student from Scratch (Baseline)

```bash
cd ~/gnn-sdn-research/all_mixed/student

# Start in background (survives terminal close)
PYTHONUNBUFFERED=1 nohup /home/abaragithan/miniconda3/envs/routenet-fermi/bin/python -u train_scratch.py > logs_scratch.txt 2>&1 &
echo $! > pid_scratch.txt && echo "Started PID: $(cat pid_scratch.txt)"

# Monitor progress
tail -f logs_scratch.txt

# Stop anytime (resumes from last completed epoch)
kill $(cat pid_scratch.txt)
```

**Expected:** ~24 min/epoch, ~10–15 epochs (early stopping patience=5), ~4–6 hours total.  
**Checkpoints saved to:** `ckpt_dirs_scratch/`  
**Resume:** Just re-run the same start command — it auto-detects the latest checkpoint.

### Reset and retrain from scratch
```bash
rm -rf ~/gnn-sdn-research/all_mixed/student/ckpt_dirs_scratch
# Then re-run the start command above
```

---

## Step 2 — Train Student with Knowledge Distillation

> Run this **after** scratch training finishes.

```bash
cd ~/gnn-sdn-research/all_mixed/student

PYTHONUNBUFFERED=1 nohup /home/abaragithan/miniconda3/envs/routenet-fermi/bin/python -u train_kd.py > logs_kd.txt 2>&1 &
echo $! > pid_kd.txt && echo "Started PID: $(cat pid_kd.txt)"

# Monitor
tail -f logs_kd.txt

# Stop
kill $(cat pid_kd.txt)
```

**Requires:** Teacher checkpoint at `all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/`  
**Checkpoints saved to:** `ckpt_dirs_kd_alpha0.5/`  
**KD Loss:** `alpha * MAPE(student, y_true) + (1-alpha) * MSE(student, teacher)` with alpha=0.5

---

## Step 3 — Evaluate All Models

```bash
cd ~/gnn-sdn-research/all_mixed/evaluation

/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python evaluate_all.py
```

Outputs a comparison table:
```
Model                          Params    Val Loss   OOD Loss   OOD Time
Teacher (RouteNet-Fermi)        4,945      4.71%      5.37%      Xs
Student-Scratch (no KD)         1,241      X.XX%      X.XX%      Xs
Student-KD (alpha=0.5)          1,241      X.XX%      X.XX%      Xs
```

Saves results to `results_comparison.json`.

---

## Step 4 — Evaluate Student Only

```bash
cd ~/gnn-sdn-research/all_mixed/student
/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python evaluate.py
```

---

## Useful One-Liners

```bash
# Check if training is running
ps aux | grep train_scratch | grep -v grep

# Check current epoch from log
grep "Epoch " ~/gnn-sdn-research/all_mixed/student/logs_scratch.txt | tail -3

# Check best val_loss so far
grep "val_loss improved" ~/gnn-sdn-research/all_mixed/student/logs_scratch.txt

# Count parameters (teacher vs student)
cd ~/gnn-sdn-research/all_mixed/evaluation
/home/abaragithan/miniconda3/envs/routenet-fermi/bin/python count_parameters.py
```

---

## Known Results

| Model | Val MAPE | Test OOD MAPE | Params | MP Stages |
|-------|---------|--------------|--------|-----------|
| Teacher (RouteNet-Fermi) | **4.71%** | **5.37%** | ~4,945 | 24 (8×3) |
| Student-Scratch | TBD | TBD | ~1,241 | 8 (4×2) |
| Student-KD (α=0.5) | TBD | TBD | ~1,241 | 8 (4×2) |

---

## Architecture Summary

| | Teacher | Student |
|--|---------|---------|
| Graph type | Tripartite (Link ↔ Queue ↔ Path) | **Bipartite (Interface ↔ Path)** |
| Key innovation | — | Queue-Link Fusion (mean-pool queues into links) |
| GRU cells | 3 | 2 |
| State dim | 32 | 16 |
| Iterations | 8 | 4 |
| MP stages/iter | 3 | 2 |
| Total MP stages | 24 | **8 (67% reduction)** |
| Parameters | ~4,945 | ~1,241 (**75% reduction**) |
| Readout formula | queue_delay + trans_delay | same (preserved) |
