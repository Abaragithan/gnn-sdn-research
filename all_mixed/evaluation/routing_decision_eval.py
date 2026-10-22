# evaluation/routing_decision_eval.py
#
# ROUTING DECISION EVALUATION — Real Use Case Validation
# =======================================================
# Validates each model in a realistic SDN controller scenario:
#
#   Given: A fixed network topology + traffic matrix
#   Task:  Choose the BEST routing from K candidate routings (SP, W-1, W-2, etc.)
#   How:   Each model predicts avg delay per routing → picks the lowest
#   Metric: Did the model pick the SAME routing as the ground truth (simulator)?
#
# DESIGN: Uses the teacher/student validation split (10-170 nodes) for speed.
# Streams samples from disk without pre-loading all into memory.
#
# Metrics:
#   - Decision Accuracy: % scenarios where model picks the optimal routing
#   - Top-2 Accuracy:    % scenarios where model's choice is in the best 2
#   - Mean Regret:       average extra delay (ms) incurred by wrong decisions
#   - Throughput:        routing evaluations per second
#
# Run:
#   cd ~/gnn-sdn-research/all_mixed/evaluation
#   /home/abaragithan/miniconda3/envs/routenet-fermi/bin/python routing_decision_eval.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import time
import json
import collections

import numpy as np
import networkx
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_generator import network_to_hypergraph, hypergraph_to_input_data, input_fn
from datanetAPI import DatanetAPI
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

# Use the v1 validation split: topologies 10-170 nodes.
# NOTE: this split WAS seen during training (it's the validation split),
# but this is still valid for routing decision evaluation — we want to measure
# how well each model RANKS routing candidates, not just predict absolute values.
# The test/ split only has 200-300 node topologies (too slow to process here).
EVAL_TOPOLOGY_SIZES = []   # empty = all sizes in the dataset

MIN_CANDIDATES = 2     # skip scenario groups with fewer than this many candidates
MAX_GROUPS     = 100   # max scenario groups to evaluate (set None for all)


# ─────────────────────────────────────────────────────────────
# Load and group scenarios
# ─────────────────────────────────────────────────────────────

def load_scenario_groups(test_path, topology_sizes, max_groups=None):
    """
    Stream samples from the dataset, group by (graph_file, traffic_intensity)
    so each group = same topology+traffic, different routing configs.
    """
    print(f"\n  Scanning dataset: {test_path}")
    sizes_str = topology_sizes if topology_sizes else "all"
    print(f"  Topology sizes:  {sizes_str}")

    api = DatanetAPI(test_path, topology_sizes=topology_sizes, shuffle=False)

    # Group by graph_file only.
    # All samples sharing the same physical topology graph file are candidates:
    # they represent different (routing, traffic) combinations on the same network.
    # The model's job: predict which routing gives the lowest avg delay.
    # We fix traffic by taking same-intensity groups within each graph.
    raw_groups = collections.defaultdict(list)
    n_loaded = 0

    for sample in api:
        try:
            G = networkx.DiGraph(sample.get_topology_object())
            T = sample.get_traffic_matrix()
            R = sample.get_routing_matrix()
            P = sample.get_performance_matrix()

            avg_delay    = float(sample.get_global_delay())
            graph_file   = sample._graph_file
            routing_name = os.path.basename(sample._routing_file).replace('.txt', '')
            intensity    = float(sample.maxAvgLambda)

            # Build GNN input
            HG = network_to_hypergraph(G=G, R=R, T=T, P=P)
            inputs, _ = hypergraph_to_input_data(HG)

            # Group by graph_file (same physical topology)
            raw_groups[graph_file].append({
                'routing_name': routing_name,
                'avg_delay':    avg_delay,
                'intensity':    intensity,
                'inputs':       inputs,
                'n_nodes':      G.number_of_nodes(),
            })
            n_loaded += 1
            if n_loaded % 200 == 0:
                n_valid = sum(1 for g in raw_groups.values() if len(g) >= MIN_CANDIDATES)
                print(f"    Loaded {n_loaded} samples → {len(raw_groups)} topology groups "
                      f"({n_valid} with ≥{MIN_CANDIDATES} candidates)...")

        except Exception as e:
            print(f"    Warning: skipped sample — {e}")
            continue

    print(f"\n  Done loading: {n_loaded} samples → {len(raw_groups)} raw groups")

    # Filter to groups with enough routing candidates
    valid_groups = [g for g in raw_groups.values() if len(g) >= MIN_CANDIDATES]
    valid_groups.sort(key=lambda g: (-len(g), -g[0]['n_nodes']))  # most candidates first

    print(f"  Valid groups (≥{MIN_CANDIDATES} candidates): {len(valid_groups)}")
    if valid_groups:
        sizes = [len(g) for g in valid_groups]
        nodes = [g[0]['n_nodes'] for g in valid_groups]
        print(f"  Candidates per group: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        print(f"  Node counts in groups: {sorted(set(nodes))}")

    if max_groups:
        valid_groups = valid_groups[:max_groups]
        print(f"  Using first {len(valid_groups)} groups")

    return valid_groups


# ─────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────

def build_and_load(model_cls, ckpt_dir, val_path, **kwargs):
    """Build model, init weights via input_fn (correct tensor types), then load checkpoint."""
    model = model_cls(**kwargs)
    model.compile(loss="mape", optimizer="adam", run_eagerly=False)
    # Use input_fn to get a correctly-typed sample for weight initialisation
    ds = input_fn(val_path, shuffle=False).take(1)
    for x, _ in ds:
        _ = model(x, training=False)
        break
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint in: {ckpt_dir}")
    model.load_weights(ckpt).expect_partial()
    print(f"    Loaded: {ckpt}")
    return model


# ─────────────────────────────────────────────────────────────
# Routing decision evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_routing_decisions(model, scenario_groups):
    """
    For each scenario group (K routing candidates, same topology+traffic):
      1. Model predicts avg delay for each candidate
      2. Model picks the candidate with LOWEST predicted delay
      3. Compare against ground truth (lowest actual simulated delay)
    """
    n_correct    = 0
    n_top2       = 0
    total        = 0
    regrets      = []    # extra actual delay incurred by wrong decisions (seconds)
    latencies_ms = []

    for group in scenario_groups:
        gt_delays   = np.array([c['avg_delay'] for c in group])
        gt_best_idx = int(np.argmin(gt_delays))
        gt_sorted   = np.argsort(gt_delays)

        pred_delays = []
        for candidate in group:
            t0   = time.perf_counter()
            pred = model(candidate['inputs'], training=False)
            t1   = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)
            pred_delays.append(float(tf.reduce_mean(pred).numpy()))

        model_best_idx = int(np.argmin(pred_delays))

        if model_best_idx == gt_best_idx:
            n_correct += 1
        if model_best_idx in gt_sorted[:2]:
            n_top2 += 1

        # Regret = actual delay of model's choice − actual delay of best choice
        regret = gt_delays[model_best_idx] - gt_delays[gt_best_idx]
        regrets.append(float(regret))
        total += 1

    acc        = n_correct / total * 100 if total else 0
    top2       = n_top2 / total * 100    if total else 0
    mean_reg   = float(np.mean(regrets))
    median_reg = float(np.median(regrets))
    zero_reg   = float(np.mean([r < 1e-9 for r in regrets]) * 100)
    mean_lat   = float(np.mean(latencies_ms))

    return {
        "n_scenarios":          total,
        "decision_accuracy_%":  round(acc, 2),
        "top2_accuracy_%":      round(top2, 2),
        "mean_regret_s":        round(mean_reg, 6),
        "mean_regret_ms":       round(mean_reg * 1000, 4),
        "median_regret_ms":     round(median_reg * 1000, 4),
        "pct_zero_regret":      round(zero_reg, 2),
        "mean_inference_ms":    round(mean_lat, 3),
        "decisions_per_sec":    round(1000 / mean_lat, 1),
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    ALL_MIXED    = os.path.join(SCRIPT_DIR, '..')

    TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed_v1', 'validation')

    # ── Load scenarios ────────────────────────────────────────
    scenario_groups = load_scenario_groups(
        TEST_PATH,
        topology_sizes=EVAL_TOPOLOGY_SIZES,
        max_groups=MAX_GROUPS,
    )
    if not scenario_groups:
        print("ERROR: No valid scenario groups found. Check TEST_PATH and topology sizes.")
        return

    # val_path used for model weight initialisation
    val_path = TEST_PATH

    # ── Model configs ─────────────────────────────────────────
    model_configs = [
        {
            "name":   "Baseline (10-node only)",
            "cls":    RouteNet_Fermi,
            "ckpt":   os.path.join(ALL_MIXED, 'ckpt_dirs', 'ckpt_dir_0_10000'),
            "kwargs": {},
        },
        {
            "name":   "Teacher (RouteNet-Fermi)",
            "cls":    RouteNet_Fermi,
            "ckpt":   os.path.join(ALL_MIXED, 'ckpt_dirs_teacher', 'ckpt_dir_0_9972'),
            "kwargs": {},
        },
        {
            "name":   "Student-Scratch (no KD)",
            "cls":    StudentRouteNet,
            "ckpt":   os.path.join(ALL_MIXED, 'student', 'ckpt_dirs_scratch'),
            "kwargs": {"state_dim": 16, "iterations": 4},
        },
    ]

    kd_ckpt = os.path.join(ALL_MIXED, 'student', 'ckpt_dirs_kd_alpha0.5')
    if tf.train.latest_checkpoint(kd_ckpt):
        model_configs.append({
            "name":   "Student-KD (alpha=0.5)",
            "cls":    StudentRouteNet,
            "ckpt":   kd_ckpt,
            "kwargs": {"state_dim": 16, "iterations": 4},
        })

    # ── Run evaluation ────────────────────────────────────────
    all_results = {}
    for cfg in model_configs:
        print(f"\n{'='*65}")
        print(f"  Evaluating: {cfg['name']}")
        print(f"{'='*65}")
        try:
            model = build_and_load(cfg["cls"], cfg["ckpt"], val_path, **cfg["kwargs"])
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        t_start = time.time()
        metrics = evaluate_routing_decisions(model, scenario_groups)
        elapsed = time.time() - t_start

        all_results[cfg["name"]] = metrics
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  Decision Accuracy:  {metrics['decision_accuracy_%']:.1f}%")
        print(f"  Top-2 Accuracy:     {metrics['top2_accuracy_%']:.1f}%")
        print(f"  Mean Regret:        {metrics['mean_regret_ms']:.4f} ms extra delay")
        print(f"  Zero-Regret Pct:    {metrics['pct_zero_regret']:.1f}% of decisions were optimal")
        print(f"  Inference Latency:  {metrics['mean_inference_ms']:.2f} ms/candidate")
        print(f"  Throughput:         {metrics['decisions_per_sec']:.1f} evaluations/sec")

    # ── Summary table ─────────────────────────────────────────
    n_scen = len(scenario_groups)
    print(f"\n\n{'='*90}")
    print(f"  ROUTING DECISION EVALUATION — RESULTS")
    print(f"  {n_scen} scenario groups | Topologies: {sorted(set(g[0]['n_nodes'] for g in scenario_groups))} nodes")
    print(f"{'='*90}")
    print(f"  {'Model':<30} {'Decision Acc':>13} {'Top-2 Acc':>10} {'Regret':>12} {'Latency':>10} {'Evals/sec':>10}")
    print(f"  {'-'*86}")
    for name, m in all_results.items():
        print(f"  {name:<30} "
              f"{m['decision_accuracy_%']:>12.1f}% "
              f"{m['top2_accuracy_%']:>9.1f}% "
              f"{m['mean_regret_ms']:>10.4f}ms "
              f"{m['mean_inference_ms']:>9.2f}ms "
              f"{m['decisions_per_sec']:>10.1f}")
    print(f"{'='*90}")

    # ── Key research findings ─────────────────────────────────
    print(f"\n  KEY RESEARCH FINDINGS:")
    names = list(all_results.keys())
    teacher_name  = "Teacher (RouteNet-Fermi)"
    student_name  = "Student-Scratch (no KD)"
    kd_name       = "Student-KD (alpha=0.5)"
    baseline_name = "Baseline (10-node only)"

    if baseline_name in all_results and teacher_name in all_results:
        b = all_results[baseline_name]
        t = all_results[teacher_name]
        print(f"  • Teacher vs Baseline: +{t['decision_accuracy_%'] - b['decision_accuracy_%']:.1f}% "
              f"decision accuracy, {b['mean_regret_ms'] - t['mean_regret_ms']:.4f}ms less regret")

    if student_name in all_results and teacher_name in all_results:
        s = all_results[student_name]
        t = all_results[teacher_name]
        acc_gap  = t['decision_accuracy_%'] - s['decision_accuracy_%']
        speed_up = s['decisions_per_sec'] / t['decisions_per_sec']
        print(f"  • Student-Scratch vs Teacher: {acc_gap:+.1f}% decision accuracy gap, "
              f"but {speed_up:.1f}x more routing evaluations/sec")

    if kd_name in all_results and teacher_name in all_results:
        k = all_results[kd_name]
        t = all_results[teacher_name]
        acc_gap  = t['decision_accuracy_%'] - k['decision_accuracy_%']
        speed_up = k['decisions_per_sec'] / t['decisions_per_sec']
        print(f"  • Student-KD vs Teacher: {acc_gap:+.1f}% decision accuracy gap, "
              f"but {speed_up:.1f}x more routing evaluations/sec")

    # ── Save ──────────────────────────────────────────────────
    out_path = os.path.join(SCRIPT_DIR, "routing_decision_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
