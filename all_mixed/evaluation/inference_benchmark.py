# evaluation/inference_benchmark.py
#
# PURPOSE: Prove that the student model is faster at INFERENCE (deployment time).
#
# Measures per-sample prediction latency, model size, and memory footprint.
# This is the evidence that the student is "lightweight" in the way that matters:
# when an SDN controller calls the model to make routing decisions in real time.
#
# Run:
#   cd ~/gnn-sdn-research/all_mixed/evaluation
#   /home/abaragithan/miniconda3/envs/routenet-fermi/bin/python inference_benchmark.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only, same as deployment

import time
import json
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_generator import input_fn
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
N_WARMUP  = 5    # warm-up calls (TF graph tracing, caching) — not counted
N_REPEATS = 30   # timed inference calls per sample


def count_params(model):
    return sum(int(tf.reduce_prod(v.shape).numpy()) for v in model.trainable_variables)


def model_size_kb(model):
    """Estimate model weight size in KB (float32 = 4 bytes per param)."""
    return count_params(model) * 4 / 1024


def build_and_load(model_cls, ckpt_dir, sample_x, **kwargs):
    """Build model, run one forward pass to initialise weights, then load checkpoint."""
    model = model_cls(**kwargs)
    model.compile(loss="mape", optimizer="adam", run_eagerly=False)
    _ = model(sample_x)                               # initialise
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")
    model.load_weights(ckpt)
    print(f"  Loaded checkpoint: {ckpt}")
    return model


def benchmark_inference(model, sample_x, n_warmup=N_WARMUP, n_repeats=N_REPEATS):
    """
    Time single-sample inference.
    Returns latency stats in milliseconds.
    """
    # Warm-up: TF traces the graph on first few calls; don't count these
    for _ in range(n_warmup):
        _ = model(sample_x, training=False)

    latencies_ms = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = model(sample_x, training=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    return {
        "mean_ms":   round(float(np.mean(latencies_ms)), 3),
        "median_ms": round(float(np.median(latencies_ms)), 3),
        "min_ms":    round(float(np.min(latencies_ms)), 3),
        "max_ms":    round(float(np.max(latencies_ms)), 3),
        "std_ms":    round(float(np.std(latencies_ms)), 3),
        "n_repeats": n_repeats,
        "all_ms":    [round(x, 3) for x in latencies_ms],
    }


def main():
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    ALL_MIXED    = os.path.join(SCRIPT_DIR, '..')

    # Use validation set — grab just 1 sample
    VAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed', 'val_teacher')
    ds = input_fn(VAL_PATH, shuffle=False)
    sample_x, sample_y = next(iter(ds))  # single graph sample

    n_flows = int(sample_y.shape[0]) if hasattr(sample_y, 'shape') else '?'
    print(f"\n  Benchmark sample: {n_flows} flows/paths in this graph\n")

    # ─── Model configs ───────────────────────────────────────
    configs = [
        {
            "name": "Baseline (10-node only)",
            "cls":  RouteNet_Fermi,
            "ckpt": os.path.join(ALL_MIXED, 'ckpt_dirs', 'ckpt_dir_0_10000'),
            "kwargs": {},
            "mp_stages": "8 iters × 3 stages = 24 total",
        },
        {
            "name": "Teacher (RouteNet-Fermi)",
            "cls":  RouteNet_Fermi,
            "ckpt": os.path.join(ALL_MIXED, 'ckpt_dirs_teacher', 'ckpt_dir_0_9972'),
            "kwargs": {},
            "mp_stages": "8 iters × 3 stages = 24 total",
        },
        {
            "name": "Student-Scratch (no KD)",
            "cls":  StudentRouteNet,
            "ckpt": os.path.join(ALL_MIXED, 'student', 'ckpt_dirs_scratch'),
            "kwargs": {"state_dim": 16, "iterations": 4},
            "mp_stages": "4 iters × 2 stages = 8 total",
        },
    ]

    # Add Student-KD if checkpoint exists
    kd_ckpt = os.path.join(ALL_MIXED, 'student', 'ckpt_dirs_kd_alpha0.5')
    if tf.train.latest_checkpoint(kd_ckpt):
        configs.append({
            "name": "Student-KD (alpha=0.5)",
            "cls":  StudentRouteNet,
            "ckpt": kd_ckpt,
            "kwargs": {"state_dim": 16, "iterations": 4},
            "mp_stages": "4 iters × 2 stages = 8 total",
        })

    # ─── Run benchmark ───────────────────────────────────────
    results = {}
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"  {cfg['name']}")
        print(f"{'='*60}")
        try:
            model = build_and_load(cfg["cls"], cfg["ckpt"], sample_x, **cfg["kwargs"])
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        params    = count_params(model)
        size_kb   = model_size_kb(model)
        latency   = benchmark_inference(model, sample_x)

        print(f"  Parameters:        {params:,}")
        print(f"  Model weight size: {size_kb:.1f} KB")
        print(f"  MP stages:         {cfg['mp_stages']}")
        print(f"  Inference latency ({N_REPEATS} runs, single sample):")
        print(f"    Mean:   {latency['mean_ms']:.3f} ms")
        print(f"    Median: {latency['median_ms']:.3f} ms")
        print(f"    Min:    {latency['min_ms']:.3f} ms")
        print(f"    Max:    {latency['max_ms']:.3f} ms")
        print(f"    Std:    {latency['std_ms']:.3f} ms")

        results[cfg["name"]] = {
            "trainable_params":   params,
            "model_size_kb":      round(size_kb, 1),
            "mp_stages":          cfg["mp_stages"],
            "inference_latency":  latency,
        }

    # ─── Comparison table ────────────────────────────────────
    print(f"\n\n{'='*75}")
    print(f"  INFERENCE BENCHMARK RESULTS  ({N_REPEATS} runs per model, single sample)")
    print(f"{'='*75}")
    print(f"  {'Model':<32} {'Params':>8}  {'Size':>7}  {'MP Stages':>8}  {'Latency (mean)':>14}")
    print(f"  {'-'*70}")
    for name, res in results.items():
        lat  = res["inference_latency"]["mean_ms"]
        p    = res["trainable_params"]
        s    = res["model_size_kb"]
        mp   = res["mp_stages"].split("=")[1].strip()
        print(f"  {name:<32} {p:>8,}  {s:>6.1f}KB  {mp:>8}  {lat:>12.3f} ms")
    print(f"{'='*75}")

    # ─── Speedup ratio ───────────────────────────────────────
    if "Teacher (RouteNet-Fermi)" in results and len(results) > 1:
        teacher_lat = results["Teacher (RouteNet-Fermi)"]["inference_latency"]["mean_ms"]
        teacher_p   = results["Teacher (RouteNet-Fermi)"]["trainable_params"]
        print(f"\n  Speedup vs Teacher (mean inference latency):")
        for name, res in results.items():
            if name == "Teacher (RouteNet-Fermi)":
                continue
            lat   = res["inference_latency"]["mean_ms"]
            p     = res["trainable_params"]
            speedup = teacher_lat / lat
            p_red   = (1 - p / teacher_p) * 100
            print(f"    {name}: {speedup:.2f}x faster  |  {p_red:.1f}% fewer params")

    print()

    # ─── Save results ────────────────────────────────────────
    out_path = os.path.join(SCRIPT_DIR, "inference_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
