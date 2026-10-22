# evaluation/evaluate_all.py
# Unified evaluation comparing Teacher, Student-Scratch, and Student-KD.
# Reports overall MAPE + parameter counts + timing.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time
import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # for data_generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # for models

from data_generator import input_fn
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet


def build_and_load(model_cls, ckpt_dir, data_path, **kwargs):
    model = model_cls(**kwargs)
    model.compile(loss="mape", optimizer="adam", run_eagerly=False)

    ds = input_fn(data_path, shuffle=False).take(1)
    for x, y in ds:
        _ = model(x)
        break

    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    model.load_weights(ckpt)
    print(f"  Loaded: {ckpt}")
    return model


def evaluate_model(model, data_path, name):
    ds = input_fn(data_path, shuffle=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    print(f"\n  Evaluating {name}...")
    t0 = time.time()
    result = model.evaluate(ds, return_dict=True, verbose=1)
    elapsed = time.time() - t0
    n_params = sum(tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables)
    return {"loss": float(result["loss"]), "time_seconds": round(elapsed, 1),
            "trainable_params": int(n_params)}


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    ALL_MIXED_DIR = os.path.join(SCRIPT_DIR, '..')

    VAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed_v1', 'validation')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed_v1', 'test')

    models_config = [
        {"name": "Teacher (RouteNet-Fermi)",
         "cls": RouteNet_Fermi,
         "ckpt_dir": os.path.join(ALL_MIXED_DIR, 'teacher', 'ckpt_dirs', 'ckpt_dir_0_9972'),
         "alt_ckpt_dir": os.path.join(ALL_MIXED_DIR, 'ckpt_dirs_teacher', 'ckpt_dir_0_9972'),
         "kwargs": {}},
        {"name": "Student-Scratch (no KD)",
         "cls": StudentRouteNet,
         "ckpt_dir": os.path.join(ALL_MIXED_DIR, 'student', 'ckpt_dirs_scratch'),
         "alt_ckpt_dir": None,
         "kwargs": {"state_dim": 16, "iterations": 4}},
        {"name": "Student-KD (alpha=0.5)",
         "cls": StudentRouteNet,
         "ckpt_dir": os.path.join(ALL_MIXED_DIR, 'student', 'ckpt_dirs_kd_alpha0.5'),
         "alt_ckpt_dir": None,
         "kwargs": {"state_dim": 16, "iterations": 4}},
    ]

    results = {}
    for cfg in models_config:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        ckpt_dir = cfg["ckpt_dir"]
        if not os.path.exists(ckpt_dir) and cfg.get("alt_ckpt_dir"):
            ckpt_dir = cfg["alt_ckpt_dir"]

        try:
            model = build_and_load(cfg["cls"], ckpt_dir, TEST_PATH, **cfg["kwargs"])
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        results[name] = {
            "val_teacher": evaluate_model(model, VAL_PATH, f"{name} on val_teacher"),
            "test_ood": evaluate_model(model, TEST_PATH, f"{name} on test (OOD)"),
        }

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<30} {'Params':>10} {'Val Loss':>10} {'OOD Loss':>10} {'OOD Time':>10}")
    print(f"  {'-'*70}")
    for name, res in results.items():
        p = res["val_teacher"]["trainable_params"]
        v = res["val_teacher"]["loss"]
        t = res["test_ood"]["loss"]
        s = res["test_ood"]["time_seconds"]
        print(f"  {name:<30} {p:>10,} {v:>10.4f} {t:>10.4f} {s:>9.1f}s")
    print(f"{'='*80}")

    with open("results_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results_comparison.json")


if __name__ == "__main__":
    main()
