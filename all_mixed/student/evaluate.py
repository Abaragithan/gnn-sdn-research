# student/evaluate.py
# Evaluate the trained student model on validation and OOD test sets.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import time
import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_generator import input_fn
from student_delay_model import StudentRouteNet


def build_model():
    model = StudentRouteNet(state_dim=16, iterations=4)
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(loss=loss_object, optimizer="adam", run_eagerly=False)
    return model


def eval_split(model, path, name):
    ds = input_fn(path, shuffle=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    print(f"\n  Evaluating on {name}...")
    t0 = time.time()
    result = model.evaluate(ds, return_dict=True, verbose=1)
    elapsed = time.time() - t0
    result["time_seconds"] = round(elapsed, 1)
    return result


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

    VAL_PATH  = os.path.join(PROJECT_ROOT, 'data', 'all_mixed', 'val_teacher')
    TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed', 'test')

    # Try scratch checkpoint first, then KD
    ckpt_configs = [
        ("Student-Scratch", os.path.join(SCRIPT_DIR, "ckpt_dirs_scratch")),
        ("Student-KD (alpha=0.5)", os.path.join(SCRIPT_DIR, "ckpt_dirs_kd_alpha0.5")),
    ]

    all_results = {}

    for name, ckpt_dir in ckpt_configs:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt is None:
            print(f"  SKIPPED: No checkpoint in {ckpt_dir}")
            continue

        print(f"  Loading: {ckpt}")
        model = build_model()
        model.load_weights(ckpt)

        results = {"checkpoint": ckpt}
        results["val_teacher"] = eval_split(model, VAL_PATH, "val_teacher")
        results["test_ood"] = eval_split(model, TEST_PATH, "test (OOD: 200-300 nodes)")
        all_results[name] = results

    # Summary
    print(f"\n{'='*60}")
    print(f"  STUDENT MODEL RESULTS")
    print(f"{'='*60}")
    for name, res in all_results.items():
        val = res["val_teacher"]["loss"]
        test = res["test_ood"]["loss"]
        print(f"  {name:<30} val={val:.4f}  test_ood={test:.4f}")
    print(f"{'='*60}")

    out = "results_student.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
