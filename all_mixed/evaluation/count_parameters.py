# evaluation/count_parameters.py
# Count and compare trainable parameters between Teacher and Student models.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_generator import input_fn
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet


def count_params(model, name, data_path):
    model.compile(loss="mape", optimizer="adam", run_eagerly=True)
    ds = input_fn(data_path, shuffle=False)
    for x, y in ds.take(1):
        _ = model(x, training=True)
        break

    total = 0
    print(f"\n{'='*60}")
    print(f"  {name} — Parameter Breakdown")
    print(f"{'='*60}")
    for var in model.trainable_variables:
        params = tf.reduce_prod(var.shape).numpy()
        total += params
        print(f"  {var.name:45s} {str(var.shape):20s} {params:>8,}")
    print(f"{'='*60}")
    print(f"  TOTAL: {total:,} trainable parameters")
    print(f"{'='*60}")
    return total


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed_v1', 'validation')

    print("\n" + "=" * 60)
    print("  MODEL PARAMETER COMPARISON")
    print("=" * 60)

    teacher = RouteNet_Fermi()
    teacher_params = count_params(teacher, "Teacher (RouteNet-Fermi)", DATA_PATH)

    student = StudentRouteNet(state_dim=16, iterations=4)
    student_params = count_params(student, "Student (Queue-Link Fusion)", DATA_PATH)

    print(f"\n{'='*60}")
    print(f"  COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Teacher parameters:  {teacher_params:>10,}")
    print(f"  Student parameters:  {student_params:>10,}")
    print(f"  Reduction:           {(1 - student_params / teacher_params) * 100:.1f}%")
    print(f"  Compression ratio:   {teacher_params / student_params:.1f}x")
    print(f"  Teacher MP:          8 iterations x 3 stages = 24 stages")
    print(f"  Student MP:          4 iterations x 2 stages = 8 stages")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
