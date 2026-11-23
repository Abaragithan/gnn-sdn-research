"""T5: train the Queue-Link Fusion student from scratch or with KD.

Scratch and KD share one training step; scratch is alpha = 1 with no teacher,
so the teacher signal is the only difference within a seed pair. For a given
seed, both modes load the same initial weights (runs/init/student_init_s<seed>)
and see the same data order. The checkpoint is selected on flow-weighted
validation MAPE, the metric Keras reports for every other model.

Run from the repository root, e.g.
  python all_mixed/student/train_student_v2.py --mode kd --soft-loss mape --seed 1 --out-dir runs/kdrel_s1
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/all_mixed_v1/train"
VAL = ROOT / "data/all_mixed_v1/validation"
TEACHER = ROOT / "all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("scratch", "kd"), required=True)
    p.add_argument("--soft-loss", choices=("mse", "mape"), default="mape",
                   help="kd only: mse = original loss (seconds^2); mape = percent error to the teacher")
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the ground-truth term (kd only)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-seed", type=int, help="training shuffle seed (default: --seed)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=9972)
    p.add_argument("--resume-from", type=Path, help="checkpoint to continue from after a crash (use a new --out-dir)")
    p.add_argument("--max-steps", type=int, help="smoke test only: overrides --steps-per-epoch")
    p.add_argument("--val-take", type=int, help="smoke test only: number of validation scenarios")
    args = p.parse_args()
    if args.mode == "scratch":
        args.alpha, args.soft_loss = 1.0, None
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.max_steps:
        args.steps_per_epoch = args.max_steps
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    return args


def sha256_files(prefix):
    files = sorted(prefix.parent.glob(prefix.name + ".*"))
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    args = parse_args()
    if (args.out_dir / "config.json").exists():
        sys.exit("Output folder already used: {}. Choose a new --out-dir.".format(args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Logging only: this harmless gather-gradient warning repeats at every trace.
    warnings.filterwarnings("ignore", message="Converting sparse IndexedSlices")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "all_mixed"))
    import numpy as np
    import tensorflow as tf
    from data_generator import input_fn
    from delay_model import RouteNet_Fermi
    from student_delay_model import StudentRouteNet

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    class Trainer(tf.keras.Model):
        """Wraps the student; the teacher (if any) is frozen and never trained."""

        def __init__(self, student, teacher, alpha, soft_loss):
            super().__init__()
            self.student, self.teacher = student, teacher
            self.alpha, self.soft_loss = alpha, soft_loss
            self.mape = tf.keras.losses.MeanAbsolutePercentageError()
            self.mse = tf.keras.losses.MeanSquaredError()
            self.trackers = {name: tf.keras.metrics.Mean(name) for name in ("loss", "hard", "soft")}
            # float64 sums: validation pools millions of flows.
            self.ape_sum = tf.keras.metrics.Sum("ape_sum", dtype=tf.float64)
            self.flow_count = tf.keras.metrics.Sum("flow_count", dtype=tf.float64)
            self.scenario_mape = tf.keras.metrics.Mean("scenario_mape")

        @property
        def metrics(self):  # Keras resets these at the start of every epoch and evaluation.
            return list(self.trackers.values()) + [self.ape_sum, self.flow_count, self.scenario_mape]

        def call(self, inputs):
            return self.student(inputs)

        def train_step(self, data):
            x, y = data
            teacher_pred = None if self.teacher is None else self.teacher(x)
            with tf.GradientTape() as tape:
                pred = self.student(x)
                hard = self.mape(y, pred)
                if teacher_pred is None:
                    soft = tf.constant(0.0)
                elif self.soft_loss == "mse":
                    soft = self.mse(teacher_pred, pred)
                else:  # Keras MAPE rule with the teacher as target: 100*|t-s|/max(|t|, 1e-7)
                    soft = self.mape(teacher_pred, pred)
                loss = self.alpha * hard + (1.0 - self.alpha) * soft
            variables = self.student.trainable_variables
            self.optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            for name, value in (("loss", loss), ("hard", hard), ("soft", soft)):
                self.trackers[name].update_state(value)
            return {name: t.result() for name, t in self.trackers.items()}

        def test_step(self, data):
            x, y = data
            pred = tf.cast(tf.reshape(self.student(x), [-1]), tf.float64)
            y = tf.cast(tf.reshape(y, [-1]), tf.float64)
            ape = tf.abs(y - pred) / tf.maximum(tf.abs(y), 1e-7)
            self.ape_sum.update_state(tf.reduce_sum(ape))
            self.flow_count.update_state(tf.cast(tf.size(y), tf.float64))
            self.scenario_mape.update_state(100.0 * tf.reduce_mean(ape))
            return {"mape_flow": 100.0 * self.ape_sum.result() / self.flow_count.result(),
                    "mape_unweighted": self.scenario_mape.result()}

    class Record(tf.keras.callbacks.Callback):
        """history.json every epoch; student weights and best.json on improvement."""

        def __init__(self):
            super().__init__()
            self.best, self.history = float("inf"), []

        def on_train_batch_end(self, batch, logs=None):
            if batch == 0 and not self.history:  # Same seed pair => same value in scratch and kd.
                print("\nFirst-step hard loss: {:.6f}".format(float(logs["hard"])), flush=True)

        def on_epoch_end(self, epoch, logs=None):
            row = {"epoch": epoch + 1, "time_unix": time.time()}
            row.update({k: float(v) for k, v in logs.items()})
            self.history.append(row)
            write_json(args.out_dir / "history.json", self.history)
            if row["val_mape_flow"] < self.best:
                self.best = row["val_mape_flow"]
                checkpoint = args.out_dir / "ckpt-{:02d}".format(epoch + 1)
                self.model.student.save_weights(str(checkpoint))
                write_json(args.out_dir / "best.json", dict(row, checkpoint=str(checkpoint)))
                print("\nval_mape_flow improved to {:.4f}; saved {}".format(self.best, checkpoint), flush=True)

    # 1. Student and its shared initial weights, before the teacher touches the RNG.
    sample_x, _ = next(iter(input_fn(str(VAL), shuffle=False)))
    student = StudentRouteNet(state_dim=16, iterations=4)
    student(sample_x)
    init = ROOT / "runs/init/student_init_s{}".format(args.seed)
    if Path(str(init) + ".index").exists():
        student.load_weights(str(init)).assert_existing_objects_matched()
        init_status = "loaded"
    else:
        init.parent.mkdir(parents=True, exist_ok=True)
        student.save_weights(str(init))
        init_status = "created"
    if args.resume_from:
        student.load_weights(str(args.resume_from)).assert_existing_objects_matched()

    # 2. Frozen teacher (kd only).
    teacher = None
    if args.mode == "kd":
        teacher = RouteNet_Fermi()
        teacher(sample_x)
        status = teacher.load_weights(str(TEACHER))
        status.expect_partial()  # The checkpoint also holds optimizer state.
        status.assert_existing_objects_matched()
        teacher.trainable = False

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                         universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(git_commit=commit, tensorflow=tf.__version__, started_unix=time.time(),
                  init_weights=str(init), init_status=init_status, init_sha256=sha256_files(init),
                  teacher_checkpoint=str(TEACHER) if teacher else None,
                  teacher_sha256=sha256_files(TEACHER) if teacher else None,
                  selection_metric="val_mape_flow (flow-weighted)")
    write_json(args.out_dir / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)

    # 3. Data: the same shuffle seed gives the same order in both runs of a pair.
    train = input_fn(str(TRAIN), seed=args.data_seed, shuffle=True)
    train = train.prefetch(tf.data.experimental.AUTOTUNE).repeat()
    val = input_fn(str(VAL), shuffle=False)
    if args.val_take:
        val = val.take(args.val_take)
    val = val.prefetch(tf.data.experimental.AUTOTUNE)

    trainer = Trainer(student, teacher, args.alpha, args.soft_loss)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_mape_flow", mode="min",
                                                  patience=args.patience, verbose=1)
    # verbose=2: one log line per epoch instead of a progress bar (the old KD log was 16 MB).
    trainer.fit(train, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                validation_data=val, callbacks=[Record(), early_stop], verbose=2)
    print("Finished. Best checkpoint:", json.loads((args.out_dir / "best.json").read_text())["checkpoint"])


if __name__ == "__main__":
    main()
