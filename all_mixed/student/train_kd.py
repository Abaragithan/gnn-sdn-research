# student/train_kd.py
# Train the Queue-Link Fusion Student model with Knowledge Distillation.
# Uses online KD: teacher runs inference on each batch during training.
# Loss = alpha * MAPE(student, y_true) + (1-alpha) * MSE(student, teacher)

seed_value = 69420
import os
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["PYTHONHASHSEED"]        = str(seed_value)
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"   # suppress TF C++ INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN rebuild messages

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")          # suppress Python-level TF warnings
import warnings
warnings.filterwarnings("ignore")          # suppress IndexedSlices UserWarnings
tf.random.set_seed(seed_value)

import json
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # for data_generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # for models

from data_generator import input_fn
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet


class KDModel(tf.keras.Model):
    def __init__(self, student, teacher, alpha):
        super(KDModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.mape_fn = tf.keras.losses.MeanAbsolutePercentageError()
        self.mse_fn = tf.keras.losses.MeanSquaredError()
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.hard_tracker = tf.keras.metrics.Mean(name="hard")
        self.soft_tracker = tf.keras.metrics.Mean(name="soft")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.hard_tracker, self.soft_tracker, self.val_loss_tracker]

    def train_step(self, data):
        inputs, y_true = data
        teacher_pred = self.teacher(inputs, training=False)
        
        with tf.GradientTape() as tape:
            student_pred = self.student(inputs, training=True)
            loss_hard = self.mape_fn(y_true, student_pred)
            loss_soft = self.mse_fn(teacher_pred, student_pred)
            total_loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
            
        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        
        self.loss_tracker.update_state(total_loss)
        self.hard_tracker.update_state(loss_hard)
        self.soft_tracker.update_state(loss_soft)
        
        return {
            "loss": self.loss_tracker.result(), 
            "hard": self.hard_tracker.result(), 
            "soft": self.soft_tracker.result()
        }

    def test_step(self, data):
        inputs, y_true = data
        student_pred = self.student(inputs, training=False)
        val_loss = self.mape_fn(y_true, student_pred)
        self.val_loss_tracker.update_state(val_loss)
        return {"loss": self.val_loss_tracker.result()}


class SaveStudentCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_dir):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            ckpt_path = os.path.join(self.ckpt_dir, f"ckpt-{epoch+1:02d}")
            self.model.student.save_weights(ckpt_path)
            print(f"\n  val_loss improved to {val_loss:.4f}, saved student weights to {ckpt_path}")


def main():
    # Use absolute paths to avoid resolution issues with TF's from_generator
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    ALL_MIXED_DIR = os.path.join(SCRIPT_DIR, '..')

    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed', 'train_teacher')
    VAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed', 'val_teacher')
    NUM_TRAIN_SAMPLES = 9972

    # Check both new and old teacher checkpoint locations
    TEACHER_CKPT_DIR = os.path.join(ALL_MIXED_DIR, 'teacher', 'ckpt_dirs', 'ckpt_dir_0_9972')
    if not os.path.exists(TEACHER_CKPT_DIR):
        TEACHER_CKPT_DIR = os.path.join(ALL_MIXED_DIR, 'ckpt_dirs_teacher', 'ckpt_dir_0_9972')

    EPOCHS = 30
    STEPS_PER_EPOCH = NUM_TRAIN_SAMPLES
    ALPHA = 0.5

    # Load Teacher
    print("Loading teacher model...")
    teacher = RouteNet_Fermi()
    teacher.compile(loss="mape", optimizer="adam", run_eagerly=False)

    teacher_ckpt = tf.train.latest_checkpoint(TEACHER_CKPT_DIR)
    if teacher_ckpt is None:
        raise FileNotFoundError(f"No teacher checkpoint in {TEACHER_CKPT_DIR}")
    print(f"Teacher checkpoint: {teacher_ckpt}")

    ds_build = input_fn(VAL_PATH, shuffle=False).take(1)
    for x, y in ds_build:
        _ = teacher(x)
        break
    teacher.load_weights(teacher_ckpt)
    teacher.trainable = False

    # Build Student
    print("Building student model...")
    student = StudentRouteNet(state_dim=16, iterations=4)

    # Datasets
    ds_train = input_fn(TRAIN_PATH, seed=0, shuffle=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()

    ds_val = input_fn(VAL_PATH, shuffle=False)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Train with KD
    ckpt_dir = os.path.join(SCRIPT_DIR, f"ckpt_dirs_kd_alpha{ALPHA}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    kd_model = KDModel(student, teacher, alpha=ALPHA)
    kd_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    save_student_cb = SaveStudentCallback(ckpt_dir)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5,
        restore_best_weights=True, verbose=1,
    )

    print(f"\nStarting KD training (alpha={ALPHA})...")
    history = kd_model.fit(
        ds_train, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=ds_val, callbacks=[save_student_cb, early_stop],
        use_multiprocessing=True
    )
    
    best_val = min(history.history["val_loss"])

    result = {"alpha": ALPHA, "best_val_loss": float(best_val)}
    out_file = f"results_kd_alpha{ALPHA}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
