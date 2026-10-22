# student/train_scratch.py
# Train the Queue-Link Fusion Student model from scratch (no KD).
# This is the critical BASELINE to prove KD adds value.

seed_value = 69420
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONHASHSEED"] = str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # for data_generator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # for student_delay_model

from data_generator import input_fn
from student_delay_model import StudentRouteNet


def main():
    # Use absolute paths to avoid resolution issues with TF's from_generator
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed_v1', 'train')
    VAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'all_mixed_v1', 'validation')
    NUM_TRAIN_SAMPLES = 9972

    EPOCHS = 30
    STEPS_PER_EPOCH = NUM_TRAIN_SAMPLES

    ckpt_dir = os.path.join(SCRIPT_DIR, "ckpt_dirs_scratch")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build student model
    model = StudentRouteNet(state_dim=16, iterations=4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(loss=loss_object, optimizer=optimizer, run_eagerly=False)

    # Dataset
    ds_train = input_fn(TRAIN_PATH, seed=0, shuffle=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()

    ds_val = input_fn(VAL_PATH, shuffle=False)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Callbacks
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt-{epoch:02d}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_prefix, verbose=1, monitor="val_loss",
        mode="min", save_best_only=True, save_weights_only=True, save_freq="epoch",
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5,
        restore_best_weights=True, verbose=1,
    )

    # Resume or start fresh
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        print("Resuming from:", latest)
        model.load_weights(latest)
    else:
        print("Starting student training from scratch (random init)...")

    model.fit(
        ds_train, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=ds_val, callbacks=[cp_callback, early_stop],
        use_multiprocessing=True,
    )


if __name__ == "__main__":
    main()
