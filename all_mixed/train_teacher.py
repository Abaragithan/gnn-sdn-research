# train_teacher.py
# Full fixed training script: resume + early stopping + correct checkpoint naming

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

from data_generator import input_fn

import sys
sys.path.append("../")
from delay_model import RouteNet_Fermi


def build_model():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = RouteNet_Fermi()

    # Keep same loss you used (MAPE). For robustness you may later compare with MAE too.
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(
        loss=loss_object,
        optimizer=optimizer,
        run_eagerly=False
    )
    return model


def main():
    # Paths for teacher dataset
    TRAIN_PATH = "../data/all_mixed/train_teacher"
    VAL_PATH   = "../data/all_mixed/val_teacher"

    # Your counted usable train samples (used for full-pass epochs)
    NUM_TRAIN_SAMPLES = 9972

    # Training config
    EPOCHS = 30
    STEPS_PER_EPOCH = NUM_TRAIN_SAMPLES   # 1 epoch = full pass (since no batch)
    # If you want faster experiments, set STEPS_PER_EPOCH=1000 temporarily.

    # Repro seed for dataset shuffle
    seed = 0

    # Checkpoint directory
    ckpt_dir = f"./ckpt_dirs_teacher/ckpt_dir_0_{NUM_TRAIN_SAMPLES}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build model
    model = build_model()

    # Build datasets
    ds_train = input_fn(TRAIN_PATH, seed=seed, shuffle=True)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.repeat()  # infinite stream

    ds_val = input_fn(VAL_PATH, shuffle=False)
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # ---- Checkpointing that supports tf.train.latest_checkpoint ----
    # IMPORTANT: use a stable prefix like "ckpt-XX"
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt-{epoch:02d}")

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_prefix,
        verbose=1,
        monitor="val_loss",
        mode="min",
        save_best_only=True,      # keep only best epoch weights
        save_weights_only=True,
        save_freq="epoch"
    )

    # Early stopping (stops training if val_loss doesn't improve)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Resume if checkpoint exists, else start from initial weights
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        print("Resuming from:", latest)
        model.load_weights(latest)
    else:
        print("Starting from initial weights:", "./initial_weights/initial_weights")
        model.load_weights("./initial_weights/initial_weights")

    # Train
    model.fit(
        ds_train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=ds_val,
        callbacks=[cp_callback, early_stop],
        use_multiprocessing=True
    )

if __name__ == "__main__":
    main()
