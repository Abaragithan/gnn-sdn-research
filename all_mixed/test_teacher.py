import os
import json
import tensorflow as tf
from data_generator import input_fn

import sys
sys.path.append("../")
from delay_model import RouteNet_Fermi


def build_model():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model = RouteNet_Fermi()
    loss_object = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(loss=loss_object, optimizer=optimizer, run_eagerly=False)
    return model


def eval_split(model, path, workers=10, take_n=None):
    ds = input_fn(path, shuffle=False)
    if take_n is not None:
        ds = ds.take(take_n)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return model.evaluate(ds, return_dict=True, use_multiprocessing=True, workers=workers)


def main():
    NUM_TRAIN_SAMPLES = 9972
    ckpt_dir = f"./ckpt_dirs_teacher/ckpt_dir_0_{NUM_TRAIN_SAMPLES}"

    VAL_PATH  = "../data/all_mixed/val_teacher"
    TEST_PATH = "../data/all_mixed/test"

    best_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    if best_ckpt is None:
        raise FileNotFoundError(f"No checkpoint found in: {ckpt_dir}")

    print("Loading checkpoint:", best_ckpt)

    model = build_model()
    model.load_weights(best_ckpt)

    results = {
        "ckpt_dir": ckpt_dir,
        "checkpoint": best_ckpt,
        "val_teacher": None,
        "test": None
    }

    print("\nEvaluating on VAL_TEACHER...")
    results["val_teacher"] = eval_split(model, VAL_PATH)

    print("\nEvaluating on TEST (OOD: 200–300 nodes)...")
    results["test"] = eval_split(model, TEST_PATH)

    out = "results_teacher.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
