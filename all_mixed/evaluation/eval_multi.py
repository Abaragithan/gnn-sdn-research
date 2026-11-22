"""T2: read each scenario once and evaluate exact checkpoints on both orders.

Run from any directory. Custom checkpoint paths are relative to the repo root.
Example: python all_mixed/evaluation/eval_multi.py --limit 5 --keras-check
Resume requires identical split, orders, model classes and checkpoint contents.
Forward timings exclude tracing/warm-up; construction excludes dataset parsing
and the additional hop-order correction/check (recorded separately).
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "all_mixed"))

import networkx as nx
import numpy as np
import tensorflow as tf
from datanetAPI import DatanetAPI
from data_generator import input_fn, network_to_hypergraph, hypergraph_to_input_data
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet
from hop_order import input_signature, to_tensors, with_hop_order

DEFAULT_MODELS = [
    "baseline:fermi:all_mixed/ckpt_dirs/ckpt_dir_0_10000/15-6.41",
    "teacher:fermi:all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26",
    "scratch:student:all_mixed/student/ckpt_dirs_scratch/ckpt-08",
    "kd:student:all_mixed/student/ckpt_dirs_kd_alpha0.5/ckpt-19",
]


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
    temp.replace(path)


def model_configs(specs):
    configs = {}
    for spec in specs:
        name, kind, checkpoint = spec.split(":", 2)
        if kind not in ("fermi", "student") or not name or name in configs:
            raise ValueError("Expected unique name:fermi|student:exact_checkpoint")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        files = [Path(str(checkpoint) + ".index")] + sorted(
            checkpoint.parent.glob(checkpoint.name + ".data-*"))
        if len(files) < 2 or not all(p.is_file() for p in files):
            raise FileNotFoundError(str(checkpoint))
        hashes = {}
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as src:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path.name] = digest.hexdigest()
        configs[name] = dict(kind=kind, checkpoint=str(checkpoint), sha256=hashes)
    return configs


def load_model(config, inputs):
    model = (RouteNet_Fermi() if config["kind"] == "fermi" else
             StudentRouteNet(state_dim=16, iterations=4))
    model(inputs).numpy()  # Build all variables before asserting checkpoint coverage.
    status = model.load_weights(config["checkpoint"])
    status.expect_partial()  # Historical checkpoints can also contain optimizer state.
    status.assert_existing_objects_matched()
    model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer="adam")
    return model


def read_completed(path):
    """Recover only a torn final write; preserve its bytes before truncation."""
    rows = []
    if not path.exists():
        return rows
    with path.open("rb+") as src:
        offset = 0
        while True:
            line = src.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                backup = path.with_name(path.name + ".incomplete-" + str(time.time_ns()))
                backup.write_bytes(line)
                src.truncate(offset)
                print("Recovered unfinished final record; saved bytes:", backup, flush=True)
                break
            rows.append(json.loads(line))  # Malformed complete records are errors.
            offset = src.tell()
    ids = [row["scenario_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario IDs in output")
    return rows


def metrics(y, pred, forward_ms):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if pred.shape != y.shape:
        raise ValueError("Prediction/label shape mismatch")
    finite = bool(np.all(np.isfinite(pred)))
    error = pred - y
    return dict(n_flows=len(y),
                sum_ape=float(np.sum(np.abs(error) / np.abs(y))) if finite else None,
                sum_ae=float(np.sum(np.abs(error))) if finite else None,
                sum_se=float(np.sum(error ** 2)) if finite else None,
                mean_true=float(y.mean()), mean_pred=float(pred.mean()) if finite else None,
                forward_ms=forward_ms, n_nonfinite=int(np.sum(~np.isfinite(pred))),
                n_negative=int(np.sum(pred < 0)))


def keras_check(models, rows, args, out):
    checks = {}
    for name in ("teacher", "kd"):
        if name not in models:
            continue
        # Independent historical pipeline, including its original delay input.
        result = models[name].evaluate(input_fn(str(args.split), shuffle=False).take(args.limit),
                                       return_dict=True, verbose=2)
        items = [row["models"][name]["ref"] for row in rows]
        pooled = 100 * sum(item["sum_ape"] for item in items) / sum(item["n_flows"] for item in items)
        difference = abs(pooled - float(result["loss"]))
        checks[name] = dict(pooled_mape=pooled, keras_mape=float(result["loss"]),
                            absolute_difference=difference, passed=difference <= 0.01)
        print("Keras comparison", name, checks[name], flush=True)
    write_json(out, dict(n_scenarios=len(rows), tolerance=0.01, models=checks))
    if not checks or not all(item["passed"] for item in checks.values()):
        raise ValueError("Keras reproduction check failed")


def run(args):
    args.split = args.split.resolve()
    configs = model_configs(args.models)
    orders = args.orders.split(",")
    if not orders or len(set(orders)) != len(orders) or any(o not in ("ref", "hop") for o in orders):
        raise ValueError("--orders must be ref, hop or ref,hop")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.keras_check and (not args.limit or "ref" not in orders):
        raise ValueError("--keras-check requires --limit and ref order")
    if not args.split.is_dir():
        raise FileNotFoundError(str(args.split))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.out.with_suffix(".meta.json")
    metadata = dict(schema_version=1, split=str(args.split), models=configs, orders=orders,
                    inference="fixed_signature_python_function_checked_against_original",
                    timing="warm forward; construction excludes parsing and hop correction")
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise ValueError("Resume configuration differs; use a new output file")
    else:
        if args.out.exists() and args.out.stat().st_size:
            raise ValueError("Existing JSONL has no metadata; refusing to mix runs")
        write_json(meta_path, metadata)
    rows = read_completed(args.out)
    completed = {r["scenario_id"] for r in rows}
    print("Resume:", len(completed), "completed scenarios", flush=True)
    models, forward = {}, {}
    archive_counts = {}
    seen = set()
    iterator = iter(DatanetAPI(str(args.split), shuffle=False))
    if args.limit:
        iterator = itertools.islice(iterator, args.limit)
    with args.out.open("a") as output:
        for sample in iterator:
            archive = Path(sample._get_data_set_file_name()).relative_to(args.split).as_posix()
            ordinal = archive_counts.get(archive, 0)
            archive_counts[archive] = ordinal + 1
            topology = Path(sample._graph_file).relative_to(args.split).as_posix()
            scenario_id = "{}::{}::{}".format(archive, topology, ordinal)
            seen.add(scenario_id)
            if scenario_id in completed:
                continue
            graph = nx.DiGraph(sample.get_topology_object())
            started = time.perf_counter()
            hg = network_to_hypergraph(graph, sample.get_routing_matrix(),
                                      sample.get_traffic_matrix(), sample.get_performance_matrix())
            raw, labels = hypergraph_to_input_data(hg)
            reference = to_tensors(raw)
            labels = tf.convert_to_tensor(labels, dtype=tf.float32).numpy().astype(np.float64)
            construct_ms = (time.perf_counter() - started) * 1000
            if labels.ndim != 1 or not len(labels) or not np.all(np.isfinite(labels)) or np.any(labels <= 0):
                raise ValueError("MAPE requires finite positive labels: " + scenario_id)
            inputs = {"ref": reference}
            correction_ms = None
            if "hop" in orders:
                started = time.perf_counter()
                inputs["hop"] = with_hop_order(hg, raw)
                correction_ms = (time.perf_counter() - started) * 1000
            record = dict(scenario_id=scenario_id, topology_id=topology,
                          n_nodes=graph.number_of_nodes(), n_links=int(reference["capacity"].shape[0]),
                          n_queues=int(reference["queue_size"].shape[0]), construct_ms=construct_ms,
                          hop_correction_ms=correction_ms, models={})
            if not models:
                for name, config in configs.items():
                    print("Loading", name, config["checkpoint"], flush=True)
                    models[name] = load_model(config, reference)
                    # Rewrap the original Python body, avoiding nested shape-specific
                    # traces in the historical @tf.function. Check both input orders.
                    forward[name] = tf.function(models[name].call.python_function,
                                                input_signature=[input_signature(reference)])
                    for order in orders:
                        original = models[name](inputs[order]).numpy()
                        stable = forward[name](inputs[order]).numpy()
                        np.testing.assert_allclose(stable, original, rtol=1e-6, atol=1e-9)
                    print("Fixed signature matches original:", name, flush=True)
            for name in models:
                record["models"][name] = {}
                for order in orders:
                    started = time.perf_counter()
                    pred = forward[name](inputs[order]).numpy()  # Synchronize inside timing.
                    elapsed = (time.perf_counter() - started) * 1000
                    record["models"][name][order] = metrics(labels, pred, elapsed)
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            rows.append(record)
            completed.add(scenario_id)
            print("Saved scenario", len(rows), scenario_id, "nodes=", record["n_nodes"], flush=True)
    if completed - seen:
        raise ValueError("Output contains scenarios outside this split/limit")
    if not rows:
        raise ValueError("No scenarios yielded")
    if args.keras_check:
        if not models:  # A repeated smoke check can reuse completed JSONL safely.
            first = next(iter(input_fn(str(args.split), shuffle=False)))
            reference = {k: v for k, v in first[0].items() if k != "delay"}
            models = {name: load_model(cfg, reference) for name, cfg in configs.items()}
        keras_check(models, rows, args, args.out.with_suffix(".keras_check.json"))
    print("Finished:", len(rows), "scenarios in", args.out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "data/all_mixed_v1/test")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--orders", default="ref,hop")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=ROOT / "results/dec1/eval_v0_test.jsonl")
    parser.add_argument("--keras-check", action="store_true", help="Compare limited ref run with Keras teacher/kd")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        summary = ROOT / "results/dec1/SUMMARY.md"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("a") as out:
            out.write("\n## Problems\n\nT2 evaluator stopped: {}: {}. Output: {}.\n".format(
                type(error).__name__, error, args.out))
        raise


if __name__ == "__main__":
    main()
