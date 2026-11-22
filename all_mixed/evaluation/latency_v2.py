"""T3 CPU latency benchmark; one fresh process per model/thread setting.

R2 check (run manually before the full queue):
  python -u all_mixed/evaluation/latency_v2.py --model \
    teacher:fermi:all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26 \
    --threads default --scenarios validation --out results/dec1/latency_v2_smoke.json

Each worker writes its own JSON, updated after each scenario. --merge combines
the eight complete workers without importing TensorFlow. Existing output files
are refused. All inputs use historical reference order, with labels removed.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ["validation", "test200", "test240", "test260", "test280", "test300"]
MODELS = [
    "baseline:fermi:all_mixed/ckpt_dirs/ckpt_dir_0_10000/15-6.41",
    "teacher:fermi:all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26",
    "scratch:student:all_mixed/student/ckpt_dirs_scratch/ckpt-08",
    "kd:student:all_mixed/student/ckpt_dirs_kd_alpha0.5/ckpt-19",
]


def save(path, document):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as out:
        json.dump(document, out, indent=2, allow_nan=False)
        out.write("\n")
    temp.replace(path)


def stats(values):
    import numpy as np
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                std_ms=float(np.std(values, ddof=0)), p95_ms=float(np.percentile(values, 95)),
                all_ms=values, n_repeats=len(values))


def peak_memory():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Fedora/Linux reports KiB; macOS reports bytes.
    kib = raw / 1024 if sys.platform == "darwin" else raw
    return dict(peak_rss_kib=float(kib), peak_rss_mib=float(kib / 1024))


def first_sample(split, n_nodes=None):
    """Select the same first yielded scenario as an unshuffled split iterator.

    For test sizes, inspect graph metadata to exclude folders of other sizes,
    retaining DatanetAPI's original archive order. This avoids parsing hundreds
    of irrelevant traffic matrices in each of the eight benchmark processes.
    Validate all graph variants in each folder; mixed-size folders are errors.
    No T2 prediction/results files are needed for this selection.
    """
    import networkx as nx
    from datanetAPI import DatanetAPI
    api = DatanetAPI(str(split), shuffle=False)
    if n_nodes is not None:
        sizes, selected = {}, []
        for folder, archive in api.get_available_files():
            if folder not in sizes:
                graphs = sorted((Path(folder) / "graphs").iterdir())
                counts = {nx.read_gml(str(p), destringizer=int).number_of_nodes()
                          for p in graphs if p.is_file()}
                if len(counts) != 1:
                    raise ValueError("Expected one topology size per folder: " + folder)
                sizes[folder] = counts.pop()
            if sizes[folder] == n_nodes:
                selected.append((folder, archive))
        if not selected:
            raise ValueError("No archives for topology size {}".format(n_nodes))
        # The bundled set_files_to_process shadows the built-in `tuple` in
        # `for tuple in ...`, so `type(tuple) is tuple` rejects every entry.
        # These tuples come directly from get_available_files above. Set the
        # exact same selection field as that method, preserving archive order,
        # without modifying the historical DatanetAPI implementation.
        api._selected_tuple_files = selected.copy()
    iterator = iter(api)
    try:
        sample = next(iterator)
    except StopIteration:
        raise ValueError("No scenarios yielded from " + str(split))
    finally:
        iterator.close()
    if n_nodes is not None and sample.get_topology_object().number_of_nodes() != n_nodes:
        raise ValueError("Selected scenario has the wrong topology size")
    return sample


def build_inputs(sample):
    import networkx as nx
    from data_generator import network_to_hypergraph, hypergraph_to_input_data
    from hop_order import to_tensors
    graph = nx.DiGraph(sample.get_topology_object())
    hg = network_to_hypergraph(graph, sample.get_routing_matrix(),
                              sample.get_traffic_matrix(), sample.get_performance_matrix())
    raw, _ = hypergraph_to_input_data(hg)
    return to_tensors(raw)


def compare_historical(name, threads, scenario, mean_ms):
    if threads != "default" or scenario != "validation" or name not in ("teacher", "scratch"):
        return None
    expected = {"teacher": 178.5, "scratch": 34.2}[name]
    difference = (mean_ms - expected) / expected
    return dict(expected_mean_ms=expected, obtained_mean_ms=mean_ms,
                relative_difference=difference, within_20_percent=abs(difference) <= 0.2,
                note="A mismatch is reported, not hidden; machine load and protocol can affect latency.")


def worker(args):
    # These MUST be set before importing eval_multi, data_generator or TF.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        if args.threads == "1":
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)  # default means TF's automatic threading.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "all_mixed"))
    import networkx as nx
    import numpy as np
    import tensorflow as tf
    from eval_multi import load_model, model_configs
    from hop_order import input_signature

    if args.threads == "1":
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    configs = model_configs([args.model])
    name, config = next(iter(configs.items()))
    document = dict(schema_version=1, status="running", model=dict(name=name, **config),
                    threads=args.threads, requested_scenarios=args.scenarios,
                    n_warmup=5, n_timed=30, n_construct=3, input_order="ref",
                    pid=os.getpid(), started_unix=time.time(), scenarios=[],
                    environment=dict(python=sys.version, tensorflow=tf.__version__,
                                     networkx=nx.__version__, platform=platform.platform(),
                                     cpu_count=os.cpu_count(),
                                     intra_op_threads=tf.config.threading.get_intra_op_parallelism_threads(),
                                     inter_op_threads=tf.config.threading.get_inter_op_parallelism_threads(),
                                     thread_environment={k: os.environ.get(k) for k in (
                                         "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS",
                                         "MKL_NUM_THREADS", "TF_ENABLE_ONEDNN_OPTS")}),
                    timing_policy="Construction includes DiGraph + hypergraph + tensors; excludes sample parsing. "
                                  "Forward synchronizes to NumPy; excludes tracing, warm-up and validation checks.",
                    memory_policy="ru_maxrss is the cumulative high-water mark of this model/thread process, "
                                  "including TF, parsing, construction and earlier scenarios; not per-scenario memory.")
    try:
        document["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    except (OSError, subprocess.CalledProcessError):
        document["git_commit"] = None
    save(args.out, document)
    model = forward = None
    try:
        for scenario in args.scenarios:
            print("Selecting", name, args.threads, scenario, flush=True)
            split = ROOT / "data/all_mixed_v1" / ("validation" if scenario == "validation" else "test")
            started = time.perf_counter()
            sample = first_sample(split, None if scenario == "validation" else int(scenario[4:]))
            select_ms = (time.perf_counter() - started) * 1000
            construction, inputs = [], None
            for repeat in range(3):
                inputs = None  # Free the previous tensors before the next timed construction.
                gc.collect()
                started = time.perf_counter()
                inputs = build_inputs(sample)
                # Materialize tensor work before stopping the clock (CPU-only).
                inputs["traffic"].numpy()
                construction.append((time.perf_counter() - started) * 1000)
                print("Construction", scenario, repeat + 1, "/ 3:", construction[-1], "ms", flush=True)
            if model is None:
                print("Loading exact checkpoint:", config["checkpoint"], flush=True)
                model = load_model(config, inputs)
                original = model(inputs).numpy()
                forward = tf.function(model.call.python_function, input_signature=[input_signature(inputs)])
                np.testing.assert_allclose(forward(inputs).numpy(), original, rtol=1e-6, atol=1e-9)
                document["fixed_signature_matches_original"] = True
                document["parameter_count"] = int(sum(np.prod(v.shape) for v in model.trainable_variables))
                del original
            n_flows = int(inputs["traffic"].shape[0])

            def check(prediction):
                if prediction.size != n_flows or not np.all(np.isfinite(prediction)) or np.any(prediction < 0):
                    raise ValueError("Invalid prediction in " + scenario)

            for _ in range(5):
                check(forward(inputs).numpy())
            elapsed = []
            for _ in range(30):
                started = time.perf_counter()
                prediction = forward(inputs).numpy()
                elapsed.append((time.perf_counter() - started) * 1000)
                check(prediction)  # Validation is outside the timed interval.
            archive = Path(sample._get_data_set_file_name()).relative_to(split).as_posix()
            topology = Path(sample._graph_file).relative_to(split).as_posix()
            result = dict(scenario=scenario, split=str(split), scenario_id=archive + "::" + topology + "::0",
                          topology_id=topology, n_nodes=sample.get_topology_object().number_of_nodes(),
                          n_links=int(inputs["capacity"].shape[0]), n_queues=int(inputs["queue_size"].shape[0]),
                          n_flows=n_flows, selection_and_parse_ms=select_ms,
                          construction=stats(construction), forward=stats(elapsed),
                          n_nonfinite=0, n_negative=0, **peak_memory())
            result["construct_plus_forward_ms"] = result["construction"]["median_ms"] + result["forward"]["mean_ms"]
            result["historical_comparison"] = compare_historical(name, args.threads, scenario, result["forward"]["mean_ms"])
            document["scenarios"].append(result)
            document.update(peak_memory())
            save(args.out, document)
            print("Finished", scenario, "forward mean_ms=", result["forward"]["mean_ms"],
                  "peak RSS MiB=", result["peak_rss_mib"], flush=True)
            if result["historical_comparison"] is not None:
                print("Historical comparison:", result["historical_comparison"], flush=True)
            inputs = sample = prediction = None
            gc.collect()
        document["status"] = "complete"
        document["finished_unix"] = time.time()
        document.update(peak_memory())
        save(args.out, document)
    except Exception as error:
        document["status"] = "failed"
        document["error"] = "{}: {}".format(type(error).__name__, error)
        document.update(peak_memory())
        save(args.out, document)
        summary = ROOT / "results/dec1/SUMMARY.md"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("a") as out:
            out.write("\n## Problems\n\nT3 benchmark stopped: {}. See {}.\n".format(document["error"], args.out))
        raise
    print("Saved complete worker:", args.out, flush=True)


def merge(paths):
    runs = [json.loads(path.read_text()) for path in paths]
    expected = {(spec.split(":", 1)[0], threads) for spec in MODELS for threads in ("default", "1")}
    actual = [(r["model"]["name"], r["threads"]) for r in runs]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Merge requires exactly four models x two thread settings")
    identities, checkpoints = {}, {}
    for run in runs:
        if (run["status"] != "complete" or run["requested_scenarios"] != SCENARIOS
                or run["n_warmup"] != 5 or run["n_timed"] != 30 or run["n_construct"] != 3
                or run["input_order"] != "ref" or not run["fixed_signature_matches_original"]):
            raise ValueError("Incomplete worker or different benchmark protocol")
        name = run["model"]["name"]
        if name in checkpoints and checkpoints[name] != run["model"]:
            raise ValueError("Checkpoint differs between thread settings")
        checkpoints[name] = run["model"]
        if [r["scenario"] for r in run["scenarios"]] != SCENARIOS:
            raise ValueError("Each worker must contain all six scenarios in order")
        for row in run["scenarios"]:
            identity = tuple(row[k] for k in ("scenario_id", "split", "n_nodes", "n_links", "n_queues", "n_flows"))
            if row["scenario"] in identities and identities[row["scenario"]] != identity:
                raise ValueError("Scenario differs between benchmark workers")
            identities[row["scenario"]] = identity
    return dict(schema_version=1, status="complete", n_benchmark_combinations=48,
                worker_files=[str(p.resolve()) for p in paths], runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="name:fermi|student:exact_checkpoint (relative to repo root)")
    parser.add_argument("--threads", choices=("default", "1"), default="default")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--merge", nargs="+", type=Path, help="Merge eight complete worker JSON files; no TF import")
    parser.add_argument("--list-models", action="store_true", help="Print fixed model specs for the shell loop")
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(MODELS))
        return
    if args.out is None or (not args.merge and not args.model):
        parser.error("--out and either --model or --merge are required")
    if args.out.exists():
        parser.error("Output already exists; use a new path to preserve the previous run")
    if len(args.scenarios) != len(set(args.scenarios)):
        parser.error("Duplicate scenarios")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.merge:
        if args.model:
            parser.error("--merge cannot be combined with --model")
        save(args.out, merge(args.merge))
        print("Saved all 48 benchmark combinations:", args.out)
    else:
        worker(args)


if __name__ == "__main__":
    main()
