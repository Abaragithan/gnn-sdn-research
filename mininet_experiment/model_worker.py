"""One frozen model per process. Uses the existing converter without modifying it."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from settings import ROOT, CHECKPOINTS, WORKER_PORT, capacity, digest, save, scenario
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "all_mixed"))
import networkx as nx
import numpy as np
import tensorflow as tf
from data_generator import network_to_hypergraph, hypergraph_to_input_data
from datanetAPI import TimeDist
from delay_model import RouteNet_Fermi
from student_delay_model import StudentRouteNet

RAGGED = {"link_to_path", "queue_to_path", "queue_to_link", "path_to_queue", "path_to_link"}
INTEGER = {"length", "model", "policy", "priority"}


def build_inputs(config, candidate, preserve_hop_order=True):
    """Time unit = second; bits include Ethernet header, exclude FCS/preamble/IFG.

    P is required only by the unmodified converter. Its dummy labels are removed
    before inference; measured outcomes are never accepted by this interface.
    """
    n = config["n"]
    graph = nx.DiGraph()
    for node in range(n):
        graph.add_node(node, schedulingPolicy="FIFO", levelsQoS=1,
                       schedulingWeights="-", tosToQoSqueue="0,1,2",
                       bufferSizes=str(config["queue_bits"]))
    for a, b in config["edges"]:
        graph.add_edge(a, b, bandwidth=capacity(config, a, b))
        graph.add_edge(b, a, bandwidth=capacity(config, b, a))
    traffic, routes, performance = (np.empty((n, n), dtype=object) for _ in range(3))
    for a in range(n):
        for b in range(n):
            traffic[a, b] = {"Flows": []}
            performance[a, b] = {"Flows": []}
            routes[a, b] = []
    for flow in config["flows"]:
        src, dst = flow["src"], flow["dst"]
        rate = flow["rate_bps"]
        packets = rate / (config["frame_bytes"] * 8)
        traffic[src, dst] = {"Flows": [{
            "AvgBw": rate, "PktsGen": packets, "TimeDist": TimeDist.DETERMINISTIC_T,
            "TimeDistParams": {"EqLambda": rate, "AvgPktsLambda": packets}, "ToS": 0}]}
        performance[src, dst] = {"Flows": [{"AvgDelay": 0.0}]}
        routes[src, dst] = (config["candidates"][candidate]
                            if flow["id"] == config["foreground"] else flow["path"])
        path = routes[src, dst]
        if path[0] != src or path[-1] != dst or len(set(path)) != len(path):
            raise ValueError("Invalid route")
        if any(not graph.has_edge(a, b) for a, b in zip(path, path[1:])):
            raise ValueError("Route traverses nonexistent link")
    hg = network_to_hypergraph(graph, routes, traffic, performance)
    result = convert_inputs(hg, preserve_hop_order)
    links = [v for v in hg if v.startswith("l_")]
    for flow in config["flows"]:
        route = config["candidates"][candidate] if flow["id"] == config["foreground"] else flow["path"]
        expected = ["l_%d_%d" % edge for edge in zip(route, route[1:])]
        actual = [links[i] for i in result["link_to_path"][flow["id"]].numpy().tolist()]
        if preserve_hop_order and actual != expected:
            raise ValueError("Converted hop order differs from requested route")
    return result


def convert_inputs(hg, preserve_hop_order=True):
    """Convert an author's hypergraph; the audit can also inspect historical order."""
    raw, _ = hypergraph_to_input_data(hg)
    # nx.relabel_nodes(copy=True) can reorder INCOMING edges by global node
    # insertion order. The converter's path_to_link positions instead use
    # OUTGOING path edges, whose order is preserved. Make both agree here.
    # This is deployment-only; the historical converter is untouched.
    links = [v for v in hg if v.startswith("l_")]
    queues = [v for v in hg if v.startswith("q_")]
    path_nodes = [v for v in hg if v.startswith("p_")]
    link_ids, queue_ids = {v: i for i, v in enumerate(links)}, {v: i for i, v in enumerate(queues)}
    if preserve_hop_order:
        raw["link_to_path"] = tf.ragged.constant([
            [link_ids[v] for _, v in hg.out_edges(p) if v.startswith("l_")] for p in path_nodes], dtype=tf.int32)
        raw["queue_to_path"] = tf.ragged.constant([
            [queue_ids[v] for _, v in hg.out_edges(p) if v.startswith("q_")] for p in path_nodes], dtype=tf.int32)
    raw.pop("delay")
    result = {}
    for key, value in raw.items():
        if key in RAGGED:
            result[key] = tf.cast(value, tf.int32)
        else:
            result[key] = tf.convert_to_tensor(value, dtype=tf.int32 if key in INTEGER else tf.float32)
    if preserve_hop_order:
        for forward, reverse in (("link_to_path", "path_to_link"), ("queue_to_path", "path_to_queue")):
            for node, positions in enumerate(result[reverse].to_list()):
                if any(int(result[forward][path][hop]) != node for path, hop in positions):
                    raise ValueError("Inconsistent path connectivity tensors")
    return result


def load_model(method, config):
    model = RouteNet_Fermi() if method == "teacher" else StudentRouteNet(state_dim=16, iterations=4)
    model(build_inputs(config, 0)).numpy()
    status = model.load_weights(str(ROOT / CHECKPOINTS[method]))
    status.expect_partial()
    status.assert_existing_objects_matched()
    return model


def prediction(model, inputs):
    result = model(inputs).numpy().reshape(-1)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Invalid model prediction; do not clip or silently select a route")
    return result


def physical_checks(model, inputs):
    """Inspect readout without clipping or changing weights; outside timed inference.

    Per-hop checks apply to this experiment's single FIFO queue per link.
    Eager execution exposes the existing readout; normal inference stays compiled.
    """
    captured = {}
    original = model.readout_path.call
    def capture(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        captured["bits"] = result.numpy()[..., 0]
        return result
    model.readout_path.call = capture
    try:
        predicted = model.call.python_function(inputs).numpy().reshape(-1)
    finally:
        model.readout_path.call = original
    mask = np.arange(captured["bits"].shape[1])[None, :] < inputs["length"].numpy()[:, None]
    bits = captured["bits"][mask]
    limits = tf.gather(inputs["queue_size"], inputs["queue_to_path"]).flat_values.numpy().reshape(-1)
    capacities = tf.gather(inputs["capacity"], inputs["link_to_path"])
    transmission = ((inputs["traffic"] / inputs["packets"]) * tf.reduce_sum(1 / capacities, axis=1)).numpy().reshape(-1)
    upper = transmission + tf.reduce_sum(
        tf.gather(inputs["queue_size"], inputs["queue_to_path"]) / capacities, axis=1).numpy().reshape(-1)
    violations = {"nonfinite": int(np.sum(~np.isfinite(bits)) + np.sum(~np.isfinite(predicted))),
                  "negative_queue_hops": int(np.sum(bits < -0.001)),
                  "overfull_queue_hops": int(np.sum(bits > limits + 0.001)),
                  "below_serialization_flows": int(np.sum(predicted < transmission - 1e-8)),
                  "above_queue_bound_flows": int(np.sum(predicted > upper + 1e-8))}
    return dict(valid=not any(violations.values()), violations=violations,
                minimum_queued_bits=float(np.min(bits)) if np.all(np.isfinite(bits)) else None,
                maximum_queued_bits=float(np.max(bits)) if np.all(np.isfinite(bits)) else None)



def serve(method, config):
    model = load_model(method, config)
    inputs = [build_inputs(config, k) for k in range(3)]
    checks = [physical_checks(model, x) for x in inputs]
    if all(check["valid"] for check in checks):
        for _ in range(5):
            for x in inputs:
                prediction(model, x)
    prefix = ROOT / CHECKPOINTS[method]
    identity = {"method": method, "checkpoint": CHECKPOINTS[method], "pid": os.getpid(),
                "scenario_sha256": digest(config), "physical_checks": checks,
                "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(prefix.parent.glob(prefix.name + ".*"))}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(identity)  # Server starts only after weights load and warm-up.

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if digest(request["scenario"]) != digest(config):
                    raise ValueError("Runner and worker must use the same fixed scenario")
                if not all(check["valid"] for check in checks):
                    raise ValueError("Physical prediction checks failed; inspect worker identity")
                t0 = time.perf_counter()
                inputs = [build_inputs(config, k) for k in range(3)]
                preparation_ms = (time.perf_counter() - t0) * 1000
                t1 = time.perf_counter()
                scores = [float(prediction(model, x)[config["foreground"]]) for x in inputs]
                inference_ms = (time.perf_counter() - t1) * 1000
                self.reply(dict(identity, selected=int(np.argmin(scores)), predicted_delay_s=scores,
                                preparation_ms=preparation_ms, inference_ms=inference_ms))
            except Exception as exc:
                self.reply({"error": str(exc)}, 400)

    class Server(HTTPServer):
        allow_reuse_address = True
    print(json.dumps(identity), flush=True)
    Server(("127.0.0.1", WORKER_PORT), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(CHECKPOINTS))
    parser.add_argument("--scenario", help="Prepared scenario JSON; omit for the six-switch pilot")
    parser.add_argument("--check", action="store_true", help="Check predictions only; no server or Mininet")
    parser.add_argument("--output", help="New JSON file for --check results")
    args = parser.parse_args()
    config = scenario()
    if args.scenario:
        with open(args.scenario) as handle:
            config = json.load(handle)
    if args.check:
        if args.output and os.path.exists(args.output):
            parser.error("Check output already exists; choose a new filename")
        model = load_model(args.method, config)
        checks = [physical_checks(model, build_inputs(config, k)) for k in range(3)]
        report = dict(method=args.method, checkpoint=CHECKPOINTS[args.method],
                      scenario_sha256=digest(config), physical_checks=checks)
        if args.output:
            save(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
        sys.exit(0 if all(check["valid"] for check in checks) else 2)
    else:
        serve(args.method, config)
