"""One fixed pilot configuration, shared by the runner and model worker."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PYTHON = ROOT.parent / "miniconda3/envs/routenet-fermi/bin/python"
CHECKPOINTS = {
    "teacher": "all_mixed/ckpt_dirs_teacher/ckpt_dir_0_9972/ckpt-26",
    "scratch": "all_mixed/student/ckpt_dirs_scratch/ckpt-08",
    "kd": "all_mixed/student/ckpt_dirs_kd_alpha0.5/ckpt-19",
}
METHODS = ["shortest", "teacher", "scratch", "kd"]
# Fedora SELinux permits OVS to connect to the OpenFlow-labelled port 6653.
CONTROLLER_PORT, API_PORT, WORKER_PORT, UDP_PORT = 6653, 18080, 18878, 19000
FRAME_BYTES = 125  # Ethernet + IPv4 + UDP + payload; excludes FCS/preamble/IFG.
PAYLOAD_BYTES = FRAME_BYTES - 14 - 20 - 8
CAPACITY_BPS, QUEUE_BITS = 20000, 32000
QUEUE_PACKETS = QUEUE_BITS // (FRAME_BYTES * 8)
WARMUP, DURATION = 2, 10  # Seconds; this is a short integration pilot.


def scenario():
    edges = [(0, 1), (1, 5), (0, 2), (2, 3), (3, 5), (0, 4), (4, 5)]
    candidates = [[0, 1, 5], [0, 4, 5], [0, 2, 3, 5]]
    neighbors = {i: [] for i in range(6)}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    def shortest(src, dst):
        pending = [[src]]
        while pending:
            path = pending.pop(0)
            if path[-1] == dst:
                return path
            pending.extend(path + [v] for v in sorted(neighbors[path[-1]]) if v not in path)
        raise ValueError("Disconnected topology")

    # One flow per ordered host pair. Only foreground flow 0 -> 5 changes path.
    flows, link_load = [], {}
    for src in range(6):
        for dst in range(6):
            if src == dst:
                continue
            path = shortest(src, dst)
            weight = 0.5 + ((src * 7 + dst * 3) % 6) / 10.0
            flows.append(dict(id=len(flows), src=src, dst=dst, path=path, rate_bps=weight))
            for edge in zip(path, path[1:]):
                link_load[edge] = link_load.get(edge, 0) + weight
    scale = 0.6 * CAPACITY_BPS / max(link_load.values())
    for flow in flows:
        flow["rate_bps"] *= scale
    foreground = next(f["id"] for f in flows if f["src"] == 0 and f["dst"] == 5)
    return dict(n=6, edges=edges, candidates=candidates, flows=flows, foreground=foreground,
                capacity_bps=CAPACITY_BPS, queue_bits=QUEUE_BITS, frame_bytes=FRAME_BYTES)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def capacity(config, src, dst):
    """Real-data scenarios retain each directed link's published capacity."""
    return config.get("link_capacities_bps", {}).get("%d,%d" % (src, dst), config["capacity_bps"])


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
