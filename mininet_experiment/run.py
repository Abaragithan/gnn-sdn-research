"""Run ONE method on the pilot or a prepared real-data scenario."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from urllib.request import Request, urlopen

from settings import (HERE, MODEL_PYTHON, METHODS, CONTROLLER_PORT, API_PORT,
                      WORKER_PORT, FRAME_BYTES, PAYLOAD_BYTES, QUEUE_PACKETS,
                      WARMUP, DURATION, capacity, digest, save, scenario)


def api(path, data=None, port=API_PORT, timeout=60):
    request = Request("http://127.0.0.1:%d%s" % (port, path),
                      data=None if data is None else json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for(check, seconds=45):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (OSError, ValueError) as exc:
            last = exc
        time.sleep(0.2)
    raise RuntimeError("Readiness timeout: " + str(last))



def percentiles(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def read_packets(directory, config, start, warmup, duration):
    low, high = round((start + warmup) * 1e9), round((start + warmup + duration) * 1e9)
    sent, received, failures, arrivals = {}, {}, [], set()
    for name, target in (("sent", sent), ("received", received)):
        for path in directory.glob(name + "_*.csv"):
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    row = {k: int(v) for k, v in row.items()}
                    if name == "received" and low <= row["received_ns"] < high:
                        arrivals.add((row["flow"], row["seq"]))
                    if low <= row["scheduled_ns"] < high:
                        key = row["flow"], row["seq"]
                        if key in target:
                            failures.append("duplicate_" + name)
                        target[key] = row
    flows = []
    for flow in config["flows"]:
        fid = flow["id"]
        tx = {k: r for k, r in sent.items() if k[0] == fid}
        rx = {k: r for k, r in received.items() if k[0] == fid}
        if not tx:
            failures.append("flow_%d_no_sent_packets" % fid)
            continue
        if not set(rx).issubset(tx):
            failures.append("flow_%d_unmatched_received_packets" % fid)
        delays = [(r["received_ns"] - r["sent_ns"]) / 1e6 for k, r in rx.items() if k in tx]
        late = [r["late_ns"] / 1e9 for r in tx.values()]
        # Endpoint quantization is inevitable at low packet rates; allow one packet.
        actual_rate = len(tx) * FRAME_BYTES * 8 / duration
        tolerance = max(0.03 * flow["rate_bps"], FRAME_BYTES * 8 / duration)
        if abs(actual_rate - flow["rate_bps"]) > tolerance:
            failures.append("flow_%d_offered_rate_mismatch" % fid)
        if percentiles(late, 0.99) > 0.01:
            failures.append("flow_%d_pacing_late" % fid)
        if any(d < 0 for d in delays):
            failures.append("negative_measured_delay")
        if not rx:
            failures.append("flow_%d_no_received_packets" % fid)
        flows.append({"flow": fid, "sent": len(tx), "received": len(rx),
                      "loss_percent": 100 * (len(tx) - len(rx)) / len(tx),
                      "mean_delay_ms": sum(delays) / len(delays) if delays else None,
                      "p95_delay_ms": percentiles(delays, 0.95),
                      "goodput_payload_bps": sum(key[0] == fid for key in arrivals) * PAYLOAD_BYTES * 8 / duration,
                      "offered_frame_bps": actual_rate, "target_frame_bps": flow["rate_bps"],
                      "pacing_late_p99_ms": percentiles(late, 0.99) * 1000})
    return {"flows": flows, "foreground": next((f for f in flows if f["flow"] == config["foreground"]), None),
            "failures": sorted(set(failures))}


class Experiment:
    def __init__(self, output):
        self.output = output
        self.net = self.controller = self.worker = None
        self.logs, self.children = [], []
        self.prefix = "gn%04x" % (os.getpid() % 65536)

    def launch(self, command, logfile, env=None):
        handle = open(logfile, "w")
        self.logs.append(handle)
        return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)

    @staticmethod
    def stop(process):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def start(self, config):
        from mininet.net import Mininet
        from mininet.node import RemoteController, OVSSwitch
        from mininet.link import Link
        # Refuse occupied service ports; never reuse another experiment's service.
        for port in (CONTROLLER_PORT, API_PORT, WORKER_PORT):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        existing = subprocess.check_output(["ovs-vsctl", "--timeout=3", "list-br"], text=True).split()
        save(self.output / "existing_bridges.json", existing)
        if any(name.startswith(self.prefix) for name in existing):
            raise RuntimeError("Experiment switch name collision")
        # A free port may still be referenced by an unrelated, disconnected bridge.
        local_targets = {proto + host + suffix for proto in ("tcp:", "ssl:")
                         for host in ("127.0.0.1", "localhost", "[::1]", "::1")
                         for suffix in ("", ":%d" % CONTROLLER_PORT)}
        for bridge in existing:
            targets = subprocess.check_output(["ovs-vsctl", "--timeout=3", "get-controller", bridge], text=True).splitlines()
            if local_targets.intersection(targets):
                raise RuntimeError("Existing bridge %s references this local controller port; leaving it unchanged" % bridge)
        self.controller = self.launch([
            str(HERE / ".venv/bin/ryu-manager"), "--ofp-listen-host", "127.0.0.1",
            "--ofp-tcp-listen-port", str(CONTROLLER_PORT), "--wsapi-host", "127.0.0.1",
            "--wsapi-port", str(API_PORT), str(HERE / "controller.py")], self.output / "controller.log",
            env=dict(os.environ, PYTHONPATH=str(HERE)))
        wait_for(lambda: self.controller.poll() is None and api("/status", timeout=1))
        self.net = Mininet(controller=None, switch=OVSSwitch, link=Link, build=False,
                           autoSetMacs=False, autoStaticArp=False)
        self.net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=CONTROLLER_PORT)
        self.hosts, self.switches, self.ports = [], [], {}
        for i in range(config["n"]):
            host = self.net.addHost("h%d" % i, ip="10.77.0.%d/24" % (i + 1),
                                    mac="02:77:00:00:00:%02x" % (i + 1))
            switch = self.net.addSwitch(self.prefix + str(i), protocols="OpenFlow13",
                                       dpid="%016x" % (0x470000000000 + os.getpid() * 16 + i),
                                       failMode="secure")
            self.hosts.append(host)
            self.switches.append(switch)
            link = self.net.addLink(host, switch)
            self.ports[i, "host"] = switch.ports[link.intf2]
        self.shaped = []
        for a, b in config["edges"]:
            link = self.net.addLink(self.switches[a], self.switches[b])
            self.ports[a, b] = self.switches[a].ports[link.intf1]
            self.ports[b, a] = self.switches[b].ports[link.intf2]
            self.shaped.extend([(link.intf1, capacity(config, a, b)),
                                (link.intf2, capacity(config, b, a))])
        self.net.build()
        self.net.start()
        expected = {int(s.dpid, 16) for s in self.switches}
        print("Waiting for %d switches to connect to Ryu on port %d..." % (config["n"], CONTROLLER_PORT), flush=True)
        try:
            wait_for(lambda: set(api("/status")["dpids"]) == expected)
        except RuntimeError as exc:
            raise RuntimeError("Switch connection timeout; Ryu status: " + json.dumps(api("/status"))) from exc
        for host in self.hosts:
            for other in self.hosts:
                if host != other:
                    host.setARP(other.IP(), other.MAC())
        for node in self.hosts + self.switches:
            for interface in node.intfList():
                if str(interface) != "lo":
                    # Disable segmentation/coalescing: one skb must mean one frame.
                    result = node.pexec(["ethtool", "-K", str(interface), "tso", "off", "gso", "off", "gro", "off"])
                    if result[2] != 0:
                        raise RuntimeError("Could not disable offloads: " + str(result))
                    out, err, code = node.pexec(["sysctl", "-qw",
                        "net.ipv6.conf.%s.disable_ipv6=1" % interface])
                    if code:
                        raise RuntimeError("Cannot disable experiment interface IPv6: " + out + err)
        save(self.output / "ports.json", {str(k): v for k, v in self.ports.items()})

    def shape(self, config):
        records = {}
        for interface, rate_bps in self.shaped:
            # netem rate serializes fixed-size frames; its sole queue is FIFO.
            # No HTB/TBF token bucket or additional unmodelled child queue.
            command = ["tc", "qdisc", "replace", "dev", str(interface), "root", "handle", "10:",
                       "netem", "limit", str(QUEUE_PACKETS), "rate", str(rate_bps) + "bit"]
            out, err, code = interface.node.pexec(command)
            if code:
                raise RuntimeError("Shaping failed: " + out + err)
            out, err, code = interface.node.pexec(["tc", "-j", "-s", "qdisc", "show", "dev", str(interface)])
            data = json.loads(out)
            q = next(q for q in data if q["kind"] == "netem")
            if q["options"]["limit"] != QUEUE_PACKETS:
                raise RuntimeError("Queue limit does not match configured packet count")
            # tc JSON reports bytes/second, unlike command-line bit/second.
            rate = q["options"]["rate"]
            rate = rate["rate"] if isinstance(rate, dict) else rate
            if abs(rate * 8 - rate_bps) > 1:
                raise RuntimeError("Queue rate units do not match configured capacity")
            records[str(interface)] = data
        save(self.output / "qdisc_before.json", records)

    def rules(self, config):
        result = []
        for candidate in range(3):
            flows = []
            for f in config["flows"]:
                path = config["candidates"][candidate] if f["id"] == config["foreground"] else f["path"]
                outputs = [{"dpid": int(self.switches[a].dpid, 16), "port": self.ports[a, b]}
                           for a, b in zip(path, path[1:])]
                outputs.append({"dpid": int(self.switches[path[-1]].dpid, 16),
                                "port": self.ports[path[-1], "host"]})
                flows.append({"id": f["id"], "src_ip": self.hosts[f["src"]].IP(),
                              "dst_ip": self.hosts[f["dst"]].IP(), "outputs": outputs})
            result.append(flows)
        return result

    def counters(self, rules, directory, suffix):
        dumps, counts = {}, {}
        for switch in self.switches:
            text = subprocess.check_output(["ovs-ofctl", "--no-names", "-O", "OpenFlow13", "dump-flows", switch.name], text=True)
            dumps[switch.name] = text
            for line in text.splitlines():
                if "cookie=0x474e4e01" not in line:
                    continue
                src = re.search(r"nw_src=([\d.]+)", line)
                dst = re.search(r"nw_dst=([\d.]+)", line)
                packets = re.search(r"n_packets=(\d+)", line)
                port = re.search(r"actions=output:(\d+)\s*$", line)
                if not all((src, dst, packets, port)):
                    raise RuntimeError("Unexpected flow format: " + line)
                counts[int(switch.dpid, 16), src[1], dst[1], int(port[1])] = int(packets[1])
        save(directory / ("flow_dump_" + suffix + ".json"), dumps)
        failures, expected = [], set()
        for flow in rules:
            for step in flow["outputs"]:
                key = step["dpid"], flow["src_ip"], flow["dst_ip"], step["port"]
                expected.add(key)
                if key not in counts or (suffix != "before" and counts[key] <= 0):
                    failures.append("missing_or_unused_expected_flow:" + str(key))
        if set(counts) != expected:
            failures.append("unexpected_forwarding_rules")
        return failures

    def measure(self, config, method, candidate=None):
        """Load one model, install its chosen route, send packets, then summarize."""
        startup = time.perf_counter()
        duration = config.get("measurement_s", DURATION)
        if method != "shortest":
            self.worker = self.launch([str(MODEL_PYTHON), str(HERE / "model_worker.py"), method,
                                       "--scenario", str(self.output / "scenario.json")],
                                      self.output / "worker.log")
            def ready():
                if self.worker.poll() is not None:
                    raise RuntimeError("Worker failed; inspect worker.log")
                return api("/", port=WORKER_PORT, timeout=1)
            identity = wait_for(ready, seconds=90)
            save(self.output / "prediction_checks.json", identity)
            if identity["scenario_sha256"] != digest(config) or identity["method"] != method:
                raise RuntimeError("Worker configuration does not match this pilot")
            if not all(check["valid"] for check in identity["physical_checks"]):
                raise RuntimeError("Invalid queue predictions; inspect prediction_checks.json. No route selected.")
        startup_ms = (time.perf_counter() - startup) * 1000
        started = time.perf_counter()
        rules = self.rules(config)
        request = {"method": method, "scenario": config, "candidate_rules": rules}
        if candidate is not None:
            request["candidate"] = candidate
        decision = api("/decision", request)
        decision["decision_to_install_ms"] = (time.perf_counter() - started) * 1000
        decision["worker_startup_and_warmup_ms"] = startup_ms if self.worker else 0
        save(self.output / "decision.json", decision)
        selected = decision["selected"]
        failures = self.counters(rules[selected], self.output, "before")
        if failures:
            raise RuntimeError("Installed rules do not match the requested paths: " + str(failures))

        # Keep receivers alive long enough to receive packets still in the queues.
        hops = max(len(p) - 1 for p in config["candidates"] + [f["path"] for f in config["flows"]])
        slowest = min(rate for _, rate in self.shaped)
        drain = math.ceil(hops * (config["queue_bits"] + FRAME_BYTES * 8) / slowest) + 2
        start = time.monotonic() + 5
        for host in self.hosts:
            self.children.append(host.popen([
                "/usr/sbin/python", str(HERE / "traffic.py"), "receive", "--start", str(start),
                "--seconds", str(WARMUP + duration + drain),
                "--output", str(self.output / ("received_%s.csv" % host.name))],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        wait_for(lambda: len(list(self.output.glob("received_*.csv.ready"))) == config["n"], seconds=4)
        for i, host in enumerate(self.hosts):
            self.children.append(host.popen([
                "/usr/sbin/python", str(HERE / "traffic.py"), "send", "--host", str(i),
                "--config", str(self.output / "scenario.json"), "--start", str(start),
                "--seconds", str(WARMUP + duration),
                "--output", str(self.output / ("sent_%s.csv" % host.name))],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        while any(p.poll() is None for p in self.children):
            if time.monotonic() > start + WARMUP + duration + drain + 10:
                raise RuntimeError("Traffic helper timed out")
            if self.controller.poll() is not None or (self.worker and self.worker.poll() is not None):
                raise RuntimeError("Controller or model worker exited")
            time.sleep(0.25)
        for process in self.children:
            if process.returncode:
                failures.append("traffic_helper_failed:" + process.stderr.read().decode())
            process.stderr.close()
        self.children.clear()

        failures.extend(self.counters(rules[selected], self.output, "after"))
        qdiscs = {}
        for interface, _ in self.shaped:
            out, err, code = interface.node.pexec(["tc", "-j", "-s", "qdisc", "show", "dev", str(interface)])
            if code:
                raise RuntimeError("Cannot read queue counters: " + err)
            qdiscs[str(interface)] = json.loads(out)
            q = next(q for q in qdiscs[str(interface)] if q["kind"] == "netem")
            if q.get("packets", 0) and q.get("bytes", 0) != q["packets"] * FRAME_BYTES:
                failures.append("qdisc_frame_size_mismatch:" + str(interface))
        save(self.output / "qdisc_after.json", qdiscs)
        packets = read_packets(self.output, config, start, WARMUP, duration)
        failures.extend(packets.pop("failures"))
        if api("/status")["errors"]:
            failures.append("openflow_errors")
        return dict(method=method, selected_path=config["candidates"][selected], decision=decision,
                    forced_candidate=candidate,
                    warmup_s=WARMUP, measurement_s=duration, drain_s=drain,
                    accepted=not failures, failures=sorted(set(failures)), **packets)


    def close(self):
        try:
            for p in self.children:
                self.stop(p)
            self.stop(self.worker)
            if self.net is not None:
                self.net.stop()
        finally:
            self.stop(self.controller)
            for handle in self.logs:
                handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--name", required=True, help="New output folder name, e.g. shortest_01")
    parser.add_argument("--scenario", type=Path, help="Prepared real-data JSON; omit for the original pilot")
    parser.add_argument("--candidate", type=int, choices=[0, 1, 2], help="Measure a fixed candidate; --method shortest only")
    parser.add_argument("--seconds", type=float, help="Measurement duration; warm-up and drainage are separate")
    args = parser.parse_args()
    if args.candidate is not None and args.method != "shortest":
        parser.error("--candidate is for fixed-path measurements, not model selection")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if os.geteuid() != 0:
        parser.error("Run this pilot from your terminal with sudo")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("Use a single folder name")
    output = HERE / "outputs" / args.name
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    config = json.loads(args.scenario.read_text()) if args.scenario else scenario()
    if args.seconds is not None:
        config["measurement_s"] = args.seconds
    save(output / "scenario.json", config)
    experiment = Experiment(output)
    result = {"method": args.method, "accepted": False}
    preflight_ms = 0
    try:
        if args.method != "shortest":
            print("Checking model predictions before starting Mininet...", flush=True)
            preflight_started = time.perf_counter()
            with (output / "preflight.log").open("w") as log:
                checked = subprocess.run([str(MODEL_PYTHON), str(HERE / "model_worker.py"), args.method,
                    "--scenario", str(output / "scenario.json"), "--check",
                    "--output", str(output / "preflight_checks.json")], stdout=log, stderr=subprocess.STDOUT)
            preflight_ms = (time.perf_counter() - preflight_started) * 1000
            if checked.returncode:
                raise RuntimeError("Model preflight failed; inspect preflight_checks.json and preflight.log. Mininet not started.")
        print("1/4 Starting the network and Ryu...", flush=True)
        experiment.start(config)       # 1. Create this scenario's switches and start Ryu.
        print("2/4 Configuring link rates and FIFO queues...", flush=True)
        experiment.shape(config)       # 2. Apply and check rates and FIFO limits.
        print("3/4 Selecting a route and measuring packets...", flush=True)
        result = experiment.measure(config, args.method, args.candidate)  # 3. Select, install, measure.
        result["preflight_ms"] = preflight_ms  # One-off validation, outside decision timing.
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        result["preflight_ms"] = preflight_ms
        result["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        try:
            save(output / "result.json", result)
        finally:
            print("4/4 Stopping this pilot's network and processes...", flush=True)
            experiment.close()         # 4. Stop only this pilot's resources.
        print("Saved:", output)
    if not result["accepted"]:
        raise SystemExit("Pilot failed acceptance checks; inspect result.json")


if __name__ == "__main__":
    main()
