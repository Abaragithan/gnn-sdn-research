"""Timestamped fixed-size CBR UDP traffic. Runs inside Mininet host namespaces."""
import argparse
import csv
import heapq
import json
import socket
import struct
import time
from pathlib import Path

from settings import PAYLOAD_BYTES, FRAME_BYTES, UDP_PORT

HEADER = struct.Struct("!4sIQQQ")
MAGIC = b"GNN1"


def send(args, config):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    queue = []
    by_id = {f["id"]: f for f in config["flows"] if f["src"] == args.host}
    for fid, flow in by_id.items():
        interval = FRAME_BYTES * 8 / flow["rate_bps"]
        # Identical phase offsets and offered rates across manually selected methods.
        heapq.heappush(queue, (args.start + interval * ((fid % 7) / 7), fid, 0))
    with open(args.output, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["flow", "seq", "scheduled_ns", "sent_ns", "late_ns", "bytes"])
        while queue:
            scheduled, fid, seq = heapq.heappop(queue)
            if scheduled >= args.start + args.seconds:
                continue
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            sent = time.monotonic_ns()
            packet = HEADER.pack(MAGIC, fid, seq, round(scheduled * 1e9), sent)
            packet += b"\0" * (PAYLOAD_BYTES - len(packet))
            size = sock.sendto(packet, ("10.77.0.%d" % (by_id[fid]["dst"] + 1), UDP_PORT))
            out.writerow([fid, seq, round(scheduled * 1e9), sent,
                          sent - round(scheduled * 1e9), size])
            interval = FRAME_BYTES * 8 / by_id[fid]["rate_bps"]
            heapq.heappush(queue, (scheduled + interval, fid, seq + 1))
    sock.close()


def receive(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.25)
    with open(args.output, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["flow", "seq", "scheduled_ns", "sent_ns", "received_ns", "bytes"])
        Path(args.output + ".ready").touch()
        while time.monotonic() < args.start + args.seconds:
            try:
                packet, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            received = time.monotonic_ns()
            if len(packet) != PAYLOAD_BYTES:
                raise ValueError("Unexpected UDP payload size")
            magic, fid, seq, scheduled, sent = HEADER.unpack_from(packet)
            if magic != MAGIC:
                raise ValueError("Unexpected packet header")
            out.writerow([fid, seq, scheduled, sent, received, len(packet)])
    sock.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["send", "receive"])
    p.add_argument("--config")
    p.add_argument("--host", type=int)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.mode == "send":
        send(a, json.loads(Path(a.config).read_text()))
    else:
        receive(a)
