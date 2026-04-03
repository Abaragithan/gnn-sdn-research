import os
import argparse
import numpy as np

import networkx as nx
from datanetAPI import DatanetAPI

from data_generator import network_to_hypergraph, hypergraph_to_input_data


def collect_delays(data_dir, max_raw=None):
    """
    Returns:
      delays_all: all delays from usable samples (after filtering)
      delays_raw: delays from all samples (even if they'd be filtered) [optional for insight]
      counts: dict with raw, filtered, usable
    """
    raw = 0
    filtered = 0
    usable = 0

    delays_all = []
    delays_raw = []

    tool = DatanetAPI(data_dir, shuffle=False)
    it = iter(tool)

    for sample in it:
        raw += 1
        if max_raw is not None and raw > max_raw:
            break

        G = nx.DiGraph(sample.get_topology_object())
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        P = sample.get_performance_matrix()

        HG = network_to_hypergraph(G=G, R=R, T=T, P=P)
        x, y = hypergraph_to_input_data(HG)  # y is list of per-path delays (floats)

        # store raw delays for extra insight
        delays_raw.extend(list(y))

        # apply same filter as training pipeline
        if not all(v > 0 for v in y):
            filtered += 1
            continue

        usable += 1
        delays_all.extend(list(y))

    return np.array(delays_all, dtype=np.float64), np.array(delays_raw, dtype=np.float64), {
        "raw_samples": raw,
        "filtered_samples": filtered,
        "usable_samples": usable,
        "usable_ratio": (usable / raw) if raw else 0.0
    }


def summarize(delays, thresholds=(1e-6, 1e-5, 1e-4, 1e-3)):
    if delays.size == 0:
        return {"count": 0}

    stats = {
        "count": int(delays.size),
        "min": float(np.min(delays)),
        "max": float(np.max(delays)),
        "mean": float(np.mean(delays)),
        "median": float(np.median(delays)),
        "p1": float(np.percentile(delays, 1)),
        "p5": float(np.percentile(delays, 5)),
        "p95": float(np.percentile(delays, 95)),
        "p99": float(np.percentile(delays, 99)),
    }

    for t in thresholds:
        stats[f"pct_delay_lt_{t:g}"] = float(np.mean(delays < t) * 100.0)

    return stats


def print_stats(title, counts, stats):
    print(f"\n=== {title} ===")
    print(f"Raw samples        : {counts['raw_samples']}")
    print(f"Filtered samples   : {counts['filtered_samples']}")
    print(f"Usable samples     : {counts['usable_samples']}")
    print(f"Usable ratio       : {counts['usable_ratio']:.3f}")

    if stats.get("count", 0) == 0:
        print("No delays collected.")
        return

    print(f"Delay count (paths): {stats['count']}")
    print(f"min    : {stats['min']}")
    print(f"p1     : {stats['p1']}")
    print(f"p5     : {stats['p5']}")
    print(f"median : {stats['median']}")
    print(f"mean   : {stats['mean']}")
    print(f"p95    : {stats['p95']}")
    print(f"p99    : {stats['p99']}")
    print(f"max    : {stats['max']}")

    # tiny-delay percentages
    for k, v in stats.items():
        if k.startswith("pct_delay_lt_"):
            print(f"{k.replace('pct_', '')}: {v:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to split directory (train/val/test)")
    parser.add_argument("--max_raw", type=int, default=None, help="Optional cap for debugging (e.g., 1000)")
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Split path not found or not a directory: {args.data}")

    usable_delays, raw_delays, counts = collect_delays(args.data, max_raw=args.max_raw)

    usable_stats = summarize(usable_delays)
    raw_stats = summarize(raw_delays)

    print_stats("USABLE (after filter v>0)", counts, usable_stats)
    print_stats("RAW (before filtering)", counts, raw_stats)

    # quick guidance
    if usable_stats.get("count", 0) > 0:
        pct_tiny = usable_stats.get("pct_delay_lt_0.001", 0.0)  # <1e-3
        if pct_tiny >= 5.0:
            print("\nNOTE: Many delays are very small (<1e-3). MAPE can blow up.")
            print("Recommendation: report MAE and/or train with MAE/Huber, keep MAPE only for comparison.")


if __name__ == "__main__":
    main()


# python check_delay_distribution.py --data ../data/all_mixed/train
# python check_delay_distribution.py --data ../data/all_mixed/validation
# python check_delay_distribution.py --data ../data/all_mixed/test
