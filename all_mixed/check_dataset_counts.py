import os
import argparse
import numpy as np

# import your generator directly so we can count filtered samples
from data_generator import generator


def count_split(data_dir, max_raw=None):
    """
    Counts:
      - raw samples seen from DatanetAPI iterator (before filtering)
      - filtered out samples due to non-positive targets
      - yielded usable samples
    """
    raw = 0
    filtered = 0
    usable = 0

    # generator(data_dir, shuffle) yields only usable samples (after filtering)
    # But we also want raw count. So we recreate the logic by iterating DatanetAPI ourselves.
    # Easiest: use your generator AND also track raw by copying its internal structure.
    #
    # Since your generator currently hides raw count, we implement a local "instrumented" generator
    # by importing DatanetAPI and calling the same preprocessing as in your code.

    # --- We need the same imports as your data_generator.py uses ---
    import networkx as nx
    import tensorflow as tf
    from datanetAPI import DatanetAPI
    from data_generator import network_to_hypergraph, hypergraph_to_input_data

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
        x, y = hypergraph_to_input_data(HG)  # x = dict inputs, y = list delays

        # Filter rule same as yours
        if not all(v > 0 for v in y):
            filtered += 1
            continue

        usable += 1

    return raw, filtered, usable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train directory")
    parser.add_argument("--val", required=True, help="Path to validation directory")
    parser.add_argument("--test", required=True, help="Path to test directory")
    parser.add_argument("--max_raw", type=int, default=None,
                        help="Optional cap for debugging (e.g., 1000). Counts only first N raw samples.")
    args = parser.parse_args()

    for name, path in [("TRAIN", args.train), ("VAL", args.val), ("TEST", args.test)]:
        if not os.path.isdir(path):
            raise SystemExit(f"{name} path not found or not a directory: {path}")

        raw, filtered, usable = count_split(path, max_raw=args.max_raw)
        print(f"\n{name}: {path}")
        print(f"  Raw samples seen        : {raw}")
        print(f"  Filtered (non-positive) : {filtered}")
        print(f"  Usable yielded          : {usable}")
        if raw > 0:
            print(f"  Usable ratio            : {usable/raw:.3f}")


if __name__ == "__main__":
    main()
