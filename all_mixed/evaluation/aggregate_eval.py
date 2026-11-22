"""Aggregate T2 JSONL without TensorFlow or SciPy; time/error units are explicit."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def ranking_metrics(items):
    """Spearman (average ties), Kendall tau-b, strict pairwise accuracy.

    True ties have no correct ordering and are excluded from pairwise accuracy;
    a predicted tie on a non-tied true pair counts as incorrect. Counts are kept.
    Undefined correlations are null, never represented as zero.
    """
    true = np.array([x["mean_true"] for x in items])
    pred = np.array([x["mean_pred"] for x in items])
    rank_true, rank_pred = average_ranks(true), average_ranks(pred)
    spearman = None
    if len(items) > 1 and np.ptp(rank_true) and np.ptp(rank_pred):
        spearman = float(np.corrcoef(rank_true, rank_pred)[0, 1])
    concordant = discordant = true_ties = pred_ties = joint_ties = 0
    for i in range(len(items) - 1):
        a, b = np.sign(true[i + 1:] - true[i]), np.sign(pred[i + 1:] - pred[i])
        concordant += int(np.sum(a * b > 0))
        discordant += int(np.sum(a * b < 0))
        true_ties += int(np.sum((a == 0) & (b != 0)))
        pred_ties += int(np.sum((b == 0) & (a != 0)))
        joint_ties += int(np.sum((a == 0) & (b == 0)))
    untied = concordant + discordant
    denominator = float(np.sqrt((untied + true_ties) * (untied + pred_ties)))
    comparable = untied + pred_ties
    return dict(n_scenarios=len(items), spearman=spearman,
                kendall_tau_b=(concordant - discordant) / denominator if denominator else None,
                pairwise_accuracy=concordant / comparable if comparable else None,
                comparable_pairs=comparable, concordant_pairs=concordant,
                discordant_pairs=discordant, true_only_ties=true_ties,
                predicted_only_ties=pred_ties, joint_ties=joint_ties)


def pooled(items):
    n = sum(x["n_flows"] for x in items)
    mae = sum(x["sum_ae"] for x in items) / n
    rmse = float(np.sqrt(sum(x["sum_se"] for x in items) / n))
    return dict(n_scenarios=len(items), n_flows=n,
                pooled_mape=100 * sum(x["sum_ape"] for x in items) / n,
                unweighted_scenario_mape=float(np.mean([100 * x["sum_ape"] / x["n_flows"] for x in items])),
                pooled_mae_seconds=mae, pooled_mae_ms=mae * 1000,
                pooled_rmse_seconds=rmse, pooled_rmse_ms=rmse * 1000)


def timing(values):
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)))


def topology_folder(topology_id):
    """'200/<topo-folder>/graphs/graph_x.txt' -> '200/<topo-folder>'.

    A graph file is one topology plus one scheduling/QoS configuration; the
    folder is the topology itself, so it gives the larger, main ranking groups.
    """
    return Path(topology_id).parent.parent.as_posix()


def ranking_summary(groups):
    rankings = {key: ranking_metrics(values) for key, values in sorted(groups.items())}
    means = {}
    for key in ("spearman", "kendall_tau_b", "pairwise_accuracy"):
        values = [v[key] for v in rankings.values() if v[key] is not None]
        means[key] = dict(mean=float(np.mean(values)) if values else None,
                          n_defined=len(values), n_groups=len(rankings))
    return rankings, means


def aggregate(rows):
    if not rows or len({r["scenario_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty JSONL with unique scenario IDs")
    expected = {(name, order) for name, orders in rows[0]["models"].items() for order in orders}
    groups = defaultdict(list)
    for row in rows:
        actual = {(name, order) for name, orders in row["models"].items() for order in orders}
        if expected != actual:
            raise ValueError("Model/order coverage differs between scenarios")
        for name, order in expected:
            groups[name, order].append((row, row["models"][name][order]))
    summary = dict(n_scenarios=len(rows), models={},
                   pairwise_policy="True ties excluded; predicted ties on comparable pairs incorrect.",
                   timing_policy="parse = DatanetAPI next() (schema >= 2); construct = DiGraph + "
                                 "hypergraph + tensors; warm forward; hop correction excluded.",
                   ranking_policy="Main: per topology folder. Secondary: per graph file "
                                  "(topology + scheduling config). Groups differ mostly in "
                                  "traffic intensity, so high scores may be easy to reach.")
    for (name, order), entries in sorted(groups.items()):
        items = [item for _, item in entries]
        invalid = sum(x["n_nonfinite"] for x in items)
        result = dict(n_nonfinite=invalid, n_negative=sum(x["n_negative"] for x in items))
        if invalid:
            # Never silently pool only valid flows or scenarios.
            result["accuracy_unavailable"] = "Nonfinite predictions; inspect JSONL. No flows excluded."
        else:
            result.update(pooled(items))
            by_size, by_folder, by_graph = defaultdict(list), defaultdict(list), defaultdict(list)
            for row, item in entries:
                by_size[str(row["n_nodes"])].append(item)
                by_folder[topology_folder(row["topology_id"])].append(item)
                by_graph[row["topology_id"]].append(item)
            result["by_topology_size"] = {size: pooled(values) for size, values in sorted(by_size.items())}
            # Main ranking view: per topology folder. Secondary: per graph file
            # (topology + scheduling configuration), which gives small groups.
            # Within a group, scenarios differ mostly in traffic intensity.
            result["ranking_by_topology"], result["ranking_mean_over_topologies"] = \
                ranking_summary(by_folder)
            result["ranking_by_graph_config"], result["ranking_mean_over_graph_configs"] = \
                ranking_summary(by_graph)
        construction = [row["construct_ms"] for row, _ in entries]
        forwards = [x["forward_ms"] for x in items]
        result["timing"] = dict(construct=timing(construction), forward=timing(forwards),
                                forward_fraction_of_construct_plus_forward=sum(forwards) / (
                                    sum(construction) + sum(forwards)),
                                mean_scenario_forward_fraction=float(np.mean([
                                    f / (c + f) for c, f in zip(construction, forwards)])))
        if all(row.get("parse_ms") is not None for row, _ in entries):  # schema_version >= 2
            parsing = [row["parse_ms"] for row, _ in entries]
            result["timing"]["parse"] = timing(parsing)
            result["timing"]["forward_fraction_of_end_to_end"] = sum(forwards) / (
                sum(parsing) + sum(construction) + sum(forwards))
        result["n_nonpositive_labels"] = sum(row.get("n_nonpositive_labels", 0) for row, _ in entries)
        summary["models"].setdefault(name, {})[order] = result
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/dec1/eval_v0_test.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/dec1/eval_v0_test_summary.json")
    args = parser.parse_args()
    with args.input.open() as source:
        rows = [json.loads(line) for line in source]
    summary = aggregate(rows)
    summary["source"] = str(args.input.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write("\n")
    for name, orders in summary["models"].items():
        for order, result in orders.items():
            print(name, order, "MAPE:", result.get("pooled_mape", "unavailable"))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
