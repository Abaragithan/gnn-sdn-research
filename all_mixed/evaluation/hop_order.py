"""Tensor conversion and an optional hop-order correction for offline audits."""

import tensorflow as tf

RAGGED = {"link_to_path", "queue_to_path", "queue_to_link", "path_to_queue", "path_to_link"}
INTEGER = {"length", "model", "policy", "priority"}


def to_tensors(raw):
    """Match data_generator.input_fn dtypes/ragged ranks; exclude the label."""
    return {
        key: (tf.cast(value, tf.int32) if key in RAGGED else
              tf.convert_to_tensor(value, dtype=tf.int32 if key in INTEGER else tf.float32))
        for key, value in raw.items() if key != "delay"
    }


def with_hop_order(hg, raw):
    """Copy of model_worker.convert_inputs' correction and consistency check.

    Source: mininet_experiment/model_worker.py, convert_inputs (22 Nov 2026).
    Do not import that module: it changes TF threading and imports Mininet settings.
    Incoming edges may follow global node order after nx.relabel_nodes(copy=True).
    Outgoing path edges preserve the hop positions used by the reverse tensors.
    The historical converter and the reference raw dictionary remain untouched.
    """
    result = dict(raw)
    links = [v for v in hg if v.startswith("l_")]
    queues = [v for v in hg if v.startswith("q_")]
    paths = [v for v in hg if v.startswith("p_")]
    link_ids = {v: i for i, v in enumerate(links)}
    queue_ids = {v: i for i, v in enumerate(queues)}
    result["link_to_path"] = tf.ragged.constant([
        [link_ids[v] for _, v in hg.out_edges(p) if v.startswith("l_")]
        for p in paths], dtype=tf.int32, ragged_rank=1)
    result["queue_to_path"] = tf.ragged.constant([
        [queue_ids[v] for _, v in hg.out_edges(p) if v.startswith("q_")]
        for p in paths], dtype=tf.int32, ragged_rank=1)
    result = to_tensors(result)
    for forward, reverse in (("link_to_path", "path_to_link"),
                             ("queue_to_path", "path_to_queue")):
        rows = result[forward].to_list()
        for node, positions in enumerate(result[reverse].to_list()):
            if any(rows[path][hop] != node for path, hop in positions):
                raise ValueError("Inconsistent path connectivity tensors")
    return result


def input_signature(inputs):
    """Keep variable graph dimensions out of TF's tracing cache."""
    signature = {}
    for key, value in inputs.items():
        if key in RAGGED:
            shape = [None, None, 2] if key.startswith("path_to_") else [None, None]
            signature[key] = tf.RaggedTensorSpec(
                shape, tf.int32, ragged_rank=1, row_splits_dtype=value.row_splits.dtype)
        else:
            signature[key] = tf.TensorSpec([None] + value.shape.as_list()[1:], value.dtype)
    return signature
