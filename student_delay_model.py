"""
Student Delay Model: Queue-Link Fusion Bipartite GNN

Key architectural innovations vs. the RouteNet-Fermi teacher:
1. BIPARTITE GRAPH: Fuses Queue+Link into a single "Interface" entity,
   reducing the tripartite (Link ↔ Queue ↔ Path) to bipartite (Interface ↔ Path).
2. 2-STAGE MESSAGE PASSING: Eliminates the Queue→Link stage entirely.
   Teacher: Link+Queue→Path, Path→Queue, Queue→Link (3 stages × 8 iters)
   Student: Interface→Path, Path→Interface (2 stages × 4 iters)
3. REDUCED DIMENSIONS: state_dim=16 (vs teacher's 32).
4. SAME PHYSICS-INFORMED READOUT: queue_delay + trans_delay formula preserved.

Queue-Link Fusion rationale: In real networks, queues exist ON link interfaces—
they are not independent entities. We aggregate per-link queue features
(queue_size, priority, weight) into the link's "interface" representation,
yielding a single entity that captures both link capacity and queuing behavior.
"""

import tensorflow as tf


class StudentRouteNet(tf.keras.Model):
    def __init__(self, state_dim=16, iterations=4):
        super(StudentRouteNet, self).__init__()

        self.max_num_models = 7
        self.num_policies = 4
        self.max_num_queues = 3

        self.iterations = iterations
        self.path_state_dim = state_dim
        self.interface_state_dim = state_dim

        # Same z-score normalization as teacher (dataset statistics are identical)
        self.z_score = {
            'traffic': [1385.4058837890625, 859.8118896484375],
            'packets': [1.4015231132507324, 0.8932565450668335],
            'eq_lambda': [1350.97119140625, 858.316162109375],
            'avg_pkts_lambda': [0.9117304086685181, 0.9723503589630127],
            'exp_max_factor': [6.663637638092041, 4.715115070343018],
            'pkts_lambda_on': [0.9116322994232178, 1.651275396347046],
            'avg_t_off': [1.6649284362792969, 2.356407403945923],
            'avg_t_on': [1.6649284362792969, 2.356407403945923],
            'ar_a': [0.0, 1.0], 'sigma': [0.0, 1.0],
            'capacity': [27611.091796875, 20090.62109375],
            'queue_size': [30259.10546875, 21410.095703125],
        }

        # --- GRU Cells: only 2 (vs teacher's 3) ---
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.interface_update = tf.keras.layers.GRUCell(self.interface_state_dim)

        # Pre-build the RNN wrapper in __init__ — NOT inside call().
        # Creating layers inside @tf.function forces retracing every call.
        self.path_update_rnn = tf.keras.layers.RNN(
            self.path_update, return_sequences=True, return_state=True
        )

        # --- Embedding MLPs: only 2 (vs teacher's 3) ---
        # Path embedding: same 17-dim input as teacher (10 traffic features + 7 model one-hot)
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=10 + self.max_num_models),
            tf.keras.layers.Dense(self.path_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.path_state_dim, activation='relu')
        ])

        # Interface embedding: FUSED link + aggregated queue features
        # Input: load(1) + policy(4) + mean_queue_size(1) + mean_priority(3) + mean_weight(1) + num_queues(1) = 11
        self.interface_input_dim = 1 + self.num_policies + 1 + self.max_num_queues + 1 + 1
        self.interface_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.interface_input_dim),
            tf.keras.layers.Dense(self.interface_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.interface_state_dim, activation='relu')
        ])

        # --- Readout MLP: smaller (16→8→8→1 vs teacher's 32→16→16→1) ---
        self.readout_path = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(self.path_state_dim // 2, activation='relu'),
            tf.keras.layers.Dense(self.path_state_dim // 2, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")

    def _fuse_queue_into_link(self, queue_size, priority, weight, queue_to_link, n_links):
        """
        Aggregate per-queue features into per-link interface features.

        Data layout (from data_generator.py):
          queue_to_link: RaggedTensor shape (n_links, None) — but stored as
                         a list per link of which queues belong to it.
          We use unsorted_segment_mean to scatter queue features → links.

        Args:
            queue_size: (n_queues, 1) normalized queue sizes
            priority:   (n_queues, max_num_queues) one-hot priority vectors
            weight:     (n_queues, 1) scheduling weights
            queue_to_link: RaggedTensor (n_links, None) — each row = queue
                           indices belonging to that link
            n_links:    scalar, total number of links

        Returns:
            mean_queue_size: (n_links, 1)
            mean_priority:   (n_links, max_num_queues)
            mean_weight:     (n_links, 1)
            num_queues:      (n_links, 1)  normalized by max_num_queues=3
        """
        # Build a flat array: for each queue, which link does it belong to?
        # queue_to_link is ragged (n_links, None); invert it to get
        # segment_ids of shape (n_queues,)
        # Each row i of queue_to_link contains the queue indices for link i.
        # We need: for each queue q → link index l.
        # Use queue_to_link.flat_values (queue indices) and row_ids (link indices).
        link_ids = queue_to_link.value_rowids()   # (n_queues,) — which link each queue belongs to
        # queue_to_link.flat_values gives the queue indices in link order
        queue_indices = queue_to_link.flat_values  # (n_queues,)

        # Gather queue features in the order defined by queue_to_link
        qs_ordered = tf.gather(queue_size, queue_indices)   # (n_queues, 1)
        pr_ordered = tf.gather(priority, queue_indices)     # (n_queues, max_num_queues)
        wt_ordered = tf.gather(weight, queue_indices)       # (n_queues, 1)

        # Aggregate: mean over all queues per link
        mean_qs = tf.math.unsorted_segment_mean(qs_ordered, link_ids, n_links)
        mean_pr = tf.math.unsorted_segment_mean(pr_ordered, link_ids, n_links)
        mean_wt = tf.math.unsorted_segment_mean(wt_ordered, link_ids, n_links)

        # Number of queues per link (normalized)
        counts = tf.cast(queue_to_link.row_lengths(), tf.float32)   # (n_links,)
        num_q = tf.expand_dims(counts / 3.0, axis=1)                # (n_links, 1)

        return mean_qs, mean_pr, mean_wt, num_q


    @tf.function
    def call(self, inputs):
        # ============================================================
        # Extract inputs (identical to teacher — same data pipeline)
        # ============================================================
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model']
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        pkts_lambda_on = inputs['pkts_lambda_on']
        avg_t_off = inputs['avg_t_off']
        avg_t_on = inputs['avg_t_on']
        ar_a = inputs['ar_a']
        sigma = inputs['sigma']

        capacity = inputs['capacity']
        policy = tf.one_hot(inputs['policy'], self.num_policies)

        queue_size = inputs['queue_size']
        priority = tf.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']

        # Connectivity tensors
        link_to_path = inputs['link_to_path']     # (n_paths, None) → interface_to_path
        path_to_link = inputs['path_to_link']      # (n_links, None, 2) → path_to_interface
        queue_to_link = inputs['queue_to_link']    # (n_links, None) → used for fusion only

        # ============================================================
        # Compute load (same as teacher)
        # ============================================================
        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity
        pkt_size = traffic / packets

        # ============================================================
        # INNOVATION: Queue-Link Fusion
        # ============================================================
        n_links = tf.shape(capacity)[0]  # number of link/interface nodes
        queue_size_norm = (queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1]
        mean_qs, mean_pr, mean_wt, num_q = self._fuse_queue_into_link(
            queue_size_norm, priority, weight, queue_to_link, n_links
        )

        # ============================================================
        # Initialize embeddings
        # ============================================================
        # Path state: same as teacher (17-dim input → path_state_dim)
        path_state = self.path_embedding(tf.concat([
            (traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
            (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
            tf.one_hot(model, self.max_num_models),
            (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
            (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
            (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
            (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
            (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
            (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
            (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
            (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1],
        ], axis=1))

        # Interface state: FUSED link + queue features (11-dim input → interface_state_dim)
        interface_state = self.interface_embedding(tf.concat([
            load, policy, mean_qs, mean_pr, mean_wt, num_q
        ], axis=1))

        # ============================================================
        # Message Passing: 2 stages × iterations (vs teacher's 3 × 8)
        # Uses pre-built self.path_update_rnn (defined in __init__)
        # ============================================================
        for _ in range(self.iterations):
            # ---- Stage 1: Interface → Path ----
            # Gather the interface state sequence along each path's hops
            interface_gather = tf.gather(interface_state, link_to_path)

            # Run GRU over each path's hop sequence
            previous_path_state = path_state
            path_state_sequence, path_state = self.path_update_rnn(
                interface_gather, initial_state=path_state
            )
            # Prepend the previous state for per-hop readout
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            # ---- Stage 2: Path → Interface ----
            # Aggregate path states into each interface
            path_gather = tf.gather_nd(path_state_sequence, path_to_link)
            path_sum = tf.math.reduce_sum(path_gather, axis=1)
            interface_state, _ = self.interface_update(path_sum, [interface_state])

            # ---- NO Stage 3 (Queue → Link eliminated by fusion) ----

        # ============================================================
        # Readout: same physics-informed formula as teacher
        # ============================================================
        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()

        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

        queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

        return queue_delay + trans_delay
