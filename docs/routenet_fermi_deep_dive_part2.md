# RouteNet-Fermi: Research-Level Deep Dive — Part 2
## Code-Level Architecture Analysis & Data Pipeline

> [!NOTE]
> This is Part 2 of 2. Read Part 1 first for GNN fundamentals, GRU internals, and the tripartite graph concept.

---

## 1. Model Initialization — Code Walkthrough

All three models ([delay_model.py](file:///home/abaragithan/gnn-sdn-research/delay_model.py), [jitter_model.py](file:///home/abaragithan/gnn-sdn-research/jitter_model.py), [loss_model.py](file:///home/abaragithan/gnn-sdn-research/loss_model.py)) share **identical** `__init__` and message passing code. They differ **only** in the readout layer.

### 1.1 Hyperparameters (Lines 11-19)

```python
self.max_num_models = 7      # 7 traffic distribution types
self.num_policies = 4        # WFQ, SP, DRR, FIFO
self.max_num_queues = 3      # Max QoS priority levels per link
self.iterations = 8          # Message passing rounds (T)
self.path_state_dim = 32     # Latent dimension for all entities
self.link_state_dim = 32
self.queue_state_dim = 32
```

### 1.2 Z-Score Normalization (Lines 21-29)

Features are normalized as: $x_{\text{norm}} = \frac{x - \mu}{\sigma}$

These are **precomputed** from the training dataset — not learned. Each entry is `[mean, std_dev]`.

### 1.3 Three GRU Cells (Lines 32-34)

```python
self.path_update  = GRUCell(32)   # Input: 64 (queue[32] || link[32])
self.queue_update = GRUCell(32)   # Input: 32 (aggregated path states)
self.link_update  = GRUCell(32)   # Input: 32 (queue states sequence)
```

### 1.4 Embedding MLPs (Lines 36-52)

Each entity maps raw features → 32-dim hidden state via a 2-layer MLP with ReLU:

| Entity | Input Dim | Architecture | Output |
|:--|:--|:--|:--|
| Path | 17 (10 scalars + 7 one-hot) | Dense(17→32,ReLU) → Dense(32→32,ReLU) | $\mathbf{h}_p^{(0)} \in \mathbb{R}^{32}$ |
| Link | 5 (1 load + 4 one-hot policy) | Dense(5→32,ReLU) → Dense(32→32,ReLU) | $\mathbf{h}_l^{(0)} \in \mathbb{R}^{32}$ |
| Queue | 5 (1 size + 3 one-hot priority + 1 weight) | Dense(5→32,ReLU) → Dense(32→32,ReLU) | $\mathbf{h}_q^{(0)} \in \mathbb{R}^{32}$ |

---

## 2. The `call()` Method — Forward Pass

### 2.1 Feature Extraction (Lines 65-89)

The input dictionary contains raw features + graph structure tensors. Key computed features:

```python
# Link load: sum of traffic from all paths on link / link capacity
path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
load = tf.math.reduce_sum(path_gather_traffic, axis=1) / capacity

# Average packet size (used in delay readout)
pkt_size = traffic / packets
```

> [!IMPORTANT]
> **Link load is computed dynamically**, not stored as a static feature. `path_to_link[:, :, 0]` extracts path indices for each link, gathers their traffic values, sums them, and divides by capacity. This is a domain-knowledge injection — the model knows load = total_demand / capacity.

### 2.2 Initial Embedding (Lines 97-116)

All features are Z-score normalized, then passed through embedding MLPs:

```python
path_state = self.path_embedding(tf.concat([
    (traffic - μ_traffic) / σ_traffic,
    (packets - μ_packets) / σ_packets,
    tf.one_hot(model, 7),          # Traffic model type
    (eq_lambda - μ) / σ, ...       # 8 more distribution params
], axis=1))                        # Shape: [num_paths, 32]

link_state = self.link_embedding(tf.concat([load, policy], axis=1))
queue_state = self.queue_embedding(tf.concat([queue_size_norm, priority, weight], axis=1))
```

---

## 3. Message Passing — The Core Loop (Lines 119-149)

This loop runs 8 times. Each iteration has 3 phases:

### Phase 1: Link+Queue → Path (Lines 120-134)

```python
# GATHER: For each path, collect its queue and link states along the route
queue_gather = tf.gather(queue_state, queue_to_path)  # [num_paths, path_len, 32]
link_gather  = tf.gather(link_state, link_to_path)    # [num_paths, path_len, 32]

# CONCATENATE: Each hop = [queue_state || link_state] → 64-dim input per hop
# Then process the SEQUENCE of hops through a GRU-RNN
path_update_rnn = tf.keras.layers.RNN(self.path_update,
                                       return_sequences=True,  # Need ALL hop outputs
                                       return_state=True)      # Need final state

path_state_sequence, path_state = path_update_rnn(
    tf.concat([queue_gather, link_gather], axis=2),  # [num_paths, path_len, 64]
    initial_state=path_state                          # Previous path state
)

# PREPEND the initial state to the sequence (for indexing in Phase 2)
path_state_sequence = tf.concat([tf.expand_dims(previous_path_state, 1),
                                  path_state_sequence], axis=1)
```

**What happens here conceptually:**

For a path $p$ with 3 hops through queues $[q_0, q_1, q_2]$ and links $[l_0, l_1, l_2]$:

```
Hop 0: GRU input = [h_q0 || h_l0] (64-dim), hidden = h_p^(t-1)
       → output: h_p,0^seq (what path "knows" after hop 0)

Hop 1: GRU input = [h_q1 || h_l1] (64-dim), hidden = h_p,0^seq
       → output: h_p,1^seq (after hop 0 AND hop 1)

Hop 2: GRU input = [h_q2 || h_l2] (64-dim), hidden = h_p,1^seq
       → output: h_p,2^seq (after all hops) = new path_state
```

> [!TIP]
> **Why sequential (RNN) and not parallel aggregation?** Because network delay is **cumulative and order-dependent**. Congestion at hop 1 affects how packets arrive at hop 2, creating cascading delays. The GRU's sequential processing mirrors this physical reality — each hop's output carries information about ALL previous hops.

### Phase 2: Path → Queue (Lines 136-141)

```python
# GATHER: For each queue, get the sequence states of paths at the SPECIFIC HOP
# where they pass through this queue
path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
# path_to_queue[q] = [[path_idx, hop_idx], [path_idx, hop_idx], ...]

# AGGREGATE: Sum all path contributions to this queue
path_sum = tf.math.reduce_sum(path_gather, axis=1)  # [num_queues, 32]

# UPDATE: Queue GRU takes aggregated path info + previous queue state
queue_state, _ = self.queue_update(path_sum, [queue_state])
```

**Critical detail**: `tf.gather_nd` with `path_to_queue` extracts the **sequence state at the exact hop** where path $p$ encounters queue $q$ — NOT the final path state. This means the queue receives information about the path's condition *at that specific point in its journey*, including the cumulative effects of all prior hops.

### Phase 3: Queue → Link (Lines 143-149)

```python
# GATHER: For each link, collect states of its queues
queue_gather = tf.gather(queue_state, queue_to_link)  # [num_links, num_queues, 32]

# UPDATE: Process queues sequentially through GRU-RNN (ordered by priority)
link_gru_rnn = tf.keras.layers.RNN(self.link_update, return_sequences=False)
link_state = link_gru_rnn(queue_gather, initial_state=link_state)
```

The link processes its queues as a **sequence** (typically ordered by priority: high → low). This allows the GRU to learn scheduling behaviors — e.g., in Strict Priority, high-priority queues dominate, and the GRU can learn to suppress the influence of low-priority queue states.

### Complete Message Passing Flow Diagram

```
┌─────────────────── ITERATION t ───────────────────┐
│                                                     │
│  ┌─────────┐   gather    ┌──────────┐              │
│  │Queue    │────────────►│          │              │
│  │States   │             │ GRU-RNN  │──► Path      │
│  └─────────┘   concat    │ (seq)    │   States     │
│  ┌─────────┐────────────►│          │   + Sequence │
│  │Link     │             └──────────┘              │
│  │States   │                   │                    │
│  └─────────┘                   │ gather_nd          │
│       ▲                        ▼ (at specific hop)  │
│       │                  ┌──────────┐              │
│       │    GRU-RNN       │ reduce   │              │
│       │◄────(seq)────────│ _sum     │              │
│       │                  └────┬─────┘              │
│  ┌─────────┐                  │                    │
│  │Queue    │◄────GRU Cell─────┘                    │
│  │States   │   (single step)                       │
│  │(updated)│                                       │
│  └─────────┘                                       │
└─────────────────────────────────────────────────────┘
```

---

## 4. Readout Layers — Where the Three Models Diverge

After 8 iterations, the models extract predictions differently.

### 4.1 Delay Model Readout ([delay_model.py:151-162](file:///home/abaragithan/gnn-sdn-research/delay_model.py#L151-L162))

```python
capacity_gather = tf.gather(capacity, link_to_path)          # Per-hop capacities
input_tensor = path_state_sequence[:, 1:].to_tensor()        # Remove prepended initial state

occupancy_gather = self.readout_path(input_tensor)            # MLP → scalar per hop
occupancy_gather = tf.RaggedTensor.from_tensor(occupancy_gather, lengths=length)

queue_delay = tf.math.reduce_sum(occupancy_gather / capacity_gather, axis=1)
trans_delay = pkt_size * tf.math.reduce_sum(1 / capacity_gather, axis=1)

return queue_delay + trans_delay
```

**Physics-informed decomposition:**

$$\text{Delay}_p = \underbrace{\sum_{i=1}^{N_p} \frac{\text{MLP}(\mathbf{h}_{p,i}^{\text{seq}})}{C_{l_{p,i}}}}_{\text{Queueing delay}} + \underbrace{\frac{\text{traffic}_p}{\text{packets}_p} \cdot \sum_{i=1}^{N_p} \frac{1}{C_{l_{p,i}}}}_{\text{Transmission delay}}$$

> [!IMPORTANT]
> **The transmission delay is NOT learned** — it is computed analytically from known packet sizes and link capacities using the formula $d_{\text{trans}} = \text{pkt\_size} / C$. Only the **queueing delay** is learned by the neural network. This is a powerful inductive bias: the model only needs to learn what it cannot compute analytically.

### 4.2 Jitter Model Readout ([jitter_model.py:151-160](file:///home/abaragithan/gnn-sdn-research/jitter_model.py#L151-L160))

```python
ratio_gather = self.readout_path(input_tensor)    # MLP → scalar ratio per hop
ratio_gather = tf.RaggedTensor.from_tensor(ratio_gather, lengths=length)

jitter = tf.math.reduce_sum(ratio_gather / capacity_gather, axis=1)
```

$$\text{Jitter}_p = \sum_{i=1}^{N_p} \frac{\text{MLP}(\mathbf{h}_{p,i}^{\text{seq}})}{C_{l_{p,i}}}$$

Similar structure to delay but without the analytical transmission component (jitter is purely stochastic).

### 4.3 Loss Model Readout ([loss_model.py:151-153](file:///home/abaragithan/gnn-sdn-research/loss_model.py#L151-L153))

```python
losses = self.readout_path(path_state)    # Uses FINAL path state, not sequence
return losses                              # Sigmoid output ∈ [0, 1]
```

$$\text{Loss}_p = \sigma\left(\text{MLP}(\mathbf{h}_p^{(T)})\right)$$

**Key differences from delay/jitter:**
1. Uses the **final aggregated path state** $\mathbf{h}_p^{(T)}$, not per-hop sequence states
2. **Sigmoid activation** constrains output to $[0, 1]$ (probability)
3. No capacity scaling — loss is modeled as a direct function of path state
4. Readout input shape is `(path_state_dim,)` not `(None, path_state_dim)`

### 4.4 Summary Table

| | Delay | Jitter | Loss |
|:--|:--|:--|:--|
| **Readout input** | Per-hop sequence states $\mathbf{h}_{p,i}^{\text{seq}}$ | Per-hop sequence states $\mathbf{h}_{p,i}^{\text{seq}}$ | Final path state $\mathbf{h}_p^{(T)}$ |
| **MLP output activation** | Linear | Linear | Sigmoid |
| **Post-processing** | ÷ capacity, + transmission delay | ÷ capacity | None |
| **Aggregation** | Sum over hops | Sum over hops | Direct output |
| **Domain knowledge** | Queuing theory ($D=Q/C$) + transmission formula | Capacity scaling | Probability bound |

---

## 5. Data Pipeline — From Simulation to Tensors

### 5.1 The Transformation Chain

```
OMNeT++ Simulation → tar.gz files → DatanetAPI → NetworkX DiGraph
→ network_to_hypergraph() → Tripartite DiGraph → hypergraph_to_input_data()
→ Feature dict + Adjacency tensors → tf.data.Dataset → Model
```

### 5.2 `network_to_hypergraph()` ([data_generator.py:134-218](file:///home/abaragithan/gnn-sdn-research/all_mixed/data_generator.py#L134-L218))

This function converts a physical network (topology G, traffic matrix T, routing matrix R, performance matrix P) into the tripartite graph:

1. **Creates link nodes** (`l_src_dst`) with capacity and policy features
2. **Creates path nodes** (`p_src_dst_flowid`) with all 17 traffic features + ground-truth delay
3. **Creates queue nodes** (`q_src_dst_priority`) with buffer size, priority, weight
4. **Creates edges** between all three entity types based on routing and QoS mapping

### 5.3 ToS-to-Queue Mapping (Lines 198-212)

```python
# Maps Type-of-Service field to QoS queue
q_map = [['0'], ['1'], ['2']]  # Default: ToS 0 → Queue 0, etc.

# A flow is connected to the queue matching its ToS
if str(int(flow_tos)) in q_map[q]:
    D_G.add_edge(path_node, queue_node)  # Bidirectional
    D_G.add_edge(queue_node, path_node)
```

This is how **QoS differentiation** enters the graph — different flows get routed to different priority queues on the same link.

---

## 6. Why This Architecture Works — Design Insights

### 6.1 Inductive Bias Alignment

The architecture mirrors network physics:

| Network Reality | Architectural Choice |
|:--|:--|
| Delay accumulates hop-by-hop | Sequential GRU-RNN processes hops in order |
| Multiple flows share queues | Sum aggregation pools all path states per queue |
| Queue priority affects scheduling | Queues processed as ordered sequence by link GRU |
| Delay = Occupancy / Capacity | Readout explicitly divides by capacity |
| Packet size determines transmission time | Analytical transmission delay computation |

### 6.2 What the GRU Learns vs. What is Hard-Coded

| Learned (Neural Network) | Hard-Coded (Domain Knowledge) |
|:--|:--|
| Queue occupancy estimation | $D_{\text{trans}} = \text{pkt\_size} / C$ |
| Non-linear congestion effects | $D_{\text{queue}} = \text{occupancy} / C$ |
| Impact of scheduling policies | Link load = Σ traffic / capacity |
| Traffic burstiness effects | Z-score normalization constants |
| Cross-flow interference patterns | Graph structure (routing, QoS mapping) |

### 6.3 Generalization Mechanism

The model generalizes to unseen topologies because:
1. **Parameter sharing**: The same GRU weights process ALL paths/links/queues — no topology-specific parameters
2. **Local computation**: Each message passing step is a local operation (node + neighbors) — independent of global graph size
3. **Iterative refinement**: 8 iterations allow information to propagate across the full graph regardless of topology
