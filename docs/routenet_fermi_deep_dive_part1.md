# RouteNet-Fermi: Research-Level Deep Dive — Part 1
## From Graph Theory Fundamentals to the GNN Architecture

> [!NOTE]
> This is Part 1 of 2. This part covers foundational concepts (graph theory, GNNs, GRUs, heterogeneous graphs) and builds up to the RouteNet-Fermi representation. Part 2 covers the line-by-line code analysis, data pipeline, readout mechanisms, and research directions.

---

## Table of Contents
1. [Prerequisites: Graph Theory Essentials](#1-prerequisites-graph-theory-essentials)
2. [Why Traditional ML Fails for Computer Networks](#2-why-traditional-ml-fails-for-computer-networks)
3. [Graph Neural Networks from First Principles](#3-graph-neural-networks-from-first-principles)
4. [The GRU Cell: The Engine Inside RouteNet-Fermi](#4-the-gru-cell-the-engine-inside-routenet-fermi)
5. [Homogeneous vs. Heterogeneous GNNs](#5-homogeneous-vs-heterogeneous-gnns)
6. [The RouteNet Lineage](#6-the-routenet-lineage)
7. [The Tripartite Graph Representation](#7-the-tripartite-graph-representation)
8. [Reading List & References](#8-reading-list--references)

---

## 1. Prerequisites: Graph Theory Essentials

### 1.1 What is a Graph?

A **graph** $G = (V, E)$ consists of:
- $V$ = a finite set of **vertices** (nodes)
- $E \subseteq V \times V$ = a set of **edges** connecting pairs of nodes

In a **directed graph** (digraph), edges have direction: $(u, v) \neq (v, u)$. Computer networks are naturally directed graphs — a fiber optic cable from router A to router B is a distinct physical entity from the cable going B to A.

### 1.2 Key Terminology

| Term | Definition | Network Analogy |
|:--|:--|:--|
| **Adjacency** | Node $u$ is adjacent to $v$ if $(u,v) \in E$ | Router A has a direct cable to Router B |
| **Neighborhood** $\mathcal{N}(v)$ | Set of all nodes adjacent to $v$ | All routers directly connected to Router B |
| **Degree** $d(v)$ | Number of edges incident to $v$ | Number of interfaces on a router |
| **Path** | Sequence of nodes $(v_1, v_2, \ldots, v_k)$ where each $(v_i, v_{i+1}) \in E$ | A route from source to destination |
| **Adjacency Matrix** $\mathbf{A}$ | $n \times n$ matrix where $A_{ij} = 1$ if $(i,j) \in E$ | The connectivity matrix of the network |
| **Feature Matrix** $\mathbf{X}$ | $n \times d$ matrix where row $i$ is the feature vector of node $i$ | Each router's attributes (capacity, policy, etc.) |

### 1.3 Adjacency Matrix Example

Consider a 4-node network:
```
    [0]--->[1]
     |      |
     v      v
    [2]--->[3]
```

$$\mathbf{A} = \begin{bmatrix} 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

The neighborhood of node 0 is $\mathcal{N}(0) = \{1, 2\}$ (outgoing edges).

### 1.4 Feature Vectors on Graphs

Each node $v$ carries a **feature vector** $\mathbf{x}_v \in \mathbb{R}^d$. For example, a router might have:

$$\mathbf{x}_v = [\underbrace{\text{capacity}}_{1}, \underbrace{\text{load}}_{1}, \underbrace{\text{policy (one-hot)}}_{4}] \in \mathbb{R}^6$$

Edges can also carry features $\mathbf{e}_{uv}$ (e.g., link bandwidth, propagation delay).

---

## 2. Why Traditional ML Fails for Computer Networks

### 2.1 The Core Problem

A computer network has **variable-size, non-Euclidean structure**. The number of nodes, links, and paths changes across different topologies. Routes change when a link fails. Traffic patterns shift over time.

### 2.2 MLPs (Multi-Layer Perceptrons)

MLPs require **fixed-length input vectors**. To feed a network topology into an MLP, you would need to:
- Flatten the adjacency matrix into a vector → $O(n^2)$ dimensions
- Pad smaller topologies to a maximum size
- **Critical flaw**: This destroys structural information. The MLP cannot distinguish between "node 3 is connected to node 7" and "node 3 has high traffic" — both are just numbers in a flat vector.
- **Generalization failure**: A model trained on 14-node topologies cannot process a 50-node topology without retraining.

### 2.3 CNNs (Convolutional Neural Networks)

CNNs assume **grid-structured (Euclidean) data** — images have a fixed 2D spatial layout where "nearby pixels are related." Networks violate this:
- There is no canonical spatial ordering of routers
- The number of neighbors per node varies (irregular structure)
- Distance in a network is measured in hops or latency, not Euclidean distance

### 2.4 RNNs (Recurrent Neural Networks)

RNNs process **sequential data**. While you *can* represent a path as a sequence, a network is fundamentally **not** a sequence — it is a graph with complex dependencies:
- Multiple paths share links and queues simultaneously
- A congestion event on one link affects all paths traversing it
- Pure sequence models cannot capture this shared-resource interaction

### 2.5 What We Need

We need an architecture that:
1. Operates directly on **graph-structured** data of arbitrary size
2. Is **permutation invariant** — the output should not depend on how we number the nodes
3. Captures **relational dependencies** — information propagates through the graph structure
4. **Generalizes** to unseen topologies, routing schemes, and traffic patterns

This is precisely what **Graph Neural Networks (GNNs)** provide.

---

## 3. Graph Neural Networks from First Principles

### 3.1 The Core Intuition

The fundamental idea of a GNN is simple: **a node's representation should be informed by its neighborhood**.

Instead of processing nodes in isolation, a GNN allows each node to "look at" its neighbors, gather information from them, and update its own representation. After several rounds of this, each node's representation encodes information about its local (and progressively global) graph neighborhood.

### 3.2 The Message Passing Neural Network (MPNN) Framework

Gilmer et al. (2017) unified most GNN architectures under the **MPNN framework**. It defines two phases:

#### Phase 1: Message Passing (repeated for $T$ iterations)

At each iteration $t = 1, 2, \ldots, T$, every node $v$ performs three operations:

**Step 1 — Message Computation:**
Each neighbor $u \in \mathcal{N}(v)$ computes a message to send to $v$:

$$m_{u \to v}^{(t)} = \text{MSG}\left(\mathbf{h}_u^{(t-1)}, \mathbf{h}_v^{(t-1)}, \mathbf{e}_{uv}\right)$$

where $\mathbf{h}_u^{(t-1)}$ is node $u$'s hidden state from the previous iteration, and $\mathbf{e}_{uv}$ is the edge feature.

**Step 2 — Aggregation:**
All incoming messages are aggregated using a **permutation-invariant function** (sum, mean, or max):

$$\mathbf{M}_v^{(t)} = \bigoplus_{u \in \mathcal{N}(v)} m_{u \to v}^{(t)}$$

> [!IMPORTANT]
> **Why permutation invariance?** Graphs have no canonical node ordering. If we relabel node 3 as node 7, the aggregated message must remain identical. Sum, mean, and max satisfy this; concatenation does not (it depends on order).

**Step 3 — Update:**
The node updates its hidden state using the aggregated messages and its previous state:

$$\mathbf{h}_v^{(t)} = \text{UPD}\left(\mathbf{h}_v^{(t-1)}, \mathbf{M}_v^{(t)}\right)$$

The UPDATE function is a learnable neural network — in RouteNet-Fermi, this is a **GRU cell** (explained in Section 4).

#### Phase 2: Readout

After $T$ iterations, the final node states $\{\mathbf{h}_v^{(T)}\}$ are used to make predictions:

$$\hat{y} = \text{READOUT}\left(\{\mathbf{h}_v^{(T)} \mid v \in V\}\right)$$

The readout function depends on the task:
- **Node-level prediction**: Use individual $\mathbf{h}_v^{(T)}$ directly (e.g., predict delay per path)
- **Graph-level prediction**: Pool all node states (e.g., predict overall network health)
- **Edge-level prediction**: Combine endpoint states (e.g., predict link utilization)

### 3.3 Receptive Field and the Number of Iterations

After $t$ message passing iterations, each node has information about nodes up to $t$ **hops** away. This is called the **receptive field**.

```
t=0: Node knows only its own features
t=1: Node knows its immediate neighbors (1-hop)
t=2: Node knows neighbors-of-neighbors (2-hop)
...
t=T: Node knows its T-hop neighborhood
```

In RouteNet-Fermi, $T = 8$ iterations. This means information can propagate up to 8 hops through the graph in each direction, allowing the model to capture long-range dependencies in the network.

> [!TIP]
> **Why exactly 8?** This is a hyperparameter. Too few iterations → nodes don't have enough context. Too many → over-smoothing (all node representations converge to the same vector). 8 iterations were empirically found to balance these tradeoffs for network topologies up to 300 nodes.

### 3.4 Concrete Example: One Message Passing Iteration

Consider a simple 3-node path: $A \to B \to C$

**Initial states** (from embedding):
- $\mathbf{h}_A^{(0)} = [0.2, 0.8, 0.1, ...]$ (32-dim vector)
- $\mathbf{h}_B^{(0)} = [0.5, 0.3, 0.9, ...]$
- $\mathbf{h}_C^{(0)} = [0.1, 0.6, 0.4, ...]$

**Iteration 1:**
1. Node B receives messages from $\mathcal{N}(B) = \{A, C\}$:
   - $m_{A \to B} = \text{MSG}(\mathbf{h}_A^{(0)}, \mathbf{h}_B^{(0)}, \mathbf{e}_{AB})$
   - $m_{C \to B} = \text{MSG}(\mathbf{h}_C^{(0)}, \mathbf{h}_B^{(0)}, \mathbf{e}_{CB})$
2. Aggregate: $\mathbf{M}_B = m_{A \to B} + m_{C \to B}$
3. Update: $\mathbf{h}_B^{(1)} = \text{GRU}(\mathbf{M}_B, \mathbf{h}_B^{(0)})$

After this, $\mathbf{h}_B^{(1)}$ encodes not just B's own features, but also information about A and C.

### 3.5 Specific GNN Variants (Context)

Different GNN architectures differ in how they define MSG, AGG, and UPD:

| Architecture | MSG Function | AGG | UPD | Key Paper |
|:--|:--|:--|:--|:--|
| **GCN** (Kipf 2017) | $\frac{1}{\sqrt{d_u d_v}} \mathbf{W} \mathbf{h}_u$ | Sum | ReLU | Semi-Supervised Classification |
| **GraphSAGE** (Hamilton 2017) | $\mathbf{W} \mathbf{h}_u$ | Mean/LSTM/Pool | ReLU(concat) | Inductive Representation Learning |
| **GAT** (Veličković 2018) | $\alpha_{uv} \mathbf{W} \mathbf{h}_u$ (attention) | Sum | LeakyReLU | Graph Attention Networks |
| **GGNN** (Li 2016) | $\mathbf{W} \mathbf{h}_u$ | Sum | **GRU** | Gated Graph Sequence NN |
| **RouteNet-Fermi** | Gather (index select) | Sum / RNN-sequence | **GRU** | Network Modeling with GNNs |

RouteNet-Fermi is closest to the **GGNN** family — it uses GRU cells for updates — but extends it significantly with a heterogeneous, multi-entity graph structure.

---

## 4. The GRU Cell: The Engine Inside RouteNet-Fermi

RouteNet-Fermi uses **GRU (Gated Recurrent Unit)** cells as its core update mechanism. Understanding GRUs at the gate level is essential for understanding the model.

### 4.1 Why GRU over Simple Neural Networks?

When message passing runs for $T=8$ iterations, information must flow through 8 sequential update steps. Simple neural networks (MLP + ReLU) suffer from:
- **Vanishing gradients**: Gradients shrink exponentially through deep chains of multiplications
- **Exploding gradients**: Gradients grow exponentially, causing numerical instability
- **No selective memory**: Every update overwrites the previous state entirely

GRUs solve these problems with **gating mechanisms** that learn *what to remember and what to forget*.

### 4.2 GRU Gate Equations (Full Derivation)

Given:
- $\mathbf{x}_t$ = input at time $t$ (the aggregated message $\mathbf{M}_v^{(t)}$)
- $\mathbf{h}_{t-1}$ = previous hidden state ($\mathbf{h}_v^{(t-1)}$)

The GRU computes:

**Reset Gate** — decides how much of the previous state to "forget":
$$\mathbf{r}_t = \sigma\left(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1} + \mathbf{b}_r\right)$$

**Update Gate** — decides how much of the new candidate to use vs. keeping the old state:
$$\mathbf{z}_t = \sigma\left(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1} + \mathbf{b}_z\right)$$

**Candidate State** — new information computed from input and (optionally reset) previous state:
$$\tilde{\mathbf{h}}_t = \tanh\left(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h\right)$$

**Output State** — interpolation between old and new:
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

Where:
- $\sigma$ = sigmoid function, outputs values in $[0, 1]$
- $\odot$ = element-wise (Hadamard) product
- $\mathbf{W}_r, \mathbf{W}_z, \mathbf{W}_h$ = input weight matrices
- $\mathbf{U}_r, \mathbf{U}_z, \mathbf{U}_h$ = recurrent weight matrices

### 4.3 Intuitive Interpretation in RouteNet-Fermi's Context

Consider a **queue node** receiving aggregated path messages $\mathbf{M}_q^{(t)}$:

- **Reset gate close to 0**: "Ignore my previous queue state — the traffic pattern has changed significantly, recompute from scratch."
- **Reset gate close to 1**: "My previous state is still relevant — combine it with the new path information."
- **Update gate close to 0**: "The incoming path messages don't change my understanding — keep my current state."
- **Update gate close to 1**: "The incoming path messages are very informative — update my state significantly."

This allows the model to learn nuanced behaviors like: "If this queue is already congested (previous state indicates high occupancy), and new paths are being routed through it (incoming messages), maintain the congestion memory but incorporate the new load information."

### 4.4 Parameter Count in RouteNet-Fermi's GRU Cells

For the path update GRU with `path_state_dim = 32`:
- Input dimension: 64 (concatenation of queue state [32] + link state [32])
- Hidden dimension: 32
- Parameters per gate: $W: 64 \times 32 + U: 32 \times 32 + b: 32 = 2048 + 1024 + 32 = 3104$
- Total GRU parameters (3 gates): $3 \times 3104 = 9312$

For queue update GRU:
- Input: 32, Hidden: 32 → $3 \times (32 \times 32 + 32 \times 32 + 32) = 6240$ parameters

For link update GRU:
- Input: 32, Hidden: 32 → $3 \times (32 \times 32 + 32 \times 32 + 32) = 6240$ parameters

---

## 5. Homogeneous vs. Heterogeneous GNNs

### 5.1 Homogeneous GNNs

In a **homogeneous GNN**, all nodes are of the same type and all edges are of the same type. The graph $G = (V, E)$ has a single node feature space and a single message passing scheme.

**Example**: A social network where every node is a "person" and every edge is a "friendship."

**For networking**: Early GNN approaches modeled routers as nodes and cables as edges. This is insufficient because:
- A traffic flow (path) spanning 5 hops cannot be represented as a single node
- Queuing behavior at each hop requires per-hop state
- Different scheduling policies (FIFO, SP, WFQ, DRR) affect different queues differently

### 5.2 Heterogeneous GNNs

A **heterogeneous GNN** operates on a graph with **multiple node types** and/or **multiple edge types**. Each type has its own feature space, embedding function, and update rule.

Formally, a heterogeneous graph $G = (V, E, \tau, \phi)$ includes:
- $\tau: V \to \mathcal{T}_V$ = node type mapping (e.g., Path, Link, Queue)
- $\phi: E \to \mathcal{T}_E$ = edge type mapping (e.g., path-traverses-link, queue-belongs-to-link)

### 5.3 Why RouteNet-Fermi Requires Heterogeneity

Computer networks have fundamentally different entity types:

| Entity | What it Represents | Key Features | Behavior |
|:--|:--|:--|:--|
| **Path** | End-to-end traffic flow (src → dst) | Traffic rate, packet rate, traffic model, burstiness params | Experiences delay/jitter/loss across its route |
| **Link** | Physical transmission line | Capacity, scheduling policy | Shared by multiple paths; multiplexing point |
| **Queue** | Buffer at egress port | Buffer size, priority, weight | Where packets wait; determines per-hop delay |

These entities interact through well-defined relationships that a homogeneous model cannot capture.

---

## 6. The RouteNet Lineage

### 6.1 RouteNet (Rusek et al., 2020)

**Paper**: *"RouteNet: Leveraging Graph Neural Networks for network modeling and optimization in SDN"* — IEEE JSAC 2020

**Key innovation**: First to apply GNNs to SDN network modeling using a **bipartite graph** with two entity types:
- **Paths**: End-to-end traffic flows
- **Links**: Physical connections

**Limitation**: No explicit queue modeling → cannot handle QoS policies (Strict Priority, WFQ, DRR) or differentiated queue buffer sizes.

### 6.2 RouteNet-Erlang (2022)

**Key innovation**: Added support for **multi-queue scheduling policies** and more complex traffic models. Named after the Erlang distribution used in teletraffic theory.

### 6.3 RouteNet-Fermi (Ferriol-Galmés et al., 2022)

**Paper**: *"RouteNet-Fermi: Network Modeling with Graph Neural Networks"* — IEEE/ACM Transactions on Networking. [arXiv:2212.12070](https://arxiv.org/abs/2212.12070)

**Key innovations over predecessors**:
1. **Tripartite graph**: Three entity types (Paths, Links, **Queues**)
2. **Queue-aware message passing**: Explicitly models buffer sizes, priorities, and scheduling weights
3. **Non-Markovian traffic**: Supports 7+ traffic distribution models (Poisson, ON-OFF, autocorrelated, etc.)
4. **Physics-informed readout**: Delay prediction incorporates queuing theory formula $D = Q/C$
5. **Scalability**: Validated on topologies up to 300 nodes (trained on much smaller ones)

---

## 7. The Tripartite Graph Representation

### 7.1 How a Physical Network Becomes a Tripartite Graph

This is the most critical conceptual step. The [network_to_hypergraph](file:///home/abaragithan/gnn-sdn-research/all_mixed/data_generator.py#L134-L218) function performs this transformation.

**Physical Network Input:**
```
Router 0 ──(10Gbps)──> Router 1 ──(10Gbps)──> Router 2
   │                                              ▲
   └────────────(5Gbps)────────────────────────────┘
```

With traffic flows:
- Flow A: Router 0 → Router 2 via [0, 1, 2] (3 hops, 2 links)
- Flow B: Router 0 → Router 2 via [0, 2] (2 hops, 1 link, direct path)

**Tripartite Representation:**

```
PATHS (traffic flows)           QUEUES (buffers)          LINKS (cables)
┌──────────┐                  ┌──────────────┐          ┌──────────┐
│ Path A   │◄────────────────►│ Queue 0→1, q0│─────────►│ Link 0→1 │
│ (0→1→2)  │◄────────────────►│ Queue 1→2, q0│─────────►│ Link 1→2 │
└──────────┘                  └──────────────┘          └──────────┘
┌──────────┐                  ┌──────────────┐          ┌──────────┐
│ Path B   │◄────────────────►│ Queue 0→2, q0│─────────►│ Link 0→2 │
│ (0→2)    │                  └──────────────┘          └──────────┘
└──────────┘
```

### 7.2 The Five Adjacency Tensors

The tripartite graph is encoded through **five adjacency/index tensors** that define all relationships:

```python
# From data_generator.py — these are the graph structure tensors
queue_to_path   # shape: [num_paths, variable] — which queues each path traverses
link_to_path    # shape: [num_paths, variable] — which links each path traverses  
path_to_queue   # shape: [num_queues, variable, 2] — which paths pass through each queue (with hop index)
path_to_link    # shape: [num_links, variable, 2] — which paths pass through each link (with hop index)
queue_to_link   # shape: [num_links, variable] — which queues belong to each link
```

These are implemented as `tf.RaggedTensor` because each path has a different length, each link has a different number of queues, and each queue serves a different number of paths.

> [!IMPORTANT]
> **The `path_to_queue` tensor has shape `[num_queues, variable, 2]`** where the last dimension `[path_index, hop_position]` is critical. It tells the model: "For queue $q$, path $p$ passes through me at hop position $i$." This allows the model to extract the **sequence state** at the correct hop position — not the final path state, but the intermediate state at the exact point where the path interacts with this queue.

### 7.3 Why This Representation Matters

The tripartite graph captures three crucial network phenomena that simpler representations miss:

1. **Resource sharing**: Multiple paths traverse the same queue → queue congestion depends on ALL paths passing through it, not just one.

2. **QoS differentiation**: A single link has multiple queues with different priorities and buffer sizes → high-priority traffic experiences less delay than low-priority traffic on the same link.

3. **Sequential dependency**: Delay on a path accumulates hop-by-hop → the GRU-RNN processes each hop sequentially, capturing how congestion at hop 1 affects the packet's experience at hop 2.

---

## 8. Reading List & References

### Foundational (Read These First)

| Priority | Paper | Why Read It |
|:--|:--|:--|
| ★★★ | Gilmer et al. (2017) — *"Neural Message Passing for Quantum Chemistry"* [arXiv:1704.01212](https://arxiv.org/abs/1704.01212) | Defines the MPNN framework that RouteNet-Fermi implements |
| ★★★ | Cho et al. (2014) — *"Learning Phrase Representations using RNN Encoder-Decoder"* [arXiv:1406.1078](https://arxiv.org/abs/1406.1078) | Original GRU paper — understand the gates |
| ★★★ | Ferriol-Galmés et al. (2022) — *"RouteNet-Fermi: Network Modeling with GNNs"* [arXiv:2212.12070](https://arxiv.org/abs/2212.12070) | **THE** paper for your research |

### GNN Foundations

| Priority | Paper | Why Read It |
|:--|:--|:--|
| ★★☆ | Kipf & Welling (2017) — *"Semi-Supervised Classification with GCNs"* [arXiv:1609.02907](https://arxiv.org/abs/1609.02907) | Most cited GNN paper; spectral foundations |
| ★★☆ | Hamilton et al. (2017) — *"Inductive Representation Learning on Large Graphs (GraphSAGE)"* [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) | Sampling + aggregation; inductive learning |
| ★★☆ | Li et al. (2016) — *"Gated Graph Sequence Neural Networks"* [arXiv:1511.05493](https://arxiv.org/abs/1511.05493) | GRU-based GNN — direct ancestor of RouteNet |
| ★☆☆ | Veličković et al. (2018) — *"Graph Attention Networks"* [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) | Attention-based alternative to understand design choices |

### RouteNet Lineage

| Priority | Paper | Why Read It |
|:--|:--|:--|
| ★★★ | Rusek et al. (2020) — *"RouteNet: Leveraging GNNs for SDN"* [arXiv:1901.08113](https://arxiv.org/abs/1901.08113) | Original RouteNet — bipartite paths+links |
| ★★☆ | *"RouteNet-Erlang"* | Multi-queue extension; precursor to Fermi |
| ★☆☆ | Suárez-Varela et al. (2019) — *"Challenging the Generalization Capabilities of GNNs for Network Modeling"* | Stress-tests RouteNet's generalization |

### Network Theory Background

| Priority | Resource | Why Read It |
|:--|:--|:--|
| ★★☆ | Kleinrock (1975) — *"Queueing Systems, Vol. 1"* | Classical queuing theory ($D = Q/C$) used in the delay readout |
| ★☆☆ | Cisco QoS documentation | Understanding SP, WFQ, DRR scheduling policies |

### Surveys & Tutorials

| Priority | Resource | Why Read It |
|:--|:--|:--|
| ★★☆ | Wu et al. (2020) — *"A Comprehensive Survey on GNNs"* [arXiv:1901.00596](https://arxiv.org/abs/1901.00596) | Broad GNN landscape overview |
| ★★☆ | Hamilton (2020) — *"Graph Representation Learning"* (Book) | Free textbook, excellent depth |
| ★☆☆ | Stanford CS224W lecture notes | Excellent video lectures on GNNs by Jure Leskovec |
