# From Raw Simulation to Embedding: The Complete Visual Journey

> [!NOTE]
> This document traces **one path entity** through the entire RouteNet-Fermi pipeline — from the raw numbers produced by a network simulator to the 32-dimensional embedding vector $\mathbf{h}_p^{(0)}$. Every stage is explained with concrete numbers and a real-life network analogy.

---

## 1. What is "Simulation Data" Actually?

### The Real-Life Scenario

Imagine you are an engineer at an ISP (Internet Service Provider) like Airtel or Jio. You have:

- **A physical network**: 14 routers connected by fiber optic cables across Tamil Nadu
- **Traffic demands**: Thousands of users streaming Netflix, making video calls, browsing websites
- **A question**: "If I route User-A's video call from Chennai to Madurai via Trichy, what delay will they experience?"

You **cannot** test every possible scenario on your live network — that would disrupt real users. Instead, you use a **network simulator** (like OMNeT++/INET) that:

1. Takes your network topology as input (which routers, which cables, what bandwidth)
2. Takes traffic demands as input (who sends how much data where)
3. Takes routing configuration as input (which path each flow takes)
4. **Runs a virtual experiment** for simulated time (e.g., 60 seconds)
5. **Measures performance** — average delay, jitter, packet loss for every flow

### What the Simulator Produces (Raw Data)

For **one simulation run** on a 14-node network, the simulator outputs four matrices:

```
┌─────────────────────────────────────────────────────────┐
│                    SIMULATION OUTPUT                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  G = Topology Graph                                     │
│  ├── 14 nodes (routers/switches)                        │
│  ├── ~40 directed edges (fiber links)                   │
│  ├── Per-node: schedulingPolicy, bufferSizes,           │
│  │            schedulingWeights, levelsQoS               │
│  └── Per-edge: bandwidth (e.g., 10000 bits/s)           │
│                                                         │
│  T = Traffic Matrix [14 × 14]                           │
│  ├── T[src][dst] = list of Flows                        │
│  ├── Per-flow: AvgBw, PktsGen, ToS, TimeDist,           │
│  │            TimeDistParams (burstiness etc.)           │
│  └── Example: T[0][2].Flows[0].AvgBw = 2500 bits/s     │
│                                                         │
│  R = Routing Matrix [14 × 14]                           │
│  ├── R[src][dst] = [hop0, hop1, ..., hopN]              │
│  └── Example: R[0][2] = [0, 1, 2]  (via router 1)      │
│                                                         │
│  P = Performance Matrix [14 × 14]                       │
│  ├── P[src][dst] = measured metrics                     │
│  └── Per-flow: AvgDelay, Jitter, PktLoss (GROUND TRUTH) │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **The simulation is the "experiment."** The GNN is trying to **learn to predict** the performance matrix P, given only G, T, and R — without running the expensive simulation. Think of it as: "Can an AI learn physics well enough to predict the outcome of an experiment without actually running it?"

---

## 2. From Raw Simulation → Raw Feature Vector (One Path)

Let's trace a **single traffic flow**: Router 0 → Router 2, via route [0, 1, 2].

### Step 2a: The Simulator Recorded These Raw Numbers

From the code in [data_generator.py](file:///home/abaragithan/gnn-sdn-research/all_mixed/data_generator.py#L134-L218), the `network_to_hypergraph` function extracts:

```
Flow: Router 0 → Router 2, Flow ID 0
Route: [0, 1, 2] (2 hops: link 0→1, then link 1→2)

Raw values extracted from T[0][2]['Flows'][0]:
┌────────────────────┬───────────┬──────────────────────────────────────┐
│ Feature Name       │ Raw Value │ What It Means (Real Life)            │
├────────────────────┼───────────┼──────────────────────────────────────┤
│ traffic (AvgBw)    │ 2500.0    │ Average bandwidth: 2500 bits/sec     │
│ packets (PktsGen)  │ 2.5       │ Generating 2.5 packets/sec           │
│ model (TimeDist)   │ 3         │ Traffic model index (e.g., ON-OFF)   │
│ eq_lambda          │ 2400.0    │ Equivalent Poisson arrival rate      │
│ avg_pkts_lambda    │ 1.2       │ Avg packets per burst                │
│ exp_max_factor     │ 10.0      │ Max burst multiplier                 │
│ pkts_lambda_on     │ 1.8       │ Packet rate during "ON" period       │
│ avg_t_off          │ 2.0       │ Avg time the source is silent (sec)  │
│ avg_t_on           │ 2.0       │ Avg time the source is active (sec)  │
│ ar_a               │ 0.0       │ Autoregressive coefficient (unused)  │
│ sigma              │ 0.0       │ Std dev parameter (unused for model) │
└────────────────────┴───────────┴──────────────────────────────────────┘

Ground truth (from P matrix):
│ delay (AvgDelay)   │ 0.00342   │ Measured: 3.42 ms average delay      │
└────────────────────┴───────────┴──────────────────────────────────────┘
```

### Real-Life Analogy

This is like a hospital recording a patient's vitals:

| Network Feature | Hospital Analogy |
|:--|:--|
| `traffic` = 2500 bits/s | Heart rate = 72 bpm |
| `packets` = 2.5 pkt/s | Blood pressure = 120/80 |
| `model` = 3 (ON-OFF) | Condition = "Intermittent fever" |
| `eq_lambda` = 2400 | Temperature = 38.5°C |
| `delay` = 0.00342 s (**label**) | Diagnosis = "Mild infection" (**what we predict**) |

The raw measurements are just numbers from instruments. They need **processing** before a doctor (or neural network) can use them.

---

## 3. Why Can't We Use Raw Numbers Directly?

### The Scale Problem

Look at the raw feature values:

```
traffic    = 2500.0      ← Huge number
packets    = 2.5         ← Small number  
eq_lambda  = 2400.0      ← Huge number
ar_a       = 0.0         ← Zero
```

If you feed these directly into a neural network, the large values **dominate** the computation. The network's weight updates are proportional to input magnitudes — so `traffic` (2500) would have **1000× more influence** than `packets` (2.5), even though both are equally important.

### The Solution: Z-Score Normalization

Z-score normalization transforms every feature to have **mean ≈ 0** and **standard deviation ≈ 1**:

$$x_{\text{normalized}} = \frac{x - \mu}{\sigma}$$

The mean ($\mu$) and std ($\sigma$) are **pre-computed from the training dataset** and hardcoded in the model. From [delay_model.py](file:///home/abaragithan/gnn-sdn-research/delay_model.py#L21-L29):

```python
self.z_score = {
    'traffic':   [1385.41, 859.81],   # [mean, std]
    'packets':   [1.40, 0.89],
    'eq_lambda': [1350.97, 858.32],
    ...
}
```

### Applying Normalization to Our Example

```
                    Raw Value    Mean (μ)    Std (σ)    Normalized
                    ─────────    ────────    ───────    ──────────
traffic:            2500.0    -  1385.41  /  859.81  =  +1.296
packets:               2.5   -     1.40  /    0.89  =  +1.236
eq_lambda:          2400.0    -  1350.97  /  858.32  =  +1.222
avg_pkts_lambda:       1.2   -     0.91  /    0.97  =  +0.299
exp_max_factor:       10.0   -     6.66  /    4.72  =  +0.708
pkts_lambda_on:        1.8   -     0.91  /    1.65  =  +0.539
avg_t_off:             2.0   -     1.66  /    2.36  =  +0.144
avg_t_on:              2.0   -     1.66  /    2.36  =  +0.144
ar_a:                  0.0   -     0.00  /    1.00  =   0.000
sigma:                 0.0   -     0.00  /    1.00  =   0.000
```

Now all values are in the **same small range** (~-3 to +3). The neural network can treat each feature with equal importance.

---

## 4. Building the Raw Feature Vector $\mathbf{x}_p$

The model concatenates **all normalized scalars + the one-hot model vector** into a single vector. From [delay_model.py lines 97-108](file:///home/abaragithan/gnn-sdn-research/delay_model.py#L97-L108):

```
One-hot encoding of model=3 (with max_num_models=7):
model → [0, 0, 0, 1, 0, 0, 0]    (position 3 is "hot")

Final concatenated vector x_p (17 dimensions):
┌─────┬─────┬───┬───┬───┬───┬───┬───┬───┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│1.296│1.236│ 0 │ 0 │ 0 │ 1 │ 0 │ 0 │ 0 │1.222│0.299│0.708│0.539│0.144│0.144│0.000│0.000│
└──┬──┴──┬──┴───┴───┴───┴─┬─┴───┴───┴───┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
   │     │                │                │     │     │     │     │     │     │
traffic packets   model (one-hot)      eq_λ  avg   exp  pkts  t_off t_on  ar_a sigma
                                             pkts  max  λ_on
                                              λ   fact
```

This 17-dimensional vector $\mathbf{x}_p \in \mathbb{R}^{17}$ is the **raw feature vector**. It contains everything the model knows about this traffic flow, but it's **not yet an embedding**.

---

## 5. The Embedding: What, Why, and How

### What IS an Embedding?

An embedding is a **learned, dense, fixed-size representation** that captures the "meaning" of the input in a way that's useful for the downstream task.

Think of it this way:

```
Raw Feature Vector x_p (17-dim)          Embedding h_p^(0) (32-dim)
┌──────────────────────────┐             ┌────────────────────────────────┐
│ Human-readable numbers   │    MLP      │ "Learned language" of the GNN  │
│ (traffic, packets, etc.) │ ────────►   │ (abstract representation)      │
│ Directly interpretable   │  2-layer    │ NOT directly interpretable     │
│ Fixed meaning per dim    │  neural     │ Each dim = learned combination │
│ 17 dimensions            │  network    │ 32 dimensions                  │
└──────────────────────────┘             └────────────────────────────────┘
```

### Why Not Just Use the 17-dim Raw Vector?

Three critical reasons:

**1. Dimensionality mismatch**: The GRU cells expect 32-dimensional inputs/states. All three entity types (paths, links, queues) must speak the **same-sized language** (32-dim) so they can exchange messages during message passing.

**2. Non-linear feature interactions**: Raw features are independent numbers. But in reality, `traffic=2500` combined with `model=ON-OFF` and `avg_t_on=2.0` together create a **bursty traffic pattern** that's qualitatively different from `traffic=2500` with `model=Poisson`. A linear layer can't capture this. The MLP with ReLU activations learns these non-linear combinations.

**3. Shared representation space**: After embedding, paths, links, and queues all live in $\mathbb{R}^{32}$. This allows the GRU to process concatenations like $[\mathbf{h}_q \| \mathbf{h}_l] \in \mathbb{R}^{64}$ meaningfully.

### Real-Life Analogy: The Translation

```
SCENARIO: Three doctors from different countries examine the same patient

Doctor 1 (Path):     "தீவிரமான காய்ச்சல், இடைவிட்டு வரும்"     (Tamil)
Doctor 2 (Link):     "Lien saturé, politique FIFO"               (French)  
Doctor 3 (Queue):    "バッファ使用率高い、優先度2"                  (Japanese)

They CANNOT collaborate because they speak different languages.

EMBEDDING = TRANSLATOR

After embedding, all three speak the same "32-dimensional medical language":
Doctor 1 → [0.2, 0.8, 0.1, -0.3, ...]  (32 numbers)
Doctor 2 → [0.5, 0.3, 0.9, 0.1, ...]   (32 numbers)
Doctor 3 → [0.1, 0.6, 0.4, -0.5, ...]  (32 numbers)

Now they can exchange messages and collaborate!
```

---

## 6. The Embedding MLP: Step-by-Step Computation

The path embedding is a 2-layer neural network. From [delay_model.py lines 36-40](file:///home/abaragithan/gnn-sdn-research/delay_model.py#L36-L40):

```python
self.path_embedding = tf.keras.Sequential([
    Input(shape=17),           # x_p ∈ R^17
    Dense(32, activation=relu), # Layer 1: R^17 → R^32
    Dense(32, activation=relu)  # Layer 2: R^32 → R^32
])
```

### Layer 1: $\mathbb{R}^{17} \to \mathbb{R}^{32}$

```
Input: x_p = [1.296, 1.236, 0, 0, 0, 1, 0, 0, 0, 1.222, 0.299, 0.708, 0.539, 0.144, 0.144, 0, 0]

        W₁ (learned weight matrix)              b₁ (learned bias)
        ┌                        ┐               ┌     ┐
        │ w₁₁  w₁₂  ...  w₁,₁₇ │               │ b₁  │
        │ w₂₁  w₂₂  ...  w₂,₁₇ │               │ b₂  │
        │  :    :    ...   :     │    32×17       │ :   │   32×1
        │ w₃₂,₁ ...      w₃₂,₁₇│  = 544 params  │ b₃₂ │
        └                        ┘               └     ┘

Computation:
  z₁ = W₁ · x_p + b₁         (linear transform)
  a₁ = ReLU(z₁)               (activation: set negatives to 0)
  
Example for ONE output neuron (neuron j=1):
  z₁[1] = w₁₁·(1.296) + w₁₂·(1.236) + w₁₃·(0) + ... + w₁,₁₇·(0) + b₁
        = (some learned weighted sum)
        = e.g., -0.34
  
  a₁[1] = ReLU(-0.34) = 0.0  (negative → zeroed out!)

Example for neuron j=2:
  z₁[2] = w₂₁·(1.296) + w₂₂·(1.236) + ... + b₂
        = e.g., 0.72
  
  a₁[2] = ReLU(0.72) = 0.72  (positive → kept!)

After Layer 1: a₁ ∈ R^32 (intermediate representation)
```

### Layer 2: $\mathbb{R}^{32} \to \mathbb{R}^{32}$

```
Input: a₁ (32-dim output from Layer 1)

  z₂ = W₂ · a₁ + b₂          (W₂ is 32×32 = 1024 params)
  h_p^(0) = ReLU(z₂)          (final embedding!)

Output: h_p^(0) ∈ R^32
```

### The Complete Transformation

```
x_p ∈ R^17                     h_p^(0) ∈ R^32
[1.296, 1.236, 0, 0, 0,        [0.2, 0.8, 0.1, 0.0, 0.45,
 1, 0, 0, 0, 1.222,    ───►     0.0, 0.33, 0.91, 0.0, 0.67,
 0.299, 0.708, 0.539,            0.12, 0.0, 0.78, 0.55, 0.0,
 0.144, 0.144, 0, 0]             0.23, 0.89, 0.0, 0.41, 0.0,
                                  0.0, 0.56, 0.34, 0.0, 0.88,
 17 interpretable features        0.15, 0.0, 0.62, 0.0, 0.44,
                                  0.27, 0.73]

                                 32 learned abstract features
```

> [!IMPORTANT]
> **The embedding is NOT designed by humans.** The 32 dimensions have no predefined meaning. During training, the network learns weight matrices $W_1, W_2$ such that similar traffic patterns produce **nearby** embeddings, and different patterns produce **distant** embeddings. Dimension 5 might learn to encode "burstiness" while dimension 12 might encode "heaviness" — but this emerges from training, not from human design.

---

## 7. The Same Process for Links and Queues

### Link Embedding (5-dim → 32-dim)

```
Raw link data for link 0→1:
  capacity = 10000 bits/s (from G.edges[0,1]['bandwidth'])
  policy   = "FIFO"       (from G.nodes[0]['schedulingPolicy'])

Step 1 — Compute load dynamically:
  load = (sum of traffic from ALL paths using this link) / capacity
       = (2500 + 1800 + 900) / 10000 = 0.52   (52% utilized)

Step 2 — One-hot encode policy:
  FIFO → [0, 0, 0, 1]  (index 3 in ['WFQ','SP','DRR','FIFO'])

Step 3 — Concatenate:
  x_l = [0.52, 0, 0, 0, 1]    (5 dimensions)
         load   ← policy →

Step 4 — Embed via MLP:
  h_l^(0) = LinkMLP(x_l) ∈ R^32
           = [0.5, 0.3, 0.9, ...]
```

### Queue Embedding (5-dim → 32-dim)

```
Raw queue data for queue at link 0→1, priority 0:
  queue_size = 32000 bytes (from G.nodes[0]['bufferSizes'])
  priority   = 0
  weight     = 0.33        (scheduling weight, normalized)

Step 1 — Z-score normalize queue_size:
  (32000 - 30259.1) / 21410.1 = 0.081

Step 2 — One-hot encode priority:
  priority=0 → [1, 0, 0]  (with max_num_queues=3)

Step 3 — Concatenate:
  x_q = [0.081, 1, 0, 0, 0.33]    (5 dimensions)
         q_size  priority   weight

Step 4 — Embed via MLP:
  h_q^(0) = QueueMLP(x_q) ∈ R^32
           = [0.1, 0.6, 0.4, ...]
```

---

## 8. The Complete Picture: Your 3-Node Example

Now you can fully understand the example from your document. Here is the **complete flow** for the path A→B→C:

```
════════════════════════════════════════════════════════════════════════
                     THE COMPLETE PIPELINE
════════════════════════════════════════════════════════════════════════

STAGE 1: NETWORK SIMULATOR (OMNeT++)
═══════════════════════════════════════

Physical Network:         Router A ──(10Gbps)──► Router B ──(10Gbps)──► Router C

Simulator runs for 60 simulated seconds...
Measures: avg delay for flow A→C = 3.42 ms

                              │
                              ▼

STAGE 2: DATA EXTRACTION (data_generator.py → network_to_hypergraph)
════════════════════════════════════════════════════════════════════════

Creates a NetworkX directed graph with three types of nodes:

  Path node "p_0_2_0":                    Link nodes:              Queue nodes:
  ┌─────────────────────────┐     ┌──────────────────┐    ┌───────────────────────┐
  │ traffic = 2500          │     │ l_0_1:           │    │ q_0_1_0:              │
  │ packets = 2.5           │     │   capacity=10000 │    │   queue_size=32000    │
  │ model = 3 (ON-OFF)      │     │   policy=FIFO    │    │   priority=0          │
  │ eq_lambda = 2400        │     │                  │    │   weight=0.33         │
  │ avg_pkts_lambda = 1.2   │     │ l_1_2:           │    │                       │
  │ exp_max_factor = 10     │     │   capacity=10000 │    │ q_1_2_0:              │
  │ pkts_lambda_on = 1.8    │     │   policy=FIFO    │    │   queue_size=32000    │
  │ avg_t_off = 2.0         │     └──────────────────┘    │   priority=0          │
  │ avg_t_on = 2.0          │                             │   weight=0.33         │
  │ ar_a = 0.0              │                             └───────────────────────┘
  │ sigma = 0.0             │
  │ delay = 0.00342 (LABEL) │
  └─────────────────────────┘

  Edges created:
    p_0_2_0 ↔ l_0_1,  p_0_2_0 ↔ l_1_2    (path traverses links)
    p_0_2_0 ↔ q_0_1_0, p_0_2_0 ↔ q_1_2_0 (path traverses queues)
    q_0_1_0 → l_0_1,  q_1_2_0 → l_1_2    (queues belong to links)

                              │
                              ▼

STAGE 3: Z-SCORE NORMALIZATION (inside model.call())
════════════════════════════════════════════════════════

  traffic:  (2500 - 1385.41) / 859.81  = +1.296
  packets:  (2.5 - 1.40) / 0.89       = +1.236
  model:    one_hot(3, depth=7)        = [0,0,0,1,0,0,0]
  ...10 scalar features normalized...

                              │
                              ▼

STAGE 4: CONCATENATION → RAW FEATURE VECTOR
════════════════════════════════════════════

  x_p = [1.296, 1.236, 0,0,0,1,0,0,0, 1.222, 0.299, 0.708, 0.539, 0.144, 0.144, 0, 0]
         ├─────────────────── 17 dimensions ──────────────────────────────────────────┤

                              │
                              ▼

STAGE 5: MLP EMBEDDING → INITIAL HIDDEN STATE
══════════════════════════════════════════════

  h_A^(0) = PathMLP(x_A) = [0.2, 0.8, 0.1, ...]   ← 32 dimensions
  h_B^(0) = LinkMLP(x_B) = [0.5, 0.3, 0.9, ...]   ← 32 dimensions  
  h_C^(0) = QueueMLP(x_C)= [0.1, 0.6, 0.4, ...]   ← 32 dimensions

  These ARE the initial states from your 3-node example!
  NOW ready for message passing.

                              │
                              ▼

STAGE 6: MESSAGE PASSING (8 iterations)
═══════════════════════════════════════

  Entities exchange information through the graph structure.
  After 8 rounds, each entity "knows" about the entire network.

                              │
                              ▼

STAGE 7: READOUT → PREDICTION
═════════════════════════════

  Predicted delay ≈ 0.00342 seconds
  Compared against ground truth from simulator.
  Loss computed, backpropagated, weights updated.
```

---

## 9. What the Embedding "Means" — An Intuitive Explanation

After training, the embedding space develops **semantic structure**. Similar flows cluster together:

```
                    EMBEDDING SPACE (32-dim, visualized as 2D)
                    
                         High Traffic
                              ▲
                              │
                    ●         │         ●
                  (heavy      │      (heavy
                   Poisson)   │      ON-OFF)
                              │
         ─────────────────────┼─────────────────── Bursty →
                              │
                    ○         │         ○
                  (light      │      (light
                   Poisson)   │      ON-OFF)
                              │
                              ▼
                         Low Traffic


● Flow with traffic=2500, model=ON-OFF → embedded near top-right
○ Flow with traffic=200, model=Poisson → embedded near bottom-left

Flows that BEHAVE SIMILARLY in the network (cause similar congestion)
are placed NEAR EACH OTHER in embedding space.
```

> [!TIP]
> **The key insight**: The embedding doesn't just compress the 17 numbers into 32 numbers. It **reorganizes** them so that the GNN can easily extract what matters for predicting delay. Two flows with different raw numbers but similar network impact will have similar embeddings.

---

## 10. Summary: The Translation Chain

```
Real Network          Simulator         Raw Numbers        Normalized        Embedding
(physical cables,  → (virtual       →  (traffic=2500,  →  (z-scored,     → h_p^(0) ∈ R^32
 routers, users)     experiment)       packets=2.5,       one-hot          [0.2, 0.8, ...]
                                        model=3, ...)      encoded)
                                                           x_p ∈ R^17

   "The real          "The               "The raw           "Cleaned         "The language
    world"             experiment"         measurements"      data"            the GNN speaks"
```

Each stage answers a question:

| Stage | Question Answered |
|:--|:--|
| Simulation | "What happens in this network scenario?" |
| Data Extraction | "What are the relevant numbers?" |
| Normalization | "How do we make all features comparable?" |
| Embedding | "How do we represent this in a form the GNN can reason about?" |
| Message Passing | "How does information from other entities affect this one?" |
| Readout | "Given all this context, what's the predicted delay?" |
