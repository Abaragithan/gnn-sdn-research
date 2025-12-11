---
title: "Research Infrastructure Documentation: Lightweight GNN for QoS-Aware SDN Routing"
author: "Abaragithan"
date: "2025-12-11"
---

# 📋 Document Overview
This document details the software infrastructure installed to support the research: **"Lightweight Graph Neural Networks for Real-Time QoS-Aware Routing in Software-Defined Networks"**. All tools were installed on a **Fedora Linux system** to ensure native performance, direct kernel access for networking, and optimal compatibility with networking and machine learning frameworks.

# 🏗️ 1. Core Operating System & Virtualization

## 1.1 Fedora Linux
- **Installation Command:** Native OS installation
- **Verification:** `cat /etc/fedora-release`
- **Purpose in Research:**
  - Provides native Linux environment essential for SDN tools like Mininet.
  - Eliminates virtualization overhead ensuring accurate performance measurement.
- **Research Contribution:**
  - Forms the foundational layer enabling direct hardware access.

## 1.2 Docker & Docker Compose
- **Installation:** `sudo dnf install docker docker-compose`
- **Verification:** `docker --version`, `docker compose version`
- **Purpose in Research:** Containerization platform for creating reproducible, isolated environments.
- **Research Contribution:**
  - Enables packaging of GNN models and SDN controllers into portable containers.
  - Simulates edge-device deployment.
  - Facilitates reproducible experiments.

# 🌐 2. SDN Network Emulation & Control Plane

## 2.1 Mininet
- **Installation:** `git clone https://github.com/mininet/mininet && sudo ./util/install.sh -n`
- **Verification:** `sudo mn --version`, `sudo mn --test pingall`
- **Purpose in Research:** Network emulator for virtual networks of hosts, switches, and links.
- **Research Contribution:**
  - Primary testbed for routing algorithm evaluation.
  - Simulates custom network topologies.
  - Generates dynamic traffic patterns.

## 2.2 RYU SDN Controller
- **Installation:** `pip3 install ryu`
- **Verification:** `ryu-manager --version`
- **Purpose in Research:** Python-based SDN controller implementing OpenFlow.
- **Research Contribution:**
  - Hosts GNN-based routing intelligence.
  - Collects real-time network state.
  - Translates GNN decisions into OpenFlow rules.

## 2.3 Open vSwitch (OVS)
- **Installation:** Installed automatically with Mininet (`-n` flag)
- **Verification:** `ovs-vsctl --version`
- **Purpose in Research:** Virtual switch implementing OpenFlow protocol.
- **Research Contribution:**
  - Programmable data plane.
  - Provides flow statistics.
  - Supports VLANs, QoS, and tunneling.

# 🧠 3. Machine Learning & Graph Neural Networks Stack

## 3.1 PyTorch
- **Installation:** `pip3 install torch torchvision torchaudio`
- **Verification:** `python3 -c "import torch; print(torch.__version__)"`
- **Purpose in Research:** Deep learning framework with GPU acceleration.
- **Research Contribution:**
  - Implements lightweight GNN architectures.
  - Provides optimization algorithms.
  - Supports model compression and edge deployment.

## 3.2 PyTorch Geometric (PyG)
- **Installation:**
```bash
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip3 install torch-geometric
```
- **Verification:** `python3 -c "import torch_geometric; print(torch_geometric.__version__)"`
- **Purpose in Research:** Deep learning on graph-structured data.
- **Research Contribution:**
  - Models network topology as graphs.
  - Provides GNN layers (GCN, GAT, GraphSAGE).

## 3.3 Stable-Baselines3
- **Installation:** `pip3 install stable-baselines3`
- **Verification:** `python3 -c "import stable_baselines3; print('SB3 imported successfully')"`
- **Purpose in Research:** Reinforcement learning algorithms.
- **Research Contribution:**
  - Benchmark DRL routing agents.
  - Provides baseline for GNN comparison.

## 3.4 Scientific Python Stack
- **Installation:** `pip3 install numpy pandas matplotlib scikit-learn jupyter`
- **Verification:** Individual import tests
- **Purpose in Research:** Data manipulation, visualization, and analysis.
- **Research Contribution:**
  - NumPy/Pandas: Process topology and traffic data.
  - Matplotlib/Seaborn: Visualize performance results.
  - Scikit-learn: ML utilities and evaluation.
  - Jupyter: Interactive experimentation.

# 📊 4. Monitoring, Measurement & Traffic Generation

## 4.1 iPerf3 & D-ITG
- **Installation:** `sudo dnf install iperf3` + compile D-ITG from source
- **Verification:** `iperf3 --version`, `which ditg`
- **Purpose in Research:** Generate controlled network traffic.
- **Research Contribution:**
  - Create dynamic traffic loads.
  - Measure bandwidth.
  - Stress test routing algorithms.

## 4.2 Network Analysis Tools
- **Installation:** `sudo dnf install wireshark tcpdump net-tools`
- **Verification:** `tcpdump --version`, `which wireshark`
- **Purpose in Research:** Packet capture and analysis.
- **Research Contribution:**
  - Debug OpenFlow messages.
  - Capture traffic for offline analysis.

## 4.3 System Monitoring Libraries
- **Installation:** `pip3 install psutil scapy`
- **Verification:** Python import tests
- **Purpose in Research:** Programmatic system and network monitoring.
- **Research Contribution:**
  - Monitor CPU, memory, disk I/O.
  - Collect network interface stats.

# 🔄 5. Integrated Research Pipeline Architecture

## 5.1 Experimental Workflow
```
Network Topology (Mininet)
    → Traffic Generation (iPerf3/D-ITG)
    → Data Collection (psutil, OVS stats)
    → Graph Construction (PyG Data objects)
    → GNN Training/Inference (PyTorch+PyG)
    → Routing Decisions
    → Control Implementation (RYU)
    → Data Plane Configuration (OVS)
    → Performance Measurement
    → Analysis & Iteration
```

## 5.2 Tool Interaction Mapping
| Research Phase | Primary Tools | Supporting Tools | Output |
|----------------|---------------|-----------------|--------|
| Environment Setup | Mininet, Docker | Fedora, OVS | Virtual network testbed |
| Data Generation | iPerf3, D-ITG | Python scripts | Traffic matrices, QoS metrics |
| Model Development | PyTorch, PyG | NumPy, Jupyter | Lightweight GNN architecture |
| Model Compression | PyTorch utilities | Custom scripts | Pruned/quantized model |
| Controller Integration | RYU, Python | OpenFlow library | Intelligent SDN controller |
| Evaluation | Mininet, psutil | Matplotlib, pandas | Performance comparisons |
| Deployment Testing | Docker, Fedora | System monitoring | Edge deployment metrics |

# 🎯 6. Alignment with Research Objectives

- **Objective 1:** Design lightweight GNN for real-time routing
  - Enabled by: PyTorch, PyG, Fedora
  - Validation: Measure inference latency on RYU with Mininet loads

- **Objective 2:** Apply model compression techniques
  - Enabled by: PyTorch pruning/quantization APIs, Docker
  - Validation: Compare model size/accuracy before/after compression

- **Objective 3:** Evaluate on edge-level SDN controllers
  - Enabled by: RYU, Mininet, psutil
  - Validation: Deploy on Docker containers, measure QoS vs. resources

- **Objective 4:** Analyze trade-offs
  - Enabled by: Complete stack integration
  - Validation: Experiments measuring QoS vs. CPU/memory usage

# 📁 7. Directory Structure for Research Organization
```
~/sdn-gnn-research/
├── data/
│   ├── topologies/
│   ├── traffic_patterns/
│   └── results/
├── models/
│   ├── baseline_gnn.py
│   ├── lightweight_gnn.py
│   └── trained_models/
├── sdn_controller/
│   ├── gnn_routing_app.py
│   ├── monitoring.py
│   └── ryu_requirements.txt
├── experiments/
│   ├── train_gnn.py
│   ├── evaluate_routing.py
│   └── resource_benchmark.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_development.ipynb
│   └── result_visualization.ipynb
└── docker/
    ├── edge-controller.Dockerfile
    ├── training-environment.Dockerfile
    └── docker-compose.experiment.yml
```

# ✅ 8. Verification Checklist
```bash
# verify_stack.sh
#!/bin/bash

echo "Research Stack Verification - $(date)"
echo "======================================"

check_tool() {
    if $1 $2 >/dev/null 2>&1; then
        echo "✓ $3"
    else
        echo "✗ $3"
    fi
}

check_tool "python3 --version" "" "Python 3"
check_tool "docker --version" "" "Docker"
check_tool "sudo mn --version" "" "Mininet"
check_tool "ovs-vsctl --version" "" "Open vSwitch"
check_tool "ryu-manager --version" "" "RYU Controller"
check_tool "python3 -c \"import torch\"" "" "PyTorch"
check_tool "python3 -c \"import torch_geometric\"" "" "PyTorch Geometric"
check_tool "iperf3 --version" "" "iPerf3"

echo "======================================"
```

# 📝 9. Maintenance & Updates
- **System packages:** `sudo dnf update`
- **Python packages:**
```bash
pip3 list --outdated | grep -v "^-e" | cut -d = -f 1 | xargs -n1 pip3 install -U
```
- **Research code:** `cd ~/mininet && git pull`

- **Environment Recreation:**
  1. Fedora minimal installation
  2. `dnf install` commands documented above
  3. `pip3 install` from requirements files
  4. Git clone of Mininet with specific tag

# 🔗 10. Key References & Resources
- Mininet Documentation: http://mininet.org
- RYU Documentation: https://ryu-sdn.org
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io
- Open vSwitch Documentation: https://docs.openvswitch.org
- GEANT2 Topology Data: https://www.geant.org/Networks
- Internet2 Topology Data: https://www.internet2.edu

