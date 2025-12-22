# Lightweight GNN-based SDN Routing in GEANT2

## Setup
1. Requires Python 3.10 and Mininet.
2. Create environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`

## Running
1. Start Controller: `osken-manager src/controller/gnn_routing_app.py`
2. Start Topology: `sudo python3 experiments/geant2_topo.py`
