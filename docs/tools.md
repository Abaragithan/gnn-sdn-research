## 🛠️ Tools Used for Research 

### Operating system
Fedora 43

### Python and development tools
python3 
python3-pip 
python3-devel 
gcc-c++ 
make 
cmake 
git

### Network tools

wireshark 
tcpdump 
net-tools 
iproute

### Docker



```bash
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
### Log out and back in

Step 3: Install SDN Tools
A. Mininet (Network Emulator)
bash

### Clone and install Mininet
git clone https://github.com/mininet/mininet.git
cd mininet
git checkout -b 2.3.1d6 2.3.1d6  # Stable version
./util/install.sh -n  # Install Mininet + dependencies

# Test Mininet
sudo mn --test pingall

B. RYU Controller (SDN Controller)
bash

# Install RYU
pip3 install ryu

# Or from source (recommended for development)
git clone https://github.com/faucetsdn/ryu.git
cd ryu
pip3 install -r tools/pip-requires -r tools/optional-requires
pip3 install .

C. ONOS (Alternative SDN Controller - Optional)
bash

# ONOS runs in Docker (easy!)
docker pull onosproject/onos:latest
# Run later when needed

Step 4: Install ML/Deep Learning Stack
A. PyTorch with CUDA (if you have NVIDIA GPU)
bash

# Check GPU
lspci | grep -i nvidia

# Install PyTorch (choose based on your setup)
# Option 1: With CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Option 2: CPU only
pip3 install torch torchvision torchaudio

B. PyTorch Geometric (PyG) for GNNs
bash

# Install PyG dependencies
sudo dnf install openblas-devel -y
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip3 install torch-geometric

# Additional PyG libraries you might need
pip3 install torch-geometric-temporal

Step 5: Install Research-Specific Python Packages
bash

pip3 install numpy pandas scikit-learn matplotlib seaborn jupyter notebook
pip3 install networkx tensorboard scipy tqdm

# For your research specifically
pip3 install gym  # For reinforcement learning environments
pip3 install stable-baselines3  # For DRL baselines
pip3 install opencv-python  # For any visualization

Step 6: Dataset Preparation Tools
A. GEANT2 and Internet2 Topologies
bash

# Create project directory
mkdir ~/sdn-research
cd ~/sdn-research

# For topology data, you'll typically need:
git clone https://github.com/nsg-ethz/p4-utils.git  # Has topology examples
# Or download from official sites:
# GEANT2: https://www.geant.org/Networks
# Internet2: https://www.internet2.edu/

B. Traffic Generation Tools
bash

# iPerf3 for traffic generation
sudo dnf install iperf3 -y

# D-ITG (Distributed Internet Traffic Generator)
git clone https://github.com/luigirizzo/ditg.git
cd ditg
./configure
make
sudo make install

Step 7: Development Environment Setup
A. VS Code (Recommended)
bash

sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc
sudo sh -c 'echo -e "[code]\nname=Visual Studio Code\nbaseurl=https://packages.microsoft.com/yumrepos/vscode\nenabled=1\ngpgcheck=1\ngpgkey=https://packages.microsoft.com/keys/microsoft.asc" > /etc/yum.repos.d/vscode.repo'
sudo dnf install code -y

B. Useful VS Code Extensions:

    Python

    Docker

    Jupyter

    GitLens

    Remote - SSH (if using remote servers)

Step 8: Test Your Setup

Create a test file test_setup.py:
python

import torch
import torch_geometric
import ryu
import mininet
import numpy as np

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ PyTorch Geometric version:", torch_geometric.__version__)
print("✓ NumPy version:", np.__version__)

Step 9: Docker Compose for Multi-Container SDN Tests
bash

# Install Docker Compose (if not already)
sudo dnf install docker-compose -y

# Create a sample docker-compose.yml for SDN testing
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  mininet:
    image: mininet/mininet:latest
    container_name: mininet
    network_mode: "host"
    privileged: true
    volumes:
      - ./topologies:/topologies
  
  ryu-controller:
    image: osrg/ryu:latest
    container_name: ryu-controller
    ports:
      - "8080:8080"  # Web GUI
      - "6633:6633"  # OpenFlow
    volumes:
      - ./ryu-apps:/ryu-apps
  
  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
EOF

📊 Complete Architecture for Your Research
text

Your Fedora System
├── Docker (for containerized components)
├── Mininet (network emulation)
├── RYU Controller (SDN control plane)
├── PyTorch + PyG (GNN models)
├── Python ML stack (training/evaluation)
└── Jupyter Notebook (experimentation)

🚀 Quick Start Commands
bash

# 1. Start a simple SDN test
sudo mn --controller=remote,ip=127.0.0.1 --topo=linear,3

# 2. Run RYU controller in another terminal
ryu-manager ryu.app.simple_switch_13

# 3. Test your GNN model
python3 your_gnn_model.py

# 4. Monitor resources
htop  # System monitoring
nvidia-smi  # GPU monitoring (if available)

💡 Pro Tips for Your Research:

    Use Git from Day 1: Track all experiments

    Containerize Everything: Use Docker for reproducibility

    Leverage Jupyter Notebooks: For exploratory analysis

    Set up TensorBoard: For experiment tracking

    Consider using MLflow: For model management

    Use VS Code Remote SSH: If you need more powerful servers

❓ If You MUST Use Windows:

If you absolutely need Windows for other reasons, use:

    WSL2 with Ubuntu 22.04

    Docker Desktop with WSL2 backend

    VS Code with Remote WSL extension

But really, stick with Fedora - you'll save countless hours debugging OS-specific issues.

Your research will progress much faster on Fedora! The native Linux environment is exactly what SDN and networking researchers use.

