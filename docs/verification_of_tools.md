📦 Core Tools Verification
# 1. Check OS and Kernel (for networking features)

░▒▓ ~  python3 --version                                                ✔ ▓▒░
Python 3.14.2
░▒▓ ~  pip3 --version                                                   ✔ ▓▒░
pip 25.1.1 from /usr/lib/python3.14/site-packages/pip (python 3.14)
░▒▓ ~  python3 -c "import sys; print(f'Python path: {sys.executable}')"
Python path: /usr/bin/python3
░▒▓ ~ 


🐳 Docker & Containerization Verification
# 3. Docker Engine

░▒▓ ~  docker --version                                                 ✔ ▓▒░
Docker version 29.0.4, build 1.fc43
░▒▓ ~  sudo docker info | grep -E "Server Version|Containers|Images"
 Containers: 7
 Images: 2
 Server Version: 29.0.4
░▒▓ ~                                                              ✔  4s ▓▒░

# 4. Docker Compose (plugin version)

░▒▓ ~  docker-compose --version                                         ✔ ▓▒░
Docker Compose version 2.40.3
░▒▓ ~ 

🌐 SDN Network Emulation Verification
# 6. Mininet

░▒▓ ~  sudo mn --version                                                 ✔ ▓▒░
2.3.0

# 7. Open vSwitch (OVS)

░▒▓ ~  ovs-vsctl --version                                               ✔ ▓▒░
ovs-vsctl (Open vSwitch) 3.6.0-2.fc43
DB Schema 8.8.0
░▒▓ ~  which ovs-ofctl                                                   ✔ ▓▒░
/usr/bin/ovs-ofctl
░▒▓ ~ 

# 8. RYU Controller



# need to install( version compatibility issue )





