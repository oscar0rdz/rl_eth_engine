#!/usr/bin/env bash
# bootstrap_remote_vm.sh
# This script sets up the remote VM environment (Ubuntu/Debian recommended).
set -e

echo "--- 1. Installing System Dependencies ---"
sudo apt-get update
sudo apt-get install -y git python3-pip curl ca-certificates gnupg lsb-release

echo "--- 2. Installing Docker ---"
if ! command -v docker &> /dev/null; then
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "Docker installed. Note: You might need to re-login for group changes."
else
    echo "Docker already installed."
fi

echo "--- 3. Installing NVIDIA Drivers & Container Toolkit ---"
# Note: This assumes Ubuntu. User may need to manual install drivers if this fails.
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get install -y nvidia-driver-535 # Example version
fi

if ! command -v nvidia-ctk &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --plugin=docker
    sudo systemctl restart docker
fi

echo "--- 4. Validating GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[WARNING] GPU not detected via nvidia-smi. Check drivers."
fi

echo "--- 5. Preparing Project Folders ---"
mkdir -p ~/rl_eth_engine/data/raw
mkdir -p ~/rl_eth_engine/data/processed
mkdir -p ~/rl_eth_engine/checkpoints

echo "Bootstrap completed. Ready for 'git clone' and 'run_v3_pipeline.sh'."
